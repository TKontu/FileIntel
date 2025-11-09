"""
Unified LLM Provider - Single configurable provider for multiple LLM APIs.

Replaces separate OpenAI and Anthropic providers with a single, simplified implementation.
Removes async patterns, complex connection pooling, and circuit breakers.
"""

import logging
from typing import Optional, Dict, Any
from enum import Enum
from abc import ABC, abstractmethod
from pathlib import Path
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    Retrying,
)

from .base import LLMResponse, validate_response
from ..storage.postgresql_storage import PostgreSQLStorage
from ..core.config import Settings
from ..prompt_management import AnswerFormatManager, load_prompt_template

logger = logging.getLogger(__name__)


class LLMProviderType(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""

    pass


class LLMAPIStrategy(ABC):
    """Abstract base class for LLM API strategies."""

    @abstractmethod
    def call_api(
        self,
        http_client: httpx.Client,
        api_key: str,
        base_url: str,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> LLMResponse:
        """Call the specific LLM API and return a response."""
        pass


class OpenAIStrategy(LLMAPIStrategy):
    """Strategy for OpenAI API calls."""

    def call_api(
        self,
        http_client: httpx.Client,
        api_key: str,
        base_url: str,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> LLMResponse:
        """Call OpenAI API."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        response = http_client.post(
            f"{base_url}/chat/completions", json=payload, headers=headers
        )
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            model=model,
            provider="openai",
            usage=usage,
            metadata={"response_id": data.get("id")},
        )


class AnthropicStrategy(LLMAPIStrategy):
    """Strategy for Anthropic API calls."""

    def call_api(
        self,
        http_client: httpx.Client,
        api_key: str,
        base_url: str,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> LLMResponse:
        """Call Anthropic API."""
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }

        response = http_client.post(
            f"{base_url}/messages", json=payload, headers=headers
        )
        response.raise_for_status()

        data = response.json()
        content = data["content"][0]["text"] if data.get("content") else ""
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            model=model,
            provider="anthropic",
            usage=usage,
            metadata={"response_id": data.get("id")},
        )


class LLMAPIError(LLMProviderError):
    """Raised when LLM API returns an error."""

    pass


class LLMConfigurationError(LLMProviderError):
    """Raised when LLM provider configuration is invalid."""

    pass


class UnifiedLLMProvider:
    """
    Unified LLM provider supporting multiple APIs through configuration.

    Eliminates code duplication between OpenAI and Anthropic providers.
    Uses synchronous HTTP requests compatible with Celery task architecture.
    """

    def __init__(self, config: Settings, storage: PostgreSQLStorage = None):
        self.config = config
        self.storage = storage
        self.provider_type = LLMProviderType(config.llm.provider)

        # Initialize HTTP client with configurable timeout (default 15 minutes for high queue depths)
        self.http_client = httpx.Client(
            timeout=httpx.Timeout(float(config.llm.http_timeout_seconds)),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

        # Setup provider-specific configuration and strategy
        self._setup_provider_config()
        self._setup_strategy()

        # Configure retry strategy from config (replaces hardcoded @retry decorator)
        self.retry_strategy = Retrying(
            wait=wait_exponential(
                multiplier=1,
                min=config.llm.retry_backoff_min,
                max=config.llm.retry_backoff_max
            ),
            stop=stop_after_attempt(config.llm.max_retries),
            retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
        )

        # Get prompts directory from environment or use relative path (shared across initializations)
        import os
        prompts_dir = os.getenv('FILEINTEL_PROMPTS_DIR')

        # Initialize answer format manager with robust path resolution
        try:
            if prompts_dir:
                formats_dir = Path(prompts_dir) / "examples"
            else:
                # Fallback to relative path
                project_root = Path(__file__).parent.parent.parent.parent
                formats_dir = project_root / "prompts" / "examples"

            if formats_dir.exists():
                self.format_manager = AnswerFormatManager(formats_dir)
                logger.info(f"AnswerFormatManager initialized: {formats_dir}")
            else:
                logger.warning(f"Format templates directory not found: {formats_dir}")
                logger.warning("Answer format templates unavailable - using default format only")
                self.format_manager = None
        except Exception as e:
            logger.error(f"Failed to initialize AnswerFormatManager: {e}")
            self.format_manager = None

        # Load vector RAG templates with robust path resolution
        try:
            if prompts_dir:
                templates_dir = Path(prompts_dir) / "templates" / "vector_rag"
            else:
                project_root = Path(__file__).parent.parent.parent.parent
                templates_dir = project_root / "prompts" / "templates" / "vector_rag"

            if templates_dir.exists():
                self._load_vector_rag_templates(templates_dir)
            else:
                logger.warning(f"Vector RAG templates not found: {templates_dir}")
                logger.warning("Using fallback hardcoded prompts")
                # Set to None to trigger fallback
                self.base_instruction_template = None
                self.citation_rules_template = None
                self.query_type_templates = None
        except Exception as e:
            logger.error(f"Failed to load vector RAG templates: {e}")
            self.base_instruction_template = None
            self.citation_rules_template = None
            self.query_type_templates = None

        logger.info(
            f"Unified LLM Provider initialized for {self.provider_type.value} "
            f"(timeout={config.llm.http_timeout_seconds}s, retries={config.llm.max_retries}, "
            f"backoff={config.llm.retry_backoff_min}-{config.llm.retry_backoff_max}s)"
        )

    def _setup_strategy(self) -> None:
        """Initialize the appropriate API strategy."""
        if self.provider_type == LLMProviderType.OPENAI:
            self.api_strategy = OpenAIStrategy()
        elif self.provider_type == LLMProviderType.ANTHROPIC:
            self.api_strategy = AnthropicStrategy()
        else:
            raise LLMConfigurationError(
                f"Unsupported provider type: {self.provider_type}"
            )

    def _setup_provider_config(self) -> None:
        """Setup provider-specific configuration and validation."""
        if self.provider_type == LLMProviderType.OPENAI:
            self.api_key = self.config.llm.openai.api_key or self.config.llm.api_key
            self.base_url = (
                self.config.llm.openai.base_url
                or self.config.llm.base_url
                or "https://api.openai.com/v1"
            )
            self.default_model = self.config.llm.model or "gemma3-12b-awq"

        elif self.provider_type == LLMProviderType.ANTHROPIC:
            self.api_key = self.config.llm.anthropic.api_key
            self.base_url = "https://api.anthropic.com/v1"
            self.default_model = self.config.llm.model or "claude-3-opus-20240229"

        else:
            raise LLMConfigurationError(
                f"Unsupported provider type: {self.provider_type}"
            )

        if not self.api_key or self.api_key == "dummy-key":
            logger.warning(
                f"No valid API key configured for {self.provider_type.value}"
            )

    def _load_vector_rag_templates(self, templates_dir: Path) -> None:
        """Load vector RAG prompt templates."""
        try:
            self.base_instruction_template = load_prompt_template(
                str(templates_dir / "base_instruction.md")
            )
            self.citation_rules_template = load_prompt_template(
                str(templates_dir / "citation_rules.md")
            )

            # Load query type specific instructions
            query_types_dir = templates_dir / "query_type_instructions"
            self.query_type_templates = {
                "factual": load_prompt_template(str(query_types_dir / "factual.md")),
                "analytical": load_prompt_template(str(query_types_dir / "analytical.md")),
                "summarization": load_prompt_template(str(query_types_dir / "summarization.md")),
                "comparison": load_prompt_template(str(query_types_dir / "comparison.md")),
                "general": load_prompt_template(str(query_types_dir / "general.md")),
            }

            logger.info(f"Loaded vector RAG templates from {templates_dir}")
        except Exception as e:
            logger.warning(f"Failed to load vector RAG templates: {e}. Using fallback hardcoded prompts.")
            # Set to None so _build_rag_prompt can fall back to hardcoded prompts
            self.base_instruction_template = None
            self.citation_rules_template = None
            self.query_type_templates = None

    def generate_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.1,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate response from LLM API with configurable retry logic.

        Args:
            prompt: Input text prompt
            model: Model name (uses default if not specified)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with generated text and metadata

        Raises:
            LLMAPIError: When API returns an error
            LLMConfigurationError: When configuration is invalid
        """
        model = model or self.default_model
        max_tokens = max_tokens or self.config.llm.max_tokens

        # Check for cached response
        cache_key = self._generate_cache_key(
            prompt, model, max_tokens, temperature, kwargs
        )
        if self.storage:
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                logger.debug(f"Cache hit for {self.provider_type.value} request")
                return cached_response

        # Generate new response using strategy pattern with configurable retry
        try:
            response = self.retry_strategy(
                self.api_strategy.call_api,
                self.http_client,
                self.api_key,
                self.base_url,
                prompt,
                model,
                max_tokens,
                temperature,
                **kwargs,
            )

            # Validate and cache response
            validated_response = validate_response(response)
            if self.storage:
                self._cache_response(cache_key, validated_response)

            return validated_response

        except httpx.HTTPStatusError as e:
            logger.error(
                f"{self.provider_type.value} API error: {e.response.status_code} - {e.response.text}"
            )
            raise LLMAPIError(f"API request failed: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"{self.provider_type.value} request error: {e}")
            raise LLMAPIError(f"Request failed: {e}")

    def _generate_cache_key(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        kwargs: Dict[str, Any],
    ) -> str:
        """Generate simple cache key using string concatenation instead of SHA256 hashing."""
        # Simplified cache key - no complex hashing needed
        kwargs_str = "_".join(f"{k}:{v}" for k, v in sorted(kwargs.items()))
        return f"{self.provider_type.value}_{model}_{max_tokens}_{temperature}_{hash(prompt)}_{kwargs_str}"

    def _get_cached_response(self, cache_key: str) -> Optional[LLMResponse]:
        """Retrieve cached response from simple cache."""
        if not self.storage:
            return None

        try:
            from ..storage.simple_cache import get_cache

            cache = get_cache()
            key = f"llm_response:{cache_key}"
            cached_data = cache.get(key)
            if cached_data:
                return LLMResponse(**cached_data)
            return None
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None

    def _cache_response(self, cache_key: str, response: LLMResponse) -> None:
        """Cache response using simple cache."""
        if not self.storage:
            return

        try:
            from ..storage.simple_cache import get_cache, DEFAULT_TTL_SECONDS

            cache = get_cache()
            key = f"llm_response:{cache_key}"
            # Convert LLMResponse to dict for caching
            response_data = response._asdict()
            cache.set(key, response_data, DEFAULT_TTL_SECONDS)
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.http_client.close()

    def generate_rag_response(
        self,
        query: str,
        context_chunks: list,
        query_type: str = "general",
        max_tokens: Optional[int] = None,
        temperature: float = 0.1,
        answer_format: str = "default",
        **kwargs,
    ) -> LLMResponse:
        """
        Generate RAG response with context-aware prompting.

        Args:
            query: User's question
            context_chunks: List of relevant document chunks
            query_type: Type of query ('factual', 'analytical', 'summarization', 'comparison')
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            answer_format: Answer format template name (default: "default")
            **kwargs: Additional parameters

        Returns:
            LLMResponse with contextual answer
        """
        # Prepare context from chunks with enhanced citation formatting
        context_parts = []
        for i, chunk in enumerate(context_chunks[:8], 1):  # Limit to top 8 chunks
            chunk_text = (
                chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            )

            # Use in-text citation format (with page numbers) for context
            if isinstance(chunk, dict):
                try:
                    from fileintel.citation import format_in_text_citation
                    # Use in-text citation so LLM sees the page-specific format to use
                    source_info = format_in_text_citation(chunk)
                except ImportError:
                    # Fallback to filename if citation module not available
                    source_info = chunk.get("original_filename", chunk.get("filename", f"Source {i}"))
            else:
                source_info = f"Source {i}"

            context_parts.append(f"[{source_info}]: {chunk_text}")

        context = "\n\n".join(context_parts)

        # Generate query-type specific prompt with optional answer format
        prompt = self._build_rag_prompt(query, context, query_type, answer_format)

        # LOG THE ACTUAL PROMPT
        logger.debug("="*80)
        logger.debug("FULL RAG PROMPT BEING SENT TO LLM:")
        logger.debug("="*80)
        logger.debug(prompt)
        logger.debug("="*80)

        return self.generate_response(
            prompt=prompt,
            max_tokens=max_tokens or 600,
            temperature=temperature,
            **kwargs,
        )

    def _build_rag_prompt(
        self,
        query: str,
        context: str,
        query_type: str,
        answer_format: str = "default"
    ) -> str:
        """
        Build query-type specific RAG prompt.

        Args:
            query: User's question
            context: Retrieved context
            query_type: Type of query for specialized prompting
            answer_format: Answer format template name (default: "default")

        Returns:
            Formatted prompt string
        """
        # Try template-based approach first, fall back to hardcoded if templates not loaded
        if self.query_type_templates is not None:
            return self._build_rag_prompt_from_templates(query, context, query_type, answer_format)
        else:
            return self._build_rag_prompt_fallback(query, context, query_type)

    def _build_rag_prompt_from_templates(
        self,
        query: str,
        context: str,
        query_type: str,
        answer_format: str = "default"
    ) -> str:
        """Build RAG prompt using loaded templates."""
        # Get base instruction
        base_instruction = self.base_instruction_template

        # Get query-type specific instruction
        specific_instruction = self.query_type_templates.get(
            query_type,
            self.query_type_templates["general"]
        )

        # Get answer format template if not default
        answer_format_section = ""
        if answer_format != "default":
            if self.format_manager is not None:
                try:
                    format_template = self.format_manager.get_format_template(answer_format)
                    answer_format_section = f"\n\n{format_template}\n"
                except (ValueError, IOError) as e:
                    logger.warning(f"Failed to load answer format '{answer_format}': {e}")
                    # Continue without format section
            else:
                logger.warning(f"Format manager unavailable - cannot apply format '{answer_format}'")

        # Get citation rules
        citation_rules = self.citation_rules_template

        # Assemble prompt
        prompt = f"""{base_instruction} {specific_instruction}

Question: {query}

Retrieved Sources:
{context}

Please provide your answer based on the sources above. If the sources don't contain sufficient information to fully answer the question, indicate what information is available and what might be missing.
{answer_format_section}
{citation_rules}"""

        return prompt

    def _build_rag_prompt_fallback(
        self,
        query: str,
        context: str,
        query_type: str
    ) -> str:
        """Fallback to hardcoded prompts if templates fail to load."""
        base_instruction = "Based on the following retrieved documents, answer the user's question accurately and comprehensively."

        if query_type == "factual":
            specific_instruction = "Focus on providing specific facts, dates, numbers, and concrete information. If exact information isn't available, clearly state what is known and what is uncertain."
        elif query_type == "analytical":
            specific_instruction = "Provide an analytical response that examines relationships, patterns, and implications. Use the sources to support your analysis and reasoning."
        elif query_type == "summarization":
            specific_instruction = "Provide a comprehensive summary that captures the key points and main themes from the sources. Organize the information logically."
        elif query_type == "comparison":
            specific_instruction = "Compare and contrast the information from different sources. Highlight similarities, differences, and any conflicting information."
        else:  # general
            specific_instruction = "Provide a clear and well-reasoned answer. Use evidence from the sources to support your response."

        return f"""{base_instruction} {specific_instruction}

Question: {query}

Retrieved Sources:
{context}

Please provide your answer based on the sources above. If the sources don't contain sufficient information to fully answer the question, indicate what information is available and what might be missing.

CRITICAL CITATION REQUIREMENTS:
- You MUST cite sources using the EXACT citation format shown in square brackets [like this] before each source text
- When a source citation includes a page number (e.g., "(Author, Year, p. X)"), you MUST include that page number in your citation
- Harvard style in-text citation format:
  * Single page: (Author Year, p.X)
  * Consecutive pages: (Author Year, pp.X-Y)
  * Non-consecutive pages: (Author Year, pp.X,Y)
- Example: If the source shows "[(Smith, 2023, p. 45)]: Some text", cite it as (Smith 2023, p.45) in your answer
- ALWAYS preserve page numbers from the source citations - they are critical for academic accuracy"""

    def generate_summary(
        self,
        content: str,
        summary_type: str = "brief",
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate content summary with specialized prompting.

        Args:
            content: Content to summarize
            summary_type: Type of summary ('brief', 'detailed', 'bullet_points', 'executive')
            max_tokens: Maximum tokens for summary
            **kwargs: Additional parameters

        Returns:
            LLMResponse with summary
        """
        if summary_type == "bullet_points":
            prompt = f"Summarize the following content as clear, concise bullet points:\n\n{content}\n\nBullet Point Summary:"
        elif summary_type == "detailed":
            prompt = f"Provide a comprehensive detailed summary of the following content:\n\n{content}\n\nDetailed Summary:"
        elif summary_type == "executive":
            prompt = f"Provide an executive summary highlighting the key points and implications:\n\n{content}\n\nExecutive Summary:"
        else:  # brief
            prompt = f"Provide a brief, clear summary of the following content:\n\n{content}\n\nSummary:"

        return self.generate_response(
            prompt=prompt, max_tokens=max_tokens or 300, temperature=0.1, **kwargs
        )
