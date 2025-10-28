"""This module defines config adapters for converting FileIntel settings to GraphRAG format."""

import os
import logging
from fileintel.core.config import Settings
from .._graphrag_imports import (
    GraphRagConfig,
    StorageConfig,
    InputConfig,
    OutputConfig,
    LanguageModelConfig,
    ModelType,
)

# Configuration constants
DEFAULT_REQUEST_TIMEOUT = 7200.0  # Increased from 30s to 5 minutes for entity extraction
DEFAULT_MAX_RETRIES = 5
DEFAULT_MAX_RETRY_WAIT = 3.0
# Ultra-high rate limits for local LLM server (effectively disables rate limiting)
HIGH_RATE_LIMIT_RPM = 100000  # 100k requests per minute
HIGH_RATE_LIMIT_TPM = 100000000  # 100M tokens per minute
DEFAULT_RETRY_STRATEGY = "native"
DEFAULT_ENCODING_MODEL = "cl100k_base"

logger = logging.getLogger(__name__)


class GraphRAGConfigAdapter:
    """A class to adapt FileIntel settings for GraphRAG."""

    def adapt_config(
        self, settings: Settings, collection_id: str, root_path: str
    ) -> GraphRagConfig:
        """Adapts FileIntel settings to the format expected by GraphRAG."""

        workspace_path = os.path.join(root_path, collection_id)
        input_path = os.path.join(workspace_path, "input")
        output_path = os.path.join(workspace_path, "output")
        os.makedirs(input_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)

        # Determine embedding base URL (use separate embedding server if configured)
        chat_base_url = settings.llm.openai.base_url
        embedding_base_url = (
            settings.llm.openai.embedding_base_url or settings.llm.openai.base_url
        )

        # Override any OpenAI environment variables that might interfere
        original_openai_base_url = os.environ.get("OPENAI_BASE_URL")
        original_openai_api_base = os.environ.get("OPENAI_API_BASE")

        logger.info(
            f"Creating GraphRAG config - Chat model: {settings.rag.llm_model}"
        )
        logger.info(
            f"Creating GraphRAG config - Embedding model: {settings.rag.embedding_model}"
        )
        logger.debug(f"GRAPHRAG DEBUG: Chat base URL: {chat_base_url}")
        logger.debug(f"GRAPHRAG DEBUG: Embedding base URL: {embedding_base_url}")
        logger.debug(
            f"GRAPHRAG DEBUG: Original OPENAI_BASE_URL env: {original_openai_base_url}"
        )
        logger.debug(
            f"GRAPHRAG DEBUG: Original OPENAI_API_BASE env: {original_openai_api_base}"
        )
        logger.debug(f"GRAPHRAG DEBUG: ModelType.OpenAIChat: {ModelType.OpenAIChat}")
        logger.debug(
            f"GRAPHRAG DEBUG: ModelType.OpenAIEmbedding: {ModelType.OpenAIEmbedding}"
        )

        # Log all current environment variables that could affect OpenAI
        env_vars = {k: v for k, v in os.environ.items() if "OPENAI" in k or "API" in k}
        logger.debug(f"GRAPHRAG DEBUG: All relevant environment variables: {env_vars}")

        # CRITICAL DEBUG: Log the actual URLs being used
        logger.debug(f"GRAPHRAG DEBUG: CRITICAL - Config chat_base_url: {chat_base_url}")
        logger.debug(
            f"GRAPHRAG DEBUG: CRITICAL - Config embedding_base_url: {embedding_base_url}"
        )
        logger.debug(
            f"GRAPHRAG DEBUG: CRITICAL - Expected: http://192.168.0.247:9003/v1"
        )
        logger.debug(
            f"GRAPHRAG DEBUG: CRITICAL - Error shows: http://172.19.0.4:8000/v1/embeddings"
        )

        # Temporarily clear ALL OpenAI and API environment variables to ensure our config is used
        env_vars_to_clear = [
            "OPENAI_BASE_URL",
            "OPENAI_API_BASE",
            "OPENAI_API_URL",
            "OPENAI_ENDPOINT",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "OPENAI_API_KEY_PATH",
            "OPENAI_ORGANIZATION",
            # Clear any potential GraphRAG-specific environment variables
            "GRAPHRAG_API_BASE",
            "GRAPHRAG_OPENAI_BASE_URL",
            "GRAPHRAG_BASE_URL",
            # Clear any potential Docker service environment variables that might interfere
            "API_BASE_URL",
            "LLM_BASE_URL",
            "EMBEDDING_BASE_URL",
            "FNLLM_BASE_URL",
        ]
        for env_var in env_vars_to_clear:
            if env_var in os.environ:
                logger.debug(
                    f"GRAPHRAG DEBUG: Clearing environment variable {env_var}={os.environ[env_var]}"
                )
                del os.environ[env_var]

        # Force set our URLs to prevent any override - set both chat and embedding URLs
        os.environ[
            "OPENAI_BASE_URL"
        ] = embedding_base_url  # Use embedding URL for consistency
        logger.debug(
            f"GRAPHRAG DEBUG: Force set OPENAI_BASE_URL to: {embedding_base_url}"
        )

        try:
            # Check for rate limiting bypass environment variable
            bypass_rate_limiting = (
                os.environ.get("GRAPHRAG_BYPASS_RATE_LIMITING", "true").lower()
                == "true"
            )
            logger.debug(
                f"GRAPHRAG DEBUG: Rate limiting bypass enabled: {bypass_rate_limiting}"
            )

            # CRITICAL FIX: Use direct embedding approach that works in FileIntel RAG
            use_direct_embeddings = (
                os.environ.get("GRAPHRAG_USE_DIRECT_EMBEDDINGS", "true").lower()
                == "true"
            )
            logger.debug(
                f"GRAPHRAG DEBUG: Direct embeddings bypass enabled: {use_direct_embeddings}"
            )

            # Create model configs with explicit debugging and performance settings
            chat_model_config = LanguageModelConfig(
                type=ModelType.OpenAIChat,
                model=settings.rag.llm_model,
                api_key=settings.llm.openai.api_key or settings.llm.api_key,
                api_base=chat_base_url,
                encoding_model=DEFAULT_ENCODING_MODEL,
                # Performance overrides to fix the ~60 second delays
                requests_per_minute=HIGH_RATE_LIMIT_RPM,
                tokens_per_minute=HIGH_RATE_LIMIT_TPM,
                concurrent_requests=settings.rag.async_processing.max_concurrent_requests,
                request_timeout=DEFAULT_REQUEST_TIMEOUT,
                max_retries=DEFAULT_MAX_RETRIES,
                max_retry_wait=DEFAULT_MAX_RETRY_WAIT,
                retry_strategy=DEFAULT_RETRY_STRATEGY,
            )

            embedding_model_config = LanguageModelConfig(
                type=ModelType.OpenAIEmbedding,
                model=settings.rag.embedding_model,
                api_key=settings.llm.openai.api_key or settings.llm.api_key,
                api_base=embedding_base_url,
                encoding_model=DEFAULT_ENCODING_MODEL,
                # Performance overrides to fix the ~60 second delays
                requests_per_minute=HIGH_RATE_LIMIT_RPM,
                tokens_per_minute=HIGH_RATE_LIMIT_TPM,
                concurrent_requests=settings.rag.async_processing.max_concurrent_requests,
                request_timeout=DEFAULT_REQUEST_TIMEOUT,
                max_retries=DEFAULT_MAX_RETRIES,
                max_retry_wait=DEFAULT_MAX_RETRY_WAIT,
                retry_strategy=DEFAULT_RETRY_STRATEGY,
            )

            # CRITICAL DEBUG: Log the exact api_base values before model creation
            logger.debug(
                f"GRAPHRAG DEBUG: Chat model config - model: {chat_model_config.model}, api_base: {chat_model_config.api_base}"
            )
            logger.debug(
                f"GRAPHRAG DEBUG: Chat model config - requests_per_minute: {chat_model_config.requests_per_minute}, tokens_per_minute: {chat_model_config.tokens_per_minute}, concurrent_requests: {chat_model_config.concurrent_requests}"
            )
            logger.debug(
                f"GRAPHRAG DEBUG: Embedding model config - model: {embedding_model_config.model}, api_base: {embedding_model_config.api_base}"
            )
            logger.debug(
                f"GRAPHRAG DEBUG: Embedding model config - requests_per_minute: {embedding_model_config.requests_per_minute}, tokens_per_minute: {embedding_model_config.tokens_per_minute}, concurrent_requests: {embedding_model_config.concurrent_requests}"
            )
            logger.debug(
                f"GRAPHRAG DEBUG: CRITICAL - Embedding model should be 'bge-large-en', actual: '{embedding_model_config.model}'"
            )
            logger.debug(
                f"GRAPHRAG DEBUG: CRITICAL - Embedding API base should be '{embedding_base_url}', actual: '{embedding_model_config.api_base}'"
            )

            # CRITICAL: Verify the api_base is exactly what we expect before creating the config
            if embedding_model_config.api_base != embedding_base_url:
                logger.error(
                    f"GRAPHRAG CRITICAL ERROR: api_base mismatch! Expected: '{embedding_base_url}', Got: '{embedding_model_config.api_base}'"
                )
                # Force correct the api_base if it's wrong
                logger.debug(
                    f"GRAPHRAG DEBUG: Force correcting api_base from '{embedding_model_config.api_base}' to '{embedding_base_url}'"
                )
                embedding_model_config.api_base = embedding_base_url

            # Log the exact values being used to create the models dictionary
            logger.debug(
                f"GRAPHRAG DEBUG: FINAL api_base values - Chat: '{chat_model_config.api_base}', Embedding: '{embedding_model_config.api_base}'"
            )

            models = {
                "default_chat_model": chat_model_config,
                "default_embedding_model": embedding_model_config,
            }

            # Import TextEmbeddingConfig to set custom token limits
            from .._graphrag_imports import TextEmbeddingConfig

            # Create embedding config with FileIntel's token limit
            embed_text_config = TextEmbeddingConfig(
                batch_max_tokens=settings.rag.embedding_batch_max_tokens
            )

            logger.debug(
                f"GRAPHRAG DEBUG: Setting embed_text batch_max_tokens to {settings.rag.embedding_batch_max_tokens}"
            )

            config = GraphRagConfig(
                root_dir=workspace_path,
                models=models,
                storage=StorageConfig(base_dir=output_path),
                input=InputConfig(base_dir=input_path),
                output=OutputConfig(base_dir=output_path),
                embed_text=embed_text_config,
            )

            # Debug log the final config
            logger.debug(
                f"GRAPHRAG DEBUG: Final config models keys: {list(config.models.keys())}"
            )
            if "default_embedding_model" in config.models:
                final_embedding_model = config.models["default_embedding_model"]
                logger.debug(
                    f"GRAPHRAG DEBUG: Final embedding model api_base: {getattr(final_embedding_model, 'api_base', 'NOT_SET')}"
                )
                logger.debug(
                    f"GRAPHRAG DEBUG: Final embedding model type: {type(final_embedding_model)}"
                )
                logger.debug(
                    f"GRAPHRAG DEBUG: Final embedding model dict: {final_embedding_model.__dict__ if hasattr(final_embedding_model, '__dict__') else 'NO_DICT'}"
                )

            # Check environment one more time before returning
            final_env_vars = {k: v for k, v in os.environ.items() if "OPENAI" in k}
            logger.debug(f"GRAPHRAG DEBUG: Final environment state: {final_env_vars}")

            # COMPREHENSIVE DEBUG: Log all environment variables that could affect GraphRAG
            all_openai_env = {
                k: v for k, v in os.environ.items() if "OPENAI" in k or "API" in k
            }
            logger.debug(
                f"GRAPHRAG COMPREHENSIVE DEBUG: All API-related env vars: {all_openai_env}"
            )

            # COMPREHENSIVE DEBUG: Log the actual model configurations that will be used
            logger.debug("=== GRAPHRAG MODEL CONFIG COMPREHENSIVE DEBUG ===")
            for model_name, model_config in config.models.items():
                logger.debug(f"Model: {model_name}")
                logger.debug(f"  Type: {model_config.type}")
                logger.debug(f"  Model: {model_config.model}")
                logger.debug(f"  API Key: {'***' if model_config.api_key else 'NONE'}")
                logger.debug(f"  API Base: {model_config.api_base}")
                logger.debug(
                    f"  Requests per minute: {model_config.requests_per_minute}"
                )
                logger.debug(f"  Tokens per minute: {model_config.tokens_per_minute}")
                logger.debug(f"  Max retries: {model_config.max_retries}")
                logger.debug(f"  Retry strategy: {model_config.retry_strategy}")
                logger.debug(f"  Request timeout: {model_config.request_timeout}")
                logger.debug(f"  All config dict: {model_config.__dict__}")

            # COMPREHENSIVE DEBUG: Check if there are any other environment variables that could interfere
            suspicious_env_vars = [
                "OPENAI_BASE_URL",
                "OPENAI_API_BASE",
                "OPENAI_API_URL",
                "OPENAI_ENDPOINT",
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_API_KEY",
                "GRAPHRAG_EMBEDDING_URL",
                "GRAPHRAG_LLM_URL",
                "FNLLM_BASE_URL",
            ]
            for env_var in suspicious_env_vars:
                value = os.environ.get(env_var, "NOT_SET")
                logger.debug(f"GRAPHRAG ENV CHECK: {env_var} = {value}")

            logger.debug("=== END GRAPHRAG CONFIG COMPREHENSIVE DEBUG ===")
            logger.debug(
                "GRAPHRAG DEBUG: Config created successfully with explicit base URLs"
            )
            return config

        finally:
            # Restore original environment variables
            if original_openai_base_url is not None:
                os.environ["OPENAI_BASE_URL"] = original_openai_base_url
            if original_openai_api_base is not None:
                os.environ["OPENAI_API_BASE"] = original_openai_api_base
