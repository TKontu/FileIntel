import re
import os
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Any, Dict
import yaml


class OpenAISettings(BaseModel):
    api_key: Optional[str] = Field(default="ollama")
    base_url: Optional[str] = Field(default=None)
    embedding_base_url: Optional[str] = Field(
        default=None
    )  # Separate embedding server URL
    # Rate limit moved to LLMSettings to avoid duplication


class AnthropicSettings(BaseModel):
    api_key: Optional[str] = Field(default=None)
    # Rate limit moved to LLMSettings to avoid duplication


class LLMSettings(BaseModel):
    provider: str = Field(default="openai")
    model: str = Field(default="gemma3-12b-awq")
    max_tokens: int = Field(default=4000)
    context_length: int = Field(default=4096)
    temperature: float = Field(default=0.1)
    rate_limit: int = Field(default=60)
    base_url: Optional[str] = Field(default=None)
    api_key: Optional[str] = Field(default="ollama")  # Default for Ollama
    # Task timeout limits (prevents worker exhaustion on long-running tasks)
    task_timeout_seconds: int = Field(
        default=300, ge=60, description="Soft timeout for LLM tasks (seconds)"
    )
    task_hard_limit_seconds: int = Field(
        default=360, ge=60, description="Hard kill timeout for LLM tasks (seconds)"
    )
    # HTTP client timeout and retry settings (for handling high server load)
    http_timeout_seconds: int = Field(
        default=900, ge=60, description="HTTP request timeout in seconds (increased for high queue depths)"
    )
    max_retries: int = Field(
        default=5, ge=1, le=10, description="Maximum retry attempts for failed requests"
    )
    retry_backoff_min: int = Field(
        default=2, ge=1, description="Minimum retry backoff in seconds"
    )
    retry_backoff_max: int = Field(
        default=60, ge=10, description="Maximum retry backoff in seconds"
    )
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)


class AsyncProcessingSettings(BaseModel):
    """Async processing configuration for GraphRAG batch operations."""

    enabled: bool = Field(
        default=False, description="Enable/disable async batch processing"
    )
    batch_size: int = Field(
        default=4, ge=1, le=8, description="Concurrent requests per batch"
    )
    max_concurrent_requests: int = Field(
        default=25, ge=1, le=100, description="Total concurrent HTTP connections (must match vLLM max_num_seqs)"
    )
    batch_timeout: int = Field(
        default=30, ge=10, description="Timeout per batch in seconds"
    )
    fallback_to_sequential: bool = Field(
        default=True, description="Fallback to sequential on batch failure"
    )


class GraphRAGCacheSettings(BaseModel):
    enabled: bool = Field(default=True)
    ttl_seconds: int = Field(default=3600)
    max_size_mb: int = Field(default=500)
    redis_host: str = Field(default="redis")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    warmup_collections: List[str] = Field(default_factory=list)


class GapPreventionSettings(BaseModel):
    """Gap prevention and retry configuration for GraphRAG indexing."""

    enabled: bool = Field(
        default=False,
        description="Enable in-phase gap prevention (retries failed items before proceeding to next phase)"
    )
    max_retries_per_item: int = Field(
        default=20,
        description="Maximum number of retry attempts for each failed item"
    )
    retry_backoff_base: float = Field(
        default=2.0,
        description="Exponential backoff base multiplier (e.g., 2.0 means 2s, 4s, 8s, 16s...)"
    )
    retry_backoff_max: float = Field(
        default=120.0,
        description="Maximum backoff time in seconds (caps exponential growth)"
    )
    retry_jitter: bool = Field(
        default=True,
        description="Add random jitter to backoff times (reduces thundering herd effect)"
    )
    gap_fill_concurrency: int = Field(
        default=5,
        description="Lower concurrency limit for gap filling pass (reduces 503 errors)"
    )


# GraphRAGSettings class removed - consolidated into RAGSettings to eliminate duplication


class ChunkingSettings(BaseModel):
    """Unified chunking configuration for all RAG operations."""

    chunk_size: int = Field(
        default=800, description="Default chunk size for text processing (optimized for 512-token embedding models)"
    )
    chunk_overlap: int = Field(default=80, description="Overlap between chunks (adjusted for smaller chunk size)")
    target_sentences: int = Field(default=3, description="Target sentences per chunk")
    overlap_sentences: int = Field(
        default=1, description="Sentence overlap between chunks"
    )


class EmbeddingProcessingSettings(BaseModel):
    """Batch processing configuration for embedding generation."""

    batch_size: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Number of chunks to process per batch task (1=disable batching, 25=optimal for most workloads, 50-100=high throughput)"
    )

    fallback_to_single: bool = Field(
        default=True,
        description="Fall back to single-chunk processing if batch fails (recommended for reliability)"
    )

    retry_failed_individually: bool = Field(
        default=True,
        description="Retry failed chunks individually after batch completes (ensures no data loss)"
    )


class RerankerSettings(BaseModel):
    """Reranker configuration for improving retrieval relevance."""

    enabled: bool = Field(
        default=False,
        description="Enable result reranking (improves relevance at cost of latency)"
    )

    # API settings (for remote vLLM/OpenAI servers)
    base_url: str = Field(
        default="http://192.168.0.136:9003/v1",
        description="Base URL for reranking API (vLLM or OpenAI-compatible server)"
    )

    api_key: str = Field(
        default="ollama",
        description="API key for reranking server authentication"
    )

    timeout: Optional[int] = Field(
        default=120,
        description="HTTP timeout in seconds for reranking requests (handles cold model loading)"
    )

    model_name: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Reranker model (bge-reranker-v2-m3, bge-reranker-large, etc.)"
    )

    # Legacy local model settings (DEPRECATED - unused in API mode, kept for backward compatibility)
    model_type: str = Field(
        default="normal",
        description="[DEPRECATED - UNUSED] Was for local model type selection. Kept for config compatibility."
    )

    use_fp16: bool = Field(
        default=True,
        description="[DEPRECATED - UNUSED] Was for local model FP16. Kept for config compatibility."
    )

    # Strategy settings
    rerank_vector_results: bool = Field(
        default=True,
        description="Rerank vector search results"
    )

    rerank_graph_results: bool = Field(
        default=True,
        description="Rerank GraphRAG results"
    )

    rerank_hybrid_results: bool = Field(
        default=True,
        description="Rerank hybrid (combined) results"
    )

    # Performance settings
    initial_retrieval_k: int = Field(
        default=20,
        description="Initial number of chunks to retrieve (before reranking)"
    )

    final_top_k: int = Field(
        default=5,
        description="Final number of chunks to return after reranking"
    )

    batch_size: int = Field(
        default=32,
        description="[DEPRECATED - UNUSED] Was for local batching. Kept for config compatibility."
    )

    normalize_scores: bool = Field(
        default=True,
        description="[DEPRECATED - UNUSED] Was for score normalization. API handles this. Kept for config compatibility."
    )

    # Device settings (DEPRECATED - unused in API mode, kept for backward compatibility)
    device: str = Field(
        default="auto",
        description="[DEPRECATED - UNUSED] Was for device selection. Kept for config compatibility."
    )

    cache_model: bool = Field(
        default=True,
        description="[DEPRECATED - UNUSED] Was for model caching. Server handles this. Kept for config compatibility."
    )

    # Advanced settings
    min_score_threshold: Optional[float] = Field(
        default=None,
        description="Minimum reranked score to include (filter low-relevance results)"
    )


class RAGSettings(BaseModel):
    """Unified RAG configuration consolidating vector and graph RAG settings."""

    strategy: str = Field(default="merge")
    embedding_provider: str = Field(default="openai")
    embedding_model: str = Field(default="bge-large-en")
    embedding_max_tokens: int = Field(default=450, description="Maximum tokens for individual embedding requests")
    enable_two_tier_chunking: bool = Field(default=False, description="Enable two-tier vector/graph chunking system")
    exclude_bibliography_sections: bool = Field(default=True, description="Exclude bibliography/reference sections from embeddings")

    # Unified chunking configuration
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)

    # Embedding batch processing configuration
    embedding_processing: EmbeddingProcessingSettings = Field(
        default_factory=EmbeddingProcessingSettings,
        description="Batch processing settings for embedding generation (improves throughput 10-25x)"
    )

    # Query routing and classification settings (previously QueryRoutingConfig)
    classification_threshold: float = Field(
        default=0.7, description="Threshold for query classification confidence"
    )
    default_strategy: str = Field(
        default="vector", description="Default query routing strategy"
    )
    enable_hybrid_queries: bool = Field(
        default=True, description="Enable hybrid query routing"
    )

    # LLM-based query classification settings
    classification_method: str = Field(
        default="hybrid",
        description="Classification method: 'llm', 'keyword', or 'hybrid' (LLM with keyword fallback)"
    )
    classification_model: str = Field(
        default="gemma3-4B", description="LLM model for query classification (small/fast model recommended)"
    )
    classification_temperature: float = Field(
        default=0.0, description="Temperature for classification (0.0 = deterministic)"
    )
    classification_max_tokens: int = Field(
        default=150, description="Max tokens for classification response"
    )
    classification_timeout_seconds: int = Field(
        default=5, description="Timeout for LLM classification before falling back to keywords"
    )
    classification_cache_enabled: bool = Field(
        default=True, description="Enable caching of classification results"
    )
    classification_cache_ttl: int = Field(
        default=3600, description="Cache TTL in seconds (1 hour default)"
    )

    # Query classification keywords (for keyword-based routing)
    graph_keywords: Optional[List[str]] = Field(
        default=None,
        description="Keywords that trigger GraphRAG routing (relationship/entity queries)"
    )
    vector_keywords: Optional[List[str]] = Field(
        default=None,
        description="Keywords that trigger VectorRAG routing (factual/search queries)"
    )
    hybrid_keywords: Optional[List[str]] = Field(
        default=None,
        description="Keywords that trigger hybrid routing (complex multi-part queries)"
    )

    # GraphRAG-specific settings (moved here to eliminate duplication)
    llm_model: str = Field(default="gemma3-12b-awq")
    community_levels: int = Field(default=3)
    max_cluster_size: int = Field(default=50, description="Leiden algorithm max cluster size (higher = fewer levels, less redundancy)")
    leiden_resolution: float = Field(default=1.0, description="Leiden algorithm resolution parameter (lower = larger communities, e.g., 0.5; higher = smaller communities, e.g., 2.0)")
    max_tokens: int = Field(default=12000)
    root_dir: str = Field(default="/data/graphrag_indices")
    auto_index_after_upload: bool = Field(default=True)
    auto_index_delay_seconds: int = Field(default=30)
    embedding_batch_max_tokens: int = Field(default=400)

    # GraphRAG checkpoint & resume settings
    enable_checkpoint_resume: bool = Field(
        default=True,
        description="Enable checkpoint detection and automatic resume from last successful workflow step"
    )
    validate_checkpoints: bool = Field(
        default=True,
        description="Validate checkpoint data consistency before resume (recommended for data integrity)"
    )

    # Gap prevention & completeness settings
    gap_prevention: GapPreventionSettings = Field(
        default_factory=GapPreventionSettings,
        description="Gap prevention and retry configuration for in-phase gap filling"
    )
    validate_completeness: bool = Field(
        default=True,
        description="Enable completeness validation after indexing"
    )
    completeness_threshold: float = Field(
        default=0.99,
        description="Warn if completeness falls below this threshold (0.99 = 99%)"
    )

    # Cache settings
    cache: GraphRAGCacheSettings = Field(default_factory=GraphRAGCacheSettings)

    # Async processing
    async_processing: AsyncProcessingSettings = Field(
        default_factory=AsyncProcessingSettings
    )

    # Result reranking
    reranking: RerankerSettings = Field(
        default_factory=RerankerSettings,
        description="Reranker settings for improving retrieval relevance"
    )


class MinerUSettings(BaseModel):
    # API type selection: "selfhosted" for FastAPI or "commercial" for async task API
    api_type: str = Field(default="selfhosted")

    # Common settings for both API types
    base_url: str = Field(default="http://192.168.0.136:8000")
    timeout: Optional[int] = Field(default=600, description="Timeout in seconds (None to disable)")
    enable_formula: bool = Field(default=False)
    enable_table: bool = Field(default=True)
    language: str = Field(default="en")

    # Backend/model selection (dual purpose field)
    # - Selfhosted API: backend selection ("pipeline" or "vlm")
    # - Commercial API: model version string
    model_version: str = Field(default="vlm")

    # Debug output settings (for troubleshooting)
    save_outputs: bool = Field(default=False)
    output_directory: str = Field(default="/home/appuser/app/mineru_outputs")

    # Element-level type preservation (Phase 1 of structure utilization)
    # When True: creates one TextElement per content_list item (preserves element boundaries)
    # When False: concatenates all elements per page (backward compatible)
    use_element_level_types: bool = Field(default=False)

    # Element filtering (Phase 2 - requires use_element_level_types=True)
    # When True: filters out TOC/LOF elements before chunking (prevents oversized chunks)
    # When False: all elements pass through to chunking (backward compatible)
    enable_element_filtering: bool = Field(default=False)

    # Commercial API specific settings (used when api_type="commercial")
    # Optional fields that can be None (for self-hosted API which doesn't need them)
    api_token: Optional[str] = Field(default="")
    poll_interval: int = Field(default=10)
    max_retries: int = Field(default=3)
    shared_folder_path: str = Field(default="/shared/uploads")
    shared_folder_url_prefix: str = Field(default="file:///shared/uploads")

    @model_validator(mode='after')
    def validate_feature_flags(self) -> 'MinerUSettings':
        """Validate feature flag dependencies."""
        import logging
        logger = logging.getLogger(__name__)

        # CRITICAL: enable_element_filtering requires use_element_level_types
        if self.enable_element_filtering and not self.use_element_level_types:
            raise ValueError(
                "Invalid MinerU configuration: enable_element_filtering=true requires "
                "use_element_level_types=true. Filtering needs element-level semantic types. "
                "Fix: Set use_element_level_types=true or disable filtering."
            )

        # WARNING: element types without filtering may create large chunks
        if self.use_element_level_types and not self.enable_element_filtering:
            logger.warning(
                "MinerU element-level types enabled without filtering. "
                "TOC/LOF elements will be included in chunks, potentially creating "
                "oversized chunks (>450 tokens). Consider enabling enable_element_filtering=true."
            )

        return self


class DocumentProcessingSettings(BaseModel):
    chunk_size: int = Field(default=800)
    overlap: int = Field(default=80)
    max_file_size: str = Field(default="100MB")
    supported_formats: List[str] = Field(
        default_factory=lambda: ["pdf", "epub", "mobi"]
    )
    # PDF processor selection (always falls back to traditional on MinerU failure)
    primary_pdf_processor: str = Field(default="mineru")
    # OCR settings consolidated from separate OCRSettings class
    ocr_primary_engine: str = Field(default="pdf_extract_kit")
    ocr_fallback_engines: List[str] = Field(
        default_factory=lambda: ["tesseract", "google_vision"]
    )
    # MinerU configuration
    mineru: MinerUSettings = Field(default_factory=MinerUSettings)
    # Type-aware chunking (Phase 1)
    use_type_aware_chunking: bool = Field(
        default=False,
        description="Enable type-aware chunking based on element semantic types (tables, images, etc.)"
    )


class LoggingSettings(BaseModel):
    level: str = Field(default="INFO")
    max_file_size_mb: int = Field(default=5, description="Maximum log file size in MB")
    backup_count: int = Field(
        default=5, description="Number of backup log files to keep"
    )
    # Component-specific log levels (optional overrides)
    # When root level is WARNING, these can be set to INFO to see progress
    component_levels: Dict[str, str] = Field(
        default_factory=dict,
        description="Per-component log levels (e.g., {'graphrag_service': 'INFO'})"
    )


# RetrySettings removed - Celery handles retry configuration through task decorators


class AuthenticationSettings(BaseModel):
    enabled: bool = Field(default=False)
    api_key: Optional[str] = Field(default=None)


class OutputSettings(BaseModel):
    default_format: str = Field(default="json")
    output_directory: str = Field(default="./results")
    include_metadata: bool = Field(default=True)
    # Concurrency, retries, and timeouts now handled by Celery configuration


class APISettings(BaseModel):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    rate_limit: int = Field(default=100)
    authentication: AuthenticationSettings = Field(
        default_factory=AuthenticationSettings
    )
    # HTTP client timeout configuration for CLI
    request_timeout_connect: int = Field(
        default=30, description="HTTP connection timeout in seconds"
    )
    request_timeout_read: Optional[int] = Field(
        default=14400, description="HTTP read timeout in seconds (None to disable)"
    )


class StorageSettings(BaseModel):
    type: str = Field(default="redis")
    connection_string: str = Field(default="sqlite:///./database.db")
    cache_ttl: int = Field(default=3600)
    # Connection pool settings (prevents exhaustion)
    pool_size: int = Field(
        default=20, ge=5, le=100, description="Base database connection pool size"
    )
    max_overflow: int = Field(
        default=30, ge=0, le=100, description="Additional connections allowed beyond pool_size"
    )
    pool_timeout: int = Field(
        default=30, ge=5, le=300, description="Seconds to wait for connection"
    )
    # Redis settings consolidated into GraphRAGCacheSettings to avoid duplication


class PathSettings(BaseModel):
    uploads: str = Field(default="/home/appuser/app/uploads")
    prompts: str = Field(default="/home/appuser/app/prompts")
    input: str = Field(default="/home/appuser/app/input")
    output: str = Field(default="/home/appuser/app/output")
    logs: str = Field(default="logs/fileintel.log")
    celery_logs: str = Field(default="logs/celery.log")


class CelerySettings(BaseModel):
    broker_url: str = Field(default="redis://redis:6379/1")
    result_backend: str = Field(default="redis://redis:6379/1")
    task_serializer: str = Field(default="json")
    accept_content: List[str] = Field(default_factory=lambda: ["json"])
    result_serializer: str = Field(default="json")
    timezone: str = Field(default="UTC")
    enable_utc: bool = Field(default=True)
    worker_concurrency: int = Field(default=4)
    worker_prefetch_multiplier: int = Field(default=1)
    task_acks_late: bool = Field(default=True)
    worker_max_tasks_per_child: Optional[int] = Field(default=None)  # None = never restart workers
    task_routes: dict = Field(
        default_factory=lambda: {
            "fileintel.tasks.document.*": {"queue": "document_processing"},
            "fileintel.tasks.rag.*": {"queue": "rag_processing"},
            "fileintel.tasks.llm.*": {"queue": "llm_processing"},
            "fileintel.tasks.graphrag.*": {"queue": "graphrag_indexing"},
        }
    )
    # Task timeout limits (configurable from YAML)
    task_soft_time_limit: Optional[int] = Field(
        default=14400, description="Soft time limit for tasks in seconds (None to disable)"
    )
    task_time_limit: Optional[int] = Field(
        default=18000, description="Hard time limit for tasks in seconds (None to disable)"
    )


class CLISettings(BaseModel):
    """CLI timeout configuration."""

    task_wait_timeout: Optional[int] = Field(
        default=14400, description="CLI task wait timeout in seconds (None to disable)"
    )


class BatchProcessingSettings(BaseModel):
    """Batch processing limits to prevent resource exhaustion and DoS attacks."""

    directory_input: str = Field(default="/home/appuser/app/input")
    directory_output: str = Field(default="/home/appuser/app/output")
    default_format: str = Field(default="json")
    max_upload_batch_size: int = Field(
        default=50, ge=1, le=200, description="Maximum files per batch upload"
    )
    max_file_size_mb: int = Field(
        default=100, ge=1, le=1000, description="Maximum individual file size in MB"
    )
    max_processing_batch_size: int = Field(
        default=20, ge=1, le=100, description="Maximum collections per batch process"
    )


class Settings(BaseModel):
    llm: LLMSettings = Field(default_factory=LLMSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    # graphrag field removed - consolidated into rag for cleaner architecture
    document_processing: DocumentProcessingSettings = Field(
        default_factory=DocumentProcessingSettings
    )
    output: OutputSettings = Field(default_factory=OutputSettings)
    api: APISettings = Field(default_factory=APISettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    # retry field removed - Celery handles retry configuration
    celery: CelerySettings = Field(default_factory=CelerySettings)
    cli: CLISettings = Field(default_factory=CLISettings)
    batch_processing: BatchProcessingSettings = Field(default_factory=BatchProcessingSettings)

    @model_validator(mode="after")
    def validate_critical_settings(self):
        """Validate critical configuration settings and provide clear error messages."""
        errors = []

        # Check LLM provider configuration
        if self.llm.provider == "openai":
            if not self.llm.openai.api_key or self.llm.openai.api_key == "ollama":
                if not self.llm.openai.base_url:
                    errors.append(
                        "OpenAI provider requires either 'api_key' or 'base_url' to be configured. "
                        "Set OPENAI_API_KEY environment variable or configure local LLM endpoint."
                    )
        elif self.llm.provider == "anthropic":
            if not self.llm.anthropic.api_key:
                errors.append(
                    "Anthropic provider requires 'api_key' to be configured. "
                    "Set ANTHROPIC_API_KEY environment variable."
                )

        # Check storage configuration
        if self.storage.type == "postgresql":
            if (
                not self.storage.connection_string
                or self.storage.connection_string == "sqlite:///./database.db"
            ):
                errors.append(
                    "PostgreSQL storage requires proper 'connection_string'. "
                    "Set DATABASE_URL environment variable with PostgreSQL connection string."
                )

        # Check Redis configuration for caching
        if self.rag.cache.enabled:
            if not self.rag.cache.redis_host:
                errors.append(
                    "Redis caching is enabled but 'redis_host' is not configured. "
                    "Set REDIS_HOST environment variable or disable caching."
                )
            if not isinstance(self.rag.cache.redis_port, int) or not (
                1 <= self.rag.cache.redis_port <= 65535
            ):
                errors.append(
                    f"Redis port {self.rag.cache.redis_port} is invalid. "
                    "Must be an integer between 1 and 65535."
                )
            if not isinstance(self.rag.cache.redis_db, int) or not (
                0 <= self.rag.cache.redis_db <= 15
            ):
                errors.append(
                    f"Redis database {self.rag.cache.redis_db} is invalid. "
                    "Must be an integer between 0 and 15."
                )

        # Check Celery broker/backend configuration
        if "redis://" in self.celery.broker_url:
            # Basic validation for Redis URLs
            if not self.celery.broker_url.startswith("redis://"):
                errors.append(
                    "Celery broker_url appears to be Redis but doesn't start with 'redis://'. "
                    "Check CELERY_BROKER_URL environment variable format."
                )

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {error}" for error in errors
            )
            raise ValueError(error_msg)

        return self

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a value from the nested settings using dot notation.
        Example: settings.get('llm.model', 'default-model')
        """
        keys = key.split(".")
        value = self
        for k in keys:
            if isinstance(value, BaseModel):
                value = getattr(value, k, None)
            elif isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value


def substitute_environment_variables(config_str: str) -> str:
    """
    Replace environment variable placeholders in configuration string.

    Supports format: ${VAR_NAME} and ${VAR_NAME:-default_value}

    Args:
        config_str: Configuration string with environment variable placeholders

    Returns:
        Configuration string with placeholders replaced by actual values

    Raises:
        ValueError: If required environment variable is not set
    """
    # Find all environment variable placeholders with optional defaults
    placeholders = re.findall(r"\$\{([^}]+)\}", config_str)

    # Replace placeholders with environment variable values
    for placeholder in placeholders:
        # Handle default values in format ${VAR:-default}
        if ":-" in placeholder:
            var_name, default_value = placeholder.split(":-", 1)
            value = os.environ.get(var_name, default_value)
        else:
            var_name = placeholder
            value = os.environ.get(var_name)
            if value is None:
                raise ValueError(
                    f"Required environment variable '{var_name}' is not set. "
                    f"Either set this environment variable or provide a default value "
                    f"using the format '${{VAR_NAME:-default_value}}' in your configuration."
                )

        config_str = config_str.replace(f"${{{placeholder}}}", value)

    return config_str


def load_config(path: str = "config/default.yaml") -> "Settings":
    """
    Load configuration from YAML file with environment variable substitution.

    Args:
        path: Path to configuration file

    Returns:
        Validated Settings object

    Raises:
        ValueError: If config file not found, YAML parsing fails, validation fails,
                   or required environment variables missing
    """
    try:
        with open(path, "r") as f:
            config_str = f.read()
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found at: {path}")
    except IOError as e:
        raise ValueError(f"Error reading configuration file {path}: {e}")

    try:
        # Substitute environment variables
        config_str = substitute_environment_variables(config_str)
    except ValueError as e:
        raise ValueError(f"Environment variable substitution failed: {e}")

    try:
        config_data = yaml.safe_load(config_str)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in {path}: {e}")

    if config_data is None:
        raise ValueError(
            f"Configuration file {path} is empty or contains only comments"
        )

    try:
        return Settings.model_validate(config_data)
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")


_settings: Optional["Settings"] = None


def get_config() -> "Settings":
    """
    Loads the configuration settings lazily.
    This function ensures that the configuration is loaded only once.
    """
    import logging
    import os
    logger = logging.getLogger(__name__)

    global _settings
    if _settings is None:
        _settings = load_config()
        # CRITICAL DEBUG: Log GraphRAG clustering config at load time
        logger.info(
            f"CONFIG LOAD: GraphRAG clustering - max_cluster_size={_settings.rag.max_cluster_size}, "
            f"leiden_resolution={_settings.rag.leiden_resolution}"
        )
        logger.info(
            f"CONFIG LOAD: Env vars - GRAPHRAG_MAX_CLUSTER_SIZE={os.getenv('GRAPHRAG_MAX_CLUSTER_SIZE', 'NOT_SET')}, "
            f"GRAPHRAG_LEIDEN_RESOLUTION={os.getenv('GRAPHRAG_LEIDEN_RESOLUTION', 'NOT_SET')}"
        )
    return _settings
