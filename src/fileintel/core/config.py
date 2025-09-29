import re
import os
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Any
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
    model: str = Field(default="gemma3-4B")
    max_tokens: int = Field(default=4000)
    context_length: int = Field(default=4096)
    temperature: float = Field(default=0.1)
    rate_limit: int = Field(default=60)
    base_url: Optional[str] = Field(default=None)
    api_key: Optional[str] = Field(default="ollama")  # Default for Ollama
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
        default=25, ge=1, le=32, description="Total concurrent HTTP connections"
    )
    batch_timeout: int = Field(
        default=30, ge=10, le=120, description="Timeout per batch in seconds"
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


# GraphRAGSettings class removed - consolidated into RAGSettings to eliminate duplication


class ChunkingSettings(BaseModel):
    """Unified chunking configuration for all RAG operations."""

    chunk_size: int = Field(
        default=500, description="Default chunk size for text processing (optimized for 512-token embedding models)"
    )
    chunk_overlap: int = Field(default=50, description="Overlap between chunks (adjusted for smaller chunk size)")
    target_sentences: int = Field(default=18, description="Target sentences per chunk")
    overlap_sentences: int = Field(
        default=2, description="Sentence overlap between chunks"
    )


class RAGSettings(BaseModel):
    """Unified RAG configuration consolidating vector and graph RAG settings."""

    strategy: str = Field(default="merge")
    embedding_provider: str = Field(default="openai")
    embedding_model: str = Field(default="bge-large-en")

    # Unified chunking configuration
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)

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
    classification_model: str = Field(
        default="gemma3-4B", description="Model for query classification"
    )

    # GraphRAG-specific settings (moved here to eliminate duplication)
    llm_model: str = Field(default="gemma3-4B")
    community_levels: int = Field(default=3)
    max_tokens: int = Field(default=12000)
    root_dir: str = Field(default="/data/graphrag_indices")
    auto_index_after_upload: bool = Field(default=True)
    auto_index_delay_seconds: int = Field(default=30)
    embedding_batch_max_tokens: int = Field(default=400)

    # Cache settings
    cache: GraphRAGCacheSettings = Field(default_factory=GraphRAGCacheSettings)

    # Async processing
    async_processing: AsyncProcessingSettings = Field(
        default_factory=AsyncProcessingSettings
    )


class DocumentProcessingSettings(BaseModel):
    chunk_size: int = Field(default=4000)
    overlap: int = Field(default=200)
    max_file_size: str = Field(default="100MB")
    supported_formats: List[str] = Field(
        default_factory=lambda: ["pdf", "epub", "mobi"]
    )
    # OCR settings consolidated from separate OCRSettings class
    ocr_primary_engine: str = Field(default="pdf_extract_kit")
    ocr_fallback_engines: List[str] = Field(
        default_factory=lambda: ["tesseract", "google_vision"]
    )


class LoggingSettings(BaseModel):
    level: str = Field(default="INFO")
    max_file_size_mb: int = Field(default=5, description="Maximum log file size in MB")
    backup_count: int = Field(
        default=5, description="Number of backup log files to keep"
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


class StorageSettings(BaseModel):
    type: str = Field(default="redis")
    connection_string: str = Field(default="sqlite:///./database.db")
    cache_ttl: int = Field(default=3600)
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
    worker_max_tasks_per_child: int = Field(default=1000)
    task_routes: dict = Field(
        default_factory=lambda: {
            "fileintel.tasks.document.*": {"queue": "document_processing"},
            "fileintel.tasks.rag.*": {"queue": "rag_processing"},
            "fileintel.tasks.llm.*": {"queue": "llm_processing"},
            "fileintel.tasks.graphrag.*": {"queue": "graphrag_indexing"},
        }
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
    global _settings
    if _settings is None:
        _settings = load_config()
    return _settings
