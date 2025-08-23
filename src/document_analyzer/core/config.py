from pydantic import BaseModel, Field
import yaml
from typing import Optional, List, Any


class OpenAISettings(BaseModel):
    api_key: Optional[str] = Field(default="ollama")
    base_url: Optional[str] = Field(default=None)
    rate_limit: int = Field(default=30)


class AnthropicSettings(BaseModel):
    rate_limit: int = Field(default=15)


class LLMSettings(BaseModel):
    provider: str = Field(default="openai")
    model: str = Field(default="gpt-4")
    max_tokens: int = Field(default=4000)
    context_length: int = Field(default=4096)
    temperature: float = Field(default=0.1)
    rate_limit: int = Field(default=60)
    base_url: Optional[str] = Field(default=None)
    api_key: Optional[str] = Field(default="ollama")  # Default for Ollama
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)


class RAGSettings(BaseModel):
    strategy: str = Field(default="merge")
    chunk_size: int = Field(default=1024)
    chunk_overlap: int = Field(default=200)
    embedding_provider: str = Field(default="openai")
    embedding_model: str = Field(default="text-embedding-3-small")


class DocumentProcessingSettings(BaseModel):
    chunk_size: int = Field(default=4000)
    overlap: int = Field(default=200)
    max_file_size: str = Field(default="100MB")
    supported_formats: List[str] = Field(
        default_factory=lambda: ["pdf", "epub", "mobi"]
    )


class OCRSettings(BaseModel):
    primary_engine: str = Field(default="pdf_extract_kit")
    fallback_engines: List[str] = Field(
        default_factory=lambda: ["tesseract", "google_vision"]
    )


class OutputSettings(BaseModel):
    default_format: str = Field(default="json")
    output_directory: str = Field(default="./results")
    include_metadata: bool = Field(default=True)
    max_concurrent_jobs: int = Field(default=5)
    retry_attempts: int = Field(default=3)
    timeout: int = Field(default=300)


class APISettings(BaseModel):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    rate_limit: int = Field(default=100)


class StorageSettings(BaseModel):
    type: str = Field(default="redis")
    connection_string: str = Field(default="sqlite:///./database.db")
    cache_ttl: int = Field(default=3600)
    redis_host: str = Field(default="redis")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    job_queue_name: str = Field(default="document_analyzer:job_queue")


class Settings(BaseModel):
    llm: LLMSettings = Field(default_factory=LLMSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    document_processing: DocumentProcessingSettings = Field(
        default_factory=DocumentProcessingSettings
    )
    ocr: OCRSettings = Field(default_factory=OCRSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    api: APISettings = Field(default_factory=APISettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)

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


def load_config(path: str = "config/default.yaml") -> Settings:
    with open(path, "r") as f:
        config_data = yaml.safe_load(f)
    return Settings.parse_obj(config_data)


settings = load_config()
