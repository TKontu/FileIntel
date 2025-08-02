# Description

## Overview

A Python-based document analysis system that processes EPUB, MOBI, and PDF files using LLM APIs with customizable markdown-based prompts and structured output formats.

## Data Flow

1. **Input Stage**: Documents are uploaded or batch-loaded from directory
2. **Document Type Detection**: System analyzes document to determine processing method
3. **Processor Selection**: Choose optimal processor (PDF-Extract-Kit, traditional PDF, OCR fallback)
4. **Content Extraction**: Extract text, tables, images with unified DocumentElement format
5. **Content Validation**: Ensure meaningful content was extracted
6. **LLM Bridge**: Convert to LLM-optimized format (structured or flattened)
7. **Prompt Composition**: Merge instruction.md + question.md + answer_format.md
8. **LLM Processing**: Send to appropriate LLM provider with rate limiting
9. **Output Generation**: Format response according to specified output type
10. **Result Storage**: Store results and metadata for retrieval# Document Analysis System Architecture

## System Architecture

### Core Components

#### 1. Document Processing Layer

**Purpose**: Extract and preprocess content from various document formats using specialized tools

- **UnifiedDocumentProcessor**: Main orchestrator that manages processor selection and fallback
- **DocumentTypeDetector**: Analyzes documents to determine optimal processing method
- **DocumentToLLMBridge**: Converts processed elements to LLM-optimized formats
- **Processor Implementations**:
  - **PDFExtractKitProcessor**: Advanced layout understanding using PDF-Extract-Kit API
  - **TraditionalPDFProcessor**: Fast text extraction using pdfplumber for simple PDFs
  - **EPUBReader**: EPUB parsing using `ebooklib`
  - **MOBIReader**: MOBI parsing using specialized libraries
  - **FallbackOCRProcessor**: Tesseract OCR as universal fallback
- **DocumentElement Model**: Unified representation for text, tables, images, and metadata
- **ContentValidator**: Ensures meaningful content extraction before LLM processing

#### 2. Prompt Management System

**Purpose**: Handle markdown-based prompt composition and templating

- **PromptLoader**: Loads and validates markdown prompt files
- **PromptComposer**: Merges instruction.md + question.md + answer_format.md
- **TemplateEngine**: Handles variable substitution and dynamic content
- **PromptValidator**: Ensures prompt structure and completeness

#### 3. LLM Integration Layer

**Purpose**: Abstract interface for various LLM providers

- **LLMProvider Interface**: Abstract base for LLM implementations
- **OpenAIProvider**: OpenAI GPT models integration
- **AnthropicProvider**: Claude models integration
- **LocalLLMProvider**: Local model support (Ollama, etc.)
- **RateLimiter**: Manages API rate limits and retry logic
- **ResponseParser**: Extracts and validates LLM responses

#### 4. Batch Processing Engine

**Purpose**: Orchestrate multiple file processing with queue management

- **JobManager**: Manages processing queue and job lifecycle
- **Worker**: Processes individual documents
- **ProgressTracker**: Monitors batch processing status
- **ErrorHandler**: Manages failures and retry logic
- **ResultAggregator**: Collects and formats batch results

#### 5. Output Management System

**Purpose**: Handle various output formats and destinations

- **OutputFormatter Interface**: Abstract base for output formats
- **EssayFormatter**: Structured essay output
- **ListFormatter**: Bullet/numbered list output
- **TableFormatter**: Tabular data output (CSV, JSON, Markdown)
- **JSONFormatter**: Structured JSON output
- **OutputWriter**: Writes results to files or databases

#### 6. API Layer

**Purpose**: RESTful API for external integration

- **FastAPI Application**: Main API server
- **Authentication**: API key management and validation
- **Request Models**: Pydantic models for API requests
- **Response Models**: Standardized API responses
- **WebSocket Support**: Real-time progress updates
- **Documentation**: Auto-generated OpenAPI docs

#### 7. Configuration Management

**Purpose**: Centralized configuration and settings

- **ConfigManager**: Loads and validates configuration
- **Environment Variables**: Runtime configuration
- **Secrets Management**: API keys and sensitive data
- **Logging Configuration**: Structured logging setup

#### 8. Storage Layer

**Purpose**: Persistent storage for jobs, results, and metadata

- **Database Interface**: Abstract storage layer
- **SQLiteStorage**: Local database implementation
- **PostgreSQLStorage**: Production database support
- **FileStorage**: File-based result storage
- **CacheManager**: Redis-based caching for performance

#### 9. Evaluation and Quality Assurance

**Purpose**: Automated testing and quality validation of analysis results

- **EvaluationEngine**: Automated assessment of analysis quality
- **GroundTruthManager**: Management of reference datasets for validation
- **MetricsCollector**: Accuracy, completeness, and consistency scoring
- **A/BTestFramework**: Compare different prompts, models, and processing methods
- **HumanFeedbackLoop**: Integration for human review and correction
- **QualityReporting**: Dashboards and alerts for quality monitoring

```
document_analyzer/
├── src/
│   ├── document_analyzer/
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py
│   │   │   ├── exceptions.py
│   │   │   └── logging.py
│   │   ├── document_processing/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── unified_processor.py
│   │   │   ├── type_detector.py
│   │   │   ├── bridge.py
│   │   │   ├── processors/
│   │   │   │   ├── pdf_extract_kit.py
│   │   │   │   ├── traditional_pdf.py
│   │   │   │   ├── epub_processor.py
│   │   │   │   ├── mobi_processor.py
│   │   │   │   └── fallback_ocr.py
│   │   │   ├── elements.py
│   │   │   └── validator.py
│   │   ├── prompt_management/
│   │   │   ├── __init__.py
│   │   │   ├── loader.py
│   │   │   ├── composer.py
│   │   │   ├── template_engine.py
│   │   │   └── validator.py
│   │   ├── llm_integration/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── openai_provider.py
│   │   │   ├── anthropic_provider.py
│   │   │   ├── local_provider.py
│   │   │   └── rate_limiter.py
│   │   ├── batch_processing/
│   │   │   ├── __init__.py
│   │   │   ├── job_manager.py
│   │   │   ├── worker.py
│   │   │   ├── progress_tracker.py
│   │   │   └── error_handler.py
│   │   ├── output_management/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── formatters/
│   │   │   │   ├── essay.py
│   │   │   │   ├── list.py
│   │   │   │   ├── table.py
│   │   │   │   └── json.py
│   │   │   └── writer.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   ├── auth.py
│   │   │   ├── models.py
│   │   │   ├── routes/
│   │   │   │   ├── analysis.py
│   │   │   │   ├── batch.py
│   │   │   │   └── status.py
│   │   │   └── websocket.py
│   │   └── storage/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── sqlite_storage.py
│   │       ├── postgresql_storage.py
│   │       └── cache.py
│   └── cli/
│       ├── __init__.py
│       └── main.py
├── prompts/
│   ├── templates/
│   │   ├── instruction.md
│   │   ├── question.md
│   │   └── answer_format.md
│   └── examples/
├── config/
│   ├── default.yaml
│   ├── development.yaml
│   └── production.yaml
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/
│   ├── api.md
│   ├── usage.md
│   └── examples/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   └── prod.txt
├── scripts/
│   ├── setup.sh
│   └── deploy.sh
├── pyproject.toml
├── README.md
└── .env.example
```

## Data Flow

1. **Input Stage**: Documents are uploaded or batch-loaded from directory
2. **Processing Stage**:
   - Document text extraction
   - Prompt composition
   - LLM API calls with rate limiting
3. **Output Stage**:
   - Response parsing and validation
   - Format-specific output generation
   - Result storage and delivery

## Key Design Patterns

### 1. Strategy Pattern

- Multiple document readers for different formats
- Multiple LLM providers with unified interface
- Multiple output formatters

### 2. Factory Pattern

- Document reader factory based on file extension
- LLM provider factory based on configuration
- Output formatter factory based on requested format

### 3. Observer Pattern

- Progress tracking during batch processing
- WebSocket notifications for real-time updates

### 4. Template Method Pattern

- Base processing workflow with customizable steps
- Consistent error handling across components

## Configuration Schema

```yaml
llm:
  provider: "openai" # openai, anthropic, local
  model: "gpt-4"
  max_tokens: 4000
  temperature: 0.1
  rate_limit: 60 # requests per minute

document_processing:
  chunk_size: 4000
  overlap: 200
  max_file_size: "100MB"
  supported_formats: ["pdf", "epub", "mobi"]

ocr:
  primary_engine: "pdf_extract_kit" # pdf_extract_kit, tesseract, google_vision, azure_cv
  fallback_engines: ["tesseract", "google_vision"]
  pdf_extract_kit:
    api_endpoint: "http://localhost:8080"
    timeout: 30
    layout_detection: true
    table_extraction: true
  tesseract:
    languages: ["eng", "spa", "fra"]
    config: "--psm 6"
  cloud_ocr:
    google_vision_api_key: "${GOOGLE_VISION_API_KEY}"
    azure_cv_endpoint: "${AZURE_CV_ENDPOINT}"

output:
  default_format: "json"
  output_directory: "./results"
  include_metadata: true
  max_concurrent_jobs: 5
  retry_attempts: 3
  timeout: 300 # seconds

api:
  host: "0.0.0.0"
  port: 8000
  cors_origins: ["*"]
  rate_limit: 100 # requests per minute per client

storage:
  type: "sqlite" # sqlite, postgresql
  connection_string: "sqlite:///./database.db"
  cache_ttl: 3600 # seconds
```

## Security Considerations

- API key management and rotation
- Input validation and sanitization
- Rate limiting and abuse prevention
- Secure file handling and temporary file cleanup
- Authentication and authorization for API access
- Audit logging for compliance

## Scalability Considerations

- Horizontal scaling with worker processes
- Database connection pooling
- Caching layer for frequent operations
- Asynchronous processing with job queues
- Load balancing for API endpoints
- Container orchestration support

## Error Handling Strategy

- Graceful degradation for unsupported file formats
- Retry logic with exponential backoff
- Comprehensive error logging and monitoring
- User-friendly error messages
- Partial success handling for batch operations
