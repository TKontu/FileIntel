# Document Analysis System Architecture

## 1. Overview

This document describes the current architecture of the Document Analysis System. The system is a Python-based application designed to process and analyze documents using Large Language Models (LLMs). It exposes a RESTful API for submitting documents, managing analysis jobs, and retrieving results.

The core functionality is built around a job-based, asynchronous processing pipeline. Users can upload single files or initiate batch processing on a directory of files. The system uses a PostgreSQL database to track documents and jobs, and Redis for caching, ensuring that identical documents are not processed more than once.

## 2. System Architecture

The system is composed of the following core layers and components:

![System Architecture Diagram](https://i.imgur.com/example.png)  <!-- Placeholder for a diagram -->

### Core Components

#### 2.1. API Layer (FastAPI)
**Location**: `src/document_analyzer/api/`

-   **Purpose**: Provides the external interface for the system.
-   **Implementation**: Built with **FastAPI**. It defines Pydantic models for request/response validation and includes three main routers:
    -   `single.py`: Handles single file uploads.
    -   `batch.py`: Initiates processing for a pre-configured directory.
    -   `analysis.py`: A more advanced single-file submission endpoint that prevents duplicate processing and provides endpoints for checking job status and retrieving results.
-   **Key Dependencies**: `fastapi`, `uvicorn`, `pydantic`.

#### 2.2. Batch Processing Engine
**Location**: `src/document_analyzer/batch_processing/`

-   **Purpose**: Manages the lifecycle of analysis jobs.
-   **Implementation**:
    -   `JobManager`: The central component responsible for creating and submitting jobs to the storage layer. It is used by all API endpoints that initiate a processing task.
    -   `Worker`: (Driven by `run_worker.py`) A Celery-based worker that continuously polls the database for pending jobs, processes them, and stores the results.
-   **Key Dependencies**: `celery`, `redis`.

#### 2.3. Document Processing Layer
**Location**: `src/document_analyzer/document_processing/`

-   **Purpose**: Handles the parsing and content extraction from various document formats.
-   **Implementation**: A factory-based design.
    -   `TypeDetector`: Identifies the MIME type of a file.
    -   `factory.py`: Selects the appropriate processor based on the file type.
    -   `processors/`: Contains specialized modules for different formats (`.epub`, `.mobi`, `.pdf`, `.txt`). It includes processors that use `pdfplumber` for text-based PDFs and `pytesseract` for OCR.
-   **Key Dependencies**: `pdfplumber`, `ebooklib`, `mobi`, `pytesseract`.

#### 2.4. LLM Integration Layer
**Location**: `src/document_analyzer/llm_integration/`

-   **Purpose**: Provides a standardized interface for communicating with different LLM providers.
-   **Implementation**: A provider-based design.
    -   `base.py`: Defines the abstract `LLMProvider` interface.
    -   `openai_provider.py`: Implementation for OpenAI models.
    -   `anthropic_provider.py`: Implementation for Anthropic models.
    -   `RateLimiter`: A basic mechanism to control the request frequency to LLM APIs.
-   **Key Dependencies**: `openai`, `anthropic`, `tenacity`.

#### 2.5. Prompt Management System
**Location**: `src/document_analyzer/prompt_management/`

-   **Purpose**: Constructs the final prompts sent to the LLMs.
-   **Implementation**:
    -   `PromptLoader`: Reads prompt components (instruction, question, format) from markdown files in the `/prompts` directory.
    -   `TemplateEngine`: Uses Jinja2 for variable substitution.
    -   `PromptComposer`: Assembles the different parts into a final, coherent prompt.
-   **Key Dependencies**: `Jinja2`, `mistune`.

#### 2.6. Output Management System
**Location**: `src/document_analyzer/output_management/`

-   **Purpose**: Formats the raw LLM output into the desired structure.
-   **Implementation**: A factory-based design.
    -   `factory.py`: Selects the appropriate formatter.
    -   `formatters/`: Contains modules to format the output as JSON, Markdown, a simple list, a table, or an essay.
-   **Key Dependencies**: None.

#### 2.7. Storage Layer
**Location**: `src/document_analyzer/storage/`

-   **Purpose**: Manages all data persistence.
-   **Implementation**:
    -   `models.py`: Defines the SQLAlchemy schema for the `documents`, `jobs`, and `results` tables.
    -   `PostgreSQLStorage`: The concrete implementation of the storage interface, handling all database interactions.
    -   `RedisStorage`: Used for caching to prevent re-processing of existing documents.
-   **Key Dependencies**: `SQLAlchemy`, `psycopg2-binary`, `redis`.

## 3. Data Flow

1.  **Submission**: A user submits a document via the `/api/v1/analyze` endpoint. The API reads the file, calculates its SHA256 hash, and saves it to the `uploads/` directory.
2.  **Deduplication**: The system checks if a document with the same hash already exists in the database. If so, it returns the existing job ID and stops.
3.  **Job Creation**: If the document is new, a `document` record is created in the database. A new `job` record is then created with a "pending" status.
4.  **Worker Pickup**: A Celery worker, running separately, polls the database for pending jobs. It retrieves the job and the associated file path.
5.  **Content Extraction**: The worker passes the file to the **Document Processing Layer**, which detects the file type and uses the appropriate processor to extract its text content.
6.  **Prompt Composition**: The extracted text is sent to the **Prompt Management System**, which combines it with the relevant prompt templates from the `/prompts` directory.
7.  **LLM Analysis**: The final prompt is sent to the configured LLM via the **LLM Integration Layer**.
8.  **Output Formatting**: The raw response from the LLM is passed to the **Output Management System**, which formats it into the requested structure (e.g., JSON).
9.  **Result Storage**: The formatted result is stored in the `results` table in the database, and the job's status is updated to "completed".
10. **Retrieval**: The user can poll the `/api/v1/jobs/{job_id}/result` endpoint to retrieve the final result once the job is complete.

## 4. Key Design Patterns

-   **Strategy Pattern**: Used in the Document Processing and LLM Integration layers to allow for different implementations (e.g., different file type processors, different LLM providers) to be used interchangeably.
-   **Factory Pattern**: Used in the Document Processing and Output Management layers to select the appropriate processor or formatter at runtime based on the file type or requested output format.
-   **Task Queue**: The entire processing pipeline is managed asynchronously using a task queue (Celery), which decouples the API from the resource-intensive document processing work.

## 5. Project Directory Structure (Simplified)

```
C:\code\FileIntel\
├── alembic.ini
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── config/
│   └── default.yaml
├── prompts/
│   └── default_analysis/
├── src/
│   └── document_analyzer/
│       ├── api/
│       │   ├── main.py
│       │   └── routes/
│       │       ├── analysis.py
│       │       ├── batch.py
│       │       └── single.py
│       ├── batch_processing/
│       │   ├── job_manager.py
│       │   └── worker.py
│       ├── document_processing/
│       │   ├── factory.py
│       │   └── processors/
│       ├── llm_integration/
│       │   ├── openai_provider.py
│       │   └── anthropic_provider.py
│       ├── output_management/
│       │   ├── factory.py
│       │   └── formatters/
│       ├── prompt_management/
│       │   └── composer.py
│       └── storage/
│           ├── models.py
│           └── postgresql_storage.py
└── tests/
```