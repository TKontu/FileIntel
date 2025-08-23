# Document Analysis System Architecture

## 1. Overview

This document describes the architecture of the Document Analysis System, a Python-based application designed to build and query knowledge bases from documents using a Retrieval-Augmented Generation (RAG) pipeline.

The system is built around the concept of "Collections"—user-defined knowledge bases containing multiple documents. It exposes a RESTful API for managing these collections, uploading documents for indexing, and asking questions against the aggregated knowledge of an entire collection.

The core of the system is an asynchronous, job-based processing pipeline. When a document is uploaded, an "indexing" job is created. A separate worker process chunks the document, creates vector embeddings, and stores them in a vector database. When a user asks a question, a "query" job is created, which retrieves the most relevant document chunks from the collection and uses a Large Language Model (LLM) to synthesize an answer.

## 2. System Architecture

The system is composed of the following core layers and components:

### Core Components

#### 2.1. API Layer (FastAPI)
**Location**: `src/document_analyzer/api/`

-   **Purpose**: Provides the external interface for managing collections and submitting jobs.
-   **Implementation**: Built with **FastAPI**. It defines Pydantic models for request/response validation and uses a single router for all collection-based operations.
    -   `collections.py`: Handles all CRUD operations for collections and documents, as well as submitting indexing and query jobs.
-   **Key Dependencies**: `fastapi`, `uvicorn`, `pydantic`.

#### 2.2. Asynchronous Job Processing
**Location**: `src/document_analyzer/batch_processing/`

-   **Purpose**: Manages the lifecycle of indexing and query jobs.
-   **Implementation**:
    -   `JobManager`: A central component for creating and submitting jobs to the storage layer.
    -   `Worker`: (Driven by `run_worker.py`) A process that continuously polls the database for pending jobs, processes them based on their type (`indexing` or `query`), and stores the results.
-   **Key Dependencies**: `SQLAlchemy`.

#### 2.3. Document Processing & RAG Pipeline
**Location**: `src/document_analyzer/document_processing/`

-   **Purpose**: Handles the parsing, chunking, and embedding of documents.
-   **Implementation**:
    -   `ReaderFactory`: Selects the appropriate document parser based on file type.
    -   `TextChunker`: Splits the extracted text into small, overlapping chunks suitable for embedding.
    -   `OpenAIEmbeddingProvider`: Uses an embedding model to convert text chunks into vector embeddings.

#### 2.4. LLM Integration Layer
**Location**: `src/document_analyzer/llm_integration/`

-   **Purpose**: Provides a standardized interface for communicating with different LLM providers.
-   **Implementation**: A provider-based design.
    -   `base.py`: Defines the abstract `LLMProvider` interface.
    -   `openai_provider.py`: Implementation for OpenAI models.
    -   `anthropic_provider.py`: Implementation for Anthropic models.

#### 2.5. Prompt Management System
**Location**: `src/document_analyzer/prompt_management/`

-   **Purpose**: Constructs the final prompts sent to the LLMs.
-   **Implementation**:
    -   `PromptLoader`: Reads prompt templates from the `/prompts` directory.
    -   `PromptComposer`: Assembles the different parts (e.g., retrieved context, user question) into a final, coherent prompt.
-   **Key Dependencies**: `Jinja2`.

#### 2.6. Storage Layer
**Location**: `src/document_analyzer/storage/`

-   **Purpose**: Manages all data persistence, including vector storage.
-   **Implementation**:
    -   `models.py`: Defines the SQLAlchemy schema for `collections`, `documents`, `document_chunks`, `jobs`, and `results`.
    -   `PostgreSQLStorage`: The concrete implementation of the storage interface, handling all database interactions. It uses the `pgvector` extension for efficient vector storage and similarity search.
-   **Key Dependencies**: `SQLAlchemy`, `psycopg2-binary`, `pgvector-sqlalchemy`.

## 3. Data Flow

### 3.1. Indexing Flow
1.  **Submission**: A user uploads a document to a collection via the `POST /api/v1/collections/{collection_id}/documents` endpoint.
2.  **Deduplication**: The system calculates the SHA256 hash of the file and checks if a document with the same hash already exists. If so, the process stops.
3.  **Job Creation**: If the document is new, a `document` record is created in the database, and an `indexing` job is created with a "pending" status.
4.  **Worker Pickup**: The worker retrieves the pending indexing job.
5.  **Content Extraction & Chunking**: The worker reads the document, extracts its text content, and splits it into chunks.
6.  **Embedding**: The worker uses the embedding provider to convert each chunk into a vector embedding.
7.  **Storage**: The worker stores the chunks and their embeddings in the `document_chunks` table.

### 3.2. Querying Flow
1.  **Submission**: A user submits a question to a collection via the `POST /api/v1/collections/{collection_id}/query` endpoint.
2.  **Job Creation**: A `query` job is created with a "pending" status.
3.  **Worker Pickup**: The worker retrieves the pending query job.
4.  **Question Embedding**: The worker embeds the user's question using the same embedding model.
5.  **Chunk Retrieval**: The worker performs a vector similarity search on the `document_chunks` table to find the most relevant chunks from the specified collection.
6.  **Prompt Composition**: The retrieved chunks are used as context in a prompt that is sent to the LLM along with the user's question.
7.  **LLM Analysis**: The LLM generates a response based on the provided context.
8.  **Result Storage**: The LLM's response is stored in the `results` table, and the job's status is updated to "completed".
9.  **Retrieval**: The user can poll the `/api/v1/jobs/{job_id}/result` endpoint to retrieve the final answer.
