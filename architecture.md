# FileIntel Architecture Overview

This document provides a high-level overview of the FileIntel application's architecture, its core components, and key workflows.

## Core Components

The application is built on a distributed task processing architecture, orchestrated with Docker Compose. The main components are:

- **API Service (`api`):** A FastAPI application that serves as the primary entry point for all user interactions. It provides both v1 (legacy) and v2 (task-based) endpoints for managing collections, uploading documents, and submitting Celery tasks for processing.

- **Celery Workers (`celery`):** Distributed task processors that handle all heavy, time-consuming operations asynchronously. Tasks are distributed across multiple workers and can be scaled horizontally. This includes document processing, RAG operations, GraphRAG indexing, and workflow orchestration.

- **PostgreSQL Database (`postgres`):** The primary data store. It uses the `pgvector` extension to store and query vector embeddings. It also stores all relational data, such as collections, documents, and GraphRAG metadata.

- **Redis (`redis`):** Functions as the Celery message broker and result backend for distributed task coordination. Also serves as a cache for LLM responses.

- **CLI (`fileintel`):** A command-line interface built with Typer that provides a user-friendly way to interact with both v1 and v2 API endpoints.

## Directory Structure (`src/fileintel`)

The project's source code is organized into distinct domains:

- `api/`: Contains the FastAPI application, including v1 (legacy) and v2 (task-based) routes and dependencies.
- `tasks/`: Contains Celery task definitions organized by domain: document processing, GraphRAG operations, LLM tasks, and workflow orchestration.
- `cli/`: Contains the Typer-based command-line interface for interacting with the task-based API.
- `storage/`: Manages database interactions, defining SQLAlchemy models and PostgreSQL storage implementation.
- `document_processing/`: Handles reading, chunking, and metadata extraction for various document formats.
- `rag/`: Contains the Retrieval-Augmented Generation implementations.
  - `vector_rag/`: Traditional embedding-based RAG.
  - `graph_rag/`: Integration layer for Microsoft's GraphRAG, including services and data adapters.
- `llm_integration/`: Manages connections to Large Language Models (e.g., OpenAI, Anthropic) and embedding providers. Includes resilience patterns like rate limiting and circuit breakers.
- `core/`: Contains shared application infrastructure, such as configuration, custom exceptions, and logging.
- `celery_config.py`: Celery application configuration and task discovery.

## Key Workflows

### 1. Document Ingestion and Indexing

1.  A user uploads a document via the v2 API endpoint.
2.  The **API Service** saves the file and creates a `Document` record in the PostgreSQL database.
3.  The API submits a Celery task (e.g., `process_document`) to the distributed task queue.
4.  A **Celery Worker** picks up the task, reads the document, extracts text and metadata, splits the text into chunks, and generates vector embeddings for each chunk.
5.  The worker saves the chunks and embeddings to the **PostgreSQL** database and updates task progress via Celery's result backend.

### 2. RAG Query (Vector and Graph)

1.  A user submits a query via the v2 API, specifying a collection and query parameters.
2.  The **API Service** creates a Celery task (e.g., `global_search_task` or vector RAG task) and submits it to the appropriate queue.
3.  A **Celery Worker** picks up the task.
4.  **For Vector RAG**: The worker generates an embedding for the query, finds relevant chunks in **PostgreSQL** via vector similarity search, and sends the context to an LLM.
5.  **For GraphRAG**: The worker uses the `GraphRAGService` to perform a global or local search on the pre-built graph index and synthesizes a response.
6.  The final result is returned via Celery's result backend and can be retrieved through the v2 API.

### 3. Workflow Orchestration

The system supports complex multi-step workflows using Celery's advanced patterns:

1.  **Groups**: Parallel execution of multiple tasks (e.g., processing multiple documents simultaneously)
2.  **Chains**: Sequential execution where each task feeds into the next (e.g., document processing → embedding generation → index building)
3.  **Chords**: Group execution followed by a callback (e.g., process all documents, then update collection index)

These patterns are implemented in `tasks/workflow_tasks.py` and enable sophisticated document processing pipelines.
