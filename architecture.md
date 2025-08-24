# FileIntel Architecture Overview

This document provides a high-level overview of the FileIntel application's architecture, its core components, and key workflows.

## Core Components

The application is built on a microservices-oriented architecture, orchestrated with Docker Compose. The main components are:

-   **API Service (`api`):** A FastAPI application that serves as the primary entry point for all user interactions, including the CLI. It handles HTTP requests for creating collections, uploading documents, and submitting analysis/query jobs.

-   **Worker Service (`worker`):** A background process that handles all heavy, time-consuming tasks asynchronously. This includes document indexing (text extraction, chunking, embedding generation) and running RAG analysis pipelines. It pulls jobs from a queue and processes them independently of the API.

-   **PostgreSQL Database (`postgres`):** The primary data store. It uses the `pgvector` extension to store and query the vector embeddings required for similarity searches in the RAG pipeline. It also stores all relational data, such as collections, documents, jobs, and results.

-   **Redis (`redis`):** Functions as the message broker and backend for the job queue. The API service places new jobs into Redis, and the Worker service retrieves them for processing.

-   **CLI (`fileintel`):** The command-line interface that provides a user-friendly way to interact with the API. It is built using Typer and makes direct HTTP requests to the API service.

## Directory Structure

The project's source code is organized as follows:

-   `src/document_analyzer/`: The main Python package.
    -   `api/`: Contains the FastAPI application, including routes and dependencies.
    -   `worker/`: Contains the logic for the background worker, including the `JobManager` and the core processing logic for different job types.
    -   `cli/`: Contains the Typer-based command-line interface.
    -   `storage/`: Manages database interactions, defining SQLAlchemy models and the storage interface.
    -   `document_processing/`: Handles all aspects of reading and parsing different document types (PDF, TXT, etc.).
    -   `llm_integration/`: Manages connections to Large Language Models and embedding providers.
    -   `prompt_management/`: Loads and composes prompts for the LLM from the file system.
-   `docker/`: Contains Docker-related files, including the `init.sql` for the PostgreSQL service.
-   `migrations/`: Contains Alembic database migration scripts (Note: Currently disabled in favor of `create_all`).
-   `prompts/`: Stores the modular prompt templates used for different analysis tasks.

## Key Workflows

### 1. Document Ingestion and Indexing

1.  A user uploads a document via the CLI or a direct API call.
2.  The **API Service** saves the file to the `uploads/` directory and creates a `Document` record in the PostgreSQL database.
3.  The API service then submits an `indexing` job to the **Redis** queue.
4.  A **Worker** process picks up the job, reads the document, extracts its text and metadata, splits the text into chunks, and generates vector embeddings for each chunk.
5.  The worker saves the chunks, embeddings, and extracted metadata back to the **PostgreSQL** database.

### 2. RAG Query/Analysis

1.  A user submits a query or analysis request via the CLI or API, specifying a collection or document and a task.
2.  The **API Service** creates a `query` or `analysis` job and places it in the **Redis** queue.
3.  A **Worker** process picks up the job.
4.  The worker generates an embedding for the user's query or the task's reference text.
5.  It queries the **PostgreSQL** database to find the most relevant document chunks based on vector similarity.
6.  The worker combines the retrieved chunks with the prompt template, sends the final prompt to the LLM, and receives the response.
7.  The final result, including the LLM's answer and the source document metadata, is saved to the **PostgreSQL** database.
