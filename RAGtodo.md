# RAG Implementation Plan for Document Analyzer

**Objective:** Transition the document analysis process to a full-fledged Retrieval-Augmented Generation (RAG) architecture. This will enable the system to build and query **knowledge bases** (called "Collections") from multiple documents, allowing users to ask complex questions and receive synthesized answers based on the entire corpus.

---

## Architectural Overview

The new architecture is based on a collection-centric, two-phase process:

1.  **Indexing Phase:** Users upload documents into a specific **Collection**. Each document is processed, chunked into small, overlapping segments, converted into vector embeddings, and stored in a vector database. The chunks are linked to both their source document and the parent collection.
2.  **Querying Phase:** A user submits a question *against an entire collection*. The system converts the question into an embedding, retrieves the most relevant document chunks from *all documents* within that collection, and then feeds those chunks (as context) along with the question to the LLM to generate a single, comprehensive answer.

---

## Phased Implementation Plan

The project is broken down into five distinct phases.

### Phase 1: Foundational Setup (Database & Core Services)

**Goal:** Prepare the database, configuration, and core components required for the collection-based RAG pipeline.

-   [ ] **Task 1.1: Set Up the Vector Database and Data Models**
    -   **Action:** Enable the `pgvector` extension in the PostgreSQL database (e.g., `CREATE EXTENSION vector;`).
    -   **Action:** Add the `pgvector-sqlalchemy` library to `pyproject.toml`.
    -   **File:** `src/document_analyzer/storage/models.py`
        -   **Action:** Define a new `Collection` model (`id`, `name`, `created_at`).
        -   **Action:** Modify the `Document` model to include a `collection_id`.
        -   **Action:** Define a new `DocumentChunk` model (`id`, `document_id`, `collection_id`, `chunk_text`, `embedding`, `metadata`).
    -   **Action:** Create a new Alembic migration script to apply these schema changes.
        -   **CRITICAL:** The migration must also create an **HNSW index** on the `embedding` column of the `DocumentChunk` table for efficient vector searching. Example SQL: `CREATE INDEX ON document_chunks USING hnsw (embedding vector_cosine_ops);`

-   [ ] **Task 1.2: Update Configuration**
    -   **File:** `config/default.yaml`
        -   **Action:** Add a `rag` section with settings for `chunk_size`, `chunk_overlap`, `embedding_provider`, and `embedding_model`.
    -   **File:** `src/document_analyzer/core/config.py`
        -   **Action:** Update the config loader to parse these new settings.

-   [ ] **Task 1.3: Create the Chunking Service**
    -   **File:** Create `src/document_analyzer/document_processing/chunking.py`.
    -   **Action:** Implement a configurable `TextChunker` class.

-   [ ] **Task 1.4: Create the Embedding Service**
    -   **File:** Create `src/document_analyzer/llm_integration/embedding_provider.py`.
    -   **Action:** Define a base `EmbeddingProvider` and a concrete `OpenAIEmbeddingProvider`.

-   [ ] **Task 1.5: Extend the Storage Layer**
    -   **File:** `src/document_analyzer/storage/postgresql_storage.py`
    -   **Action:** Add methods for `add_document_chunks` and `find_relevant_chunks_in_collection`.

### Phase 2: Indexing Pipeline & CRUD Implementation

**Goal:** Implement the API and worker logic for creating collections, managing documents, and indexing them.

-   [ ] **Task 2.1: Update the Job Model**
    -   **File:** `src/document_analyzer/storage/models.py`
        -   **Action:** Modify the `Job` model to include `job_type`, `document_id`, and `collection_id`.
        -   **Action:** Create an Alembic migration for this change.

-   [ ] **Task 2.2: Modify the Worker for Indexing**
    -   **File:** `src/document_analyzer/batch_processing/worker.py`
    -   **Action:** Implement the "indexing" job logic (extract, chunk, embed, store).

-   [ ] **Task 2.3: Implement Full Collections & Documents API (CRUD)**
    -   **Action:** Create a new API router: `src/document_analyzer/api/routes/collections.py`.
    -   **Endpoints:**
        -   `POST /api/v1/collections`: Create a new collection.
        -   `GET /api/v1/collections`: List all collections.
        -   `GET /api/v1/collections/{collection_id}`: Get details of a single collection, including a list of its documents.
        -   `DELETE /api/v1/collections/{collection_id}`: Delete a collection and all its contents.
        -   `POST /api/v1/collections/{collection_id}/documents`: Upload a document to a collection and trigger an indexing job.
        -   `DELETE /api/v1/documents/{document_id}`: Delete a single document (and its associated chunks) from a collection.
    -   **File:** `src/document_analyzer/api/main.py`
        -   **Action:** Register the new `collections` router.

### Phase 3: Querying Pipeline Implementation

**Goal:** Implement the logic for handling user questions against a collection.

-   [ ] **Task 3.1: Modify the Worker for Querying**
    -   **File:** `src/document_analyzer/batch_processing/worker.py`
    -   **Action:** Add logic to handle the "query" job type (embed question, retrieve chunks, compose prompt, call LLM, save result).

-   [ ] **Task 3.2: Add the API Endpoint for Querying**
    -   **File:** `src/document_analyzer/api/routes/collections.py`
    -   **Endpoint:** `POST /api/v1/collections/{collection_id}/query`
        -   **Logic:** Submit a "query" job and return the `job_id`.

### Phase 4: Refinement & Cleanup

**Goal:** Update documentation, remove obsolete code, and add tests.

-   [ ] **Task 4.1: Update Documentation**
    -   **File:** `API_USAGE.md`, `architecture.md`
        -   **Action:** Rewrite documentation to reflect the new collection-based RAG workflow.

-   [ ] **Task 4.2: Code Cleanup**
    -   **Action:** Deprecate and remove the old `analysis.py`, `single.py`, and `batch.py` routers.

-   [ ] **Task 4.3: Add Tests**
    -   **Action:** Create unit and integration tests for the new RAG system.

### Phase 5: Quality & Performance Enhancements

**Goal:** Improve the relevance and accuracy of the RAG pipeline.

-   [ ] **Task 5.1: Implement a Reranking Stage**
    -   **Action:** Modify the querying worker logic to use a two-stage retrieval process.
    -   **Step 1 (Retrieval):** Fetch a larger number of candidate chunks from the vector store (e.g., top 20-50).
    -   **Step 2 (Reranking):** Use a lightweight, specialized reranker model (e.g., Cohere Rerank, or a cross-encoder model) to score the candidate chunks for their specific relevance to the user's question.
    -   **Step 3 (Generation):** Select the top 3-5 reranked chunks to build the final context for the LLM.
    -   **Rationale:** This significantly improves the quality of the context provided to the LLM, reducing noise and leading to more accurate and relevant answers.
