# RAGv2 Implementation Plan (Corrected)

This document outlines a revised plan to upgrade the FileIntel RAG pipeline. This version introduces a clear distinction between simple, question-based RAG and a more powerful, template-driven analysis workflow. Both workflows will be available for entire collections and for specific, persistent documents within those collections.

---

## Part 1: Core Enhancements & Configuration (Completed)

### Task 1: Update Configuration
-   **Status:** **Completed.**

### Task 2: Enhance Prompt Loader
-   **Status:** **Completed.**

---

## Part 2: API and Worker Refactoring (Completed)

### Task 3: Create Collection-Level API Endpoints
-   **Status:** **Completed.** (`/collections/{id}/query` and `/collections/{id}/analyze`)

### Task 4: Update Worker for Collection-Level Jobs
-   **Status:** **Completed.** (Worker has `_process_query_job` and `_process_analysis_job`)

---

## Part 3: Single-Document RAG Implementation (Correction Needed)

### Task 5: Correct Single-Document API Endpoints
-   **File:** `src/document_analyzer/api/routes/collections.py` (Modify existing file)
-   **Action:**
    1.  Add a `POST /collections/{coll_id}/documents/{doc_id}/query` endpoint. It will take a `QueryRequest` body.
    2.  Add a `POST /collections/{coll_id}/documents/{doc_id}/analyze` endpoint. It will take an `AnalysisRequest` body.
-   **File:** `src/document_analyzer/api/routes/documents.py` (Delete this file)
-   **Action:** The temporary, on-the-fly document processing is incorrect. This file and its associated router will be removed.

### Task 6: Correct Worker Logic for Single-Document RAG
-   **File:** `src/document_analyzer/batch_processing/worker.py`
-   **Action:**
    1.  **Add Job Types:** Introduce `document_query` and `document_analysis` job types.
    2.  **Create `_process_document_query_job`:** This method will be given a `document_id`. It will perform the similarity search *only on the chunks belonging to that document* and then execute the RAG strategy.
    3.  **Create `_process_document_analysis_job`:** This method will do the same as above, but for the template-driven analysis workflow.
    4.  **Remove In-Memory Logic:** The incorrect `_process_single_file_...` methods and the `_find_relevant_chunks_in_memory` helper will be completely removed.
-   **File:** `src/document_analyzer/storage/postgresql_storage.py`
-   **Action:** Add a new method `find_relevant_chunks_in_document` that is similar to the collection-level one but also filters by `document_id`.

---

## Part 4: CLI and Verification (Correction Needed)

### Task 7: Correct the CLI
-   **File:** `src/document_analyzer/cli/analyze.py` and `query.py`
-   **Action:**
    1.  The subcommands will be changed from `collection` and `document` to `from-collection` and `from-document`.
    2.  The `from-document` commands will now require a collection identifier and a document identifier (ID or filename).
        -   `fileintel analyze from-collection <coll_id> --task-name <task>`
        -   `fileintel analyze from-document <coll_id> <doc_id> --task-name <task>`
        -   `fileintel query from-collection <coll_id> <question>`
        -   `fileintel query from-document <coll_id> <doc_id> <question>`
-   **File:** `src/document_analyzer/cli/client.py`
-   **Action:** Update the client methods to call the new, corrected API endpoints.

### Task 8: Verify and Test
-   **Action:**
    1.  Test all four corrected workflows.
    2.  Ensure that a query/analysis on a document only returns results from that specific document's content.
