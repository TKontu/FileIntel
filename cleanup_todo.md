# Project Cleanup & Refinement Plan

This document outlines the necessary tasks to address technical debt, improve security, and align the codebase with its current RAG-centric architecture.

## 1. Critical Tasks

### 1.1. Implement Security
-   **Task:** Add authentication and authorization to all API endpoints.
-   **Recommendation:** Implement OAuth2 or a similar token-based authentication mechanism.
-   **Justification:** The API is currently open to the public, which is a critical security vulnerability.

### 1.2. Update Documentation
-   **Task:** Rewrite `architecture.md` to reflect the current RAG-based, collection-centric design.
-   **Task:** Update `API_USAGE.md` with the new collection and query endpoints.
-   **Justification:** The current documentation is misleading and hinders new developer onboarding.

## 2. Codebase Refactoring

### 2.1. Remove Legacy Code
-   **Task:** Delete the standalone batch processing system (`scripts/batch_process_files.py` and `src/document_analyzer/batch_processing/batch_manager.py`).
-   **Task:** Remove the `_process_single_file_job` and `_process_batch_job` methods from `src/document_analyzer/batch_processing/worker.py`.
-   **Justification:** This code is obsolete, confusing, and not integrated with the main RAG pipeline.

### 2.2. Improve Configuration
-   **Task:** Move hardcoded settings (e.g., upload directory, LLM providers) to `config/default.yaml`.
-   **Task:** Refactor the worker to dynamically select the LLM and embedding providers based on the configuration.
-   **Justification:** Hardcoded settings make the application inflexible and difficult to configure for different environments.

### 2.3. Enhance Error Handling
-   **Task:** Refactor the file upload process in `src/document_analyzer/api/routes/collections.py` to prevent orphaned files.
-   **Recommendation:** Use a transactional approach or a try/except/finally block to ensure that the file is deleted if the database operation fails.
-   **Justification:** The current implementation can lead to orphaned files in the `uploads` directory.

## 3. Feature Integration

### 3.1. Integrate Batch Processing (Optional)
-   **Task:** If batch processing is still a required feature, it should be integrated into the main application.
-   **Recommendation:** Create a new API endpoint that accepts a directory path and creates indexing jobs for each file in the directory.
-   **Justification:** The current batch processing system is a standalone script that doesn't use the RAG pipeline.
