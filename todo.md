# Project Refinement and Hardening Plan

This document outlines the necessary refactoring, security hardening, and feature enhancements for the Document Analyzer project. It replaces the original development plan and focuses on improving the existing codebase.

---

## P1: Critical Security Vulnerabilities

*These issues must be addressed with the highest priority to protect the application and its users.*

-   [ ] **Implement Authentication & Authorization**
    -   **Description:** The API is currently open to the public, allowing unauthorized access to all endpoints.
    -   **Action:** Introduce an authentication mechanism (e.g., API Keys or OAuth2) to secure all endpoints. Ensure users can only access their own jobs and results.

-   [ ] **Restrict CORS Policy**
    -   **Description:** The current `cors_origins: ["*"]` policy is insecure and exposes the API to web-based attacks.
    -   **Action:** Change the CORS configuration to allowlist only trusted frontend domains.

-   [ ] **Fix Insecure File Handling**
    -   **Description:** The `single.py` endpoint saves files using their original, user-provided filenames, creating a potential path traversal vulnerability.
    -   **Action:** Standardize file saving logic across all endpoints. Use a secure, generated name for the file on disk (e.g., based on its content hash or a UUID) and store the original filename as metadata in the database.

---

## P2: Core Architectural Refactoring

*These are fundamental design flaws that impact the application's performance, scalability, and reliability.*

-   [ ] **Stream File Uploads to Disk**
    -   **Description:** The `analysis.py` endpoint reads entire uploaded files into memory, creating a significant bottleneck and risking memory exhaustion.
    -   **Action:** Refactor the file upload process to stream large files directly to a temporary disk location instead of loading them into memory.

-   [ ] **Eliminate File Upload Race Condition**
    -   **Description:** The "check-then-act" logic for checking a document's hash before saving is a race condition that can lead to duplicate processing.
    -   **Action:** Refactor the logic to attempt an atomic `INSERT` into the `documents` table and gracefully handle the `UNIQUE` constraint violation if the content hash already exists.

-   [ ] **Remove Hardcoded Paths and Configuration**
    -   **Description:** Several modules contain hardcoded absolute paths (e.g., `/home/appuser/app/uploads`), which makes the application brittle and difficult to configure.
    -   **Action:** Remove all hardcoded paths and values. Ensure that all configuration is loaded exclusively from the central `config.yaml` file.

-   [ ] **Implement Asynchronous I/O for File Operations**
    -   **Description:** Blocking file I/O operations are used within `async` FastAPI routes, which negates the benefits of the asynchronous framework.
    -   **Action:** Replace all synchronous file operations (`open`, `shutil.copyfileobj`) in API routes with an asynchronous library like `aiofiles` or use `FastAPI.run_in_threadpool`.

---

## P3: Code Quality and Maintainability

*These improvements will make the codebase cleaner, more consistent, and easier to maintain.*

-   [ ] **Unify Job Submission Logic**
    -   **Description:** The `JobManager` has multiple, confusing methods for submitting jobs (`submit_job`, `submit_batch_job`, `submit_file_job`).
    -   **Action:** Refactor the `JobManager` to provide a single, clear interface for job submission. The method should accept parameters to differentiate between job types (e.g., `single_file`, `batch`).

-   [ ] **Centralize Dependency Injection**
    -   **Description:** Dependency provider functions (`get_storage`, `get_job_manager`) are duplicated across multiple API route files.
    -   **Action:** Move the duplicated dependency injection functions into the central `dependencies.py` file to adhere to the DRY principle.

-   [ ] **Improve API Error Handling**
    -   **Description:** API error responses could be more descriptive and consistent.
    -   **Action:** Implement a centralized exception handling middleware in FastAPI to standardize error responses across the API.

---

## P4: Future Enhancements

*New features and capabilities to be considered after the critical and architectural issues are resolved.*

-   [ ] **Add Job Cancellation and Retry Endpoints**
    -   **Description:** Users currently cannot cancel a running job or easily retry a failed one.
    -   **Action:** Implement API endpoints (`/jobs/{job_id}/cancel`, `/jobs/{job_id}/retry`) and the corresponding `JobManager` logic.

-   [ ] **Real-time Job Progress Tracking**
    -   **Description:** Job status is only available via polling.
    -   **Action:** Implement WebSocket support to stream real-time progress updates to clients.

-   [ ] **Expand Document Processor Support**
    -   **Description:** The system could support more file types.
    -   **Action:** Add new processors for formats like `.docx`, `.pptx`, and improve handling of complex PDFs with tables and images.