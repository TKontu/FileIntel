# Project Hardening Plan

This document outlines the necessary security enhancements to prepare the FileIntel application for a public release. The tasks are prioritized to address the most critical vulnerabilities first.

---

## P1: Critical Vulnerabilities

_These issues represent immediate risks and must be addressed before any public deployment._

### DONE: 1. Implement Optional API Key Authentication

- **Risk:** If the API is exposed publicly, it is open to unauthorized use, leading to data breaches and resource abuse.
- **Action Plan:**
  1.  **Make Authentication Configurable:** Add a new section to `config/default.yaml` to allow authentication to be enabled or disabled. This provides security for production while maintaining ease of use for local development.
      ```yaml
      api:
        authentication:
          enabled: false # Default to off
          api_key: ${API_KEY} # Load from environment
      ```
  2.  **Create a Reusable Security Dependency:** In `src/document_analyzer/api/dependencies.py`, create a single FastAPI dependency that:
      - Checks if `api.authentication.enabled` is `true`.
      - If enabled, it validates the `X-API-Key` header against the `API_KEY` environment variable.
      - If disabled, it allows the request to proceed without checks.
  3.  **Apply Security to All Routes:** Apply the new dependency to all relevant API endpoints to protect them when authentication is enabled.
  4.  **Document Usage:** Update the `README.md` or `API_USAGE.md` to explain how to enable authentication and use the API key.

### DONE: 2. Fix Insecure File Uploads (Path Traversal)

- **Risk:** The current method of saving files using their original filename allows a malicious actor to use path traversal (`../../`) to overwrite critical system files.
- **Action Plan:**
  1.  **Identify All File Upload Endpoints:** Review all API routes (especially in `src/document_analyzer/api/routes/`) that accept file uploads.
  2.  **Implement Secure Filename Generation:**
      - For every uploaded file, generate a new, secure, and unique filename. A good practice is to use a UUID (e.g., `uuid.uuid4()`).
      - Save the file to the `uploads` directory using this generated UUID as the filename.
  3.  **Store Original Filename as Metadata:**
      - Add a new column to the `documents` table in the database (e.g., `original_filename`).
      - Store the user's original filename in this column so it can be displayed in the UI without being used on the filesystem.

### 3. Prevent Denial-of-Service (DoS) from Large File Uploads

- **Risk:** Reading entire uploaded files into memory can lead to memory exhaustion, causing the server to crash if a large file is uploaded.
- **Action Plan:**
  1.  **Stream Uploads to Disk:**
      - Refactor the file upload logic in all relevant API routes.
      - Instead of `file.read()`, which loads the whole file into memory, open a temporary file on disk and write the uploaded content in chunks.
      - Use an asynchronous file library like `aiofiles` to avoid blocking the FastAPI event loop during file I/O.
  2.  **Enforce File Size Limits:**
      - Ensure the `max_file_size` setting from `config/default.yaml` is strictly enforced at the beginning of the upload process to reject oversized files immediately.

---

## P2: Important Hardening Tasks

_These are significant security improvements that should be completed after the critical vulnerabilities are fixed._

### 1. Restrict CORS Policy

- **Risk:** The current permissive `cors_origins: ["*"]` policy allows any website to interact with the API from a user's browser, making it vulnerable to Cross-Site Request Forgery (CSRF) and other attacks.
- **Action Plan:**
  1.  **Update Configuration:**
      - In `config/default.yaml`, change the `cors_origins` value from `["*"]` to a specific list of trusted domains (e.g., `["http://localhost:3000", "https://your-frontend-app.com"]`).
  2.  **Verify Middleware:**
      - Ensure the FastAPI CORS middleware correctly loads and applies this restricted list.

### 2. Externalize All Secrets

- **Risk:** Committing secrets like API keys to the repository is a major security risk. While the database password was moved, the LLM API key remains.
- **Action Plan:**
  1.  **Move LLM API Key:**
      - Remove the `api_key` from the `llm.openai` section of `config/default.yaml`.
      - Add an `OPENAI_API_KEY` entry to the `.env` file.
      - Update the configuration loader (`src/document_analyzer/core/config.py`) to read the OpenAI API key from the environment variables.

### 3. Remove Hardcoded Paths

- **Risk:** Hardcoded paths (e.g., for `uploads` or `prompts` directories) make the application brittle and difficult to deploy in different environments.
- **Action Plan:**
  1.  **Audit Codebase:**
      - Search the entire `src/` directory for any hardcoded file paths (e.g., `/home/appuser/app/uploads`).
  2.  **Centralize in Configuration:**
      - For any hardcoded paths found, create a corresponding entry in `config/default.yaml`.
      - Refactor the code to read these paths from the central configuration object.
