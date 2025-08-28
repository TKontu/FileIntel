# RAG Implementation Plan for Document Analyzer

## Architectural Overview

The new architecture is based on a collection-centric, two-phase process:

1.  **Indexing Phase:** Users upload documents into a specific **Collection**. Each document is processed, chunked into small, overlapping segments, converted into vector embeddings, and stored in a vector database. The chunks are linked to both their source document and the parent collection.
2.  **Querying Phase:** A user submits a question _against an entire collection_. The system converts the question into an embedding, retrieves the most relevant document chunks from _all documents_ within that collection, and then feeds those chunks (as context) along with the question to the LLM to generate a single, comprehensive answer.

---

## Phased Implementation Plan

- [x] **Task 4: Implement LLM-Powered Smart Metadata Extraction**

  - **Description:** During document ingestion, use an LLM to analyze the first few pages of text to intelligently extract key metadata, making the system more robust against missing file properties.
  - **Workflow:**
    1.  After initial chunking, send the first `N` chunks to an LLM with a specialized prompt.
    2.  The LLM will return a JSON object containing the extracted data.
    3.  This data will be merged with any metadata found in the file's properties and saved to the database.
  - **Fields to Extract:** `title`, `authors`, `publication_date`, `publisher`, `doi`, `source_url`, and a pre-generated `harvard_citation`.

- [ ] **Task 4.1: Implement Manual Metadata Update Endpoint**
  - **Description:** Provide an endpoint to allow users to manually correct or add metadata to a document after it has been processed.
  - **API:** `PUT /documents/{document_id}/metadata`
  - **Request Body:** A JSON object containing the metadata fields to be updated.

### Phase 5: Advanced Querying and Data Ingestion

**Goal:** Introduce more powerful ways to query documents and streamline the data ingestion process.

- [x] **Task 5.1: Implement Bulk Document Ingestion**

  - **Description:** Allow users to add all files from a specified folder path to a collection with a single command.
  - **CLI:**
    - Create a new command: `file-intel documents add-folder --collection <collection_name> --path <folder_path> [--recursive]`
    - The command will scan the target directory and upload each file to the specified collection.
  - **API:**
    - Create a new endpoint: `POST /collections/{collection_id_or_name}/documents/batch`
    - This endpoint will accept a `multipart/form-data` request containing multiple files.
    - It will create a background job to process the batch of files, associating each with the given collection. This ensures the API can handle a large number of files without timing out.

- [ ] **Task 5.2: Implement Advanced Comparative Analysis Query (Dynamic)**

  - **Description:** Create a sophisticated querying mechanism to analyze claims or facts within a document by comparing them against a larger body of reference material. This involves a two-sided, dynamic similarity search followed by a composed prompt.
  - **Workflow:**
    1.  **Find Chunks to Analyze:** Use an `analysis_embedding_path` text to find the `top-k` relevant chunks in a specified `analysis_document`.
    2.  **Find Reference Chunks (Dynamic):** For each chunk found in the previous step, use the **content of that chunk itself** as a dynamic query to find the `top-k` most relevant chunks from a `reference_source` (either a collection or another document). These form a tailored `reference_truth` for each analysis chunk.
    3.  **Compose & Execute Prompt:** For each analysis chunk, dynamically construct a prompt from templates: `task.md` + `chunk_to_be_analyzed` + `reference_truth` + `answer_format.md`. Execute this prompt against an LLM.
  - **CLI:**
    - `file-intel query comparative-analysis --analysis-doc-id <doc_id> --analysis-embedding-file <path> --reference-collection-id <coll_id> --task-file <path> --answer-format-file <path> [--top-k-analysis 5] [--top-k-reference 3]`
  - **API:**
    - Create a new endpoint: `POST /query/comparative-analysis`
    - Request Body:
      ```json
      {
        "analysis_document_id": "doc_id_1",
        "analysis_embedding_path": "prompts/examples/...",
        "top_k_analysis": 5,
        "reference_source": {
          "collection_id": "coll_id_1"
        },
        "top_k_reference": 3,
        "prompt_template_paths": {
          "task": "prompts/templates/...",
          "answer_format": "prompts/templates/..."
        }
      }
      ```
  - **Implementation Notes:** The backend will perform a dynamic search on the reference source for each chunk found in the analysis document. The process will likely be asynchronous, creating a job to handle the multiple LLM calls.

- [ ] **Task 5.3: Implement Full Document Processing**

  - **Description:** Allow users to process every chunk of a specific document with a given prompt.
  - **CLI:**
    - Create a new command: `file-intel documents process <document_id> --prompt <prompt>`
    - This will trigger a job that iterates through all chunks of the document and applies the prompt to each one.
  - **API:**
    - Create a new endpoint: `POST /documents/{document_id}/process-all`
    - Request Body:
      ```json
      {
        "prompt": "Your prompt here"
      }
      ```
    - This will create a background job to process all chunks of the document individually. The results could be retrieved via the job status endpoint.

- [ ] **Task 5.4: Implement Inverse Checking for Counterarguments**

  - **Description:** Create a workflow to actively seek out counterarguments or contradictory evidence for claims made in a document by inverting the claim before searching reference material.
  - **Workflow:**
    1.  **Find Claim:** Use an `analysis_embedding_path` to find the `top-k` chunks in the `analysis_document`.
    2.  **Generate Counter-Claim:** For each claim found, use an internal, system-level LLM prompt to rewrite the claim into its logical opposite (e.g., "The product is effective" -> "The product is ineffective or harmful").
    3.  **Search for Counterarguments:** Use the embedding of the newly **generated counter-claim** to search the `reference_source`. This will find evidence that contradicts the original claim.
    4.  **Synthesize Final Answer:** Compose a final prompt using the `original_claim`, the `found_counterarguments`, and user-provided templates (`task.md`, `answer_format.md`) to analyze the claim in light of the contradictory evidence.
  - **API:**
    - Create a new endpoint: `POST /query/inverse-check`
    - The request body will be identical to the `comparative-analysis` endpoint, as the core inputs are the same. The "inversion" step is a non-configurable internal process.
  - **CLI:**
    - Create a new command: `file-intel query inverse-check` with the same arguments as `comparative-analysis`.

- [ ] **Task 5.5: Implement Document Citation and Reference Search**

  - **Description:** Create a workflow to find and format citations for a document's content by searching for source material in a reference collection.
  - **Workflow:**
    1.  **Select Chunks to Cite:** Process either all chunks of a document or a `top-k` selection based on a reference text.
    2.  **Find Source:** For each chunk, use its content to search the `reference_source` for the best match.
    3.  **Generate Citation:** Use an LLM to format the retrieved source text and metadata into a formal citation (e.g., Harvard style) and a suggested in-text citation.
  - **API:**
    - Create a new endpoint: `POST /query/find-citations`
    - Request Body:
      ```json
      {
        "document_id": "doc_id_1",
        "reference_source": { "collection_id": "coll_id_1" },
        "citation_style": "harvard",
        "chunk_selection": {
          // Optional: if omitted, process all chunks
          "embedding_reference_path": "prompts/...",
          "top_k": 10
        }
      }
      ```
  - **Prerequisite:** This feature requires that documents in the reference collection have associated metadata (title, author, etc.) stored to enable proper citation formatting.

### Phase 6: Quality & Performance Enhancements

**Goal:** Improve the relevance and accuracy of the RAG pipeline.

- [ ] **Task 6.1: Implement a Reranking Stage**
  - **Action:** Modify the querying worker logic to use a two-stage retrieval process.
  - **Step 1 (Retrieval):** Fetch a larger number of candidate chunks from the vector store (e.g., top 20-50).
  - **Step 2 (Reranking):** Use a lightweight, specialized reranker model (e.g., Cohere Rerank, or a cross-encoder model) to score the candidate chunks for their specific relevance to the user's question.
  - **Step 3 (Generation):** Select the top 3-5 reranked chunks to build the final context for the LLM.
  - **Rationale:** This significantly improves the quality of the context provided to the LLM, reducing noise and leading to more accurate and relevant answers.
