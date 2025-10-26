# FileIntel Commands Reference

## CLI Commands

### Collections Management
```bash
# Create a new collection
fileintel collections create "my-collection"

# List all collections
fileintel collections list

# Get collection details
fileintel collections get "my-collection"

# Delete a collection
fileintel collections delete "my-collection"
```

### Document Management
```bash
# Upload a single document to a collection
fileintel documents upload "my-collection" "file.pdf"

# List documents in a collection
fileintel documents list "my-collection"

# Get document details (including metadata)
fileintel documents get "document-id"
fileintel documents details "document-id"  # Same as get

# Update document metadata
fileintel documents update-metadata "document-id" --title "New Title" --authors "Author 1, Author 2"

# Delete a document from a collection
fileintel documents delete "my-collection" "document-id"

# Bulk upload documents from a folder
fileintel documents add-folder "my-collection" "path/to/folder"
fileintel documents add-folder "my-collection" "path/to/folder" --recursive
fileintel documents add-folder "my-collection" "path/to/folder" --extensions pdf,docx,txt --max-files 100

# Process all chunks of a document with a custom prompt, separate response for each chunk
fileintel documents process "document-id" "Your custom prompt here"
```

### Querying (RAG)
```bash
# Query an entire collection
fileintel query from-collection "my-collection" "What is the main topic?"

# Query a specific document
fileintel query from-document "my-collection" "document-id" "What does this document say about X?"

# Advanced comparative analysis
fileintel query comparative-analysis \
  --analysis-doc-id "doc-id" \
  --analysis-embedding-file "prompts/analysis.txt" \
  --reference-collection-id "reference-collection" \
  --task-file "prompts/task.md" \
  --answer-format-file "prompts/format.md" \
  --top-k-analysis 5 \
  --top-k-reference 3

# Inverse checking for counterarguments
fileintel query inverse-check \
  --analysis-doc-id "doc-id" \
  --analysis-embedding-file "prompts/claims.txt" \
  --reference-collection-id "reference-collection" \
  --task-file "prompts/task.md" \
  --answer-format-file "prompts/format.md"

# Find citations for ALL chunks of a document (uses citation-optimized chunking: 4-6 sentences per chunk)
fileintel query find-citations \
  --document-id "your-document-id" \
  --reference-collection-id "sources-collection" \
  --citation-style "harvard"

# Find citations for specific chunks (uses citation-optimized chunking + embedding selection)
fileintel query find-citations \
  --document-id "your-document-id" \
  --reference-collection-id "sources-collection" \
  --citation-style "harvard" \
  --embedding-reference-file "prompts/sections_to_cite.txt" \
  --top-k 10

### Analysis (Template-based)
```bash
# Analyze an entire collection with a template
fileintel analyze from-collection "my-collection" --task-name "expert_analysis"

# Analyze a specific document with a template
fileintel analyze from-document "my-collection" "document-id" --task-name "expert_analysis"
```

### Job Management
```bash
# List jobs (newest first)
fileintel jobs list

# List jobs with filtering
fileintel jobs list --status completed
fileintel jobs list --type indexing
fileintel jobs list --status pending --limit 10

# Check job status
fileintel jobs status "job-id"

# Get job result (JSON)
fileintel jobs result "job-id"

# Save job result as JSON to file
fileintel jobs result "job-id" --save "result.json"

# Get job result as markdown (display in console)
fileintel jobs result "job-id" --md

# Save job result as markdown to file (auto-detects .md extension)
fileintel jobs result "job-id" --save "result.md"

# Display markdown in console AND save to file
fileintel jobs result "job-id" --md --save "result.md"
```

## API Endpoints

### Collections Management

#### List Collections
```bash
curl -X GET "http://localhost:8000/api/v1/collections" -H "accept: application/json"
```

#### Create Collection
```bash
curl -X POST "http://localhost:8000/api/v1/collections" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "name=my-collection"
```

#### Get Collection Details
```bash
curl -X GET "http://localhost:8000/api/v1/collections/{collection_id_or_name}" -H "accept: application/json"
```

#### Delete Collection
```bash
curl -X DELETE "http://localhost:8000/api/v1/collections/{collection_id_or_name}" -H "accept: application/json"
```

### Document Management

#### Upload Single Document
```bash
curl -X POST "http://localhost:8000/api/v1/collections/{collection_id_or_name}/documents" \
  -H "accept: application/json" \
  -F "file=@/path/to/document.pdf"
```

#### Upload Multiple Documents (Batch)
```bash
curl -X POST "http://localhost:8000/api/v1/collections/{collection_id_or_name}/documents/batch" \
  -H "accept: application/json" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.pdf" \
  -F "files=@doc3.pdf"
```

#### List Documents in Collection
```bash
curl -X GET "http://localhost:8000/api/v1/collections/{collection_id_or_name}/documents" -H "accept: application/json"
```

#### Get Document Details
```bash
curl -X GET "http://localhost:8000/api/v1/documents/{document_id}" -H "accept: application/json"
```

#### Update Document Metadata
```bash
curl -X PUT "http://localhost:8000/api/v1/documents/{document_id}/metadata" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Updated Title",
    "authors": "Author 1, Author 2",
    "publication_date": "2024-01-01",
    "publisher": "Publisher Name"
  }'
```

#### Delete Document
```bash
curl -X DELETE "http://localhost:8000/api/v1/collections/{collection_id_or_name}/documents/{document_id_or_filename}" \
  -H "accept: application/json"
```

#### Process All Document Chunks
```bash
curl -X POST "http://localhost:8000/api/v1/documents/{document_id}/process-all" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Your custom prompt for each chunk"}'
```

### Querying (RAG)

#### Query Collection
```bash
curl -X POST "http://localhost:8000/api/v1/collections/{collection_id_or_name}/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?"}'
```

#### Query Document
```bash
curl -X POST "http://localhost:8000/api/v1/collections/{collection_id_or_name}/documents/{document_id_or_filename}/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What does this document say about X?"}'
```

#### Comparative Analysis
```bash
curl -X POST "http://localhost:8000/api/v1/query/comparative-analysis" \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_document_id": "doc-id",
    "analysis_embedding_path": "/path/to/analysis.txt",
    "top_k_analysis": 5,
    "reference_source": {
      "collection_id": "reference-collection-id"
    },
    "top_k_reference": 3,
    "prompt_template_paths": {
      "task": "/path/to/task.md",
      "answer_format": "/path/to/format.md"
    }
  }'
```

#### Inverse Check (Counterarguments)
```bash
curl -X POST "http://localhost:8000/api/v1/query/inverse-check" \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_document_id": "doc-id",
    "analysis_embedding_path": "/path/to/claims.txt",
    "top_k_analysis": 5,
    "reference_source": {
      "collection_id": "reference-collection-id"
    },
    "top_k_reference": 3,
    "prompt_template_paths": {
      "task": "/path/to/task.md",
      "answer_format": "/path/to/format.md"
    }
  }'
```

#### Find Citations
```bash
curl -X POST "http://localhost:8000/api/v1/query/find-citations" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "doc-id",
    "reference_source": {
      "collection_id": "reference-collection-id"
    },
    "citation_style": "harvard",
    "chunk_selection": {
      "embedding_reference_path": "/path/to/sections.txt",
      "top_k": 10
    }
  }'
```

### Analysis (Template-based)

#### Analyze Collection
```bash
curl -X POST "http://localhost:8000/api/v1/collections/{collection_id_or_name}/analyze" \
  -H "Content-Type: application/json" \
  -d '{"task_name": "expert_analysis"}'
```

#### Analyze Document
```bash
curl -X POST "http://localhost:8000/api/v1/collections/{collection_id_or_name}/documents/{document_id_or_filename}/analyze" \
  -H "Content-Type: application/json" \
  -d '{"task_name": "expert_analysis"}'
```

### Job Management

#### List Jobs
```bash
# List all jobs (newest first, limit 20)
curl -X GET "http://localhost:8000/api/v1/jobs" -H "accept: application/json"

# List with filtering
curl -X GET "http://localhost:8000/api/v1/jobs?status=completed" -H "accept: application/json"
curl -X GET "http://localhost:8000/api/v1/jobs?job_type=indexing&limit=10" -H "accept: application/json"
curl -X GET "http://localhost:8000/api/v1/jobs?status=pending&job_type=query" -H "accept: application/json"
```

#### Get Job Status
```bash
curl -X GET "http://localhost:8000/api/v1/jobs/{job_id}/status" -H "accept: application/json"
```

#### Get Job Result (JSON)
```bash
curl -X GET "http://localhost:8000/api/v1/jobs/{job_id}/result" -H "accept: application/json"
```

#### Get Job Result (Markdown)
```bash
curl -X GET "http://localhost:8000/api/v1/jobs/{job_id}/result/markdown" -H "accept: text/plain"
```

### Legacy Single-File Analysis

#### Basic Analysis (Default Template)
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -F "file=@/path/to/document.pdf"
```

#### Custom Analysis (Custom Template)
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -F "file=@/path/to/document.pdf" \
  -F "task_name=expert_analysis"
```

#### Batch Analysis
```bash
# Process all files in input/ directory with default template
curl -X POST "http://localhost:8000/api/v1/batch" -H "Content-Type: application/json" -d '{}'

# Process with custom template
curl -X POST "http://localhost:8000/api/v1/batch" \
  -H "Content-Type: application/json" \
  -d '{"task_name": "expert_analysis"}'
```

### Response Format
All API endpoints that create background jobs return a response in this format:
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-g1h2-i3j4k5l6m7n8"
}
```

Use the `job_id` to check status and retrieve results using the job management endpoints.

## Example Workflows

### Basic RAG Workflow
1. Create collection: `fileintel collections create "my-docs"`
2. Upload documents: `fileintel documents add-folder "my-docs" "./documents"`
3. Query collection: `fileintel query from-collection "my-docs" "What are the key findings?"`
4. Check results: `fileintel jobs result <job-id>`

### Advanced Analysis Workflow
1. Upload analysis document: `fileintel documents upload "analysis" "paper.pdf"`
2. Upload reference collection: `fileintel documents add-folder "references" "./literature"`
3. Run comparative analysis:
```bash
fileintel query comparative-analysis \
  --analysis-doc-id "paper-doc-id" \
  --analysis-embedding-file "./prompts/claims.txt" \
  --reference-collection-id "references" \
  --task-file "./prompts/compare_task.md" \
  --answer-format-file "./prompts/output_format.md"
```
4. Check counterarguments:
```bash
fileintel query inverse-check \
  --analysis-doc-id "paper-doc-id" \
  --analysis-embedding-file "./prompts/claims.txt" \
  --reference-collection-id "references" \
  --task-file "./prompts/counter_task.md" \
  --answer-format-file "./prompts/output_format.md"
```

### Citation Generation Workflow
1. Prepare document with content to cite
2. Prepare reference collection with source materials
3. Generate citations:
```bash
fileintel query find-citations \
  --document-id "content-doc-id" \
  --reference-collection-id "sources" \
  --citation-style "harvard" \
  --embedding-reference-file "./prompts/sections_to_cite.txt"
```
