# How to Use the FileIntel API

This guide provides a complete workflow for creating a knowledge base (a "Collection"), uploading documents to it, and asking questions against the entire collection.

## API Versions

FileIntel provides two API versions:

- **v2 API (Recommended)**: Task-based API using Celery distributed processing. Provides better scalability, monitoring, and fault tolerance.
- **v1 API (Legacy)**: Original job-based API. **Deprecated** - use v2 for new integrations.

⚠️ **Note**: v1 API endpoints are maintained for backward compatibility but are deprecated. New features and improvements are only available in v2.

---

# V2 API Workflow (Recommended)

## Authentication

V2 API requires authentication. Set the Authorization header:

```bash
# Set your API key
export API_KEY="your-api-key-here"
```

## Step 1: Create a Collection (v2)

### PowerShell
```powershell
$headers = @{ "Authorization" = "Bearer $env:API_KEY"; "Content-Type" = "application/json" }
$body = @{ name = "My Collection"; description = "Example collection" } | ConvertTo-Json
Invoke-WebRequest -Uri http://localhost:8000/api/v2/collections -Method POST -Body $body -Headers $headers
```

### Bash/cURL
```bash
curl -X POST "http://localhost:8000/api/v2/collections" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "My Collection", "description": "Example collection"}'
```

## Step 2: Upload and Process Documents (v2)

### Upload Document
```bash
curl -X POST "http://localhost:8000/api/v2/collections/{collection_id}/documents" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@document.pdf"
```

### Submit Document Processing Task
```bash
curl -X POST "http://localhost:8000/api/v2/collections/{collection_id}/process" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "complete_analysis",
    "file_paths": ["/path/to/document.pdf"],
    "build_graph": true,
    "generate_embeddings": true
  }'
```

**Response includes task_id for monitoring progress**

## Step 3: Monitor Task Progress (v2)

### Check Task Status
```bash
curl "http://localhost:8000/api/v2/tasks/{task_id}" \
  -H "Authorization: Bearer $API_KEY"
```

### List Active Tasks
```bash
curl "http://localhost:8000/api/v2/tasks/active" \
  -H "Authorization: Bearer $API_KEY"
```

### Get Task Metrics
```bash
curl "http://localhost:8000/api/v2/tasks/metrics" \
  -H "Authorization: Bearer $API_KEY"
```

## Step 4: Query Collections (v2)

### Submit Query Task
```bash
curl -X POST "http://localhost:8000/api/v2/tasks/submit" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "task_name": "global_search_task",
    "args": ["What is the main topic?", "{collection_id}"],
    "queue": "graphrag"
  }'
```

### Direct GraphRAG Query (if available)
```bash
curl -X POST "http://localhost:8000/api/v2/graphrag/query" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic?",
    "collection_id": "{collection_id}",
    "search_type": "global"
  }'
```

## Step 5: Cancel Tasks (v2)

```bash
curl -X POST "http://localhost:8000/api/v2/tasks/{task_id}/cancel" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"force": false}'
```

---

# V1 API Workflow (Legacy - Deprecated)

## Step 1: Create a Collection (v1)

First, create a collection to hold your documents.

### PowerShell
```powershell
Invoke-WebRequest -Uri http://localhost:8000/api/v1/collections -Method POST -Body @{name="My First Collection"}
```

### Bash/cURL
```bash
curl -X POST -d "name=My First Collection" http://localhost:8000/api/v1/collections
```

**➡️ Action:** A `collection_id` will be returned. **Copy this ID** for the next steps.

---

## Step 2: Upload a Document to the Collection

Next, upload a document to your newly created collection.

### PowerShell
Replace `<your_collection_id>` with the ID you copied.
```powershell
$form = @{ file = Get-Item -Path "testfile.pdf" }; Invoke-WebRequest -Uri http://localhost:8000/api/v1/collections/<your_collection_id>/documents -Method POST -Form $form
```

### Bash/cURL
Replace `<your_collection_id>` with the ID you copied.
```bash
curl -X POST -F "file=@testfile.pdf" http://localhost:8000/api/v1/collections/<your_collection_id>/documents
```

**➡️ Action:** The document will be uploaded, and an indexing job will be created in the background. You can upload multiple documents to the same collection.

---

## Step 3: Query the Collection

Once your documents have been indexed, you can ask questions against the entire collection.

### PowerShell
Replace `<your_collection_id>` with the ID you copied.
```powershell
$body = @{ question = "What is the main topic of the documents in this collection?" } | ConvertTo-Json
Invoke-WebRequest -Uri http://localhost:8000/api/v1/collections/<your_collection_id>/query -Method POST -Body $body -ContentType "application/json"
```

### Bash/cURL
Replace `<your_collection_id>` with the ID you copied.
```bash
curl -X POST -H "Content-Type: application/json" -d '{"question": "What is the main topic of the documents in this collection?"}' http://localhost:8000/api/v1/collections/<your_collection_id>/query
```

**➡️ Action:** A `job_id` will be returned. **Copy this ID** to check the status and get the result.

---

## Step 4: Check Job Status and Get the Result

You can check the status of your query job and retrieve the result once it's complete.

### Check Status (PowerShell)
```powershell
Invoke-WebRequest -Uri http://localhost:8000/api/v1/jobs/<your_job_id>/status
```

### Check Status (Bash/cURL)
```bash
curl http://localhost:8000/api/v1/jobs/<your_job_id>/status
```

### Get Result (PowerShell)
```powershell
Invoke-WebRequest -Uri http://localhost:8000/api/v1/jobs/<your_job_id>/result
```

### Get Result (Bash/cURL)
```bash
curl http://localhost:8000/api/v1/jobs/<your_job_id>/result
```

---

## Troubleshooting

### V2 API Issues

If a task fails or you get an error, check the service logs:

```bash
# Check the API service logs
docker-compose logs api

# Check the Celery worker logs
docker-compose logs celery

# Check Redis broker logs
docker-compose logs redis

# Monitor tasks with Celery Flower (if running)
# Visit http://localhost:5555
```

### Common V2 API Issues

1. **Authentication Errors (401)**
   - Verify API key is set correctly
   - Check Authorization header format: `Bearer your-api-key`

2. **Task Not Found (404)**
   - Task may have expired from result backend
   - Check task_id is correct
   - Verify task was submitted successfully

3. **Task Failures**
   - Check Celery worker logs for detailed error messages
   - Verify database connectivity
   - Check if required queues have active workers

4. **Slow Task Processing**
   - Monitor active tasks: `GET /api/v2/tasks/active`
   - Check worker metrics: `GET /api/v2/tasks/metrics`
   - Scale workers if needed

### V1 API Issues (Legacy)

For v1 API troubleshooting, check job status and logs as described in the original workflow.
