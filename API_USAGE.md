# How to Use the RAG API

This guide provides a complete workflow for creating a knowledge base (a "Collection"), uploading documents to it, and asking questions against the entire collection.

---

## Step 1: Create a Collection

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

If a job fails or you get an error, check the service logs:

```bash
# Check the API service logs
docker-compose logs api

# Check the worker service logs
docker-compose logs worker
```
