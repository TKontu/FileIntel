# Document Export API Endpoints

## Overview

New API endpoints for inspecting and exporting document chunks. Much easier than running scripts inside Docker containers!

## Base URL

```
http://localhost:8000/api/v2
```

## Endpoints

### 1. Get Document Information

Get document metadata and statistics.

**Endpoint**: `GET /documents/{document_id}`

**Example**:
```bash
curl http://localhost:8000/api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863
```

**Response**:
```json
{
  "success": true,
  "data": {
    "document_id": "3b9e6ac7-2152-4133-bd87-2cd0ffc09863",
    "filename": "8ff55694-8df4-41f4-98ca-f0b46d9ef122.pdf",
    "collection_id": "d3f19c71-91d8-46c1-9e4e-64129f8373b5",
    "created_at": "2025-10-19T14:30:00",
    "metadata": {
      "file_path": "/uploads/document.pdf",
      "file_size": 2547392
    },
    "statistics": {
      "total_chunks": 794,
      "vector_chunks": 794,
      "graph_chunks": 0
    }
  }
}
```

---

### 2. Get Document Chunks (JSON)

Retrieve all chunks for a document as JSON.

**Endpoint**: `GET /documents/{document_id}/chunks`

**Query Parameters**:
- `chunk_type` (optional): Filter by `vector` or `graph`

**Examples**:
```bash
# Get all chunks
curl http://localhost:8000/api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863/chunks

# Get only vector chunks
curl "http://localhost:8000/api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863/chunks?chunk_type=vector"

# Get only graph chunks
curl "http://localhost:8000/api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863/chunks?chunk_type=graph"
```

**Response**:
```json
{
  "document_id": "3b9e6ac7-2152-4133-bd87-2cd0ffc09863",
  "filename": "8ff55694-8df4-41f4-98ca-f0b46d9ef122.pdf",
  "collection_id": "d3f19c71-91d8-46c1-9e4e-64129f8373b5",
  "total_chunks": 794,
  "chunks": [
    {
      "id": "chunk_0",
      "text": "This is the first chunk content...",
      "position": 0,
      "metadata": {
        "chunk_type": "vector",
        "page_number": 1,
        "token_count": 387,
        "heuristic_type": "prose",
        "classification_source": "mineru"
      }
    },
    {
      "id": "chunk_1",
      "text": "This is the second chunk content...",
      "position": 1,
      "metadata": {
        "chunk_type": "vector",
        "page_number": 1,
        "token_count": 425,
        "heuristic_type": "bullet_list",
        "classification_source": "statistical"
      }
    }
    // ... 792 more chunks
  ]
}
```

---

### 3. Export Document Chunks (Markdown)

Export all chunks as a downloadable markdown file.

**Endpoint**: `GET /documents/{document_id}/export`

**Query Parameters**:
- `chunk_type` (optional): Filter by `vector` or `graph`
- `include_metadata` (optional, default: `false`): Include chunk metadata

**Examples**:
```bash
# Basic export (downloads markdown file)
curl -O -J http://localhost:8000/api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863/export

# Export with metadata
curl -O -J "http://localhost:8000/api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863/export?include_metadata=true"

# Export only graph chunks
curl -O -J "http://localhost:8000/api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863/export?chunk_type=graph"

# Export with custom output filename
curl "http://localhost:8000/api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863/export?include_metadata=true" -o my_document.md
```

**Response**: Markdown file download

**File Content Example**:
```markdown
# Document Export: 8ff55694-8df4-41f4-98ca-f0b46d9ef122.pdf

**Document ID**: `3b9e6ac7-2152-4133-bd87-2cd0ffc09863`
**Collection ID**: `d3f19c71-91d8-46c1-9e4e-64129f8373b5`
**Created**: 2025-10-19 14:30:00
**Total Chunks**: 794
**Exported**: 2025-10-19T19:45:00

## Document Metadata

- **file_path**: /uploads/document.pdf
- **file_size**: 2547392

---

## Document Content

### Chunk 1

<details>
<summary>Chunk Metadata</summary>

**Position**: 0
**Type**: vector
**Page**: 1
**Tokens**: 387
**Content Type**: prose
**Classification**: mineru

</details>

[Chunk text content here...]

---

### Chunk 2

...
```

---

## Usage Examples

### Browser

Simply open in your browser:
```
http://localhost:8000/api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863/export
```

The file will automatically download!

### Python

```python
import requests

# Get document info
response = requests.get(
    "http://localhost:8000/api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863"
)
doc_info = response.json()
print(f"Document has {doc_info['data']['statistics']['total_chunks']} chunks")

# Download markdown export
response = requests.get(
    "http://localhost:8000/api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863/export",
    params={'include_metadata': True}
)

with open('exported_document.md', 'w', encoding='utf-8') as f:
    f.write(response.text)

print("Exported to exported_document.md")
```

### wget

```bash
# Download with original filename
wget --content-disposition \
  "http://localhost:8000/api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863/export?include_metadata=true"

# Or specify output filename
wget -O my_document.md \
  "http://localhost:8000/api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863/export"
```

### JavaScript/Fetch

```javascript
// Get document info
const response = await fetch(
  'http://localhost:8000/api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863'
);
const docInfo = await response.json();
console.log(`Total chunks: ${docInfo.data.statistics.total_chunks}`);

// Download markdown
const exportResponse = await fetch(
  'http://localhost:8000/api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863/export?include_metadata=true'
);
const markdown = await exportResponse.text();

// Create download link
const blob = new Blob([markdown], { type: 'text/markdown' });
const url = URL.createObjectURL(blob);
const a = document.createElement('a');
a.href = url;
a.download = 'document_export.md';
a.click();
```

---

## Finding Document IDs

### From API

```bash
# List collections
curl http://localhost:8000/api/v2/collections

# Get documents in a collection
curl http://localhost:8000/api/v2/collections/{collection_id}/documents
```

### From Logs

```bash
# Get recently processed documents
docker-compose logs celery-worker | grep "document_id" | tail -10
```

### From Swagger UI

Visit: `http://localhost:8000/docs`

Navigate to the documents endpoints and try them interactively!

---

## API Documentation

### Interactive Docs (Swagger UI)

Visit: `http://localhost:8000/docs`

All endpoints are documented with:
- ✅ Request parameters
- ✅ Response schemas
- ✅ Try it out functionality
- ✅ Examples

### ReDoc

Visit: `http://localhost:8000/redoc`

Alternative documentation interface.

---

## Error Responses

### Document Not Found (404)

```json
{
  "detail": "Document 3b9e6ac7-2152-4133-bd87-2cd0ffc09863 not found"
}
```

### No Chunks Found (404)

```json
{
  "detail": "No chunks found for document 3b9e6ac7-2152-4133-bd87-2cd0ffc09863 with chunk_type=graph"
}
```

### Authentication Required (401)

If API key authentication is enabled:
```json
{
  "detail": "API key required"
}
```

Pass API key via header:
```bash
curl -H "X-API-Key: your-api-key" \
  http://localhost:8000/api/v2/documents/{document_id}/export
```

---

## Benefits Over Script

| Feature | Script | API Endpoint |
|---------|--------|--------------|
| **Access** | Inside Docker only | From anywhere |
| **Authentication** | Direct DB access | API key auth |
| **Download** | Manual file copy | Automatic download |
| **Integration** | Shell scripts | HTTP requests |
| **Documentation** | README | Swagger UI |
| **Error Handling** | Exit codes | HTTP status codes |
| **Format** | Markdown only | JSON + Markdown |

---

## Next Steps

1. **Try it now**: Open `http://localhost:8000/api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863/export` in your browser
2. **Explore Swagger**: Visit `http://localhost:8000/docs` to see all endpoints
3. **Integrate**: Use the JSON endpoint to build custom tools
4. **Automate**: Create scripts that fetch and process exports programmatically

The API is live and ready to use! 🚀
