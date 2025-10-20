# Using the Document Chunks Export Script

## Quick Reference

The export script must be run from **inside the Docker container** where it has access to the PostgreSQL database.

## Step 1: Find a Document ID

From the celery logs you shared, I can see this document was recently processed with type-aware chunking:

```
Document ID: 3b9e6ac7-2152-4133-bd87-2cd0ffc09863
Filename: 8ff55694-8df4-41f4-98ca-f0b46d9ef122.pdf
Chunks: 794 chunks created
```

## Step 2: Run the Export Script

### From Docker Container

```bash
# Enter the container
docker-compose exec <your-app-container> bash

# Export the document
poetry run python scripts/export_document_chunks.py 3b9e6ac7-2152-4133-bd87-2cd0ffc09863 -v

# With metadata
poetry run python scripts/export_document_chunks.py 3b9e6ac7-2152-4133-bd87-2cd0ffc09863 --include-metadata -v
```

### Output File Location

By default, the script creates: `3b9e6ac7-2152-4133-bd87-2cd0ffc09863.md`

To specify output location:
```bash
poetry run python scripts/export_document_chunks.py 3b9e6ac7-2152-4133-bd87-2cd0ffc09863 \
  -o /app/exports/my_document.md \
  --include-metadata \
  -v
```

## Step 3: Copy File Out of Container

```bash
# Copy from container to host
docker cp <container-name>:/app/3b9e6ac7-2152-4133-bd87-2cd0ffc09863.md ./exported_document.md
```

## Finding More Document IDs

### From Celery Logs

```bash
# Get recently processed documents
docker-compose logs celery-worker | grep "document_id" | grep "succeeded" | tail -10

# Get documents with type-aware chunking
docker-compose logs celery-worker | grep "Type-aware chunking created" | tail -10
```

### From PostgreSQL

```bash
# Enter postgres container
docker-compose exec postgres psql -U <user> -d <database>

# Query recent documents
SELECT id, filename, created_at
FROM documents
ORDER BY created_at DESC
LIMIT 10;

# Query documents with chunk count
SELECT d.id, d.filename, COUNT(c.id) as chunk_count
FROM documents d
LEFT JOIN document_chunks c ON d.document_id = c.document_id
GROUP BY d.id, d.filename
ORDER BY d.created_at DESC
LIMIT 10;
```

### From Python Inside Container

```bash
docker-compose exec <app-container> poetry run python -c "
from src.fileintel.storage.models import Document, DocumentChunk, SessionLocal

session = SessionLocal()
docs = session.query(Document).order_by(Document.created_at.desc()).limit(5).all()

print('Recent documents:')
for doc in docs:
    chunk_count = session.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).count()
    if chunk_count > 0:
        print(f'{doc.id} | {doc.filename} | {chunk_count} chunks')

session.close()
"
```

## Example Usage

### Basic Export
```bash
docker-compose exec app poetry run python scripts/export_document_chunks.py \
  3b9e6ac7-2152-4133-bd87-2cd0ffc09863
```

### Full Export with Metadata
```bash
docker-compose exec app poetry run python scripts/export_document_chunks.py \
  3b9e6ac7-2152-4133-bd87-2cd0ffc09863 \
  --include-metadata \
  --verbose \
  -o /app/exports/document_with_metadata.md
```

### Export Only Vector Chunks
```bash
docker-compose exec app poetry run python scripts/export_document_chunks.py \
  3b9e6ac7-2152-4133-bd87-2cd0ffc09863 \
  --chunk-type vector \
  -o /app/exports/vector_chunks.md
```

### Export Only Graph Chunks
```bash
docker-compose exec app poetry run python scripts/export_document_chunks.py \
  3b9e6ac7-2152-4133-bd87-2cd0ffc09863 \
  --chunk-type graph \
  -o /app/exports/graph_chunks.md
```

## Expected Output

When running with verbose mode, you'll see:

```
Connecting to database...
Retrieving document info for 3b9e6ac7-2152-4133-bd87-2cd0ffc09863...
  Document: 8ff55694-8df4-41f4-98ca-f0b46d9ef122.pdf
  Collection: d3f19c71-91d8-46c1-9e4e-64129f8373b5
Retrieving chunks...
  Found 794 chunks
Exporting 794 chunks to 3b9e6ac7-2152-4133-bd87-2cd0ffc09863.md...
  Processed 100/794 chunks...
  Processed 200/794 chunks...
  ...
✓ Export complete: 3b9e6ac7-2152-4133-bd87-2cd0ffc09863.md
  Total chunks: 794
  Total tokens: 342,567
  File size: 2,547,392 bytes
✓ Exported to: 3b9e6ac7-2152-4133-bd87-2cd0ffc09863.md
```

## Troubleshooting

### "Document not found"
Check the document ID is correct:
```sql
SELECT id FROM documents WHERE id = '3b9e6ac7-2152-4133-bd87-2cd0ffc09863';
```

### "No chunks found"
Check if chunks exist:
```sql
SELECT COUNT(*) FROM document_chunks WHERE document_id = '3b9e6ac7-2152-4133-bd87-2cd0ffc09863';
```

### Database connection error
Make sure you're running inside the Docker container with database access.

## What's in the Export?

The exported markdown file contains:

1. **Document Header**
   - Document ID: `3b9e6ac7-2152-4133-bd87-2cd0ffc09863`
   - Filename: `8ff55694-8df4-41f4-98ca-f0b46d9ef122.pdf`
   - Total Chunks: 794
   - Created timestamp

2. **All Chunks in Order**
   - Chunk 1, Chunk 2, ... Chunk 794
   - Each with full text content
   - Optional metadata (with `--include-metadata`)

3. **Summary**
   - Total token count
   - Average tokens per chunk
   - Export timestamp

This allows you to:
- ✅ Inspect chunk quality
- ✅ Verify type-aware chunking results
- ✅ Check corruption filtering
- ✅ Analyze chunk boundaries
- ✅ Review document content
