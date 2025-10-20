# Document Chunk Export - Complete Implementation

**Date**: 2025-10-19
**Status**: ✅ Complete and Ready to Use

## Summary

Implemented three ways to export document chunks as markdown files:
1. **Python Script** - For direct database access
2. **API Endpoints** - For HTTP-based access
3. **CLI Command** - For command-line usage

## What Was Built

### 1. Python Script ✅

**File**: `scripts/export_document_chunks.py`

**Features**:
- Retrieves chunks from PostgreSQL database
- Exports to markdown format
- Optional metadata inclusion
- Chunk type filtering (vector/graph)
- Verbose progress output

**Usage**:
```bash
# Must run inside Docker container
poetry run python scripts/export_document_chunks.py <document_id> --metadata -v
```

**Documentation**: `scripts/README.md`, `EXPORT_CHUNKS_USAGE.md`

---

### 2. API Endpoints ✅

**File**: `src/fileintel/api/routes/documents_v2.py`

**Endpoints**:

#### GET /api/v2/documents/{document_id}
Get document info and statistics
```bash
curl http://localhost:8000/api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863
```

#### GET /api/v2/documents/{document_id}/chunks
Get chunks as JSON
```bash
curl http://localhost:8000/api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863/chunks
```

#### GET /api/v2/documents/{document_id}/export
Export chunks as markdown (downloadable)
```bash
curl -O -J "http://localhost:8000/api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863/export?include_metadata=true"
```

**Browser Access**:
```
http://localhost:8000/api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863/export
```
File downloads automatically!

**Interactive Docs**: `http://localhost:8000/docs`

**Documentation**: `API_DOCUMENT_EXPORT.md`

---

### 3. CLI Command ✅

**File**: `src/fileintel/cli/documents.py`

**Command**:
```bash
fileintel documents export [DOCUMENT_ID] [OPTIONS]
```

**Options**:
- `--output, -o`: Output file path
- `--chunk-type, -t`: Filter by vector/graph
- `--metadata, -m`: Include metadata

**Usage**:
```bash
# Basic export
fileintel documents export 3b9e6ac7-2152-4133-bd87-2cd0ffc09863

# With metadata
fileintel documents export 3b9e6ac7-2152-4133-bd87-2cd0ffc09863 --metadata

# Custom output
fileintel documents export 3b9e6ac7-2152-4133-bd87-2cd0ffc09863 -o my_doc.md

# Only graph chunks
fileintel documents export 3b9e6ac7-2152-4133-bd87-2cd0ffc09863 -t graph
```

**Documentation**: `CLI_DOCUMENT_EXPORT.md`

---

## Files Created/Modified

### New Files Created
1. `scripts/export_document_chunks.py` - Python script for direct DB access
2. `src/fileintel/api/routes/documents_v2.py` - API endpoints
3. `EXPORT_CHUNKS_USAGE.md` - Script usage guide
4. `API_DOCUMENT_EXPORT.md` - API documentation
5. `CLI_DOCUMENT_EXPORT.md` - CLI documentation
6. `DOCUMENT_EXPORT_COMPLETE.md` - This file

### Files Modified
1. `src/fileintel/api/main.py` - Registered new router
2. `src/fileintel/cli/documents.py` - Added export command
3. `scripts/README.md` - Added script documentation

### Additional Documentation
- `TIMEOUT_FIX.md` - Celery timeout fixes
- `FINAL_FIXES_APPLIED.md` - Schema and fixes summary
- `CRITICAL_FIXES_APPLIED.md` - Initial pipeline fixes

---

## Quick Start Guide

### Option 1: Use the CLI (Recommended)

Works from your host machine, no Docker needed!

```bash
# Install/update
poetry install

# Export a document
fileintel documents export 3b9e6ac7-2152-4133-bd87-2cd0ffc09863

# With metadata
fileintel documents export 3b9e6ac7-2152-4133-bd87-2cd0ffc09863 -m
```

### Option 2: Use the API

Open in your browser:
```
http://localhost:8000/api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863/export
```

Or use curl:
```bash
curl -O -J "http://localhost:8000/api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863/export?include_metadata=true"
```

### Option 3: Use the Script

From inside Docker container:
```bash
docker-compose exec app poetry run python scripts/export_document_chunks.py \
  3b9e6ac7-2152-4133-bd87-2cd0ffc09863 \
  --metadata \
  -v
```

---

## Example Document IDs

From your recent celery logs:

| Document ID | Filename | Chunks |
|------------|----------|--------|
| `3b9e6ac7-2152-4133-bd87-2cd0ffc09863` | 8ff55694-8df4-41f4-98ca-f0b46d9ef122.pdf | 794 |
| `23ec6c76-fd4f-4ae2-a15b-dcc6eae34441` | 67854c87-8cee-4b6f-9ba8-89f481736c37.pdf | 963 |

Try exporting these documents:
```bash
fileintel documents export 3b9e6ac7-2152-4133-bd87-2cd0ffc09863
```

---

## Output Format

All three methods produce the same markdown format:

```markdown
# Document Export: filename.pdf

**Document ID**: `3b9e6ac7-2152-4133-bd87-2cd0ffc09863`
**Collection ID**: `d3f19c71-91d8-46c1-9e4e-64129f8373b5`
**Created**: 2025-10-19 14:30:00
**Total Chunks**: 794
**Exported**: 2025-10-19T19:45:00

## Document Content

### Chunk 1

[chunk text content...]

---

### Chunk 2

[chunk text content...]

---

## Export Summary

- **Total Chunks**: 794
- **Total Tokens**: 342,567
- **Average Tokens/Chunk**: 431
```

With `--metadata`:
```markdown
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

[chunk text content...]
```

---

## Features Comparison

| Feature | Script | API | CLI |
|---------|--------|-----|-----|
| **Access** | Inside Docker | HTTP | Anywhere |
| **Auth** | Direct DB | API Key | Config File |
| **Output** | File | Download/JSON | File |
| **Metadata** | ✅ | ✅ | ✅ |
| **Filter** | ✅ | ✅ | ✅ |
| **Progress** | ✅ Verbose | - | ✅ Status |
| **Integration** | Shell | HTTP/Browser | Shell |
| **Docs** | README | Swagger UI | --help |

---

## Use Cases

### 1. Quality Inspection
**Best Method**: CLI
```bash
fileintel documents export <doc_id> -m | less
```

### 2. Debugging Type-Aware Chunking
**Best Method**: API with metadata
```
http://localhost:8000/api/v2/documents/<doc_id>/export?include_metadata=true
```

### 3. Automation
**Best Method**: CLI in scripts
```bash
for doc_id in $(get_doc_ids); do
  fileintel documents export "$doc_id" -o "exports/${doc_id}.md"
done
```

### 4. Web Integration
**Best Method**: API JSON endpoint
```javascript
fetch('/api/v2/documents/<doc_id>/chunks')
  .then(r => r.json())
  .then(data => processChunks(data.chunks))
```

### 5. Manual Review
**Best Method**: Browser
```
Open: http://localhost:8000/api/v2/documents/<doc_id>/export
```

---

## Next Steps

### For Immediate Use

1. **Try the CLI now**:
   ```bash
   fileintel documents export 3b9e6ac7-2152-4133-bd87-2cd0ffc09863
   ```

2. **Or open in browser**:
   ```
   http://localhost:8000/api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863/export
   ```

### For Development

1. **Restart API server** to load new endpoints:
   ```bash
   docker-compose restart app
   ```

2. **Test all endpoints**:
   ```bash
   # Visit Swagger UI
   open http://localhost:8000/docs
   ```

3. **Test CLI**:
   ```bash
   fileintel documents --help
   fileintel documents export --help
   ```

---

## Related Features

### Type-Aware Chunking
- ✅ Enabled via config: `use_type_aware_chunking: true`
- ✅ Corruption filtering active
- ✅ Statistical heuristics working
- ✅ Specialized chunkers implemented

### Schema Updates
- ✅ Migration created for `filtered_content` structure type
- ✅ Filtering metadata storage ready
- ✅ Run: `poetry run alembic upgrade head` (in Docker)

### Timeout Fixes
- ✅ Document processing: No time limits
- ✅ Workers won't be killed for long-running tasks
- ✅ Restart workers to apply changes

---

## Testing Checklist

- [x] Script compiles successfully
- [x] API endpoints compile successfully
- [x] CLI command compiles successfully
- [x] Documentation complete
- [ ] API server restarted (user action)
- [ ] Test CLI export command (user action)
- [ ] Test API endpoint in browser (user action)
- [ ] Test script in Docker (user action)

---

## Documentation Index

- **Script**: `scripts/README.md`, `EXPORT_CHUNKS_USAGE.md`
- **API**: `API_DOCUMENT_EXPORT.md`, `http://localhost:8000/docs`
- **CLI**: `CLI_DOCUMENT_EXPORT.md`, `fileintel documents export --help`
- **Implementation**: This file
- **Fixes Applied**: `CRITICAL_FIXES_APPLIED.md`, `FINAL_FIXES_APPLIED.md`, `TIMEOUT_FIX.md`

---

**Status**: ✅ Complete - All three export methods implemented and documented!
**Ready to Use**: Restart API server and start exporting documents 🚀
