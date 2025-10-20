# Document Export CLI Command

## Overview

New CLI command for exporting document chunks as markdown files. No need to write scripts or make HTTP requests!

## Command

```bash
fileintel documents export [DOCUMENT_ID] [OPTIONS]
```

## Options

- **DOCUMENT_ID** (required): The UUID of the document to export
- **`--output, -o`**: Output file path (default: `<document_id>.md`)
- **`--chunk-type, -t`**: Filter by chunk type (`vector` or `graph`)
- **`--metadata, -m`**: Include chunk metadata in export
- **`--help`**: Show help message

## Usage Examples

### Basic Export

Export document with default settings:
```bash
fileintel documents export 3b9e6ac7-2152-4133-bd87-2cd0ffc09863
```

Output:
```
Exporting document 3b9e6ac7-2152-4133-bd87-2cd0ffc09863...
✓ Document exported to 3b9e6ac7-2152-4133-bd87-2cd0ffc09863.md
  File size: 2.5 MB
  Total chunks: 794
```

### Export with Metadata

Include detailed chunk metadata:
```bash
fileintel documents export 3b9e6ac7-2152-4133-bd87-2cd0ffc09863 --metadata
```

or shorthand:
```bash
fileintel documents export 3b9e6ac7-2152-4133-bd87-2cd0ffc09863 -m
```

### Export to Custom File

Specify output filename:
```bash
fileintel documents export 3b9e6ac7-2152-4133-bd87-2cd0ffc09863 --output my_document.md
```

or shorthand:
```bash
fileintel documents export 3b9e6ac7-2152-4133-bd87-2cd0ffc09863 -o my_document.md
```

### Export Only Graph Chunks

Filter by chunk type:
```bash
fileintel documents export 3b9e6ac7-2152-4133-bd87-2cd0ffc09863 --chunk-type graph
```

or shorthand:
```bash
fileintel documents export 3b9e6ac7-2152-4133-bd87-2cd0ffc09863 -t graph
```

### Export Only Vector Chunks

```bash
fileintel documents export 3b9e6ac7-2152-4133-bd87-2cd0ffc09863 -t vector -o vector_chunks.md
```

### Combined Options

Export graph chunks with metadata to custom file:
```bash
fileintel documents export 3b9e6ac7-2152-4133-bd87-2cd0ffc09863 \
  --chunk-type graph \
  --metadata \
  --output graph_analysis.md
```

## Finding Document IDs

### List Documents in Collection

```bash
fileintel documents list my-collection
```

Output:
```
Documents in 'my-collection' (3):
  3b9e6ac7-2152-4133-bd87-2cd0ffc09863 | document1.pdf | 2.5 MB | application/pdf
  5ed4be9e-7c0c-40dc-af8d-357dd617af5c | document2.pdf | 1.8 MB | application/pdf
  23ec6c76-fd4f-4ae2-a15b-dcc6eae34441 | document3.pdf | 3.2 MB | application/pdf
```

### Get Document Info

```bash
fileintel documents get 3b9e6ac7-2152-4133-bd87-2cd0ffc09863
```

Output shows full document metadata including ID.

### View Chunks Preview

Before exporting, preview chunks:
```bash
fileintel documents chunks 3b9e6ac7-2152-4133-bd87-2cd0ffc09863 --limit 5
```

## Related Commands

### Upload Document

```bash
fileintel documents upload my-collection ./document.pdf --process --wait
```

### List Collections

```bash
fileintel collections list
```

### View Collection

```bash
fileintel collections get my-collection
```

## Output Format

The exported markdown file contains:

1. **Document Header**
   - Document ID
   - Filename
   - Collection ID
   - Creation timestamp
   - Total chunks count

2. **Document Metadata** (if available)
   - File path, size, processing info
   - Custom metadata fields

3. **Chunks in Order**
   - Each chunk numbered sequentially
   - Full chunk text content
   - Optional metadata (with `--metadata`)

4. **Export Summary**
   - Total chunks
   - Total tokens
   - Average tokens per chunk

## Example Output

### Basic Export

```markdown
# Document Export: my_document.pdf

**Document ID**: `3b9e6ac7-2152-4133-bd87-2cd0ffc09863`
**Collection ID**: `d3f19c71-91d8-46c1-9e4e-64129f8373b5`
**Created**: 2025-10-19 14:30:00
**Total Chunks**: 794
**Exported**: 2025-10-19T19:45:00

---

## Document Content

### Chunk 1

This is the first chunk of content from the document...

---

### Chunk 2

This is the second chunk...

---

## Export Summary

- **Total Chunks**: 794
- **Total Tokens**: 342,567
- **Average Tokens/Chunk**: 431
```

### Export with Metadata

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

This is the first chunk of content...

---
```

## Error Handling

### Document Not Found

```bash
fileintel documents export invalid-id-12345

Error: Document invalid-id-12345 not found or has no chunks
```

### No Chunks Found

```bash
fileintel documents export 3b9e6ac7-2152-4133-bd87-2cd0ffc09863 -t graph

Error: Document 3b9e6ac7-2152-4133-bd87-2cd0ffc09863 not found or has no chunks
```
(This happens if document has no graph chunks)

### API Connection Error

```bash
fileintel documents export 3b9e6ac7-2152-4133-bd87-2cd0ffc09863

Error: Failed to connect to API at http://localhost:8000
Make sure the FileIntel API server is running.
```

## Integration Examples

### Export All Documents in Collection

```bash
#!/bin/bash

# Get all document IDs from collection
COLLECTION="my-collection"

# Use CLI to list and parse
fileintel documents list "$COLLECTION" | grep -oE '[a-f0-9-]{36}' | while read doc_id; do
  echo "Exporting $doc_id..."
  fileintel documents export "$doc_id" -o "exports/${doc_id}.md"
done

echo "All documents exported to exports/ directory"
```

### Export with Timestamp

```bash
#!/bin/bash

DOC_ID="3b9e6ac7-2152-4133-bd87-2cd0ffc09863"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

fileintel documents export "$DOC_ID" \
  -o "exports/${TIMESTAMP}_${DOC_ID}.md" \
  --metadata

echo "Exported with timestamp: ${TIMESTAMP}"
```

### Compare Before/After

```bash
#!/bin/bash

# Export before enabling type-aware chunking
fileintel documents export $DOC_ID_OLD -o before.md

# (Re-process with type-aware chunking enabled)

# Export after
fileintel documents export $DOC_ID_NEW -o after.md

# Compare
diff -u before.md after.md | less
```

## Configuration

The CLI uses the same configuration as the API:

**Config file**: `~/.fileintel/config.yaml` or environment variables

```yaml
api:
  base_url: http://localhost:8000
  api_key: your-api-key  # if authentication enabled
```

Or use environment variables:
```bash
export FILEINTEL_API_URL=http://localhost:8000
export FILEINTEL_API_KEY=your-api-key

fileintel documents export 3b9e6ac7-2152-4133-bd87-2cd0ffc09863
```

## Help

View all options:
```bash
fileintel documents export --help
```

View all document commands:
```bash
fileintel documents --help
```

View all CLI commands:
```bash
fileintel --help
```

## Advantages

### vs. Python Script
- ✅ No Docker container access needed
- ✅ Works from host machine
- ✅ Simple command-line interface
- ✅ Integrated with existing CLI

### vs. API Endpoint
- ✅ No need to construct URLs
- ✅ Automatic output file handling
- ✅ Better error messages
- ✅ Progress indicators

### vs. Manual Export
- ✅ One command instead of multiple steps
- ✅ Consistent formatting
- ✅ Automatic file naming
- ✅ Built-in validation

## Troubleshooting

### Command Not Found

Make sure FileIntel is installed:
```bash
poetry install
```

Then use:
```bash
poetry run fileintel documents export <doc_id>
```

Or activate the virtual environment:
```bash
poetry shell
fileintel documents export <doc_id>
```

### API Not Running

Start the API server:
```bash
docker-compose up -d
```

Or check if it's running:
```bash
curl http://localhost:8000/docs
```

### Permission Denied

Make sure you have write permission in the output directory:
```bash
# Export to home directory
fileintel documents export <doc_id> -o ~/exports/document.md

# Or current directory
fileintel documents export <doc_id> -o ./document.md
```

## Next Steps

1. **Try it now**:
   ```bash
   fileintel documents export 3b9e6ac7-2152-4133-bd87-2cd0ffc09863
   ```

2. **List your documents**:
   ```bash
   fileintel documents list my-collection
   ```

3. **Export with metadata**:
   ```bash
   fileintel documents export <doc_id> --metadata
   ```

4. **Automate exports**:
   Create shell scripts to export multiple documents automatically

The CLI command is ready to use! 🚀
