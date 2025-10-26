# FileIntel Utility Scripts

## Debug Scripts (`debug/`)

Ad-hoc test and debugging scripts for manual testing and validation.

**⚠️ Note**: These are NOT automated pytest tests. They are standalone scripts for debugging specific features.

### Available Debug Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `test_reranker.py` | Test reranking API functionality | `python scripts/debug/test_reranker.py` |
| `test_page_tracking.py` | Debug page number preservation | `python scripts/debug/test_page_tracking.py` |
| `test_citation_fix.py` | Test citation extraction fixes | `python scripts/debug/test_citation_fix.py` |
| `test_citation_optimization.py` | Citation extraction performance | `python scripts/debug/test_citation_optimization.py` |
| `test_bibliography_filtering.py` | Bibliography filtering logic | `python scripts/debug/test_bibliography_filtering.py` |
| `test_llm_classifier.py` | LLM-based query classification | `python scripts/debug/test_llm_classifier.py` |
| `test_embedding_load.py` | Embedding generation load test | `python scripts/debug/test_embedding_load.py` |
| `test_export_import_compatibility.py` | Metadata export/import testing | `python scripts/debug/test_export_import_compatibility.py` |
| `test_improved_detection.py` | Improved content detection | `python scripts/debug/test_improved_detection.py` |
| `test_pdf_metadata.py` | PDF metadata extraction | `python scripts/debug/test_pdf_metadata.py` |
| `test_rag_prompt_debug.py` | RAG prompt debugging | `python scripts/debug/test_rag_prompt_debug.py` |
| `test_standalone_alerting.py` | Alerting functionality | `python scripts/debug/test_standalone_alerting.py` |
| `test_celery.py` | Celery configuration check | `python scripts/debug/test_celery.py` |
| `test_imports.py` | Verify GraphRAG imports | `python scripts/debug/test_imports.py` |

### Usage

Run individual scripts as needed for debugging:

```bash
# Example: Test reranker functionality
python scripts/debug/test_reranker.py

# Example: Check page tracking
python scripts/debug/test_page_tracking.py
```

### Note on Integration Tests

For automated testing, see the `tests/` directory which contains proper pytest-based tests with fixtures and assertions.

---

## Export Document Chunks (`export_document_chunks.py`)

Export all chunks from a document into a single readable markdown file in correct order.

### Features
- ✅ Retrieves chunks in correct document order
- ✅ Supports both vector and graph chunks
- ✅ Optional metadata display (chunk type, tokens, page numbers, etc.)
- ✅ Clean markdown output with document summary
- ✅ Collapsible metadata sections for cleaner reading

### Quick Start
```bash
# Basic export
poetry run python scripts/export_document_chunks.py <document_id>

# With metadata
poetry run python scripts/export_document_chunks.py <document_id> --include-metadata -v
```

See the full documentation below for all options and examples.

---

## Prompt Preview Script (`preview_prompt.py`)

This script is a command-line utility designed for debugging and previewing the final prompts that are sent to a Large Language Model (LLM). It constructs a complete prompt by combining a document's content, a specific question, and various template files.

This allows developers to see the exact text the LLM will receive, which is useful for ensuring the prompt is well-formed and contains the correct context.

## Usage

The script is run from the command line using Python. You must provide a document and a question.

```bash
python scripts/preview_prompt.py --doc <path_to_document> --question "<your_question>"
```

### Options

The script accepts the following command-line options:

- `--doc`: (Required) The file path to the document you want to use as context for the prompt.
- `--question`: (Required) The question you want to ask about the document.
- `--instruction`: (Optional) The name of the instruction template to use. This should be the filename (without the `.md` extension) located in the `prompts/templates` directory. Defaults to `instruction`.
- `--format`: (Optional) The name of the answer format template to use. This should be the filename (without the `.md` extension) located in the `prompts/templates` directory. Defaults to `answer_format`.
- `--max-length`: (Optional) An integer to specify the maximum character length for the composed prompt. If the prompt exceeds this length, it will be truncated.

## Examples

### Basic Preview

This example uses the default instruction and answer format templates.

```bash
python scripts/preview_prompt.py --doc C:\path\to\your\document.txt --question "What is the main idea of this document?"
```

### Specifying Custom Templates

This example shows how to use different templates for the instruction and answer format.

```bash
python scripts/preview_prompt.py \
  --doc C:\path\to\your\document.txt \
  --question "Summarize the key points." \
  --instruction "summary_instruction" \
  --format "bullet_points"
```

### Setting a Maximum Length

This example demonstrates how to limit the total length of the generated prompt.

```bash
python scripts/preview_prompt.py \
  --doc C:\path\to\a\very_long_document.txt \
  --question "What are the initial findings?" \
  --max-length 2048
```

---

# Export Document Chunks - Full Documentation

## Overview

The `export_document_chunks.py` script retrieves all chunks from a document in the database and exports them to a single inspectable markdown file in the correct order.

## Installation

No additional dependencies needed - uses existing FileIntel dependencies.

## Command Line Options

```
usage: export_document_chunks.py [-h] [-o OUTPUT] [-t {vector,graph,all}] [-m] [-v] document_id

positional arguments:
  document_id           Document UUID to export

options:
  -h, --help            Show help message
  -o OUTPUT, --output OUTPUT
                        Output markdown file path (default: <document_id>.md)
  -t {vector,graph,all}, --chunk-type {vector,graph,all}
                        Chunk type to export (default: all)
  -m, --include-metadata
                        Include chunk metadata in output
  -v, --verbose         Enable verbose output
```

## Examples

### Example 1: Basic Export
```bash
poetry run python scripts/export_document_chunks.py 3b9e6ac7-2152-4133-bd87-2cd0ffc09863
# Output: 3b9e6ac7-2152-4133-bd87-2cd0ffc09863.md
```

### Example 2: Export to Custom File
```bash
poetry run python scripts/export_document_chunks.py 3b9e6ac7-2152-4133-bd87-2cd0ffc09863 -o my_document.md
```

### Example 3: Full Export with Metadata
```bash
poetry run python scripts/export_document_chunks.py 3b9e6ac7-2152-4133-bd87-2cd0ffc09863 \
  --include-metadata \
  --verbose \
  -o debug_output.md
```

### Example 4: Export Only Graph Chunks
```bash
poetry run python scripts/export_document_chunks.py 3b9e6ac7-2152-4133-bd87-2cd0ffc09863 \
  --chunk-type graph \
  -o graph_chunks.md
```

### Example 5: Compare Vector vs Graph Chunks
```bash
# Export vector chunks
poetry run python scripts/export_document_chunks.py <doc_id> -t vector -o vector.md

# Export graph chunks  
poetry run python scripts/export_document_chunks.py <doc_id> -t graph -o graph.md

# Compare
diff -u vector.md graph.md
```

## Output Format

The exported markdown file contains:

### 1. Document Header
```markdown
# Document Export: filename.pdf

**Document ID**: `3b9e6ac7-2152-4133-bd87-2cd0ffc09863`
**Collection ID**: `d3f19c71-91d8-46c1-9e4e-64129f8373b5`
**Created**: 2025-10-19 14:30:00
**Total Chunks**: 794
**Exported**: 2025-10-19T19:20:00
```

### 2. Document Metadata (if available)
```markdown
## Document Metadata

- **file_path**: /uploads/document.pdf
- **file_size**: 2547392
- **processing_method**: mineru
```

### 3. Chunks in Order
```markdown
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

### 4. Export Summary
```markdown
## Export Summary

- **Total Chunks**: 794
- **Total Tokens**: 342,567
- **Average Tokens/Chunk**: 431
```

## Metadata Fields

When using `--include-metadata`, these fields are displayed:

| Field | Description |
|-------|-------------|
| Position | Chunk order in document (0-indexed) |
| Type | `vector` or `graph` chunk |
| Page | Source page number |
| Tokens | Token count for the chunk |
| Content Type | Type-aware classification (`prose`, `bullet_list`, `citation_heavy`, etc.) |
| Classification | Source of classification (`mineru` or `statistical`) |
| Vector Chunks | Number of vector chunks in graph chunk (two-tier mode) |
| Sentences | Sentence count (graph chunks) |
| Filtering Error | Errors during filtering (marked with ⚠) |
| Truncated | If chunk was truncated (marked with ⚠) |

## Finding Document IDs

### Method 1: Using psql
```bash
psql $DATABASE_URL -c "
SELECT id, filename, collection_id, created_at
FROM documents
ORDER BY created_at DESC
LIMIT 10;
"
```

### Method 2: Using Python
```python
poetry run python -c "
from src.fileintel.core.config import get_config
from src.fileintel.storage.postgresql_storage import PostgreSQLStorage

storage = PostgreSQLStorage.from_config(get_config())

# Get recent documents
docs = storage.db.query(storage.db.query(Document).order_by(Document.created_at.desc()).limit(10).all()

for doc in docs:
    print(f'{doc.id} | {doc.filename} | {doc.collection_id}')

storage.close()
"
```

### Method 3: From API logs
```bash
# Check celery logs for recently processed documents
docker-compose logs celery-worker | grep "document_id" | tail -20
```

## Use Cases

1. **Quality Inspection**: Verify chunking produced sensible results
2. **Debugging Type-Aware Chunking**: Check if content classification is correct
3. **Comparing Strategies**: Export before/after enabling type-aware chunking
4. **Content Review**: Human review of processed documents
5. **Token Analysis**: Analyze chunk distribution and token usage
6. **Filtering Validation**: Verify corruption filtering removed correct content
7. **Documentation**: Export processed content for reference
8. **Training Data**: Export clean chunks for ML training

## Troubleshooting

### "Document not found"
**Cause**: Invalid document ID or document doesn't exist
**Solution**: 
- Verify document ID is correct UUID
- Check document exists: `SELECT * FROM documents WHERE id = '<doc_id>'`
- Verify database connection string in config

### "No chunks found"
**Cause**: Document has no chunks or filter excluded all chunks
**Solution**:
- Check if document was processed: `SELECT * FROM chunks WHERE document_id = '<doc_id>'`
- Try without chunk type filter: remove `--chunk-type`
- Check document processing status in celery logs

### "Error: cannot import name 'PostgreSQLStorage'"
**Cause**: Python path issues
**Solution**: Run with `poetry run` from project root

### Script crashes with database error
**Cause**: Database connection issues
**Solution**:
- Run with `--verbose` for detailed error
- Check `config/default.yaml` has correct `storage.connection_string`
- Verify database is accessible
- Check PostgreSQL is running

## Performance Notes

| Document Size | Chunks | Export Time | Output Size |
|---------------|--------|-------------|-------------|
| Small         | <100   | <1s         | <100KB      |
| Medium        | 100-1000 | 1-5s      | 100KB-1MB   |
| Large         | 1000-10000 | 5-30s   | 1-10MB      |
| Very Large    | >10000 | 30s+        | >10MB       |

**Tips for Large Documents**:
- Use `--chunk-type` to export subset
- Export without `--include-metadata` first
- Consider exporting to SSD for faster writes
- Use `--verbose` to monitor progress

## Integration Examples

### Export all documents in collection
```bash
#!/bin/bash
COLLECTION_ID="d3f19c71-91d8-46c1-9e4e-64129f8373b5"

# Get all document IDs
psql $DATABASE_URL -t -c "
SELECT id FROM documents WHERE collection_id = '$COLLECTION_ID'
" | while read doc_id; do
  echo "Exporting $doc_id..."
  poetry run python scripts/export_document_chunks.py "$doc_id" \
    -o "exports/${doc_id}.md"
done
```

### Export with timestamp
```bash
DOC_ID="3b9e6ac7-2152-4133-bd87-2cd0ffc09863"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

poetry run python scripts/export_document_chunks.py "$DOC_ID" \
  -o "exports/${TIMESTAMP}_${DOC_ID}.md" \
  --include-metadata \
  --verbose
```

### Compare before/after processing
```bash
# Export document processed with traditional chunking
poetry run python scripts/export_document_chunks.py $DOC_ID_OLD -o before.md

# Re-process with type-aware chunking enabled
# (set use_type_aware_chunking: true in config)

# Export new version
poetry run python scripts/export_document_chunks.py $DOC_ID_NEW -o after.md

# Compare
diff -u before.md after.md | less
```
