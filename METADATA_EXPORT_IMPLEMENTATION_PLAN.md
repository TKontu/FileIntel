# Metadata Export Implementation Plan

## Objective
Enable users to export collection documents with metadata in table formats (CSV, Markdown, JSON) for reviewing collection integrity.

## Architecture

### Separation of Concerns
- **API Layer**: Provides structured data (JSON only)
- **CLI Layer**: Handles formatting and presentation (CSV, Markdown, JSON)
- **Principle**: API returns data, CLI handles display

## Implementation Details

### 1. API Layer (`src/fileintel/api/routes/metadata_v2.py`)

#### New Endpoint
```python
GET /metadata/collection/{collection_identifier}/export
```

#### Response Model
```python
class DocumentMetadataExport(BaseModel):
    document_id: str
    filename: str
    has_extracted_metadata: bool
    metadata: Dict[str, Any]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

class CollectionMetadataExport(BaseModel):
    collection_id: str
    collection_name: str
    total_documents: int
    documents: List[DocumentMetadataExport]
```

#### Implementation Strategy
**IMPORTANT: Avoid N+1 Query Problem**

❌ **Bad (N+1 queries):**
```python
for doc in documents:
    metadata = storage.get_document(doc.id)  # N queries!
```

✅ **Good (Single query):**
```python
# Get all documents with metadata in one query
documents = storage.get_documents_by_collection(collection_id)
# Documents already have metadata attached
```

#### Code Structure
```python
@router.get("/collection/{collection_identifier}/export", response_model=ApiResponseV2)
async def export_collection_metadata(
    collection_identifier: str,
    storage=Depends(get_storage),
):
    """Export all documents with metadata for a collection."""

    # 1. Get collection
    collection = await get_collection_by_identifier(storage, collection_identifier)

    # 2. Get all documents with metadata (SINGLE QUERY)
    documents = storage.get_documents_by_collection(collection.id)

    # 3. Transform to export format
    export_docs = [
        {
            "document_id": doc.id,
            "filename": doc.original_filename,
            "has_extracted_metadata": check_has_llm_metadata(doc.document_metadata),
            "metadata": doc.document_metadata or {},
            "created_at": doc.created_at,
            "updated_at": doc.updated_at,
        }
        for doc in documents
    ]

    # 4. Return structured response
    return ApiResponseV2(
        success=True,
        data={
            "collection_id": collection.id,
            "collection_name": collection.name,
            "total_documents": len(documents),
            "documents": export_docs,
        }
    )
```

**Performance:**
- ✅ Single database query for all documents
- ✅ Metadata already loaded (no lazy loading issues)
- ✅ Scales well with large collections

---

### 2. CLI Layer (`src/fileintel/cli/metadata.py`)

#### New Command
```bash
fileintel metadata export-table <collection> [OPTIONS]
```

#### Options
- `--format, -f`: Export format (`csv`, `markdown`, `json`) - default: `markdown`
- `--output, -o`: Output file path (optional, prints to console if not specified)
- `--fields`: Comma-separated metadata fields to include (optional, includes all if not specified)

#### Examples
```bash
# Export as Markdown table to console
fileintel metadata export-table thesis_sources

# Export as CSV to file
fileintel metadata export-table thesis_sources --format csv --output metadata.csv

# Export specific fields only
fileintel metadata export-table thesis_sources --fields title,author,year --format csv

# Export as JSON
fileintel metadata export-table thesis_sources --format json --output collection.json
```

#### Implementation Structure
```python
@app.command("export-table")
def export_collection_table(
    collection_identifier: str = typer.Argument(...),
    format: str = typer.Option("markdown", "--format", "-f"),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
    fields: Optional[str] = typer.Option(None, "--fields"),
):
    """Export collection documents with metadata in table format."""

    # 1. Call API
    def _get_export_data(api):
        return api._request("GET", f"metadata/collection/{collection_identifier}/export")

    result = cli_handler.handle_api_call(_get_export_data, "export metadata")
    data = result.get("data", {})

    # 2. Parse field selection
    selected_fields = parse_field_selection(fields) if fields else None

    # 3. Format based on format option
    if format.lower() == "csv":
        output_content = format_as_csv(data, selected_fields)
    elif format.lower() == "markdown":
        output_content = format_as_markdown(data, selected_fields)
    elif format.lower() == "json":
        output_content = format_as_json(data, selected_fields)
    else:
        cli_handler.display_error(f"Unknown format: {format}")
        raise typer.Exit(1)

    # 4. Output to file or console
    if output:
        save_to_file(output, output_content)
        cli_handler.display_success(f"Exported {data['total_documents']} documents to {output}")
    else:
        display_to_console(data, output_content)
```

---

### 3. Formatter Helpers

#### CSV Formatter
```python
def format_as_csv(data: Dict, selected_fields: Optional[List[str]] = None) -> str:
    """Format collection metadata as CSV."""
    import csv
    from io import StringIO

    documents = data["documents"]
    if not documents:
        return ""

    # Determine columns
    standard_cols = ["filename", "document_id", "has_extracted_metadata"]

    # Get all unique metadata keys
    if selected_fields:
        metadata_cols = selected_fields
    else:
        all_keys = set()
        for doc in documents:
            all_keys.update(doc["metadata"].keys())
        metadata_cols = sorted([k for k in all_keys if not k.startswith("_")])

    # Write CSV
    output = StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(standard_cols + metadata_cols)

    # Rows
    for doc in documents:
        row = [
            doc["filename"],
            doc["document_id"],
            "Yes" if doc["has_extracted_metadata"] else "No"
        ]
        for field in metadata_cols:
            value = doc["metadata"].get(field, "")
            row.append(serialize_value(value))
        writer.writerow(row)

    return output.getvalue()

def serialize_value(value: Any) -> str:
    """Serialize complex values for CSV."""
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    elif isinstance(value, dict):
        return str(value)
    else:
        return str(value) if value is not None else ""
```

#### Markdown Formatter
```python
def format_as_markdown(data: Dict, selected_fields: Optional[List[str]] = None) -> str:
    """Format collection metadata as Markdown table."""
    documents = data["documents"]
    if not documents:
        return "No documents found."

    # Determine columns
    standard_cols = ["Filename", "Document ID", "Extracted"]

    if selected_fields:
        metadata_cols = [field.replace("_", " ").title() for field in selected_fields]
        metadata_keys = selected_fields
    else:
        all_keys = set()
        for doc in documents:
            all_keys.update(doc["metadata"].keys())
        metadata_keys = sorted([k for k in all_keys if not k.startswith("_")])
        metadata_cols = [key.replace("_", " ").title() for key in metadata_keys]

    # Build table
    lines = []
    headers = standard_cols + metadata_cols

    # Header row
    lines.append("| " + " | ".join(headers) + " |")
    # Separator
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    # Data rows
    for doc in documents:
        cells = [
            truncate(doc["filename"], 30),
            doc["document_id"][:8] + "...",
            "✓" if doc["has_extracted_metadata"] else "✗"
        ]
        for key in metadata_keys:
            value = doc["metadata"].get(key, "")
            cells.append(truncate(format_markdown_value(value), 40))

        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)

def format_markdown_value(value: Any) -> str:
    """Format value for markdown table cell."""
    if isinstance(value, list):
        if len(value) == 0:
            return ""
        elif len(value) <= 3:
            return ", ".join(str(v) for v in value)
        else:
            return ", ".join(str(v) for v in value[:3]) + ", ..."
    elif isinstance(value, dict):
        return "{...}"
    else:
        return str(value) if value is not None else ""

def truncate(text: str, max_len: int) -> str:
    """Truncate text to max length."""
    text = str(text)
    return text if len(text) <= max_len else text[:max_len-3] + "..."
```

#### JSON Formatter
```python
def format_as_json(data: Dict, selected_fields: Optional[List[str]] = None) -> str:
    """Format collection metadata as JSON."""
    import json

    if selected_fields:
        # Filter metadata to selected fields
        filtered_docs = []
        for doc in data["documents"]:
            filtered_metadata = {
                k: v for k, v in doc["metadata"].items()
                if k in selected_fields
            }
            filtered_docs.append({
                **doc,
                "metadata": filtered_metadata
            })
        data["documents"] = filtered_docs

    return json.dumps(data, indent=2, default=str)
```

---

### 4. Error Handling

#### API Errors
```python
# Collection not found
if not collection:
    raise HTTPException(404, f"Collection '{collection_identifier}' not found")

# No documents
if not documents:
    return ApiResponseV2(
        success=True,
        data={
            "collection_id": collection.id,
            "collection_name": collection.name,
            "total_documents": 0,
            "documents": [],
        }
    )
```

#### CLI Errors
```python
# Empty collection
if not data.get("documents"):
    cli_handler.display_warning(f"Collection '{collection_name}' has no documents")
    raise typer.Exit(0)

# Invalid format
if format not in ["csv", "markdown", "json"]:
    cli_handler.display_error(f"Invalid format '{format}'. Use: csv, markdown, or json")
    raise typer.Exit(1)

# File write error
try:
    with open(output, "w", encoding="utf-8") as f:
        f.write(content)
except IOError as e:
    cli_handler.display_error(f"Failed to write to {output}: {e}")
    raise typer.Exit(1)
```

---

### 5. Testing Strategy

#### API Tests
```python
def test_export_collection_metadata():
    # Test successful export
    # Test collection not found
    # Test empty collection
    # Test metadata fields correctly included

def test_export_performance():
    # Test with large collection (1000+ docs)
    # Verify single query (no N+1)
```

#### CLI Tests
```python
def test_csv_format():
    # Test CSV output formatting
    # Test field selection
    # Test special characters in metadata

def test_markdown_format():
    # Test markdown table formatting
    # Test truncation
    # Test unicode characters

def test_json_format():
    # Test JSON validity
    # Test field filtering
```

---

## Implementation Checklist

### Phase 1: API Foundation
- [ ] Add Pydantic response models
- [ ] Implement `/metadata/collection/{id}/export` endpoint
- [ ] Verify single-query performance
- [ ] Add error handling
- [ ] Test with sample collections

### Phase 2: CLI Implementation
- [ ] Add `export-table` command
- [ ] Implement CSV formatter
- [ ] Implement Markdown formatter
- [ ] Implement JSON formatter
- [ ] Add field selection logic
- [ ] Add file output handling

### Phase 3: Polish
- [ ] Add comprehensive error messages
- [ ] Update CLI help documentation
- [ ] Add usage examples
- [ ] Performance testing with large collections
- [ ] Edge case testing (empty metadata, special characters, etc.)

---

## Usage Examples

### Reviewing Collection Integrity

**1. Quick visual check (Markdown to console):**
```bash
fileintel metadata export-table thesis_sources
```

**2. Detailed CSV export for analysis:**
```bash
fileintel metadata export-table thesis_sources \
  --format csv \
  --output thesis_metadata.csv
```

**3. Export specific fields for validation:**
```bash
fileintel metadata export-table thesis_sources \
  --format csv \
  --fields title,author,year,doi \
  --output validation.csv
```

**4. JSON export for programmatic processing:**
```bash
fileintel metadata export-table thesis_sources \
  --format json \
  --output collection_backup.json
```

**5. Pipe to other tools:**
```bash
fileintel metadata export-table thesis_sources --format csv | grep -i "smith"
```

---

## Benefits

1. **Collection Integrity Review**: Quickly spot missing or incorrect metadata
2. **Backup/Archive**: Export metadata for archival purposes
3. **Data Analysis**: Import into Excel/Pandas for analysis
4. **Documentation**: Generate documentation tables
5. **API Access**: Programmatic access for automation

---

## Performance Considerations

### Optimization
- ✅ Single database query (no N+1 problem)
- ✅ Lazy formatting (only when needed)
- ✅ Streaming for large exports (if needed in future)

### Scalability
- Small collections (<100 docs): Instant
- Medium collections (100-1000 docs): <1 second
- Large collections (1000+ docs): Consider pagination in future

---

## Future Enhancements

1. **Pagination**: For very large collections
2. **Filtering**: Filter documents by metadata criteria
3. **Sorting**: Sort by specific metadata fields
4. **Export to Excel**: Direct .xlsx export
5. **Diff Mode**: Compare metadata between two collections
