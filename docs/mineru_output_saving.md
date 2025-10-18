# MinerU Output Saving Feature

## Overview

Added configurable saving of MinerU raw outputs for debugging and inspection of extraction issues.

**Status**: ✅ Implemented and disabled by default

---

## Configuration

### Enable Output Saving

```yaml
# config/default.yaml
document_processing:
  mineru:
    # Enable saving of raw MinerU outputs
    save_outputs: true

    # Directory where outputs will be saved
    output_directory: "/home/appuser/app/mineru_outputs"
```

### Default Settings

```yaml
# Disabled by default to conserve disk space
save_outputs: false
output_directory: "/home/appuser/app/mineru_outputs"
```

---

## What Gets Saved

### ZIP Response Format (Preferred)

When MinerU returns a ZIP file, the following are saved:

```
/home/appuser/app/mineru_outputs/
└── <document-name>/
    ├── <document-name>.zip              # Raw ZIP file from MinerU
    ├── <document-name>.md               # Extracted markdown
    ├── <document-name>_content_list.json # Structured content with types
    ├── <document-name>_model.json       # Model output data
    ├── <document-name>_middle.json      # Intermediate processing data
    └── images/                          # Extracted images (if any)
        ├── image_0.png
        ├── image_1.png
        └── ...
```

### JSON Response Format

When MinerU returns JSON directly:

```
/home/appuser/app/mineru_outputs/
└── <document-name>/
    ├── <document-name>_response.json    # Full API response
    ├── <document-name>.md               # Markdown content
    ├── <document-name>_content_list.json # Content structure
    ├── <document-name>_model.json       # Model data
    └── <document-name>_middle.json      # Processing metadata
```

---

## Use Cases

### 1. Debugging Chunking Issues

**Problem**: Chunks exceeding token limits or reversed text

**Solution**:
1. Enable `save_outputs: true`
2. Process problematic document
3. Inspect `<document>_content_list.json` to see:
   - Element types (text, table, title, etc.)
   - Raw text content
   - Bounding boxes
   - Page indices

```bash
# Enable saving
sed -i 's/save_outputs: false/save_outputs: true/' config/default.yaml

# Process document
poetry run fileintel documents batch-upload "test" ./problem.pdf --process

# Inspect outputs
cat /home/appuser/app/mineru_outputs/problem/problem_content_list.json | jq '.[] | {type, text: .text[0:100]}'
```

---

### 2. Comparing Backend Results

**Goal**: Compare pipeline vs VLM extraction quality

```yaml
# Test with pipeline
model_version: "pipeline"
save_outputs: true
output_directory: "/home/appuser/app/mineru_outputs/pipeline"
```

Process documents, then:

```yaml
# Test with VLM
model_version: "vlm"
save_outputs: true
output_directory: "/home/appuser/app/mineru_outputs/vlm"
```

Compare outputs:

```bash
diff -r /home/appuser/app/mineru_outputs/pipeline/doc1/ \
        /home/appuser/app/mineru_outputs/vlm/doc1/
```

---

### 3. Inspecting Table Extraction

**Goal**: Understand how tables are extracted

```bash
# Enable saving
save_outputs: true

# Process document with tables
# Then inspect content_list for table elements
cat mineru_outputs/document/document_content_list.json | \
  jq '.[] | select(.type == "table")'
```

**What to look for**:
- `type: "table"` elements
- `table_body` field (HTML representation)
- `table_caption` field
- Text content (may be empty for tables)

---

### 4. Markdown Comparison

**Goal**: Compare MinerU markdown vs actual extraction

```bash
# Markdown as MinerU generated it
cat mineru_outputs/document/document.md

# Check how it was processed
# (look at TextElement text in database or logs)
```

---

## Implementation Details

### Code Changes

**Files Modified**:
1. `src/fileintel/core/config.py:148-150` - Added config fields
2. `src/fileintel/document_processing/processors/mineru_selfhosted.py:198-291` - Added `_save_mineru_outputs()` method
3. `src/fileintel/document_processing/processors/mineru_selfhosted.py:112` - Call save method after API response
4. `config/default.yaml:111-115` - Added configuration

### Method: `_save_mineru_outputs()`

**Location**: `mineru_selfhosted.py:198-291`

**Behavior**:
- Returns immediately if `save_outputs: false`
- Creates directory: `<output_directory>/<document_name>/`
- Saves all available outputs based on response type
- Logs all saved files
- Never fails processing (catches exceptions)

**Features**:
- Handles both ZIP and JSON responses
- Automatically extracts ZIP contents
- Parses and pretty-prints JSON
- Organizes by document name
- Thread-safe (each document gets own directory)

---

## Disk Space Considerations

### Typical Output Sizes

| Document Type | ZIP Size | Extracted Size | Total Size |
|--------------|----------|----------------|------------|
| Simple PDF (10 pages) | 50 KB | 150 KB | ~200 KB |
| Complex PDF (50 pages) | 500 KB | 2 MB | ~2.5 MB |
| PDF with images (100 pages) | 5 MB | 20 MB | ~25 MB |

### Recommendations

**Development/Debugging**:
```yaml
save_outputs: true  # Enable for investigation
```

**Production**:
```yaml
save_outputs: false  # Disable to save disk space
```

**Selective Saving**:
- Enable only when debugging specific issues
- Process one problematic document at a time
- Disable after collecting needed outputs
- Manually clean up old outputs: `rm -rf /home/appuser/app/mineru_outputs/*`

---

## Docker Volume Mapping

To access outputs from host machine, add volume mapping:

```yaml
# docker-compose.yml
services:
  celery-worker:
    volumes:
      - ./mineru_outputs:/home/appuser/app/mineru_outputs
```

Then outputs will be available on host at:
```bash
ls ./mineru_outputs/<document-name>/
```

---

## Example Workflow

### Debugging Reversed Text Issue

1. **Enable saving**:
   ```yaml
   save_outputs: true
   ```

2. **Process problematic document**:
   ```bash
   poetry run fileintel documents batch-upload "test" ./reversed.pdf --process
   ```

3. **Check logs for document ID**:
   ```bash
   docker compose logs celery-worker | grep "Saving MinerU outputs"
   # Output: Saving MinerU outputs to /home/appuser/app/mineru_outputs/reversed
   ```

4. **Inspect content_list**:
   ```bash
   docker compose exec celery-worker cat /home/appuser/app/mineru_outputs/reversed/reversed_content_list.json | jq '.'
   ```

5. **Look for table elements**:
   ```bash
   cat reversed_content_list.json | jq '.[] | select(.type == "table") | {text, bbox, page_idx}'
   ```

6. **Check if text is actually reversed in raw output**:
   ```bash
   cat reversed_content_list.json | jq '.[].text' | grep -o "ELBAT\|elciheV"
   ```

7. **Disable saving after debugging**:
   ```yaml
   save_outputs: false
   ```

---

## Logging

### When Enabled

```
INFO: Saving MinerU outputs to /home/appuser/app/mineru_outputs/document
INFO: Saved raw ZIP: /home/appuser/app/mineru_outputs/document/document.zip
INFO: Extracted 8 files from ZIP
INFO: Successfully saved MinerU outputs for document.pdf
```

### When Disabled

No logging (method returns immediately)

### On Error

```
WARNING: Failed to save MinerU outputs for document.pdf: [Errno 13] Permission denied
```

Note: Processing continues even if saving fails

---

## Cleanup

### Manual Cleanup

```bash
# Remove all saved outputs
docker compose exec celery-worker rm -rf /home/appuser/app/mineru_outputs/*

# Remove specific document outputs
docker compose exec celery-worker rm -rf /home/appuser/app/mineru_outputs/document-name
```

### Automated Cleanup (TODO)

Future enhancement: Add config for automatic cleanup:
```yaml
save_outputs: true
output_retention_days: 7  # Auto-delete outputs older than 7 days
```

---

## Troubleshooting

### Outputs Not Being Saved

**Check**:
1. Config setting: `save_outputs: true`
2. Directory permissions: `docker compose exec celery-worker ls -la /home/appuser/app/`
3. Logs for errors: `docker compose logs celery-worker | grep "Failed to save"`

### Permission Denied

**Solution**:
```bash
# Create directory with correct permissions
docker compose exec celery-worker mkdir -p /home/appuser/app/mineru_outputs
docker compose exec celery-worker chown appuser:appuser /home/appuser/app/mineru_outputs
```

### Outputs Not Visible on Host

**Solution**: Add volume mapping to docker-compose.yml (see Docker Volume Mapping section)

---

## Related Files

- **Config**: `config/default.yaml:111-115`
- **Schema**: `src/fileintel/core/config.py:148-150`
- **Implementation**: `src/fileintel/document_processing/processors/mineru_selfhosted.py:198-291`
- **Usage**: `src/fileintel/document_processing/processors/mineru_selfhosted.py:112`

---

## Summary

✅ **Feature**: Configurable saving of MinerU raw outputs
✅ **Default**: Disabled to conserve disk space
✅ **Use case**: Debugging extraction and chunking issues
✅ **Safety**: Never fails processing (catches exceptions)
✅ **Organization**: Outputs organized by document name

**To use**: Set `save_outputs: true` in config, process documents, inspect outputs in configured directory.
