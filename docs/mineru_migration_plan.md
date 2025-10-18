# Change Plan: Add Pipeline/VLM Backend Support

## Overview
Add backend selection capability to support both Pipeline (OCR) and VLM (Vision-Language Model) modes in the self-hosted MinerU processor.

---

## 1. Configuration Changes

### 1.1 Add Backend Configuration Field
**File:** Your config file (likely `config.py` or `config.yaml`)

**Add new field:**
```python
class MinerUConfig:
    # Existing fields...
    backend: str = 'pipeline'  # Options: 'pipeline' | 'vlm'
```

**Rationale:** Allow users to choose backend mode via configuration.

---

## 2. Code Changes

### 2.1 Update `_validate_config()` Method
**Location:** Line ~40-75 in the processor file

**Changes:**
- Add validation for `backend` field
- Validate allowed values: `'pipeline'` or `'vlm'`
- Add warning if using VLM mode (first request will be slow)

**Example:**
```python
def _validate_config(self) -> None:
    """Validate configuration for self-hosted MinerU API."""
    mineru_config = self.config.document_processing.mineru

    # Existing validations...

    # NEW: Validate backend selection
    if hasattr(mineru_config, 'backend'):
        if mineru_config.backend not in ['pipeline', 'vlm']:
            raise DocumentProcessingError(
                f"Invalid backend '{mineru_config.backend}'. Must be 'pipeline' or 'vlm'"
            )
        logger.info(f"Using backend mode: {mineru_config.backend}")

        if mineru_config.backend == 'vlm':
            logger.warning(
                "VLM mode: First request will be slow (~30-60s) while model loads. "
                "Subsequent requests will be faster."
            )
    else:
        logger.info("Backend not specified, defaulting to 'pipeline'")
```

---

### 2.2 Update `_build_form_data()` Method
**Location:** Line ~180-200

**Changes:**
- Replace hardcoded `'backend': 'pipeline'`
- Read backend from config
- Set appropriate parameters for each backend
- Remove `parse_method` for VLM mode (not applicable)

**Implementation:**
```python
def _build_form_data(self) -> Dict[str, str]:
    """
    Build form data for self-hosted API request.

    FastAPI multipart form data requires all values to be strings.
    Lists and booleans must be converted appropriately.
    """
    mineru_config = self.config.document_processing.mineru

    # Determine backend mode
    backend = getattr(mineru_config, 'backend', 'pipeline')

    # Base configuration (common to all backends)
    form_data = {
        'lang_list': mineru_config.language,
        'backend': 'vlm-vllm-engine' if backend == 'vlm' else 'pipeline',
        'formula_enable': str(mineru_config.enable_formula).lower(),
        'table_enable': str(mineru_config.enable_table).lower(),
        'return_md': 'true',
        'return_content_list': 'true',
        'return_model_output': 'true',
        'return_middle_json': 'true',
        'return_images': 'true',
        'response_format_zip': 'true',
        'start_page_id': '0',
        'end_page_id': '99999'
    }

    # Backend-specific parameters
    if backend == 'pipeline':
        # Pipeline mode: Add parse_method
        form_data['parse_method'] = 'auto'  # or from config
    # else: VLM mode needs no extra params

    return form_data
```

---

### 2.3 Update `_process_with_selfhosted_api()` Method
**Location:** Line ~110-160

**Changes:**
- Adjust timeout handling for VLM first request
- Add backend-aware logging

**Implementation:**
```python
def _process_with_selfhosted_api(self, file_path: Path, log) -> Dict[str, Any]:
    """Process PDF with self-hosted MinerU FastAPI."""
    url = f"{self.config.document_processing.mineru.base_url}/file_parse"

    mineru_config = self.config.document_processing.mineru
    backend = getattr(mineru_config, 'backend', 'pipeline')

    # Prepare form data for self-hosted API
    form_data = self._build_form_data()

    # Adjust timeout for VLM first request (model loading)
    timeout = mineru_config.timeout
    if backend == 'vlm':
        # VLM first request may take longer
        timeout = max(timeout, 180)  # At least 3 minutes for first request
        log.info(f"Using extended timeout ({timeout}s) for VLM backend")

    # Use context manager for proper file handle management
    try:
        log.info(f"Uploading {file_path.name} to self-hosted MinerU API (backend: {backend})")

        with open(file_path, 'rb') as file_handle:
            files = {
                'files': (file_path.name, file_handle, 'application/pdf')
            }

            # Make request to self-hosted API
            response = requests.post(
                url,
                data=form_data,
                files=files,
                timeout=timeout  # Use adjusted timeout
            )
            response.raise_for_status()

        # Rest of the method stays the same...
```

---

### 2.4 Update `_build_metadata()` Method
**Location:** Line ~450-490

**Changes:**
- Add backend information to metadata
- Track which backend was used for processing

**Implementation:**
```python
def _build_metadata(
    self,
    json_data: Dict[str, Any],
    mineru_results: Dict[str, Any],
    file_path: Path
) -> Dict[str, Any]:
    """Build comprehensive metadata from all available sources."""
    mineru_config = self.config.document_processing.mineru
    backend = getattr(mineru_config, 'backend', 'pipeline')

    content_list = json_data.get('content_list', [])
    model_data = json_data.get('model_data', [])
    middle_data = json_data.get('middle_data', {})

    # Calculate statistics
    total_elements = len(content_list) if content_list else 0
    pages_found = len(set(item.get('page_idx', 0) for item in content_list)) if content_list else 0

    # Count element types
    element_type_counts = {}
    if content_list:
        for item in content_list:
            elem_type = item.get('type', 'unknown')
            element_type_counts[elem_type] = element_type_counts.get(elem_type, 0) + 1

    metadata = {
        'processor': 'mineru_selfhosted',
        'api_type': 'selfhosted_fastapi',
        'backend': backend,  # NEW: Track which backend was used
        'response_type': mineru_results.get('response_type'),
        'file_path': str(file_path),
        'total_pages': len(model_data) if model_data else pages_found,
        'total_elements': total_elements,
        'element_types': element_type_counts,
        'has_images': len(json_data.get('images', [])) > 0,
        'image_count': len(json_data.get('images', [])),
        'json_files_extracted': [
            key for key in ['content_list', 'model_data', 'middle_data']
            if json_data.get(key) is not None
        ]
    }

    # Rest of the method stays the same...
```

---

### 2.5 Update `_create_elements_from_json()` Method
**Location:** Line ~370-440

**Changes:**
- Update extraction_method in metadata to reflect backend

**Implementation:**
```python
def _create_elements_from_json(
    self,
    json_data: Dict[str, Any],
    markdown_content: str,
    file_path: Path,
    log
) -> List[TextElement]:
    """Create TextElements using JSON-first approach with perfect page mapping."""

    mineru_config = self.config.document_processing.mineru
    backend = getattr(mineru_config, 'backend', 'pipeline')

    content_list = json_data.get('content_list')

    if not content_list:
        log.warning("No content_list found in self-hosted API response, falling back to markdown")
        return self._create_elements_from_markdown_fallback(markdown_content, file_path)

    # ... existing code ...

    # Build rich metadata
    metadata = {
        'source': str(file_path),
        'page_number': page_idx + 1,
        'extraction_method': f'mineru_selfhosted_{backend}_json',  # Include backend
        'backend': backend,  # NEW: Track backend used
        'format': 'structured_json',
        'element_count': total_elements,
        # ... rest of metadata ...
    }
```

---

## 3. Testing Plan

### 3.1 Pipeline Mode Testing
```python
# Config
config.document_processing.mineru.backend = 'pipeline'

# Test cases:
1. Text-based PDF (parse_method='auto' → 'txt')
2. Scanned PDF (parse_method='auto' → 'ocr')
3. Complex tables
4. Mathematical formulas
```

### 3.2 VLM Mode Testing
```python
# Config
config.document_processing.mineru.backend = 'vlm'

# Test cases:
1. Complex layouts
2. Multilingual documents
3. First request (slow - model loading)
4. Subsequent requests (fast - cached model)
5. Timeout handling
```

---

## 4. Documentation Updates

### 4.1 Configuration Documentation
Add to your config documentation:

```markdown
### MinerU Backend Selection

**backend** (string, optional, default: "pipeline")
- `"pipeline"`: Use OCR + Layout detection models (faster, good for standard docs)
- `"vlm"`: Use Vision-Language Model (better for complex layouts, slower first request)

Example:
```yaml
document_processing:
  mineru:
    backend: "pipeline"  # or "vlm"
```

**Performance Notes:**
- Pipeline: ~2-10 seconds per document
- VLM: First request ~30-60s (model loading), subsequent ~5-15s per document
```

### 4.2 Usage Examples
```python
# Pipeline mode (default)
processor = MinerUSelfHostedProcessor(config)
elements, metadata = processor.read(pdf_path)

# VLM mode
config.document_processing.mineru.backend = 'vlm'
processor = MinerUSelfHostedProcessor(config)
elements, metadata = processor.read(pdf_path)
```

---

## 5. Summary of Changes

| File/Location | Change | Lines Affected | Priority |
|---------------|--------|----------------|----------|
| Config | Add `backend` field | N/A | High |
| `_validate_config()` | Add backend validation | ~40-75 | High |
| `_build_form_data()` | Dynamic backend selection | ~180-200 | **Critical** |
| `_process_with_selfhosted_api()` | Timeout adjustment | ~110-160 | Medium |
| `_build_metadata()` | Track backend in metadata | ~450-490 | Low |
| `_create_elements_from_json()` | Backend in extraction_method | ~370-440 | Low |
| Documentation | Usage examples | N/A | Medium |

---

## 6. Backward Compatibility

**Default behavior:** Pipeline mode (existing behavior maintained)
- No config change required for existing users
- Backend defaults to `'pipeline'` if not specified
- All existing code continues to work

---

## 7. Optional Enhancements (Future)

1. **Auto-detection:** Automatically choose backend based on document complexity
2. **Retry logic:** Fallback to pipeline if VLM fails
3. **Performance metrics:** Track processing time per backend
4. **Caching:** Cache backend choice per document type

---

## 8. Docker Compose Backend Selection

### 8.1 Current Setup
The docker-compose.yml file includes two profiles:
- `pipeline`: Optimized for OCR/Layout models (10GB VRAM allocation)
- `vlm`: Optimized for Vision-Language Model (9.3GB VRAM allocation)

### 8.2 Starting the Correct Backend

**For Pipeline Mode:**
```bash
docker compose --profile pipeline up -d
```

**For VLM Mode:**
```bash
docker compose --profile vlm up -d
```

**Switching Modes:**
```bash
# Stop current mode
docker compose down

# Start different mode
docker compose --profile vlm up -d
```

### 8.3 Configuration Alignment
Ensure your application's `backend` configuration matches the Docker Compose profile you're running:

```python
# When running: docker compose --profile pipeline up -d
config.document_processing.mineru.backend = 'pipeline'

# When running: docker compose --profile vlm up -d
config.document_processing.mineru.backend = 'vlm'
```

**Important:** Using mismatched backend configuration will result in inefficient resource usage:
- Sending `backend=pipeline` requests to VLM-optimized container: Pipeline models will load (~6GB) alongside idle VLM allocation
- Sending `backend=vlm` requests to Pipeline-optimized container: VLM will load with limited memory allocation

---

## 9. Migration Checklist

- [ ] Add `backend` field to configuration
- [ ] Update `_validate_config()` with backend validation
- [ ] Update `_build_form_data()` to use dynamic backend
- [ ] Update `_process_with_selfhosted_api()` with timeout handling
- [ ] Update `_build_metadata()` to track backend
- [ ] Update `_create_elements_from_json()` extraction method
- [ ] Update documentation with backend selection guide
- [ ] Test pipeline mode with various PDF types
- [ ] Test VLM mode with complex documents
- [ ] Test first VLM request (model loading)
- [ ] Verify Docker Compose profile switching
- [ ] Verify backward compatibility (default to pipeline)
- [ ] Monitor memory usage in both modes
- [ ] Document performance characteristics

---

This plan ensures minimal disruption while adding powerful VLM support!
