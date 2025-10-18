# MinerU Backend Value Fix

## Issue

The initial implementation used incorrect backend value for VLM async mode:

```python
# ❌ WRONG
backend_api_values = {
    'vlm': 'vlm-vllm-engine'  # This is for sync mode
}
```

**Error from MinerU API:**
```
404 Client Error: Not Found for url: http://192.168.0.136:8000/file_parse

Exception: vlm-vllm-engine backend is not supported in async mode,
please use vlm-vllm-async-engine backend
```

---

## Root Cause

MinerU FastAPI has two VLM modes:
- **Sync mode**: `vlm-vllm-engine` (blocking, not used by FastAPI async endpoints)
- **Async mode**: `vlm-vllm-async-engine` (non-blocking, required for FastAPI)

The original migration plan used the sync value, which doesn't work with async FastAPI endpoints.

---

## Solution

Updated the backend mapping to use the correct async value:

```python
# ✅ CORRECT
backend_api_values = {
    'pipeline': 'pipeline',
    'vlm': 'vlm-vllm-async-engine'  # Async mode for FastAPI
}
```

---

## Changes Made

### 1. Code Fix (`mineru_selfhosted.py:206-214`)

```python
# Map model_version to API backend values
# 'vlm' -> 'vlm-vllm-async-engine' (VLM backend with async vLLM inference engine)
# 'pipeline' -> 'pipeline' (OCR + Layout detection pipeline)
backend = getattr(mineru_config, 'model_version', 'pipeline')
backend_api_values = {
    'pipeline': 'pipeline',
    'vlm': 'vlm-vllm-async-engine'
}
api_backend = backend_api_values.get(backend, 'pipeline')
```

### 2. Documentation Updates

- Updated `docs/mineru_backend_selection_implemented.md`
- Updated `/tmp/migration_plan_assessment.md`

---

## Verification

### VLM Backend
```bash
$ poetry run python -c "..."
Backend value in API request: vlm-vllm-async-engine
parse_method present: False
```
✅ Correct: Uses async engine, no parse_method

### Pipeline Backend
```bash
$ poetry run python -c "..."
Backend value in API request: pipeline
parse_method present: True
parse_method value: auto
```
✅ Correct: Uses pipeline, includes parse_method

---

## Backend Value Reference

| Config Value       | API Parameter              | Mode  | Use Case              |
|--------------------|----------------------------|-------|-----------------------|
| `pipeline`         | `backend=pipeline`         | Sync  | OCR/Layout/Tables     |
| `vlm` (sync)       | `backend=vlm-vllm-engine`  | Sync  | Not used by FastAPI   |
| `vlm` (async)      | `backend=vlm-vllm-async-engine` | Async | Vision-Language Model |

---

## Testing Recommendation

Test with both backends to verify the fix works:

### Test 1: Pipeline Backend
```yaml
# config/default.yaml
mineru:
  model_version: "pipeline"
```

```bash
poetry run fileintel documents batch-upload "test-collection" ./pdfs --process
```

**Expected**: Documents processed with pipeline backend, no errors

### Test 2: VLM Backend
```yaml
# config/default.yaml
mineru:
  model_version: "vlm"
```

```bash
# Ensure VLM Docker profile is running
docker compose --profile vlm up -d

poetry run fileintel documents batch-upload "test-collection" ./pdfs --process
```

**Expected**:
- First request takes 30-60s (model loading)
- Subsequent requests faster (5-15s)
- No 404 errors
- Documents processed with VLM backend

---

## Status

✅ **Fixed** - Backend value corrected to `vlm-vllm-async-engine` for async FastAPI mode
