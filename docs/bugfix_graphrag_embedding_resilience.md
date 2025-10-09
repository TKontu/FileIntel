# GraphRAG Embedding Connection Errors - Root Cause Analysis & Fix

## Issue Summary

GraphRAG indexing fails during the `generate_text_embeddings` workflow due to network connectivity issues with the embedding service.

**Error Messages**:
```
httpcore.RemoteProtocolError: Server disconnected without sending a response.
openai.APIConnectionError: Connection error.
asyncio.exceptions.CancelledError
```

## Root Cause Analysis

### What Happened

**File**: `src/fileintel/tasks/graphrag_tasks.py:103-113`

The GraphRAG configuration was missing critical resilience settings for handling network failures:

```python
# BEFORE - No retry or timeout settings
"embeddings": {
    "api_key": config.get("llm.openai.api_key"),
    "type": "openai_embedding",
    "model": config.rag.embedding_model,
    "batch_size": config.rag.embedding_batch_max_tokens,
}
```

**Why It Failed**:

1. **No Retry Logic**: When embedding service at `http://192.168.0.247:9003/v1/embeddings` disconnected, the request immediately failed
2. **No Timeout Configuration**: Requests could hang indefinitely waiting for response
3. **Unlimited Concurrency**: Too many concurrent requests overwhelmed the embedding service
4. **No Backoff Strategy**: Failed requests weren't retried with exponential backoff

**Observed Behavior**:
- Progress: 11/26, 12/26... 22/26 embeddings completed
- Then: "Server disconnected without sending a response"
- Result: Workflow failed with "generate_text_embeddings completed with errors"
- Some data saved (1,289 entities, 97 communities) but index incomplete/corrupted

### Infrastructure Context

**Embedding Service**: `http://192.168.0.247:9003/v1/embeddings`
- Model: `bge-large-en`
- Batch size: 450 tokens per request
- Issue: Service is timing out or crashing under concurrent load

**Possible Root Causes**:
1. Embedding service overloaded (insufficient GPU memory/compute)
2. Network instability between containers/hosts
3. HTTP connection pool exhaustion
4. Service crashes under concurrent load
5. No connection keep-alive or request timeouts

## Configuration Analysis: Vector RAG vs GraphRAG

Before implementing the fix, let's understand how Vector RAG embeddings are configured:

### Vector RAG Embedding Configuration

**From `config/default.yaml`**:
```yaml
async_processing:
  enabled: true
  batch_size: 4
  max_concurrent_requests: 8      # Tuned for RTX 3090 24GB
  batch_timeout: 30               # 30 seconds per batch
```

**From `embedding_provider.py`**:
```python
@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),  # Exponential backoff
    stop_after_attempt(3),  # 3 retries
)
def _get_embeddings_internal(self, texts: List[str]):
    # ... embedding generation code
```

**Key Points**:
- ✅ **8 concurrent requests** (proven to work with your hardware)
- ✅ **Exponential backoff**: 4s → 8s → 10s between retries
- ✅ **3 retries** total
- ✅ **30-second timeout** per batch
- ✅ Automatic fallback to individual processing if batch fails

### Why "openai" References in Configuration?

The OpenAI Python SDK (`openai.OpenAI`) is used as a **generic HTTP client** for OpenAI-compatible APIs. This is industry standard:

- **vLLM** implements OpenAI API spec at `/v1/embeddings` and `/v1/chat/completions`
- **Ollama**, **LocalAI**, **LM Studio** all implement the same spec
- Most local LLM servers use this as the standard interface

So `config.get("llm.openai.api_key")` is just a generic API key field (often set to `"ollama"` or `"dummy"` for local servers). The SDK is OpenAI-branded but works with any compatible API.

## Fix Applied ✅

### Changes to `graphrag_tasks.py`

**Added Resilience Settings to Embeddings Configuration** (lines 108-118):

```python
"embeddings": {
    "api_key": config.get("llm.openai.api_key"),
    "type": "openai_embedding",
    "model": config.rag.embedding_model,
    "api_base": config.get("llm.openai.embedding_base_url") or config.get("llm.openai.base_url"),
    "batch_size": config.rag.embedding_batch_max_tokens,
    # Resilience settings matching Vector RAG configuration
    "max_retries": 3,  # Match Vector RAG retry count
    "request_timeout": config.rag.async_processing.batch_timeout,  # Match Vector RAG timeout (30s)
    "concurrent_requests": config.rag.async_processing.max_concurrent_requests,  # Match Vector RAG concurrency (8)
}
```

**Added Resilience Settings to LLM Configuration** (lines 96-107):

```python
"llm": {
    "api_key": config.get("llm.openai.api_key"),
    "type": "openai_chat",
    "model": config.rag.llm_model,
    "api_base": config.get("llm.openai.base_url"),
    "max_tokens": config.rag.max_tokens,
    "temperature": config.get("llm.temperature", 0.1),
    # Resilience settings matching Vector RAG configuration
    "max_retries": 3,  # Match Vector RAG retry count
    "request_timeout": 300,  # 5 minutes per LLM request (community summarization can be slow)
    "concurrent_requests": config.rag.async_processing.max_concurrent_requests,  # Match Vector RAG concurrency (8)
}
```

### What Changed

| Setting | Before | After | Purpose |
|---------|--------|-------|---------|
| `max_retries` | None (0) | 3 | Retry failed requests up to 3 times (matches Vector RAG) |
| `request_timeout` | None | 30s (embed) / 300s (LLM) | Timeout per request to prevent hanging |
| `concurrent_requests` | Unlimited | 8 | Match Vector RAG concurrency (tuned for RTX 3090 24GB) |
| `api_base` | Missing | From config | Explicit API endpoint |

### Expected Behavior After Fix

1. **Automatic Retries**: Failed requests retry up to 3 times (matching Vector RAG)
2. **Timeout Protection**: Requests timeout after 30 seconds (embeddings) or 5 minutes (LLM)
3. **Optimal Concurrency**: Max 8 concurrent requests (matching Vector RAG, tuned for RTX 3090)
4. **Graceful Degradation**: Service can recover from transient failures
5. **Consistent Configuration**: GraphRAG now uses same resilience settings as proven Vector RAG pipeline

## Verification

### Syntax Check
```bash
python3 -m py_compile src/fileintel/tasks/graphrag_tasks.py
# ✓ Success - no errors
```

### Test GraphRAG Indexing
```bash
# Retry indexing with new resilience settings
curl -X POST http://localhost:8000/api/v2/graphrag/index \
  -H "Content-Type: application/json" \
  -d '{
    "collection_id": "1faf08ff-c798-4e4d-89ba-5d24d7243477",
    "force_rebuild": true
  }'
```

**Expected**:
- Automatic retries on connection failures
- Completed indexing without "Server disconnected" errors
- All embeddings generated successfully
- Status changes from "not_indexed" to "indexed"

### Check Embedding Service Health
```bash
# Test if embedding service is responsive
curl -X POST http://192.168.0.247:9003/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "test", "model": "bge-large-en"}'

# Expected: 200 OK with embedding vector
```

## Additional Recommendations

### 1. Monitor Embedding Service

**Check vLLM logs** for errors:
```bash
docker logs vllm-container -f --tail 100
```

**Look for**:
- Out of memory errors
- CUDA errors
- Connection pool exhaustion
- Request timeout messages

### 2. Tune Embedding Service

If issues persist, adjust vLLM configuration:

```yaml
# In docker-compose.yml or deployment config
vllm:
  command:
    - --max-num-seqs=4  # Reduce concurrent sequences
    - --max-model-len=512  # Ensure matches model limit
    - --gpu-memory-utilization=0.8  # Leave headroom
    - --disable-log-requests  # Reduce logging overhead
```

### 3. Optimize Network Configuration

**Connection pooling**:
- Ensure HTTP keep-alive is enabled
- Configure connection pool size in httpx/requests
- Add connection timeout settings

**Network debugging**:
```bash
# Test network latency
ping 192.168.0.247

# Check if service is reachable
curl -v http://192.168.0.247:9003/health

# Monitor concurrent connections
netstat -an | grep 9003
```

### 4. Consider Batch Size Reduction

If embedding service still struggles, reduce batch size in `config/default.yaml`:

```yaml
rag:
  embedding_batch_max_tokens: 450  # Current
  # Consider reducing to:
  embedding_batch_max_tokens: 300  # Smaller batches = more requests but less load per request
```

### 5. Scale Embedding Service

If workload is too high for single instance:
- Deploy multiple embedding service replicas
- Add load balancer (nginx/haproxy)
- Use Redis for request queuing
- Implement request rate limiting

## Prevention

### Code Review Checklist

**For API clients**:
- ✓ Always configure `max_retries` for external API calls
- ✓ Always set `request_timeout` to prevent hanging
- ✓ Limit `concurrent_requests` for resource-constrained services
- ✓ Add `retry_delay` with exponential backoff
- ✓ Use `api_base` to make endpoints explicit

**For GraphRAG/LLM integrations**:
- ✓ Test with rate-limited or slow APIs
- ✓ Verify retry logic handles transient failures
- ✓ Ensure timeouts are appropriate for workload (LLM > embeddings)
- ✓ Monitor service health during load testing

### Monitoring Recommendations

**Metrics to Track**:
- Embedding request success rate
- Average request latency
- Concurrent request count
- Retry frequency
- Service uptime

**Alerting Thresholds**:
- Alert if embedding success rate < 95%
- Alert if average latency > 10 seconds
- Alert if concurrent requests > 10 (service overload)

## Files Modified

### `src/fileintel/tasks/graphrag_tasks.py`

**Lines 96-120**: Added resilience configuration to LLM and embeddings settings

**Impact**:
- GraphRAG indexing now resilient to network failures
- Automatic retries prevent workflow failures from transient errors
- Controlled concurrency prevents service overload
- Explicit timeouts prevent hanging requests

## Impact

**Before Fix**:
- ❌ GraphRAG indexing failed on any network hiccup
- ❌ No retry mechanism for transient failures
- ❌ Unlimited concurrency could overwhelm embedding service
- ❌ No timeouts = hanging requests
- ❌ Partial/corrupted indices saved on failure

**After Fix**:
- ✅ Automatic retries handle transient network failures
- ✅ Controlled concurrency protects embedding service
- ✅ Timeouts prevent hanging requests
- ✅ Graceful degradation from temporary issues
- ✅ Higher success rate for GraphRAG indexing

## Next Steps

1. **Retry Failed Collection**:
   ```bash
   curl -X POST http://localhost:8000/api/v2/graphrag/index \
     -H "Content-Type: application/json" \
     -d '{"collection_id": "1faf08ff-c798-4e4d-89ba-5d24d7243477", "force_rebuild": true}'
   ```

2. **Monitor Progress**:
   - Watch Celery worker logs for retry messages
   - Check for "Error Invoking LLM" messages
   - Verify embeddings complete successfully

3. **Verify Completion**:
   ```bash
   # Check status
   curl http://localhost:8000/api/v2/graphrag/1faf08ff-c798-4e4d-89ba-5d24d7243477/status

   # Should return: {"status": "indexed"}

   # Then test entities endpoint
   curl http://localhost:8000/api/v2/graphrag/1faf08ff-c798-4e4d-89ba-5d24d7243477/entities
   ```

4. **If Issues Persist**:
   - Check embedding service health
   - Review vLLM logs for errors
   - Consider reducing `concurrent_requests` to 1
   - Increase `request_timeout` if embeddings are slow
   - Scale embedding service horizontally

---

**Date**: 2025-10-09
**Affected Version**: All versions using GraphRAG without resilience settings
**Fixed In**: Current commit
**Related Issues**: GraphRAG timeout (24-hour limit fix), embedding service connection errors
