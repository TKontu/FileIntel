# Concurrency Configuration Standard

**Date**: 2025-10-15
**Status**: ✅ Implemented

## Standard Value: 16 Concurrent Requests

All concurrency limits across the FileIntel system are standardized to **16** to match vLLM's capacity.

## Rationale

1. **Hardware Constraint**: vLLM is configured with `VLLM_MAX_NUM_SEQS: 16`
2. **No Queueing**: Setting all limits to 16 eliminates queue waits
3. **System Consistency**: Same value throughout prevents configuration drift
4. **Resource Optimization**: Matches GPU capacity (KV cache ~50% at 16 concurrent)

## Configuration Locations

### 1. Application Configuration
**File**: `config/default.yaml:76`
```yaml
max_concurrent_requests: 16  # Must match vLLM VLLM_MAX_NUM_SEQS
```

### 2. fnllm Global Limiter Fix
**File**: `src/fileintel/api/main.py:177`
```python
LimitContext.acquire_semaphore = asyncio.Semaphore(16)
```
**Why**: Fixes fnllm library bug that serializes all requests

### 3. GraphRAG Dynamic Community Selection
**File**: `src/graphrag/query/context_builder/dynamic_community_selection.py:42`
```python
concurrent_coroutines: int = 16  # Match vLLM capacity and system config
```
**Why**: Was hardcoded to 8, causing 2x slowdown in query phase

### 4. GraphRAG Global Search Map Phase
**File**: `src/graphrag/query/structured_search/global_search/search.py:69`
```python
concurrent_coroutines: int = 16  # Match vLLM capacity and system config
```
**Why**: Was hardcoded to 32 (excessive), now matches system capacity

## Before vs After

### Before (Inconsistent)
- vLLM capacity: 16
- Application config: 8
- LimitContext fix: 25
- Dynamic selection: 8 (hardcoded)
- Global search: 32 (hardcoded)

**Result**: Mismatched limits caused queuing, timeouts, and confusion

### After (Standardized)
- vLLM capacity: 16
- Application config: 16
- LimitContext fix: 16
- Dynamic selection: 16
- Global search: 16

**Result**: Clean 1:1 mapping, no queuing, predictable performance

## Performance Impact

### GraphRAG Query (44 community ratings + 40 map calls)

**Before**:
- Dynamic selection: 44 calls ÷ 8 parallel = 5-6 batches = ~40 seconds
- Map phase: 40 calls ÷ 16 parallel = 2-3 batches = ~20 seconds
- **Total: ~60 seconds** (often timed out)

**After**:
- Dynamic selection: 44 calls ÷ 16 parallel = 3 batches = ~20 seconds
- Map phase: 40 calls ÷ 16 parallel = 2-3 batches = ~20 seconds
- **Total: ~40 seconds** (completes reliably)

### vLLM Utilization

**Before**:
```
Running: 1-2 reqs, Waiting: 0-1 reqs, GPU KV cache: 10-20%
```

**After**:
```
Running: 16 reqs, Waiting: 0 reqs, GPU KV cache: 45-50%
```

## Scaling Up

If you need higher concurrency (e.g., multiple concurrent users):

1. **Increase vLLM capacity** first:
   ```bash
   VLLM_MAX_NUM_SEQS: 24  # or 32
   ```

2. **Update all config values** to match:
   - `config/default.yaml`: `max_concurrent_requests: 24`
   - `src/fileintel/api/main.py`: `Semaphore(24)`
   - Restart services

3. **Monitor GPU memory**: KV cache should stay <80%

## Why Not Higher?

**16 is optimal for single-user workload** because:
- GraphRAG queries: max 44 parallel calls (3 batches with 16)
- KV cache: 50% usage leaves safety margin
- Diminishing returns: 16→32 only saves ~5-10 seconds
- Complexity: Higher values require more careful tuning

## Validation

After applying these changes:

```bash
# 1. Restart API
docker-compose restart api

# 2. Run GraphRAG query
poetry run fileintel graphrag query "test" "Tell me about this"

# 3. Monitor vLLM logs
docker logs -f <vllm-container> | grep "Engine 000"

# Expected:
# Running: 16 reqs, Waiting: 0 reqs
# GPU KV cache usage: 45-55%
# Query completes in 30-50 seconds
```

## Related Documentation

- [GraphRAG Concurrency Bottleneck Analysis](./graphrag_concurrency_bottleneck_analysis.md)
- [GraphRAG Query Failure Analysis](./graphrag_query_failure_analysis.md)
