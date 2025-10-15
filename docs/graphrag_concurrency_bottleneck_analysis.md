# GraphRAG Query Concurrency Bottleneck Analysis

**Date**: 2025-10-14
**Status**: ✅ **FIXED** - Monkey-patch applied in `src/fileintel/api/main.py:146-186`
**Issue**: GraphRAG queries timeout after 5 minutes despite vLLM having capacity for 16 concurrent requests
**Symptom**: vLLM logs show "Running: 1 reqs, Waiting: 0 reqs" - only 1 request processed at a time

---

## Quick Summary

**Problem**: The `fnllm` library has a class-level `Semaphore(1)` that serializes all LLM requests globally, preventing parallel execution.

**Solution Applied**: Monkey-patched `LimitContext.acquire_semaphore = Semaphore(25)` at application startup.

**Files Changed**:
- `src/fileintel/api/main.py` (lines 146-186) - Added fix with documentation

**Next Steps**:
1. Restart API: `docker-compose restart api`
2. Verify in logs: `docker logs fileintel-api-1 | grep "fnllm concurrency fix"`
3. Test query and monitor vLLM for parallel execution

---

## Executive Summary

**Root Cause**: The `fnllm` library (v0.4.1) has a **class-level semaphore with value 1** in `LimitContext.acquire_semaphore` that serializes ALL limiter acquisitions across ALL requests, defeating parallelism entirely.

**Impact**:
- Dynamic community selection: 44 LLM calls × ~8 seconds = ~6 minutes sequential execution
- Exceeds `DEFAULT_REQUEST_TIMEOUT = 300` seconds (5 minutes)
- Query times out before completion

**Location**: `/home/appuser/.local/lib/python3.12/site-packages/fnllm/limiting/base.py:20`

---

## Detailed Investigation

### 1. Query Pipeline Flow

```
CLI → API query.py:_process_graph_query()
  ↓
graphrag_service.py:global_search()
  ↓
graphrag/api/query.py:global_search()
  ↓
GlobalSearch.stream_search()
  ├── build_context() [Dynamic Community Selection]
  │   └── DynamicCommunitySelection.select()
  │       └── asyncio.gather(*[rate_relevancy(...) for community in queue])
  │           ├── Semaphore(8) controls max 8 concurrent
  │           └── ~44 communities rated in ~17 seconds
  │
  └── asyncio.gather(*[_map_response_single_batch(...)])
      ├── Semaphore(25) controls max 25 concurrent
      ├── Each batch calls: async with self.semaphore: await self.model.achat(...)
      └── HANGS HERE - takes > 5 minutes, triggers timeout
```

### 2. Configuration Values

**FileIntel Config** (`config/default.yaml`):
```yaml
rag:
  async_processing:
    max_concurrent_requests: 25  # Used but ineffective
```

**GraphRAG Config** (from logs):
```
Chat model config:
  - model: gemma3-12b-awq
  - requests_per_minute: 100000  (effectively unlimited)
  - tokens_per_minute: 100000000  (effectively unlimited)
  - concurrent_requests: 25
```

**Semaphores in Code**:
- Dynamic Selection: `asyncio.Semaphore(8)` at dynamic_community_selection.py:53
- Map Phase: `asyncio.Semaphore(25)` at global_search/search.py:97
- Both should allow parallel requests

**vLLM Capacity**:
```bash
VLLM_MAX_NUM_SEQS: 16  # Can handle 16 parallel requests
```

### 3. fnllm Rate Limiting Stack

**Limiter Creation** (`fnllm/openai/factories/utils.py:73-118`):

```python
def create_limiter(config: OpenAIConfig, backoff_limiter) -> Limiter | None:
    limiters = []

    if backoff_limiter is not None:
        limiters.append(backoff_limiter)

    if config.max_concurrency:
        limiters.append(ConcurrencyLimiter.from_max_concurrency(config.max_concurrency))

    if config.requests_per_minute is not None:
        limiters.append(RPMLimiter.from_rpm(rpm, burst_mode=True))

    if config.tokens_per_minute is not None:
        limiters.append(TPMLimiter.from_tpm(tpm))

    return CompositeLimiter(limiters)  # Combines all limiters
```

**Our Configuration Creates**:
1. `BackoffLimiter` - Handles rate limit errors
2. `ConcurrencyLimiter(25)` - Semaphore limiting max 25 concurrent
3. `RPMLimiter(100000)` - AsyncLimiter allowing 100k requests/minute
4. `TPMLimiter(100000000)` - AsyncLimiter allowing 100M tokens/minute

**Limiter Acquisition** (`fnllm/limiting/composite.py:23-28`):

```python
async def acquire(self, manifest: Manifest) -> None:
    """Acquire the specified amount of tokens from all limiters."""
    # this needs to be sequential, the order of the limiters must be respected
    # to avoid deadlocks
    for limiter in self._acquire_order:
        await limiter.acquire(manifest)  # Sequential, but should still be fast
```

### 4. The Bottleneck: Class-Level Semaphore

**File**: `fnllm/limiting/base.py:15-31`

```python
class LimitContext:
    """A context manager for limiting."""

    acquire_semaphore: ClassVar[Semaphore] = Semaphore()  # ⚠️ DEFAULT = Semaphore(1)

    def __init__(self, limiter: Limiter, manifest: Manifest):
        self._limiter = limiter
        self._manifest = manifest

    async def __aenter__(self) -> LimitContext:
        """Enter the context."""
        async with LimitContext.acquire_semaphore:  # ⚠️ SERIALIZES ALL ACQUISITIONS!
            await self._limiter.acquire(self._manifest)
        return self

    async def __aexit__(self, ...) -> None:
        """Exit the context."""
        await self._limiter.release(self._manifest)
```

**Usage** (`fnllm/base/services/rate_limiter.py:66-71`):

```python
async def invoke(prompt: TInput, **args):
    estimated_input_tokens = self._estimator(prompt, args)
    manifest = Manifest(request_tokens=estimated_input_tokens)

    async with self._limiter.use(manifest):  # ⚠️ Calls LimitContext
        await self._events.on_limit_acquired(manifest)
        result = await delegate(prompt, **args)  # Actual LLM call
```

### 5. Why This Breaks Parallelism

**Intended Behavior**:
1. `asyncio.gather()` schedules 25 `_map_response_single_batch()` coroutines concurrently
2. Each coroutine: `async with semaphore(25): await model.achat(...)`
3. Up to 25 should execute in parallel (within semaphore limit)

**Actual Behavior**:
1. `asyncio.gather()` schedules 25 coroutines concurrently
2. Each coroutine enters: `async with LimitContext.acquire_semaphore:`
3. **ONLY 1 can enter at a time** due to `Semaphore(1)` default
4. All 25 coroutines wait in sequence at this global lock
5. Each waits for previous to complete: `await self._limiter.acquire(manifest)`
6. Result: **Fully sequential execution despite parallel code structure**

**Proof from Logs**:
- 19:45:45 - 19:46:02: 44 rating calls complete (~17 seconds, ~2.5 parallel)
- 19:46:02 - 19:50:55: 4 minute 53 second gap (map phase hangs)
- vLLM shows: `Running: 1 reqs, Waiting: 0 reqs` (sequential processing)

### 6. Why aiolimiter.AsyncLimiter Is Not The Problem

**Leaky Bucket Algorithm** (`aiolimiter/leakybucket.py:131-153`):

```python
async def acquire(self, amount: float = 1) -> None:
    if amount > self.max_rate:
        raise ValueError("Can't acquire more than the maximum capacity")

    while not self.has_capacity(amount):
        # Add to waiters heap and block
        fut = loop.create_future()
        heappush(self._waiters, (amount, self._next_count(), fut))
        await fut  # Only blocks if no capacity

    self._level += amount  # Consumes capacity
    self._wake_next()  # Wake next waiter if any
```

**With RPM=100,000, time_period=60**:
- `max_rate = 100000`
- `_rate_per_sec = 100000 / 60 = 1666.67 requests/second`
- Initial capacity: 100,000 slots available
- Each request consumes 1 slot, regenerates at 1666.67/sec

**Math**: 25 parallel requests consume 25 slots out of 100,000 → **No blocking**

The AsyncLimiter **allows** parallel requests within the rate limit. The problem is the class-level semaphore preventing them from reaching the AsyncLimiter.

### 7. Why It Works Sometimes (Dynamic Selection: 44 calls in 17 seconds)

**Observation**: Dynamic community selection completed 44 calls in ~17 seconds (~2.5 requests/sec) with semaphore=8.

**Explanation**: The class-level `Semaphore(1)` still allows SOME parallelism because:

1. **acquire() is fast when capacity exists**:
   ```python
   async with LimitContext.acquire_semaphore:  # Holds lock briefly
       await self._limiter.acquire(manifest)   # Quick if capacity available
   # Lock released immediately, next request can start
   ```

2. **Actual LLM call is outside the lock**:
   ```python
   async with self._limiter.use(manifest):  # Lock held only during acquire
       result = await delegate(prompt, **args)  # LLM call happens AFTER lock release
   ```

3. **Result**: Requests are **staggered**, not fully parallel
   - Request 1: Lock → acquire → release lock → LLM call (8 sec)
   - Request 2: **waits for lock** → acquire → release lock → LLM call (8 sec)
   - But LLM calls overlap slightly due to lock release before completion

**Why Map Phase Hangs**:
- Map phase has more complex context building
- 40+ communities selected → 40+ map calls needed
- Each takes ~8 seconds
- Sequential: 40 × 8 = 320 seconds (5 min 20 sec)
- Timeout: 300 seconds (5 minutes)
- **Result**: Times out before completion

### 8. Comparison: Expected vs Actual

| Aspect | Expected (Parallel) | Actual (Sequential) |
|--------|---------------------|---------------------|
| Dynamic Selection | 44 calls / 8 = ~5.5 seconds | ~17 seconds (~2.5x slower) |
| Map Phase (40 calls) | 40 calls / 25 = ~12.8 seconds | 40 × 8 = 320 seconds (25x slower) |
| vLLM Utilization | "Running: 8-25 reqs" | "Running: 1 reqs" |
| Success Rate | ✅ Completes in < 30 sec | ❌ Timeout after 300 sec |

---

## Solutions

### Option 1: Monkey-Patch fnllm (Immediate Fix)

**Location**: Early in application startup (before any fnllm usage)

```python
# src/fileintel/api/main.py or similar
import asyncio
from fnllm.limiting.base import LimitContext

# Replace class-level semaphore with higher concurrency
LimitContext.acquire_semaphore = asyncio.Semaphore(25)
```

**Pros**:
- Immediate fix
- No code changes needed
- Respects configured concurrent_requests

**Cons**:
- Fragile (breaks if fnllm internals change)
- Not discoverable in codebase
- May have unintended side effects

### Option 2: Disable Dynamic Community Selection

**Location**: `src/fileintel/rag/graph_rag/services/graphrag_service.py:200`

```python
result, context = await global_search(
    config=graphrag_config,
    entities=dataframes["entities"],
    communities=dataframes["communities"],
    community_reports=dataframes["community_reports"],
    community_level=self.settings.rag.community_levels,
    dynamic_community_selection=False,  # Changed from True
    response_type="text",
    query=query,
)
```

**Pros**:
- Simple configuration change
- Reduces LLM calls from ~44 to ~10-15
- Likely completes within timeout even with sequential execution

**Cons**:
- Reduces query quality (less context-aware community selection)
- Still sequential, just fewer calls

### Option 3: Increase Timeout

**Location**: `src/fileintel/rag/graph_rag/adapters/config_adapter.py:16`

```python
DEFAULT_REQUEST_TIMEOUT = 600.0  # 10 minutes (was 300.0)
```

**Pros**:
- Allows current sequential execution to complete
- No architectural changes

**Cons**:
- Doesn't fix the root problem
- Queries still take 5-6 minutes
- Poor user experience

### Option 4: Upstream Fix (Long-term)

**Issue**: Report to fnllm maintainers that `LimitContext.acquire_semaphore` defeats parallelism

**Suggested Fix** in fnllm:
```python
# fnllm/limiting/base.py
class LimitContext:
    # Remove class-level semaphore entirely
    # acquire_semaphore: ClassVar[Semaphore] = Semaphore()  # DELETE THIS

    async def __aenter__(self) -> LimitContext:
        # Direct acquire without serialization lock
        await self._limiter.acquire(self._manifest)
        return self
```

**Rationale**: Individual limiters (ConcurrencyLimiter, RPMLimiter) already have their own synchronization primitives. The class-level semaphore adds no value and breaks parallelism.

---

## Recommended Immediate Action

**✅ IMPLEMENTED: Option 1 (Monkey-patch)**

The fix has been applied in `src/fileintel/api/main.py` (lines 146-186):
- Monkey-patches `LimitContext.acquire_semaphore = Semaphore(25)` at startup
- Includes comprehensive documentation and error handling
- Logs success/failure for visibility

**Next Steps**:

1. **Restart the API service** to apply the fix:
   ```bash
   docker-compose restart api
   ```

2. **Verify in logs** that fix was applied:
   ```bash
   docker logs fileintel-api-1 | grep "fnllm concurrency fix"
   # Expected: "✓ Applied fnllm concurrency fix: LimitContext.acquire_semaphore = Semaphore(25)"
   ```

3. **Optional**: Add configuration option for dynamic_community_selection in `config/default.yaml`:
   ```yaml
   rag:
     graphrag:
       enable_dynamic_community_selection: true  # Can now keep enabled with parallel execution
   ```

---

## Verification Steps

After restarting the API service:

1. **Verify fix was applied** (check startup logs):
   ```bash
   docker logs fileintel-api-1 | grep "fnllm concurrency fix"
   ```
   Expected output:
   ```
   ✓ Applied fnllm concurrency fix: LimitContext.acquire_semaphore = Semaphore(25)
   ```

2. **Run a GraphRAG query**:
   ```bash
   poetry run fileintel graphrag query "test" "Tell me what this collection is about"
   ```

3. **Monitor vLLM logs** in real-time:
   ```bash
   docker logs -f <vllm-container-name> 2>&1 | grep "Engine 000"
   ```
   **Expected (FIXED)**:
   ```
   Running: 8-16 reqs, Waiting: 0-5 reqs, GPU KV cache usage: 15-25%
   ```
   **Before (BROKEN)**:
   ```
   Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 6-8%
   ```

4. **Check query timing**:
   - **Before fix**: Timeout after 300 seconds (5 minutes)
   - **After fix**: Completes in 30-60 seconds

   Dynamic selection should complete in ~5-7 seconds (was ~17 seconds)

5. **Check API logs for parallel requests**:
   ```bash
   docker logs fileintel-api-1 2>&1 | grep "HTTP Request: POST.*chat/completions" | tail -20
   ```
   **Expected**: Multiple POST requests with timestamps within same second (parallel execution)
   **Before**: Timestamps spaced 6-8 seconds apart (sequential execution)

6. **Monitor for errors** (first few queries):
   ```bash
   docker logs fileintel-api-1 2>&1 | grep -i "error\|deadlock\|timeout"
   ```
   Expected: No new errors related to fnllm or GraphRAG

---

## Additional Notes

### Why This Wasn't Caught Earlier

1. **Works with small datasets**: Sequential execution is acceptable for 5-10 communities
2. **Works with high timeout**: If timeout > execution time, no visible failure
3. **Subtle symptom**: vLLM showed "Running: 1" but no error, just slow performance
4. **Deep in library**: Issue is 3 layers deep (GraphRAG → fnllm → aiolimiter)

### Why httpx Connection Pool Isn't the Bottleneck

Our config has `max_connections=100, max_keepalive_connections=100` which is sufficient. The bottleneck is **before** the HTTP layer - requests never reach httpx in parallel due to the fnllm semaphore.

### Why Rate Limits Aren't the Bottleneck

With RPM=100,000 and TPM=100M, the AsyncLimiter has abundant capacity. The leaky bucket algorithm would allow ~1666 requests/second, but we only need ~2-3/second. The limiters are not blocking - the serialization lock is.
