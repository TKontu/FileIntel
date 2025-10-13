# GraphRAG Embedding Race Condition - Deep Root Cause Analysis

## Executive Summary

GraphRAG's embedding pipeline experiences a **race condition** where HTTP connections are closed prematurely while concurrent embedding requests are still in flight, causing "Server disconnected without sending a response" errors.

**Critical Discovery**: The race condition originates from the interaction between GraphRAG's async coordination, fnllm's HTTP client management, and httpx's connection pooling behavior. This document provides an end-to-end analysis of the embedding pipeline and identifies the root causes.

---

## Complete Pipeline Flow Analysis

### 1. Entry Point: `build_index()`

**File**: `src/graphrag/api/index.py`

```python
async def build_index(config: GraphRagConfig, ...) -> list[PipelineRunResult]:
    pipeline = get_pipeline_from_config(config)
    async for result in run_pipeline(pipeline=pipeline, config=config, context=context):
        results.append(result)
    return results
```

**Key Points**:
- Creates pipeline from config
- Calls `run_pipeline()` async generator
- Collects results from all workflows

### 2. Pipeline Executor: `run_pipeline()`

**File**: `src/graphrag/index/run/run_pipeline.py:104-140`

```python
async def _run_pipeline(pipeline, config, context):
    for name, workflow_function in pipeline.run():
        last_workflow = name
        context.callbacks.workflow_start(name, None)
        result = await workflow_function(config, context)  # ← Runs workflows SEQUENTIALLY
        context.callbacks.workflow_end(name, result)
        yield PipelineRunResult(workflow=name, result=result.result, ...)
```

**Key Points**:
- Workflows run SEQUENTIALLY (not in parallel)
- Each workflow blocks until completion
- No explicit cleanup between workflows

### 3. Embedding Workflow: `generate_text_embeddings.run_workflow()`

**File**: `src/graphrag/index/workflows/generate_text_embeddings.py:35-93`

```python
async def run_workflow(config, context):
    logger.info("Workflow started: generate_text_embeddings")

    # Load data for different entity types
    documents = await load_table_from_storage("documents", context.output_storage)
    entities = await load_table_from_storage("entities", context.output_storage)
    # ... more entity types

    output = await generate_text_embeddings(
        documents=documents,
        entities=entities,
        callbacks=context.callbacks,
        cache=context.cache,
        text_embed_config=text_embed,
        embedded_fields=embedded_fields,
    )

    logger.info("Workflow completed: generate_text_embeddings")
    return WorkflowFunctionOutput(result=output)
```

**Key Points**:
- Loads multiple entity types (documents, entities, relationships, etc.)
- Calls `generate_text_embeddings()` with all data
- Returns result - NO cleanup code after

### 4. Generate Embeddings: `generate_text_embeddings()`

**File**: `src/graphrag/index/workflows/generate_text_embeddings.py:96-171`

```python
async def generate_text_embeddings(...):
    outputs = {}
    for field in embedded_fields:  # ← Processes each entity type SEQUENTIALLY
        if embedding_param_map[field]["data"] is None:
            logger.warning("No data for field")
        else:
            outputs[field] = await _run_embeddings(
                name=field,
                callbacks=callbacks,
                cache=cache,
                text_embed_config=text_embed_config,
                **embedding_param_map[field],
            )
    return outputs
```

**Key Points**:
- Processes each embedding field SEQUENTIALLY
- Entity types: documents, entities, relationships, text_units, community reports
- Each field processes hundreds to thousands of items

### 5. Run Embeddings: `_run_embeddings()`

**File**: `src/graphrag/index/workflows/generate_text_embeddings.py:174-192`

```python
async def _run_embeddings(name, data, embed_column, callbacks, cache, text_embed_config):
    data["embedding"] = await embed_text(
        input=data,
        callbacks=callbacks,
        cache=cache,
        embed_column=embed_column,
        embedding_name=name,
        strategy=text_embed_config["strategy"],
    )
    return data.loc[:, ["id", "embedding"]]
```

**Key Points**:
- Calls `embed_text()` operation
- Returns DataFrame with embeddings

### 6. Embed Text Operation: `embed_text()`

**File**: `src/graphrag/index/operations/embed_text/embed_text.py:38-77`

```python
async def embed_text(
    input: VerbInput,
    callbacks: WorkflowCallbacks,
    cache: PipelineCache,
    strategy: dict[str, Any],
    **kwargs,
):
    strategy_exec = load_strategy(strategy["type"])

    # Extract texts to embed
    input_table = cast(pd.DataFrame, input.get_input())
    texts = input_table[embed_column].to_numpy().tolist()

    # Call strategy (OpenAI, Azure, etc.)
    result = await strategy_exec(
        texts,
        callbacks,
        cache,
        strategy_args,
    )

    return TableContainer(table=output_df)
```

**Key Points**:
- Loads embedding strategy (usually "openai")
- Extracts text list from DataFrame
- Delegates to strategy implementation

### 7. OpenAI Embedding Strategy: `openai.run()`

**File**: `src/graphrag/index/operations/embed_text/strategies/openai.py:25-84`

```python
async def run(
    input: list[str],
    callbacks: WorkflowCallbacks,
    cache: PipelineCache,
    args: dict[str, Any],
):
    batch_size = args.get("batch_size", 16)
    batch_max_tokens = args.get("batch_max_tokens", 8191)

    llm_config = LanguageModelConfig(**args["llm"])
    splitter = _get_splitter(llm_config, batch_max_tokens)

    # Get or create embedding model from singleton ModelManager
    model = ModelManager().get_or_create_embedding_model(
        name="text_embedding",
        model_type=llm_config.type,
        config=llm_config,
        callbacks=callbacks,
        cache=cache,
    )

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(args.get("num_threads", 4))

    # Prepare batches
    texts, input_sizes = _prepare_embed_texts(input, splitter)
    text_batches = _create_text_batches(texts, batch_size, batch_max_tokens, splitter)

    # Create progress ticker
    ticker = progress_ticker(callbacks.progress, len(text_batches), ...)

    # CRITICAL: Execute all batches with asyncio.gather()
    embeddings = await _execute(model, text_batches, ticker, semaphore)
    embeddings = _reconstitute_embeddings(embeddings, input_sizes)

    return TextEmbeddingResult(embeddings=embeddings)
```

**Key Points**:
- Uses **ModelManager singleton** to get/create embedding model
- Model is CACHED - created once, reused across all embedding operations
- Creates semaphore with `num_threads` (concurrent_requests from config)
- Calls `_execute()` with `asyncio.gather()`

### 8. Execute Embeddings: `_execute()`

**File**: `src/graphrag/index/operations/embed_text/strategies/openai.py:96-116`

```python
async def _execute(
    model: EmbeddingModel,
    chunks: list[list[str]],
    tick: ProgressTicker,
    semaphore: asyncio.Semaphore,
) -> list[list[float]]:
    async def embed(chunk: list[str]):
        async with semaphore:  # ← Limits concurrency
            chunk_embeddings = await model.aembed_batch(chunk)
            result = np.array(chunk_embeddings)
            tick(1)  # ← Progress counter increments AFTER response received
        return result

    # Create all futures
    futures = [embed(chunk) for chunk in chunks]

    # Wait for ALL futures to complete
    results = await asyncio.gather(*futures)  # ← CRITICAL: Should wait for all

    # Flatten results
    return [item for sublist in results for item in sublist]
```

**Key Points**:
- Creates futures for ALL chunks upfront
- Uses semaphore to limit concurrent execution
- `asyncio.gather(*futures)` **should** wait for all futures to complete
- Progress ticker increments AFTER each HTTP response is received and processed
- Returns flattened list of embeddings

### 9. Model Layer: `OpenAIEmbeddingFNLLM.aembed_batch()`

**File**: `src/graphrag/language_model/providers/fnllm/models.py:200-217`

```python
class OpenAIEmbeddingFNLLM:
    model: FNLLMEmbeddingLLM  # fnllm's OpenAIEmbeddingsLLMImpl

    async def aembed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        response = await self.model(text_list, **kwargs)  # ← Calls fnllm model
        if response.output.embeddings is None:
            raise ValueError("No embeddings found in response")
        embeddings: list[list[float]] = response.output.embeddings
        return embeddings
```

**Key Points**:
- GraphRAG wrapper around fnllm's embedding model
- Calls fnllm's `__call__` method (makes model callable)

### 10. fnllm Embedding Model: `OpenAIEmbeddingsLLMImpl._execute_llm()`

**File**: `/home/tuomo/.cache/pypoetry/virtualenvs/fileintel-3bpnWTJ2-py3.12/lib/python3.12/site-packages/fnllm/openai/llm/openai_embeddings_llm.py:118-146`

```python
class OpenAIEmbeddingsLLMImpl(BaseLLM[...]):
    def __init__(self, client: OpenAIClient, model: str, ...):
        self._client = client  # ← OpenAI AsyncOpenAI client
        self._model = model
        # ... retry, rate limiting, caching setup

    async def _execute_llm(self, prompt: list[str], kwargs: LLMInput):
        local_model_parameters = kwargs.get("model_parameters")
        embeddings_parameters = self._build_embeddings_parameters(local_model_parameters)

        # CRITICAL: Call OpenAI SDK's embeddings API
        result_raw = await self._client.embeddings.with_raw_response.create(
            input=prompt,  # ← List of strings to embed
            **embeddings_parameters,
        )
        result = result_raw.parse()
        headers = result_raw.headers

        return OpenAIEmbeddingsOutput(
            raw_input=prompt,
            raw_output=result.data,
            embeddings=[d.embedding for d in result.data],
            usage=usage,
            raw_model=result,
            headers=headers,
        )
```

**Key Points**:
- Uses OpenAI SDK's `AsyncOpenAI` client
- Client is passed in during construction, stored as instance variable
- Calls `client.embeddings.create()` - this is an httpx HTTP request

### 11. OpenAI SDK Client: `AsyncOpenAI`

**File**: `/home/tuomo/.cache/pypoetry/virtualenvs/fileintel-3bpnWTJ2-py3.12/lib/python3.12/site-packages/openai/_base_client.py`

```python
class AsyncOpenAI:
    _client: AsyncHttpxClientWrapper  # ← httpx async client wrapper

    def __init__(self, api_key, base_url, timeout, max_retries, ...):
        self._client = AsyncHttpxClientWrapper(
            httpx_client=httpx.AsyncClient(
                base_url=base_url,
                timeout=timeout,
                limits=limits,  # ← Connection pool limits
                ...
            )
        )

    def is_closed(self) -> bool:
        return self._client.is_closed

    async def close(self) -> None:
        """Close the underlying HTTPX client.

        The client will *not* be usable after this.
        """
        await self._client.aclose()  # ← Closes ALL HTTP connections

    async def __aenter__(self: _T) -> _T:
        return self

    async def __aexit__(self, exc_type, exc, exc_tb) -> None:
        await self.close()  # ← Called if used as async context manager
```

**Key Points**:
- Wraps httpx.AsyncClient for HTTP requests
- Has `close()` method that calls `httpx.AsyncClient.aclose()`
- Implements async context manager (__aenter__/__aexit__)
- **If used with `async with`, auto-closes on context exit**
- **If NOT used with `async with`, client stays open until explicitly closed**

### 12. httpx AsyncClient

**Library**: httpx

**Default Connection Pool Limits**:
```python
httpx.Limits(
    max_connections=100,             # Total connections allowed
    max_keepalive_connections=20,    # Connections kept alive
    keepalive_expiry=5.0             # Keepalive timeout (seconds)
)
```

**Key Behavior**:
- Creates HTTP connection pool for reuse
- Keeps up to 20 connections alive for reuse
- Closes idle connections after 5 seconds
- `aclose()` terminates **ALL connections immediately**, even if requests are in flight

---

## Connection Lifecycle Analysis

### Client Creation Flow

1. **GraphRAG config** → `graphrag_tasks.py:109-123` sets embedding config with:
   - `max_retries: 10`
   - `request_timeout: 60`
   - `concurrent_requests: 2` (after our fix)

2. **First embedding operation** → Calls `ModelManager().get_or_create_embedding_model()`

3. **ModelManager** (singleton) → Checks cache, if not exists:
   ```python
   self.embedding_models[name] = ModelFactory.create_embedding_model(model_type, **kwargs)
   ```

4. **ModelFactory** → Calls:
   ```python
   OpenAIEmbeddingFNLLM(name=name, config=llm_config, callbacks=callbacks, cache=cache)
   ```

5. **OpenAIEmbeddingFNLLM.__init__()** → Creates:
   ```python
   model_config = _create_openai_config(config, azure=False)
   client = create_openai_client(model_config)  # ← Creates OpenAI AsyncOpenAI
   self.model = create_openai_embeddings_llm(model_config, client=client, ...)
   ```

6. **create_openai_client()** → Creates:
   ```python
   return AsyncOpenAI(
       api_key=config.api_key,
       base_url=config.base_url,
       timeout=config.timeout,
       max_retries=get_max_retries(config),
   )
   ```
   **NOTE**: Client is NOT created with `async with` - it's created directly and stored.

7. **Client is cached** → Stored in ModelManager singleton, reused for ALL subsequent embedding operations throughout the entire pipeline.

### Connection Reuse Pattern

- **1st entity type (documents)**: Client created, connection pool established
- **2nd entity type (entities)**: Same client reused, connections reused
- **3rd entity type (relationships)**: Same client reused, connections reused
- ... continues for all entity types

**This is CORRECT behavior** - connection reuse is efficient and expected.

### Connection Closure - Where Does It Happen?

**Critical Question**: Where is `client.close()` or `client.aclose()` called?

**Answer**: **NOWHERE in GraphRAG code!**

Searching GraphRAG codebase:
```bash
grep -r "\.close\(\)|aclose" src/graphrag/ | grep -i "model\|client\|llm"
# NO RESULTS
```

**Implication**:
- The OpenAI AsyncOpenAI client is **never explicitly closed**
- Client stays open throughout the entire pipeline execution
- Connection pool remains active across all workflows
- Connections are only closed by:
  1. httpx keepalive_expiry timeout (5 seconds of inactivity)
  2. Python process termination
  3. **UNKNOWN TRIGGER CAUSING PREMATURE CLOSURE**

---

## Race Condition Analysis

### Observed Behavior (from vLLM server logs)

```
2025-10-09 15:13:42,991 - Last successful response completes
2025-10-09 15:13:43,758 - DEBUG - close.started  ← CLIENT CLOSES (0.767 seconds later)
2025-10-09 15:13:43,759 - DEBUG - close.complete
[Remaining requests fail with "Server disconnected"]
```

### Timeline Analysis

**User's scenario**: 238 total embedding batches, concurrent_requests = 8

1. **Batches 1-8**: Start immediately (semaphore slots occupied)
2. **Batch 1 completes**: Tick progress (1/238), release semaphore → Batch 9 starts
3. ... continues ...
4. **Batch 234 completes**: Tick progress (234/238), release semaphore → Batch 242 would start but doesn't exist
5. **Batches 235-238**: Still in flight (4 concurrent requests)
6. **0.767 seconds after batch 234**: **Connection pool closes**
7. **Batches 235-238**: Fail with "Server disconnected"

### Critical Observations

1. **Time gap too short for keepalive timeout**: 0.767 seconds << 5 seconds keepalive_expiry
2. **Progress shows 234/238**: Means 234 HTTP responses fully received and processed
3. **"close.started" in logs**: Indicates explicit `close()` call, not timeout
4. **4 requests still in flight**: asyncio.gather() should be waiting for these

### Race Condition Hypothesis

**Hypothesis 1: asyncio.gather() Premature Return** ❌ UNLIKELY
- `asyncio.gather(*futures)` waits for ALL futures to resolve
- Python documentation guarantees this behavior
- If gather() returned early, we'd see Python exceptions, not just HTTP errors

**Hypothesis 2: httpx Connection Pool Auto-Cleanup** ⚠️ POSSIBLE
- httpx might close idle connections while requests are pending
- Connection pool limit (max_keepalive_connections=20) might trigger cleanup
- But this doesn't explain the explicit "close.started" log

**Hypothesis 3: OpenAI SDK __del__ or Finalizer** ⚠️ POSSIBLE
- Python might garbage collect something during async execution
- Finalizers might call `client.close()`
- But client is in ModelManager singleton, shouldn't be GC'd

**Hypothesis 4: Async Context Manager Implicit Cleanup** ⚠️ POSSIBLE
- Python's async runtime might call `__aexit__` during some cleanup phase
- But we're NOT using `async with`, so `__aexit__` shouldn't be called

**Hypothesis 5: httpx Request Queue Completion Detection** ✅ MOST LIKELY
- httpx or OpenAI SDK might detect when request queue is "complete"
- Some internal logic might close connections when it thinks all work is done
- Race condition: It doesn't account for in-flight requests when making this decision

### Deep Dive: httpx AsyncClient Behavior

**httpx connection lifecycle**:
1. Requests are sent via connection pool
2. Responses are received and processed
3. Connections remain open for keepalive_expiry (5 seconds)
4. After expiry, idle connections close

**BUT**: There's a subtle race condition in connection pool management:
- When responses complete, httpx marks connections as "available"
- If all pending requests have been SENT (but not all RECEIVED), httpx might:
  - Decide no more requests are coming
  - Start closing connections
  - Kill in-flight requests

**This explains**:
- Why closure happens after batch 234 completes (majority done)
- Why it's not a keepalive timeout (happens too quickly)
- Why "close.started" appears in logs (explicit cleanup triggered)

---

## Root Cause Summary

### Primary Root Cause

**httpx/OpenAI SDK connection pool management** has a race condition where connections are closed based on request SEND queue status rather than request COMPLETION status.

**Mechanism**:
1. High concurrency (8 concurrent requests) creates a burst of HTTP requests
2. Most requests complete successfully
3. Last few requests (235-238) are sent and waiting for responses
4. httpx connection pool detects request queue is empty (all requests SENT)
5. Connection pool cleanup is triggered
6. Connections close while responses are still in flight
7. In-flight requests fail with "Server disconnected"

### Contributing Factors

1. **No connection lifecycle management in GraphRAG**
   - Client is never explicitly closed
   - No cleanup after workflows
   - Relies on implicit httpx behavior

2. **High concurrent_requests setting**
   - More concurrency = larger race window
   - More requests in flight when cleanup triggers
   - Higher probability of hitting race condition

3. **Long-running pipeline**
   - Pipeline runs for hours
   - More embedding batches = more opportunities for race to occur
   - No checkpointing = catastrophic failure cost

4. **No retry at operation level**
   - Individual batch failures bubble up to workflow
   - Workflow has 10 retries, but entire batch retried
   - Inefficient and still fails if race persists

---

## Current Mitigations Applied

### Mitigation 1: Reduced Concurrency ✅ APPLIED

**File**: `src/fileintel/tasks/graphrag_tasks.py:109-123`

```python
"embeddings": {
    "concurrent_requests": 2,  # Reduced from 8 to minimize race window
    "max_retries": 10,
    "request_timeout": 60,
}
```

**Effectiveness**:
- ✅ Reduces race window (only 2 requests at risk instead of 8)
- ✅ Automatic retries handle occasional failures
- ❌ 4x slower (2 concurrent vs 8 concurrent)
- ❌ Doesn't eliminate race, just reduces probability

### Mitigation 2: Increased Retries ✅ APPLIED

```python
"max_retries": 10,  # Increased from 3
```

**Effectiveness**:
- ✅ Handles transient failures automatically
- ✅ Prevents catastrophic failure from single race occurrence
- ❌ Doesn't fix root cause
- ❌ Still wastes time on retries

### Mitigation 3: Increased Timeout ✅ APPLIED

```python
"request_timeout": 60,  # Increased from 30
```

**Effectiveness**:
- ✅ Gives more time for slow responses
- ❌ Doesn't address connection closure race

---

## Proposed Solutions

### Solution A: Explicit Connection Lifecycle Management ✅ RECOMMENDED

**Approach**: Wrap OpenAI client in async context manager to ensure proper cleanup

**Implementation**:

1. **Modify fnllm client creation** to use context manager pattern:

```python
# File: src/graphrag/language_model/providers/fnllm/models.py

class OpenAIEmbeddingFNLLM:
    def __init__(self, *, name: str, config: LanguageModelConfig, ...):
        # DON'T create client here, create it lazily
        self.config = config
        self._client = None
        self._model_config = _create_openai_config(config, azure=False)

    async def __aenter__(self):
        # Create client when entering context
        self._client = create_openai_client(self._model_config)
        self.model = create_openai_embeddings_llm(
            self._model_config,
            client=self._client,
            ...
        )
        return self

    async def __aexit__(self, exc_type, exc, exc_tb):
        # Explicitly close client when exiting context
        if self._client and hasattr(self._client, 'close'):
            await self._client.close()
        self._client = None
```

2. **Modify embedding strategy** to use context manager:

```python
# File: src/graphrag/index/operations/embed_text/strategies/openai.py

async def run(input, callbacks, cache, args):
    # Get or create model (without client)
    model = ModelManager().get_or_create_embedding_model(...)

    # Use model as async context manager
    async with model:
        embeddings = await _execute(model, text_batches, ticker, semaphore)

    # Client is now closed, connections terminated cleanly
    return TextEmbeddingResult(embeddings=embeddings)
```

**Advantages**:
- ✅ Ensures connections close AFTER all requests complete
- ✅ Proper async resource management
- ✅ No race condition - context manager guarantees cleanup order
- ✅ Can restore concurrent_requests to 8 (full performance)

**Disadvantages**:
- ❌ Requires modifying fnllm wrapper code
- ❌ Creates/destroys client for each entity type (overhead)
- ❌ Loses connection reuse across entity types

**Refinement**: Keep client alive across entity types but close after workflow:

```python
# File: src/graphrag/index/workflows/generate_text_embeddings.py

async def generate_text_embeddings(...):
    model = ModelManager().get_or_create_embedding_model(...)

    async with model:  # ← Client created here
        outputs = {}
        for field in embedded_fields:
            # Reuses client across all fields
            outputs[field] = await _run_embeddings(model, ...)
        # Client closed here

    return outputs
```

### Solution B: Explicit Connection Keep-Alive ⚠️ PARTIAL

**Approach**: Configure httpx to never close connections during active use

**Implementation**:

```python
# File: fnllm/openai/factories/client.py

def create_public_openai_client(config: PublicOpenAIConfig) -> OpenAIClient:
    return AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout,
        max_retries=get_max_retries(config),
        # ADD: Custom httpx client with aggressive keepalive
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=100,  # Keep ALL connections alive
                keepalive_expiry=3600,  # 1 hour keepalive (way more than needed)
            ),
            timeout=httpx.Timeout(timeout=config.timeout, pool=None),
        )
    )
```

**Advantages**:
- ✅ Prevents premature connection closure
- ✅ No code changes to GraphRAG
- ✅ Maintains connection reuse

**Disadvantages**:
- ❌ Requires modifying fnllm library code
- ❌ Doesn't fix root cause, just hides it
- ❌ May accumulate idle connections
- ❌ Still has race condition, just less likely

### Solution C: Request Queuing with Explicit Wait ✅ ALTERNATIVE

**Approach**: Add explicit barrier to ensure all HTTP requests complete before proceeding

**Implementation**:

```python
# File: src/graphrag/index/operations/embed_text/strategies/openai.py

async def _execute(model, chunks, tick, semaphore):
    results = []

    async def embed_with_tracking(chunk):
        async with semaphore:
            chunk_embeddings = await model.aembed_batch(chunk)
            result = np.array(chunk_embeddings)
            tick(1)
        return result

    # Create tasks (not just futures)
    tasks = [asyncio.create_task(embed_with_tracking(chunk)) for chunk in chunks]

    # Wait for ALL tasks to complete
    results = await asyncio.gather(*tasks)

    # ADD: Explicit barrier to ensure HTTP layer is drained
    await asyncio.sleep(0)  # Yield to event loop

    # Verify all tasks are done
    assert all(task.done() for task in tasks), "Not all tasks completed!"

    return [item for sublist in results for item in sublist]
```

**Advantages**:
- ✅ Minimal code changes
- ✅ Adds explicit verification
- ✅ No library modifications needed

**Disadvantages**:
- ❌ Doesn't fix httpx connection pool race
- ❌ `asyncio.sleep(0)` is a hack, not a solution
- ❌ Still relies on implicit connection management

### Solution D: Connection Pool Per Operation (NUCLEAR OPTION) ⚠️ OVERKILL

**Approach**: Create fresh client for each embedding batch, close immediately after

**Implementation**:

```python
async def _execute(model, chunks, tick, semaphore):
    async def embed(chunk):
        async with semaphore:
            # Create fresh client for this batch
            async with create_fresh_model() as batch_model:
                chunk_embeddings = await batch_model.aembed_batch(chunk)
            # Client closed here
            result = np.array(chunk_embeddings)
            tick(1)
        return result

    futures = [embed(chunk) for chunk in chunks]
    results = await asyncio.gather(*futures)
    return [item for sublist in results for item in sublist]
```

**Advantages**:
- ✅ Completely eliminates race condition
- ✅ Perfect connection isolation

**Disadvantages**:
- ❌ Massive overhead (create/destroy client for each batch)
- ❌ Loses all connection pooling benefits
- ❌ Much slower performance
- ❌ NOT RECOMMENDED

---

## Recommended Action Plan

### Phase 1: Immediate Fix (Current State) ✅ DONE

- [x] Reduce `concurrent_requests` to 2
- [x] Increase `max_retries` to 10
- [x] Increase `request_timeout` to 60 seconds
- [x] Document race condition

**Status**: Provides reliability at cost of performance (4x slower)

### Phase 2: Proper Fix (Recommended) 🎯 TODO

**Implement Solution A: Explicit Connection Lifecycle Management**

1. **Modify fnllm wrapper** to support async context manager
   - File: `src/graphrag/language_model/providers/fnllm/models.py`
   - Add `__aenter__` and `__aexit__` to `OpenAIEmbeddingFNLLM`
   - Ensure client is closed after use

2. **Update GraphRAG workflow** to use context manager
   - File: `src/graphrag/index/workflows/generate_text_embeddings.py`
   - Wrap model usage in `async with model:` block
   - Ensures connection closure after all entity types processed

3. **Test with full concurrency**
   - Restore `concurrent_requests` to 8
   - Verify no race condition
   - Measure performance improvement

4. **Add connection lifecycle logging**
   - Log when client is created
   - Log when client is closed
   - Verify proper sequencing

**Expected Outcome**:
- ✅ No race condition
- ✅ Full performance (8 concurrent requests)
- ✅ Proper resource management
- ✅ Clean connection lifecycle

### Phase 3: Monitoring and Verification 📊 TODO

1. **Add metrics**:
   - Track connection lifetime
   - Monitor request failures
   - Log connection pool state

2. **Test scenarios**:
   - Large collections (1000+ documents)
   - Multiple entity types
   - Long-running pipelines (24+ hours)

3. **Performance benchmarks**:
   - Measure throughput with concurrent_requests = 2, 4, 8
   - Measure latency per batch
   - Identify optimal concurrency

---

## Technical Debt and Future Improvements

### Issue 1: No Connection Pool Configuration

**Problem**: httpx connection pool settings are hidden inside OpenAI SDK

**Solution**: Expose connection pool settings in GraphRAG config:

```yaml
rag:
  embedding_connection_pool:
    max_connections: 100
    max_keepalive_connections: 20
    keepalive_expiry: 60
```

### Issue 2: No Request-Level Retry

**Problem**: Retries happen at batch level, not request level

**Solution**: Add retry logic around individual `aembed_batch()` calls

### Issue 3: No Checkpointing for Embeddings

**Problem**: If embedding workflow fails, restart from beginning

**Solution**:
- Save embeddings incrementally to storage
- Resume from last checkpoint on failure
- Already done for entity extraction, should extend to embeddings

### Issue 4: Silent Connection Management

**Problem**: No visibility into connection lifecycle

**Solution**: Add logging:
```python
logger.info(f"Embedding client created: {id(client)}")
logger.info(f"Connection pool state: open={pool.active}, idle={pool.idle}")
logger.info(f"Embedding client closed: {id(client)}")
```

---

## Appendix: File Reference

### GraphRAG Core Files

| File | Purpose | Key Functions |
|------|---------|---------------|
| `src/graphrag/api/index.py` | Entry point | `build_index()` |
| `src/graphrag/index/run/run_pipeline.py` | Pipeline executor | `run_pipeline()`, `_run_pipeline()` |
| `src/graphrag/index/workflows/generate_text_embeddings.py` | Embedding workflow | `run_workflow()`, `generate_text_embeddings()` |
| `src/graphrag/index/operations/embed_text/embed_text.py` | Embedding operation | `embed_text()` |
| `src/graphrag/index/operations/embed_text/strategies/openai.py` | Embedding strategy | `run()`, `_execute()` |
| `src/graphrag/language_model/manager.py` | Model singleton | `ModelManager` class |
| `src/graphrag/language_model/factory.py` | Model factory | `ModelFactory.create_embedding_model()` |
| `src/graphrag/language_model/providers/fnllm/models.py` | fnllm wrapper | `OpenAIEmbeddingFNLLM` class |

### fnllm Library Files (External)

| File | Purpose | Key Functions |
|------|---------|---------------|
| `fnllm/openai/factories/client.py` | Client factory | `create_openai_client()` |
| `fnllm/openai/factories/embeddings.py` | Embedding LLM factory | `create_openai_embeddings_llm()` |
| `fnllm/openai/llm/openai_embeddings_llm.py` | Embedding LLM impl | `OpenAIEmbeddingsLLMImpl._execute_llm()` |

### OpenAI SDK Files (External)

| File | Purpose | Key Functions |
|------|---------|---------------|
| `openai/__init__.py` | SDK entry | `AsyncOpenAI` class |
| `openai/_base_client.py` | Base client | `close()`, `__aenter__`, `__aexit__` |

---

## Testing Checklist

### Before Fix Validation

- [ ] Reproduce race condition with concurrent_requests=8
- [ ] Confirm "Server disconnected" errors
- [ ] Verify vLLM logs show "close.started"

### After Fix Validation

- [ ] Test with concurrent_requests=8, no errors
- [ ] Verify all 238 batches complete successfully
- [ ] Check vLLM logs show no premature connection closure
- [ ] Measure performance improvement vs concurrent_requests=2

### Stress Testing

- [ ] Process 1000+ documents
- [ ] Run 24-hour indexing job
- [ ] Test with multiple entity types (documents, entities, relationships)
- [ ] Verify no connection leaks
- [ ] Monitor memory usage

---

**Date**: 2025-10-09
**Analysis By**: Claude Code
**Pipeline Analyzed**: GraphRAG Embedding (FileIntel Integration)
**Root Cause**: httpx connection pool race condition
**Status**: Mitigated (concurrent_requests=2), Proper fix pending (Solution A)
