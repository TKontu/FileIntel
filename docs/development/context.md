# GraphRAG Performance Issue - Current Status

## Problem
GraphRAG entity extraction takes 133+ seconds when it should take ~7 seconds. vLLM server responds to individual requests in 1.4 seconds, but GraphRAG perceives massive delays.

## Root Cause Analysis
**Confirmed Fast Components:**
- vLLM server: 1.4 seconds per request (tested with curl)
- vLLM gateway: No container startup delays (already running)
- Database operations: Sub-second timing
- Response processing: 0.00 seconds (BaseModelResponse creation)

**Identified Bottleneck:**
The delay occurs specifically in `self.model()` call within fnllm library. 5 concurrent GraphRAG requests start simultaneously but one completes after 133.17 seconds instead of expected 7 seconds (5 × 1.4s sequential).

## Debugging Implemented
1. **GraphRAG extraction layer**: Timing for document processing and LLM calls
2. **FNLLM wrapper layer**: Timing for achat(), self.model() calls, and response creation
3. **Async/sync conversion**: Timing for run_coroutine_sync threading
4. **Community summarization**: Fixed aggressive rate limiter (1/60s → 10/60s)

## Current Investigation
The 126-second overhead (133 - 7 = 126) suggests a timeout, retry mechanism, or thread synchronization issue in the GraphRAG→fnllm→OpenAI client chain, not the LLM server itself.

## Next Steps
Identify why `self.model()` calls experience 20x delay compared to direct server response times.
