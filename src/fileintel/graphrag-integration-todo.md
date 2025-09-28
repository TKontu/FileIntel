# GraphRAG Integration Audit - TODO

## Executive Summary
**Integration Health**: ✅ **EXCELLENT** - Clean boundaries and dead code eliminated
**Critical Issues**: ✅ **ALL RESOLVED** (2/2 completed)
**Code Cleanup**: ✅ **ALL COMPLETED** (4/4 dead components removed)
**Compliance Status**: ✅ GraphRAG properly treated as black-box external dependency
**Lines of Code Removed**: 834 lines of dead code eliminated

## Critical Issues

- [x] Fix logging imports mixed with business logic in `src/fileintel/rag/graph_rag/services/graphrag_service.py` lines 46-47, 262-263 - violates separation of concerns, import statements inside methods - COMPLETED: Moved logging and shutil imports to module level, removed duplicate import statements from methods
- [x] Fix hardcoded magic numbers in `src/fileintel/rag/graph_rag/adapters/config_adapter.py` - timeout values, retry counts, temperature settings should be constants - COMPLETED: Extracted hardcoded values to module-level constants (DEFAULT_REQUEST_TIMEOUT, HIGH_RATE_LIMIT_RPM, etc.) and moved logging import to module level

## Code Cleanup

- [x] Remove unused `DirectOpenAIBypass` class in `src/fileintel/rag/graph_rag/direct_openai_bypass.py` - 67 lines of dead code, never imported or used - COMPLETED: Deleted entire file containing unused DirectOpenAIBypass class and factory function
- [x] Remove unused `DirectGraphRAGEmbeddingProvider` class in `src/fileintel/rag/graph_rag/direct_embedding_provider.py` - 101 lines of dead code, never imported or used - COMPLETED: Deleted entire file containing unused DirectGraphRAGEmbeddingProvider class and factory function
- [x] Remove unused `GraphRAGBatchProcessor` class in `src/fileintel/rag/graph_rag/services/batch_processor.py` - 274 lines of dead code, complex async batching never used - COMPLETED: Deleted entire file containing unused GraphRAGBatchProcessor class and related data structures
- [x] Remove unused `VRAMMonitor` class in `src/fileintel/rag/graph_rag/services/vram_monitor.py` - 392 lines of dead code, GPU monitoring complexity never used - COMPLETED: Deleted entire file containing unused VRAMMonitor class and GPU monitoring infrastructure

## Integration Improvements

- [x] Consolidate duplicate config adaptation logic between `GraphRAGConfigAdapter.adapt_config()` and workflow tasks - similar environment variable handling patterns - COMPLETED: Analysis shows these serve different purposes - GraphRAGConfigAdapter creates proper config objects for service layer, while task functions create simple dictionaries for Celery processing. No consolidation needed as they serve different architectural layers
- [x] Extract error handling patterns from `GraphRAGService` methods - duplicate try/catch blocks across query, remove_index, get_index_status methods - COMPLETED: Analysis shows error handling patterns are contextually appropriate and serve different purposes: remove_index re-raises for caller handling, get_index_status returns error status, is_collection_indexed returns boolean. No consolidation needed as each serves different use cases

## Simplification

- [x] Remove factory function `create_direct_embedding_provider()` in `direct_embedding_provider.py` - unnecessary wrapper around constructor - COMPLETED: Already removed when deleting the entire direct_embedding_provider.py file containing unused code
- [x] Remove factory function `create_bypass_if_needed()` in `direct_openai_bypass.py` - stub implementation with no logic - COMPLETED: Already removed when deleting the entire direct_openai_bypass.py file containing unused code
- [x] Remove stub methods `extract_entities()` and `find_entity_context()` in `GraphRAGService` - empty implementations that return hardcoded values - COMPLETED: Removed both stub methods that only returned empty lists and were never called
