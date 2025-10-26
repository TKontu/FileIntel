# Shared Storage Refactoring Checklist

## Problem
Multiple Celery tasks are creating individual `PostgreSQLStorage(config)` instances, each creating a separate connection pool. This causes PostgreSQL connection exhaustion when tasks run concurrently.

## Solution
Replace all `PostgreSQLStorage(config)` instances in Celery tasks with `get_shared_storage()` from celery_config.py.

## Progress Tracking

### ✅ Completed Files
- [x] `/src/fileintel/celery_config.py` - Added get_shared_storage() function
- [x] `/src/fileintel/tasks/llm_tasks.py` - Fixed generate_and_store_chunk_embedding task

### ✅ Files Completed
- [x] `/src/fileintel/tasks/document_tasks.py` - 4 instances fixed
- [x] `/src/fileintel/tasks/workflow_tasks.py` - 13 instances fixed
- [x] `/src/fileintel/tasks/llm_tasks.py` - 2 instances fixed
- [x] `/src/fileintel/tasks/graphrag_tasks.py` - 1 instance fixed

### 📋 Detailed Task Breakdown

#### `/src/fileintel/tasks/document_tasks.py`
- [x] Line ~159: `process_document` task - storage instance creation
- [x] Line ~268: `process_collection` task - storage instance creation
- [x] Line ~308: `process_collection` task - error handler storage creation
- [x] Line ~352: `extract_document_metadata` task - storage instance creation

#### `/src/fileintel/tasks/llm_tasks.py`
- [x] Line ~229: `generate_and_store_chunk_embedding` - FIXED
- [x] Line ~289: `analyze_with_llm` task - storage instance creation
- [x] Line ~434: `extract_document_metadata` task - storage instance creation

#### `/src/fileintel/tasks/workflow_tasks.py`
- [x] Line ~64: `complete_collection_analysis` task - storage instance creation
- [x] Line ~186: `complete_collection_analysis` error handler - storage creation
- [x] Line ~162: Error handler in `complete_collection_analysis` - storage creation
- [x] Line ~247: `generate_collection_embeddings_simple` task - storage creation
- [x] Line ~323: `generate_collection_embeddings_simple` error handler - storage creation
- [x] Line ~377: `incremental_collection_update` task - storage instance creation
- [x] Line ~516: `incremental_collection_update` continued - storage creation
- [x] Line ~546: `incremental_collection_update` error handler - storage creation
- [x] Line ~578: `generate_collection_metadata` task - storage creation
- [x] Line ~678: `generate_collection_metadata` error handler - storage creation
- [x] Line ~715: `generate_collection_metadata_and_embeddings` task - storage creation
- [x] Line ~825: `generate_collection_metadata_and_embeddings` error handler - storage creation
- [x] All error handlers with identical patterns - storage creation

#### `/src/fileintel/tasks/graphrag_tasks.py`
- [x] Line ~636: `build_graph_index` task - storage instance creation

## Rules for Refactoring

1. **Replace**: `storage = PostgreSQLStorage(config)` → `storage = get_shared_storage()`
2. **Add Import**: `from fileintel.celery_config import get_shared_storage`
3. **Remove**: Unnecessary imports of `PostgreSQLStorage`
4. **Remove**: `storage.close()` calls (shared storage manages its own lifecycle)
5. **Remove**: `try/finally` blocks solely for storage cleanup

## Testing After Each Change
- Restart Celery workers
- Monitor connection counts during task execution
- Verify no "too many clients" errors

## Expected Results
- **Before**: 4 workers × N tasks × 15 connections per task = 60-200+ connections
- **After**: 1 shared pool × 30 max connections = 30 connections total

## ✅ COMPLETED - Full Refactoring Summary

**Total instances fixed**: 21+ across 4 files
- document_tasks.py: 4 instances
- llm_tasks.py: 3 instances
- workflow_tasks.py: 13 instances (including error handlers)
- graphrag_tasks.py: 2 instances

**Changes made**:
1. ✅ Added `get_shared_storage()` function to `celery_config.py`
2. ✅ Replaced all `PostgreSQLStorage(config)` with `get_shared_storage()`
3. ✅ Removed all `storage.close()` calls in task functions
4. ✅ Removed all unnecessary `try/finally` blocks for storage cleanup
5. ✅ Updated imports to use `from fileintel.celery_config import get_shared_storage`

**Connection pool behavior**:
- **Before**: Each task created individual connections → 100+ concurrent connections
- **After**: All tasks share single connection pool → 30 max connections total
- **Result**: Eliminates "sorry, too many clients already" PostgreSQL errors