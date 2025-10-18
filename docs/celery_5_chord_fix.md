# Celery 5.x Chord Pattern Fix

## Issue

**Error**: `'AsyncResult' object has no attribute 'apply_async'`

**Root Cause**: In Celery 5.x, the `chord(header)(callback)` pattern **automatically calls `apply_async()` internally** and returns an `AsyncResult` object.

## Celery 5.x Chord Behavior

```python
# From celery/canvas.py in Celery 5.5.3
class chord:
    def __call__(self, body=None, **options):
        return self.apply_async((), {'body': body} if body else {}, **options)
```

When you call `chord(header)(callback)`:
1. `chord.__call__(callback)` is invoked
2. Which internally calls `self.apply_async()`
3. Returns an `AsyncResult` object immediately

## The Bug Pattern

```python
# WRONG (Celery 5.x):
workflow_result = chord(document_signatures)(
    generate_collection_metadata.s(collection_id=collection_id)
).apply_async()  # ← ERROR: AsyncResult has no apply_async()

# CORRECT (Celery 5.x):
workflow_result = chord(document_signatures)(
    generate_collection_metadata.s(collection_id=collection_id)
)  # ← Returns AsyncResult directly
```

## Files Fixed

**File**: `src/fileintel/tasks/workflow_tasks.py`

### Fixed Patterns (8 instances)

1. **Lines 98-102** - `complete_collection_analysis` - Full workflow (metadata + embeddings)
   ```python
   # Before:
   workflow_result = chord(document_signatures)(
       generate_collection_metadata_and_embeddings.s(collection_id=collection_id)
   ).apply_async()

   # After:
   workflow_result = chord(document_signatures)(
       generate_collection_metadata_and_embeddings.s(collection_id=collection_id)
   )
   ```

2. **Lines 116-120** - `complete_collection_analysis` - Metadata only workflow
   ```python
   workflow_result = chord(document_signatures)(
       generate_collection_metadata.s(collection_id=collection_id)
   )
   ```

3. **Lines 134-138** - `complete_collection_analysis` - Embeddings only workflow
   ```python
   workflow_result = chord(document_signatures)(
       generate_collection_embeddings_simple.s(collection_id=collection_id)
   )
   ```

4. **Lines 152-154** - `complete_collection_analysis` - Simple document processing
   ```python
   workflow_result = chord(document_signatures)(completion_callback)
   ```

5. **Line 368** - `generate_collection_embeddings_simple`
   ```python
   workflow_result = chord(embedding_jobs)(completion_callback)
   ```

6. **Line 476** - `incremental_collection_update`
   ```python
   chord_result = chord(new_doc_jobs)(callback)
   ```

7. **Line 580** - `update_collection_index`
   ```python
   workflow_result = chord(task_group)(completion_callback)
   ```

8. **Line 746** - `generate_collection_metadata`
   ```python
   workflow_result = chord(task_group)(completion_callback)
   ```

9. **Line 899** - `generate_collection_metadata_and_embeddings`
   ```python
   workflow_result = chord(task_group)(completion_callback)
   ```

## Migration Guide

If you're migrating from Celery 4.x to Celery 5.x:

### Celery 4.x Pattern
```python
# Celery 4.x: chord returns a signature, must call apply_async()
workflow = chord(tasks)(callback)
result = workflow.apply_async()
```

### Celery 5.x Pattern
```python
# Celery 5.x: chord already calls apply_async() internally
result = chord(tasks)(callback)
```

## Verification

After fixing, all chord patterns should be:

```python
# Pattern 1: Direct assignment
workflow_result = chord(header)(callback)

# Pattern 2: With intermediate variable (if needed for clarity)
workflow = chord(header)
workflow_result = workflow(callback)
```

**NOT**:
```python
# ❌ WRONG - This causes the error
workflow_result = chord(header)(callback).apply_async()

# ❌ WRONG - This also causes the error
workflow = chord(header)(callback)
workflow_result = workflow.apply_async()
```

## Correct `.apply_async()` Usage

Direct task invocations are still correct:
```python
# ✅ CORRECT - calling apply_async on a task signature
task_result = my_task.apply_async(args=[...])

# ✅ CORRECT - calling apply_async on a task with .s()
task_result = my_task.s(arg1, arg2).apply_async()
```

## Testing

After the fix, the workflow should:
1. Successfully create chord workflows
2. Return valid `AsyncResult` objects with `.id` attribute
3. Execute header tasks in parallel
4. Call the callback after all header tasks complete
5. No longer throw `'AsyncResult' object has no attribute 'apply_async'`

## Impact

This fix affects all document processing workflows:
- Complete collection analysis
- Metadata extraction
- Embedding generation
- Incremental updates
- GraphRAG indexing

All workflows now correctly use Celery 5.x chord patterns.
