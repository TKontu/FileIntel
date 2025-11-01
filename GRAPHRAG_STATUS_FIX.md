# GraphRAG Status Tracking Fix

## Problem
The GraphRAG indexing status is not properly tracked, causing these issues:
1. Database shows `"indexed"` even when indexing is only 21% complete
2. Resume is blocked because API thinks index is complete
3. Status field uses wrong values (`"indexed"` vs schema's `building/ready/failed`)

## Database Schema (Correct)
```sql
index_status VARCHAR NOT NULL DEFAULT 'building'
-- Values: building, ready, failed, updating
```

## Current Bugs

### Bug 1: graphrag_service.py:459
**Current (Wrong)**:
```python
return {
    "status": "indexed",  # ← Hardcoded, ignores DB status!
    ...
}
```

**Should Be**:
```python
return {
    "status": index_info.get("index_status", "unknown"),  # ← Use actual DB value
    ...
}
```

### Bug 2: graphrag_v2.py:70
**Current (Wrong)**:
```python
if status.get("status") == "indexed":  # ← Wrong value, should be "ready"
```

**Should Be**:
```python
if status.get("status") in ["ready", "indexed"]:  # ← Match schema OR keep compatibility
```

### Bug 3: Missing Status Updates

**Need to add**:
1. Set status to `"building"` when indexing starts
2. Set status to `"ready"` when indexing completes
3. Set status to `"failed"` on error

## Files to Fix

1. `src/fileintel/rag/graph_rag/services/graphrag_service.py:459`
   - Use `index_info.get("index_status")` instead of hardcoded "indexed"

2. `src/fileintel/api/routes/graphrag_v2.py:70, 208, 304, 410`
   - Change `== "indexed"` to `== "ready"` OR `in ["ready", "indexed"]`

3. `src/fileintel/tasks/graphrag_tasks.py:~660, ~727`
   - Update status to "building" at start
   - Update status to "ready" at completion
   - Update status to "failed" on error

4. `src/fileintel/storage/postgresql_storage.py`
   - Add `update_graphrag_index_status(collection_id, status)` method

## Immediate Manual Fix

Run this script to fix the current stuck status:

```bash
chmod +x fix_graphrag_status.sh
./fix_graphrag_status.sh
```

Or manually:
```sql
UPDATE graphrag_indices
SET index_status = 'building'
WHERE collection_id = '6525aacb-55b1-4a88-aaaa-a4211d03beba';
```

## Implementation Priority

1. **IMMEDIATE**: Manual DB fix (above)
2. **HIGH**: Fix graphrag_service.py to return actual status
3. **HIGH**: Fix API to check for "ready" instead of "indexed"
4. **MEDIUM**: Add status updates in task lifecycle
