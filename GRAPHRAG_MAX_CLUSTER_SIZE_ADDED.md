# GraphRAG max_cluster_size Configuration Added

## Summary

Added `max_cluster_size` parameter to control the Leiden algorithm's community detection granularity. This parameter determines how many hierarchical levels are created during indexing.

## Problem

GraphRAG was creating **7 levels** (0-6) with **3 redundant levels**:
- Level 0: 75,637 nodes
- Level 1: 75,132 nodes (-0.7% difference) ← REDUNDANT
- Level 2: 72,545 nodes (-4.1% difference) ← REDUNDANT
- Level 3: 47,053 nodes (-35% reduction) ← First meaningful level!

This caused:
- ~223K unnecessary LLM calls for community summarization
- Hours of wasted processing time
- Degraded query performance

## Solution

Added `max_cluster_size` configuration parameter that controls Leiden algorithm granularity:
- **Lower value (10)** → More granular communities → More levels
- **Higher value (50)** → Coarser communities → Fewer levels

## Changes Made

### 1. Config Parameter (config/default.yaml:109)
```yaml
graphrag:
  max_cluster_size: ${GRAPHRAG_MAX_CLUSTER_SIZE:-50}  # Leiden algorithm max cluster size
```

**Default value: 50** (was 10 in GraphRAG library)

### 2. Import Added (_graphrag_imports.py:15, 53, 71)
```python
from graphrag.config.models.cluster_graph_config import ClusterGraphConfig
```

### 3. Config Adapter Updated (config_adapter.py:13, 229-235, 244)
```python
from .._graphrag_imports import ClusterGraphConfig

# Create cluster_graph config with max_cluster_size
cluster_graph_config = ClusterGraphConfig(
    max_cluster_size=settings.rag.max_cluster_size
)

config = GraphRagConfig(
    ...
    cluster_graph=cluster_graph_config,
)
```

## Usage

### In .env file:
```bash
# Reduce redundant levels (default: 50)
GRAPHRAG_MAX_CLUSTER_SIZE=50

# For even fewer levels (more aggressive merging)
GRAPHRAG_MAX_CLUSTER_SIZE=100

# For more granular levels (original behavior)
GRAPHRAG_MAX_CLUSTER_SIZE=10
```

### Recommended Values:

| max_cluster_size | Expected Levels | Use Case |
|-----------------|-----------------|----------|
| 10 (original)   | 6-7 levels      | Maximum granularity, many redundant levels |
| 50 (new default)| 4-5 levels      | Good balance, fewer redundant levels |
| 100             | 3-4 levels      | Aggressive merging, minimal levels |

## Effect on Current Index

**Your current index** was built with the **old default (10)**. It has 7 levels with redundancy.

To rebuild with the new value:
1. Delete community caches (keep entities/relationships)
2. Set `GRAPHRAG_MAX_CLUSTER_SIZE=50` in `.env`
3. Restart services: `docker compose restart api worker`
4. Resume indexing: `fileintel graphrag index thesis_sources`

Communities will be rebuilt with:
- ✅ Fewer redundant levels (expect 4-5 instead of 7)
- ✅ ~150K fewer LLM calls
- ✅ Hours less processing time
- ✅ Better query performance

## Files Modified

1. `config/default.yaml` - Added max_cluster_size parameter
2. `src/fileintel/rag/graph_rag/_graphrag_imports.py` - Added ClusterGraphConfig import
3. `src/fileintel/rag/graph_rag/adapters/config_adapter.py` - Wired parameter through to GraphRAG config

## Verification

After deploying, check logs for:
```
GRAPHRAG DEBUG: Setting cluster_graph max_cluster_size to 50
```

After rebuilding communities, check logs for:
```
Number of nodes at level=0 => ~75K
Number of nodes at level=1 => ~40K  (should show bigger reduction now)
Number of nodes at level=2 => ~10K
Number of nodes at level=3 => ~1K
```

The gaps between levels should be larger, indicating better hierarchy.

## Notes

- This parameter only affects **community creation** (phase 7), not entity extraction
- Existing entity/relationship data (105K entities, 187K relationships) is NOT affected
- Only community caches need to be deleted and rebuilt
- The parameter is passed to GraphRAG's `hierarchical_leiden` algorithm
