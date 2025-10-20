# GraphRAG Integration with Advanced Chunking - Analysis

## Current State

### Type-Aware Chunking (Currently Active)
**Status:** ✅ Enabled in config (`use_type_aware_chunking: true`)

**How it works:**
1. MinerU extracts structured elements from PDFs
2. Type-aware chunker processes elements based on their semantic type:
   - Headers, prose, tables, images, bullet lists
   - Each gets appropriate chunking strategy
3. **All chunks are stored as "vector" type** (line 826 in document_tasks.py)
4. No separate chunk types for GraphRAG vs Vector RAG

**Key limitation:**
- Type-aware chunking **does NOT create separate graph chunks**
- All chunks get `chunk_type: "vector"` in metadata
- GraphRAG uses the **same chunks** as Vector RAG

### Two-Tier Chunking (Currently Disabled)
**Status:** ❌ Disabled in config (`enable_two_tier_chunking: false`)

**How it would work if enabled:**
1. Creates **two types of chunks**:
   - `vector` chunks: Small, sentence-based (300-400 tokens) for embedding
   - `graph` chunks: Larger (1500+ tokens) for entity/relationship extraction
2. Stores chunk_type in metadata: `{"chunk_type": "vector"}` or `{"chunk_type": "graph"}`
3. Vector RAG uses `vector` chunks
4. GraphRAG uses `graph` chunks

**Integration point:**
```python
# graphrag_tasks.py:683-687
if getattr(config.rag, 'enable_two_tier_chunking', False):
    all_chunks = storage.get_chunks_by_type_for_collection(collection_id, 'graph')
else:
    all_chunks = storage.get_all_chunks_for_collection(collection_id)
```

## Current GraphRAG Behavior

### What GraphRAG Actually Uses

**With current settings (type-aware chunking enabled, two-tier disabled):**

```
Document Processing Flow:
PDF → MinerU → Elements → Type-Aware Chunker → Chunks (all marked as "vector")
                                                    ↓
                                          Vector RAG & GraphRAG both use these
```

**Consequences:**
1. GraphRAG receives type-aware chunks (semantic boundaries preserved)
2. Chunks are **smaller and more granular** than traditional GraphRAG expects
3. GraphRAG chunk size from config is **ignored** (lines 124-127 in graphrag_tasks.py):
   ```python
   "chunks": {
       "size": config.rag.chunking.chunk_size,      # NOT USED
       "overlap": config.rag.chunking.chunk_overlap, # NOT USED
   }
   ```
4. These config values are passed to GraphRAG but don't affect input chunks

### Impact on GraphRAG Performance

**Advantages of type-aware chunks for GraphRAG:**
- ✅ Clean semantic boundaries (no mid-sentence splits)
- ✅ Contextually coherent units
- ✅ Better element type preservation

**Disadvantages:**
- ❌ Chunks may be **too small** for optimal entity extraction
  - GraphRAG traditionally uses 1000-1500 token chunks
  - Type-aware chunks are often 200-400 tokens (optimized for embedding)
- ❌ More chunks = more entity extraction calls = higher cost & longer indexing
- ❌ Relationships may be split across chunk boundaries

## Comparison: Three Chunking Strategies

### 1. Type-Aware Chunking (Current)
```
Chunk Size: Variable (200-800 tokens, depends on semantic boundaries)
Chunk Type: All "vector"
Used By: Both Vector RAG and GraphRAG

Pros:
- Semantic coherence
- Clean element boundaries
- Page numbers preserved
- Better for Vector RAG retrieval

Cons:
- Suboptimal for GraphRAG (chunks too small)
- More chunks = more API calls
- No chunk overlap
```

### 2. Two-Tier Chunking (Disabled)
```
Vector Chunks: 300-400 tokens, 3-sentence overlap
Graph Chunks: 1500+ tokens, 2-chunk overlap
Chunk Types: "vector" and "graph"
Used By: Vector RAG uses vector, GraphRAG uses graph

Pros:
- Optimal chunk size for each use case
- Overlap preserves context
- Fewer GraphRAG API calls
- Better entity/relationship extraction

Cons:
- Stores more data (2x chunks)
- More complex processing
- Requires sentence-level processing
```

### 3. Traditional Chunking (Fallback)
```
Chunk Size: Fixed 800 characters
Chunk Type: All "vector"
Used By: Both Vector RAG and GraphRAG

Pros:
- Simple, predictable
- Backward compatible

Cons:
- Can split mid-sentence
- No semantic awareness
- No page tracking
- Not optimal for either use case
```

## Metadata Structure

### Current Chunk Metadata (Type-Aware)
```json
{
  "page_number": 17,
  "chunk_type": "vector",
  "chunk_strategy": "split_at_paragraph",
  "token_count": 342,
  "content_type": "prose",
  "classification_source": "statistical",
  "start_char": 1234,
  "end_char": 2456
}
```

### Two-Tier Chunk Metadata (If Enabled)
```json
{
  "page_number": 17,
  "chunk_type": "graph",  // or "vector"
  "token_count": 1542,
  "pages": [17, 18],
  "sentence_ids": ["s1", "s2", "s3"],
  "extraction_methods": ["mineru_selfhosted_pipeline_json"]
}
```

## Recommendations

### Option 1: Keep Current Setup (Type-Aware Only)
**When to use:**
- Prioritize Vector RAG quality
- Limited GraphRAG usage
- Budget-conscious (fewer API calls for GraphRAG)

**Trade-offs:**
- GraphRAG may have lower quality entity extraction
- More GraphRAG API calls (more chunks)
- Relationships may be fragmented

### Option 2: Enable Two-Tier Chunking
**When to use:**
- Heavy GraphRAG usage
- Need optimal entity/relationship extraction
- Want proper overlap for context preservation

**Implementation:**
```yaml
# config/default.yaml
rag:
  enable_two_tier_chunking: true
  chunking:
    chunk_size: 800        # For vector chunks (characters)
    chunk_overlap: 80
    target_sentences: 3    # Target for vector chunks
    overlap_sentences: 1   # For vector chunk overlap
```

**What would change:**
1. Document processing creates both vector and graph chunks
2. Vector RAG uses small, overlapping chunks
3. GraphRAG uses larger graph chunks (1500+ tokens)
4. Storage increases (~2x chunks)

### Option 3: Hybrid Approach
**Create graph chunks from type-aware elements:**

Instead of sentence-level two-tier chunking, group type-aware chunks into larger units for GraphRAG:

```python
# Pseudo-code for hybrid approach
def create_graph_chunks_from_type_aware(elements):
    """Group type-aware chunks into larger graph chunks."""
    graph_chunks = []
    current_chunk = []
    current_tokens = 0

    for element in elements:
        element_tokens = element['token_count']

        # Group until reaching target size
        if current_tokens + element_tokens > 1500:
            graph_chunks.append(merge_elements(current_chunk))
            current_chunk = [element]
            current_tokens = element_tokens
        else:
            current_chunk.append(element)
            current_tokens += element_tokens

    return graph_chunks
```

## Current Integration Points

### 1. Document Processing
**File:** `src/fileintel/tasks/document_tasks.py`
- Lines 694-720: Type-aware chunking pathway
- Line 826: Sets `chunk_type = "vector"` for all chunks

### 2. GraphRAG Indexing
**File:** `src/fileintel/tasks/graphrag_tasks.py`
- Lines 680-688: Chunk retrieval logic
- Checks `enable_two_tier_chunking` flag
- Falls back to all chunks if two-tier disabled

### 3. Storage Layer
**File:** `src/fileintel/storage/document_storage.py`
- Lines 395-410: `get_chunks_by_type_for_collection()`
- Filters by `chunk_metadata->>'chunk_type'`

## Questions to Consider

1. **How critical is GraphRAG entity extraction quality?**
   - If very critical → Enable two-tier or implement hybrid
   - If moderate → Current setup may be acceptable

2. **What's your typical document size?**
   - Small docs (< 10 pages) → Type-aware is fine
   - Large docs (> 50 pages) → Two-tier may be better

3. **API budget for GraphRAG indexing?**
   - Limited → Prefer fewer, larger chunks (two-tier or hybrid)
   - Flexible → Current type-aware is acceptable

4. **Storage constraints?**
   - Limited → Stick with type-aware (single chunk set)
   - Flexible → Two-tier acceptable (2x chunks)

## Summary

**Current State:**
- Type-aware chunking creates semantically coherent chunks
- All chunks marked as "vector" type
- GraphRAG uses these same chunks (not ideal size)
- Page numbers preserved in metadata

**GraphRAG Impact:**
- Works, but not optimized for entity extraction
- Chunks smaller than GraphRAG prefers
- More API calls during indexing
- Relationships may span chunk boundaries

**Next Steps:**
- Monitor GraphRAG quality with current chunks
- Consider two-tier if entity extraction quality is insufficient
- Hybrid approach could be implemented if needed
