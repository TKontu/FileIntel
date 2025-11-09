# Chunk-Based GraphRAG Implementation Plan

**Goal:** Implement a new GraphRAG search mode that uses the highest-level community to select relevant scope, then answers questions using **original document chunks** instead of summaries.

**Date:** 2025-11-09
**Status:** Planning Phase

---

## 1. Executive Summary

### Current Problem
- **Global Search**: Uses only community report summaries (200-500 words)
- **Local Search**: Uses entity/relationship descriptions + community summaries
- **Both**: Never read the original document chunks for answer generation
- **Citations**: Traced retroactively, creating transparency issues

### Proposed Solution: "Deep Search"
A new search mode that:
1. **Uses highest-level community** (level 5 in your case) to identify the most relevant topical scope
2. **Retrieves original chunks** from that community's text units
3. **Answers with real document content** instead of summaries
4. **Provides accurate citations** because the LLM actually reads the cited chunks

### Trade-offs
| Aspect | Current (Summary-based) | Proposed (Chunk-based) |
|--------|------------------------|------------------------|
| **Speed** | 10-100x faster | Slower (more tokens) |
| **Cost** | 10-100x cheaper | Higher (LLM reads chunks) |
| **Accuracy** | Thematic/broad | Precise/detailed |
| **Scalability** | Millions of docs | Thousands of docs |
| **Citations** | Retroactive/misleading | Direct/transparent |
| **Verifiability** | Low | High |

---

## 2. Architecture Overview

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: Community Selection (Fast - uses summaries)           │
├─────────────────────────────────────────────────────────────────┤
│ Query → Embed Query → Similarity Search → Top-1 Level-5 Community│
│         (bge-large)    (community summaries)                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: Chunk Retrieval (Trace community → chunks)            │
├─────────────────────────────────────────────────────────────────┤
│ Community → Entity IDs → Text Unit IDs → Document IDs (chunks) │
│   (L5)       (31 ents)    (~50-200 TUs)    (~50-200 chunks)    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3: Chunk Re-ranking (Semantic relevance to query)        │
├─────────────────────────────────────────────────────────────────┤
│ All Chunks → Embed Chunks → Cosine Similarity → Top-K Chunks   │
│  (~200)       (use cached    (query vs chunks)    (K=10-20)    │
│               embeddings)                                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 4: Answer Generation (LLM reads actual chunks)           │
├─────────────────────────────────────────────────────────────────┤
│ Top-K Chunks → Format Context → LLM Prompt → Answer + Citations│
│  (10-20)        (with metadata)   (single-shot)                │
└─────────────────────────────────────────────────────────────────┘
```

### Key Insight
**Use community hierarchy for scope, but answer with chunks:**
- **Level 5** (highest): Broad topics (31 entities, ~200 chunks per community)
- **Level 0** (lowest): Specific micro-topics (1-5 entities, ~5-20 chunks)

By selecting at Level 5, we get a manageable scope (~200 chunks) that covers a complete topic, then rank those chunks by relevance.

---

## 3. Implementation Details

### 3.1 New Search Mode: `deep_search()`

**Location:** `src/fileintel/rag/graph_rag/services/graphrag_service.py`

**Function Signature:**
```python
async def deep_search(
    self,
    query: str,
    collection_id: str,
    max_chunks: int = 20,
    community_level: int = 5  # Use highest level by default
) -> Dict[str, Any]:
    """
    Perform deep search using highest-level community + original chunks.

    Args:
        query: User question
        collection_id: Collection to search
        max_chunks: Maximum chunks to send to LLM (default: 20)
        community_level: Community hierarchy level to use (default: 5 = highest)

    Returns:
        {
            "response": "Answer based on actual chunks",
            "context": {
                "selected_community": {...},
                "chunks_used": [...],
                "rerank_scores": [...]
            }
        }
    """
```

### 3.2 Phase 1: Community Selection

**Goal:** Find the single most relevant Level-5 community

**Algorithm:**
```python
async def _select_best_community(
    self,
    query: str,
    community_reports_df: pd.DataFrame,
    community_level: int = 5
) -> Dict[str, Any]:
    """
    Select the best community using semantic similarity.

    Steps:
    1. Filter communities by level (e.g., level == 5)
    2. Embed query using bge-large-en
    3. Embed community summaries (or use pre-computed from community_embeddings.npy)
    4. Compute cosine similarity
    5. Return top-1 community with highest similarity
    """
    # Filter to target level
    level_communities = community_reports_df[
        community_reports_df['level'] == community_level
    ]

    # Embed query
    query_embedding = await asyncio.to_thread(
        self.embedding_provider.get_embeddings, [query]
    )

    # Get community embeddings (use cached if available)
    community_summaries = level_communities['summary'].tolist()
    community_embeddings = await asyncio.to_thread(
        self.embedding_provider.get_embeddings, community_summaries
    )

    # Compute similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(query_embedding, community_embeddings)[0]

    # Get top community
    best_idx = np.argmax(similarities)
    best_community = level_communities.iloc[best_idx]

    return {
        "community_id": best_community['community'],
        "title": best_community['title'],
        "summary": best_community['summary'],
        "similarity": float(similarities[best_idx]),
        "entity_count": len(best_community.get('entity_ids', [])),
    }
```

**Optimization:**
- Pre-compute community embeddings and save to `community_embeddings.npy` (I see you already have this!)
- Load cached embeddings instead of re-embedding during query

### 3.3 Phase 2: Chunk Retrieval

**Goal:** Get all chunks related to the selected community

**Algorithm:**
```python
async def _get_community_chunks(
    self,
    community_id: int,
    communities_df: pd.DataFrame,
    entities_df: pd.DataFrame,
    text_units_df: pd.DataFrame,
    workspace_path: str
) -> List[str]:
    """
    Retrieve all chunk UUIDs for a community.

    Steps:
    1. Get entity IDs from community
    2. Get text unit IDs from entities
    3. Get document IDs (chunk UUIDs) from text units
    4. Return unique chunk UUIDs
    """
    # Get community row
    community = communities_df[communities_df['community'] == community_id].iloc[0]
    entity_ids = community['entity_ids']

    # Get text units from entities
    entity_mask = entities_df['id'].isin(entity_ids)
    text_unit_ids = set()
    for tu_list in entities_df[entity_mask]['text_unit_ids']:
        if tu_list is not None:
            text_unit_ids.update(tu_list)

    # Get chunk UUIDs from text units
    tu_mask = text_units_df['id'].isin(text_unit_ids)
    chunk_uuids = set()
    for doc_ids in text_units_df[tu_mask]['document_ids']:
        if doc_ids is not None:
            chunk_uuids.update(doc_ids)

    logger.info(
        f"Community {community_id}: {len(entity_ids)} entities → "
        f"{len(text_unit_ids)} text units → {len(chunk_uuids)} chunks"
    )

    return list(chunk_uuids)
```

**Data Flow:**
```
Community 0 (Level 5)
├─ 31 entities
│  ├─ Entity 1: ["text_unit_1", "text_unit_2"]
│  ├─ Entity 2: ["text_unit_3", "text_unit_4"]
│  └─ ... (31 entities total)
├─ ~50-200 text units (after deduplication)
└─ ~50-200 chunk UUIDs (each text unit → 1-3 chunks)
```

### 3.4 Phase 3: Chunk Re-ranking

**Goal:** Select the most relevant K chunks for the query

**Algorithm:**
```python
async def _rerank_chunks(
    self,
    query: str,
    chunk_uuids: List[str],
    max_chunks: int = 20
) -> List[Dict[str, Any]]:
    """
    Re-rank chunks by semantic similarity to query.

    Steps:
    1. Batch fetch chunks from PostgreSQL
    2. Use PRE-COMPUTED embeddings (chunk.embedding column)
    3. Compute cosine similarity with query
    4. Return top-K chunks with scores
    """
    # Batch fetch chunks from storage
    chunks = await asyncio.to_thread(
        self._batch_fetch_chunks, chunk_uuids
    )

    # Embed query
    query_embedding = await asyncio.to_thread(
        self.embedding_provider.get_embeddings, [query]
    )

    # Get chunk embeddings (use pre-computed from PostgreSQL)
    chunk_embeddings = []
    valid_chunks = []

    for chunk_uuid, chunk in chunks.items():
        if chunk and chunk.embedding is not None:
            chunk_embeddings.append(chunk.embedding)
            valid_chunks.append(chunk)

    # Compute similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]

    # Sort by similarity
    ranked_indices = np.argsort(similarities)[::-1][:max_chunks]

    reranked = []
    for idx in ranked_indices:
        chunk = valid_chunks[idx]
        reranked.append({
            "chunk_id": chunk.chunk_id,
            "chunk_text": chunk.chunk_text,
            "document_id": chunk.document_id,
            "chunk_metadata": chunk.chunk_metadata,
            "similarity": float(similarities[idx])
        })

    logger.info(
        f"Re-ranked {len(chunk_uuids)} chunks → "
        f"Top {len(reranked)} (similarity: {reranked[0]['similarity']:.3f} - {reranked[-1]['similarity']:.3f})"
    )

    return reranked
```

**Key Optimization:**
- Use **pre-computed embeddings** from PostgreSQL `chunk.embedding` column
- No need to re-embed chunks during query (saves time and cost)

### 3.5 Phase 4: Answer Generation

**Goal:** Generate answer using actual chunk content

**Prompt Template:**
```python
DEEP_SEARCH_PROMPT = """You are a helpful research assistant. Answer the question using ONLY the provided document excerpts.

# Question
{query}

# Relevant Document Excerpts
{chunks_context}

# Instructions
1. Answer the question using information from the excerpts above
2. Cite sources using the format: (Document Name, Page X)
3. If the excerpts don't contain enough information, say "The provided excerpts do not contain sufficient information to fully answer this question."
4. Be precise and quote specific facts when possible
5. Do NOT make up information beyond what's in the excerpts

# Answer
"""

def _format_chunks_context(chunks: List[Dict]) -> str:
    """Format chunks for LLM context."""
    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get('chunk_metadata', {})
        doc_name = metadata.get('document_name', 'Unknown')
        pages = metadata.get('pages', [])
        page_str = f"p. {pages[0]}" if pages else "unknown page"

        context_parts.append(
            f"## Excerpt {i} ({doc_name}, {page_str})\n"
            f"{chunk['chunk_text']}\n"
        )

    return "\n".join(context_parts)
```

**LLM Call:**
```python
async def _generate_answer_from_chunks(
    self,
    query: str,
    chunks: List[Dict[str, Any]]
) -> str:
    """Generate answer using LLM with chunk context."""

    chunks_context = self._format_chunks_context(chunks)

    prompt = DEEP_SEARCH_PROMPT.format(
        query=query,
        chunks_context=chunks_context
    )

    # Use unified LLM provider
    response = await asyncio.to_thread(
        self._llm_provider.generate_response,
        prompt=prompt,
        max_tokens=2000,
        temperature=0.1
    )

    return response.content if hasattr(response, 'content') else str(response)
```

---

## 4. Integration Plan

### 4.1 API Changes

**New Endpoint:** Add `deep` to search_type options

**File:** `src/fileintel/api/routes/query.py`

```python
# Line 31: Update search_type description
search_type: Optional[str] = Field(
    default="adaptive",
    description="RAG search type: 'vector', 'graph', 'adaptive', 'global', 'local', or 'deep'"
)

# Line 200: Add deep search routing
elif request.search_type == "deep":
    # Submit deep search task
    task_result = query_graph_deep.delay(
        query=request.question,
        collection_id=collection.id,
        answer_format=request.answer_format
    )
    query_type = "graph_deep"
```

### 4.2 Celery Task

**File:** `src/fileintel/tasks/graphrag_tasks.py`

```python
@app.task(
    base=BaseFileIntelTask,
    bind=True,
    queue="graphrag_queries",
    rate_limit="30/m",  # Slower rate due to heavier processing
    max_retries=3,
)
def query_graph_deep(
    self,
    query: str,
    collection_id: str,
    answer_format: str = "default",
    max_chunks: int = 20,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform deep GraphRAG query using highest-level community + original chunks.

    This is slower and more expensive than global/local search, but provides:
    - More accurate answers (based on actual document content)
    - Better citations (LLM actually reads cited chunks)
    - Higher verifiability (can audit claims against sources)

    Args:
        query: Query string
        collection_id: Collection to query
        answer_format: Answer format template name
        max_chunks: Maximum chunks to use (default: 20)

    Returns:
        Dict containing query results with chunk-based answer
    """
    self.validate_input(
        ["query", "collection_id"], query=query, collection_id=collection_id
    )

    from fileintel.rag.graph_rag.services.graphrag_service import GraphRAGService
    from fileintel.celery_config import get_shared_storage

    storage = get_shared_storage()

    try:
        self.update_progress(0, 4, "Preparing deep GraphRAG query")

        config = get_config()
        graphrag_service = GraphRAGService(storage, config)

        self.update_progress(1, 4, "Executing deep search (community selection + chunk retrieval)")

        import asyncio
        loop = asyncio.get_event_loop()
        future = asyncio.run_coroutine_threadsafe(
            graphrag_service.deep_search(
                query,
                collection_id,
                max_chunks=max_chunks
            ),
            loop
        )
        search_result = future.result()

        self.update_progress(2, 4, "Processing deep search results")

        result = {
            "query": query,
            "collection_id": collection_id,
            "answer": search_result.get("response", ""),
            "sources": search_result.get("context", {}).get("chunks_used", []),
            "search_type": "deep",
            "community_selected": search_result.get("context", {}).get("selected_community", {}),
            "chunks_analyzed": len(search_result.get("context", {}).get("chunks_used", [])),
            "confidence": 0.9,  # Higher confidence due to chunk-based answers
            "status": "completed",
        }

        self.update_progress(4, 4, "Deep GraphRAG query completed")
        return result

    except Exception as e:
        logger.error(f"Error in deep GraphRAG query: {e}")
        return {
            "query": query,
            "collection_id": collection_id,
            "error": str(e),
            "status": "failed",
        }
    finally:
        storage.close()
```

### 4.3 CLI Integration

**File:** `src/fileintel/cli/query.py`

```python
# Line 24: Update help text
rag_type: Optional[str] = typer.Option(
    "auto",
    "--type",
    "-t",
    help="RAG type: 'vector', 'graph', 'global', 'local', 'deep', or 'auto'."
),

# Line 37-44: Update search_type_map
search_type_map = {
    "auto": "adaptive",
    "vector": "vector",
    "graph": "graph",
    "global": "global",
    "local": "local",
    "deep": "deep"  # NEW
}
```

---

## 5. Performance Considerations

### 5.1 Token Usage Estimation

**Scenario:** Question about a Level-5 community (typical case)

```
Community Selection:
- Query embedding: ~10 tokens → 1 embed call
- Community summaries: Already cached (no cost)

Chunk Retrieval:
- ~200 chunks identified
- Embeddings: Pre-computed (no cost)

Re-ranking:
- Query embedding: Already computed (no cost)
- Cosine similarity: Local computation (no cost)

Answer Generation:
- Top 20 chunks @ 500 tokens each = 10,000 input tokens
- System prompt: ~200 tokens
- Answer: ~1,000 output tokens
- Total LLM cost: ~11,000 tokens

Total: ~11,000 tokens per query
```

**Comparison:**
- **Global search**: ~3,000-5,000 tokens (summaries only)
- **Deep search**: ~10,000-15,000 tokens (actual chunks)
- **Cost increase**: 2-5x more expensive

### 5.2 Latency Estimation

```
Community Selection:    ~0.5s  (1 embed call + similarity)
Chunk Retrieval:        ~2.0s  (PostgreSQL batch query for 200 chunks)
Re-ranking:             ~0.5s  (cosine similarity, local)
Answer Generation:      ~5.0s  (LLM call with 11k tokens)
─────────────────────────────
Total:                  ~8.0s
```

**Comparison:**
- **Global search**: ~3-4s (faster due to smaller context)
- **Deep search**: ~8-10s (slower but more accurate)

### 5.3 Optimization Strategies

**1. Cache Community Embeddings**
```python
# Pre-compute during indexing
community_embeddings = embed_batch(community_summaries)
np.save(f"{workspace_path}/community_embeddings.npy", community_embeddings)
```

**2. Batch Database Queries**
```python
# Fetch all chunks in single query instead of 200 individual queries
chunks = storage.get_chunks_by_ids(chunk_uuids)  # Single batch query
```

**3. Parallel Processing**
```python
# Embed query while fetching chunks
query_embed_task = asyncio.create_task(embed_query(query))
chunks_task = asyncio.create_task(fetch_chunks(chunk_uuids))
query_embed, chunks = await asyncio.gather(query_embed_task, chunks_task)
```

**4. Progressive Retrieval**
```python
# If community has > 500 chunks, use two-stage retrieval:
# Stage 1: Rank text units (fewer items)
# Stage 2: Get chunks only from top text units
```

---

## 6. Testing Plan

### 6.1 Unit Tests

**File:** `tests/unit/test_deep_search.py`

```python
import pytest
from fileintel.rag.graph_rag.services.graphrag_service import GraphRAGService

@pytest.mark.asyncio
async def test_select_best_community(graphrag_service, mock_community_reports):
    """Test community selection returns highest similarity."""
    result = await graphrag_service._select_best_community(
        query="What is time management?",
        community_reports_df=mock_community_reports,
        community_level=5
    )

    assert result['community_id'] is not None
    assert result['similarity'] > 0.5
    assert 'title' in result

@pytest.mark.asyncio
async def test_get_community_chunks(graphrag_service, mock_dataframes):
    """Test chunk retrieval from community."""
    chunk_uuids = await graphrag_service._get_community_chunks(
        community_id=0,
        communities_df=mock_dataframes['communities'],
        entities_df=mock_dataframes['entities'],
        text_units_df=mock_dataframes['text_units'],
        workspace_path="/mock/path"
    )

    assert len(chunk_uuids) > 0
    assert all(isinstance(uuid, str) for uuid in chunk_uuids)

@pytest.mark.asyncio
async def test_rerank_chunks(graphrag_service, mock_chunks):
    """Test chunk re-ranking by similarity."""
    reranked = await graphrag_service._rerank_chunks(
        query="What is time?",
        chunk_uuids=[c['chunk_id'] for c in mock_chunks],
        max_chunks=5
    )

    assert len(reranked) == 5
    assert reranked[0]['similarity'] >= reranked[-1]['similarity']
    assert all('chunk_text' in chunk for chunk in reranked)
```

### 6.2 Integration Tests

**File:** `tests/integration/test_deep_search_e2e.py`

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_deep_search_end_to_end(test_collection_id, graphrag_service):
    """Test complete deep search flow."""
    result = await graphrag_service.deep_search(
        query="What are the main concepts related to time?",
        collection_id=test_collection_id,
        max_chunks=20
    )

    assert 'response' in result
    assert len(result['response']) > 100  # Non-trivial answer
    assert 'context' in result
    assert 'selected_community' in result['context']
    assert 'chunks_used' in result['context']
    assert len(result['context']['chunks_used']) <= 20
```

### 6.3 Manual Test Cases

**Test Case 1: Broad Question**
```bash
fileintel query collection thesis_collection \
  "What are the main themes in project management?" \
  --type deep
```

**Expected:**
- Selects a Level-5 community about "Project Management"
- Returns 20 chunks from that community
- Answer cites specific pages and documents
- Answer is detailed (500+ words)

**Test Case 2: Specific Question**
```bash
fileintel query collection thesis_collection \
  "What does W.S. Jevons say about time?" \
  --type deep
```

**Expected:**
- Selects community containing W.S. Jevons
- Returns chunks mentioning Jevons
- Answer quotes specific passages
- Citations point to exact pages

**Test Case 3: Comparison with Global Search**
```bash
# Global (summary-based)
fileintel query collection thesis_collection \
  "What is scope creep?" \
  --type global

# Deep (chunk-based)
fileintel query collection thesis_collection \
  "What is scope creep?" \
  --type deep
```

**Expected Differences:**
- Global: Broader, more abstract answer
- Deep: More detailed, with specific examples and quotes

---

## 7. Rollout Strategy

### Phase 1: Core Implementation (Week 1)
- [ ] Implement `_select_best_community()` in graphrag_service.py
- [ ] Implement `_get_community_chunks()` in graphrag_service.py
- [ ] Implement `_rerank_chunks()` in graphrag_service.py
- [ ] Implement `_generate_answer_from_chunks()` in graphrag_service.py
- [ ] Implement main `deep_search()` method
- [ ] Write unit tests

### Phase 2: Integration (Week 1-2)
- [ ] Add `query_graph_deep` Celery task
- [ ] Update API routes to accept `search_type="deep"`
- [ ] Update CLI to support `--type deep`
- [ ] Write integration tests

### Phase 3: Optimization (Week 2)
- [ ] Pre-compute community embeddings during indexing
- [ ] Implement batch chunk fetching
- [ ] Add caching for frequently selected communities
- [ ] Optimize LLM context formatting

### Phase 4: Testing & Validation (Week 2-3)
- [ ] Run manual test cases
- [ ] Compare accuracy vs global/local search
- [ ] Measure latency and cost
- [ ] Gather user feedback

### Phase 5: Documentation (Week 3)
- [ ] Update API documentation
- [ ] Add user guide for choosing search types
- [ ] Document performance characteristics
- [ ] Create comparison table (vector vs graph vs global vs local vs deep)

---

## 8. Future Enhancements

### 8.1 Multi-Community Deep Search
Instead of selecting just 1 community, select top-3 and merge chunks:
```python
async def deep_search_multi(
    self,
    query: str,
    collection_id: str,
    num_communities: int = 3,
    max_chunks: int = 30
):
    """Select top-N communities and merge their chunks."""
    # Get top 3 communities by similarity
    # Retrieve chunks from each
    # Merge and re-rank all chunks
    # Generate answer
```

**Benefit:** Better coverage for complex questions spanning multiple topics

### 8.2 Adaptive Chunk Count
Dynamically adjust chunk count based on query complexity:
```python
# Simple factual question: 5-10 chunks
# Complex analytical question: 20-30 chunks
# Comparison question: 30-40 chunks

chunk_count = estimate_chunk_count_for_query(query)
```

### 8.3 Hierarchical Deep Search
Start at Level 5, drill down if needed:
```python
# Phase 1: Select Level-5 community (broad scope)
# Phase 2: If answer confidence < 0.7, drill down to Level-4 children
# Phase 3: Select most relevant Level-4 community
# Phase 4: Use its chunks for more focused answer
```

**Benefit:** Progressive detail - start broad, narrow down as needed

### 8.4 Hybrid Mode: Summary + Chunks
Combine the best of both worlds:
```python
# Step 1: Global search (fast, thematic)
# Step 2: Extract key entities from global answer
# Step 3: Deep search (slow, detailed) for those entities
# Step 4: Merge: Use global structure + deep details
```

**Benefit:** Fast thematic overview + detailed verification

---

## 9. Success Metrics

### Accuracy Metrics
- **Citation Precision**: % of citations where LLM actually read the cited chunk
  - Target: 100% (vs current ~0% due to retroactive tracing)
- **Fact Accuracy**: % of specific facts that can be verified in source chunks
  - Target: >95% (vs current ~60-70% with summaries)

### Performance Metrics
- **Latency**: p50, p95, p99 response times
  - Target: <10s (p50), <15s (p95)
- **Cost**: Average tokens per query
  - Target: <15,000 tokens per query
- **Chunk Efficiency**: % of selected chunks used in final answer
  - Target: >50% (avoid retrieving irrelevant chunks)

### User Satisfaction
- **Answer Completeness**: User ratings 1-5
  - Target: >4.0 average
- **Citation Usefulness**: Do users find citations helpful for verification?
  - Target: >80% yes
- **Preference**: Deep vs Global for detail-oriented queries
  - Target: >70% prefer deep for research/analysis tasks

---

## 10. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| **High latency** (>15s) | Medium | Optimize batch queries, cache embeddings, parallel processing |
| **High cost** (>20k tokens/query) | Medium | Implement max_chunks limit, use smaller models for re-ranking |
| **Poor community selection** | High | Validate with manual testing, add fallback to multi-community |
| **Irrelevant chunks** | Medium | Improve re-ranking algorithm, use minimum similarity threshold |
| **Memory issues** (loading 200 chunks) | Low | Stream chunks instead of loading all at once |
| **Storage bottleneck** (200 chunk queries) | Medium | Implement connection pooling, batch queries |

---

## 11. Conclusion

This implementation plan provides a **production-ready approach** to chunk-based GraphRAG that:

1. ✅ Uses **highest-level communities** for intelligent scope selection
2. ✅ Retrieves **original document chunks** instead of summaries
3. ✅ Provides **accurate, verifiable citations** (LLM reads cited content)
4. ✅ Balances **accuracy vs performance** (2-5x cost, 2-3x latency)
5. ✅ Integrates **seamlessly** with existing architecture

**Next Steps:**
1. Review and approve this plan
2. Begin Phase 1 implementation (core methods)
3. Test on your thesis collection with real queries
4. Iterate based on performance metrics

**Estimated Timeline:** 2-3 weeks for full implementation and testing

**Estimated Effort:** ~40-60 hours of development + testing
