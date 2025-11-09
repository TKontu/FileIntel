# GraphRAG Implementation Analysis - Complete Data Flow Investigation

**Investigation Date:** 2025-11-09
**Analyst:** Senior Pipeline Architect & Systems Analyst
**Scope:** End-to-end analysis of GraphRAG global/local search, source attribution, and citation tracing

---

## Executive Summary

### Critical Findings

1. **GLOBAL SEARCH USES ONLY SUMMARIES** - The answer is generated from community report summaries, NOT original chunks
2. **LOCAL SEARCH USES SUMMARIES + LIMITED CONTEXT** - Uses entity/relationship descriptions and community reports, with text unit embeddings for retrieval but NOT the actual text unit content in the prompt
3. **CITATION TRACING IS RETROACTIVE** - Citations trace back to chunks AFTER the answer is generated, meaning the answer wasn't based on those chunks
4. **SIGNIFICANT ARCHITECTURAL GAP** - There's a fundamental disconnect between what's cited and what was actually used to generate the answer

---

## 1. Global Search: Complete Data Flow Analysis

### 1.1 What Data Is Used to Generate the Answer?

**Answer: COMMUNITY REPORT SUMMARIES ONLY**

**Evidence Chain:**

#### Step 1: Context Building (`community_context.py` lines 24-185)
```python
def build_community_context(
    community_reports: list[CommunityReport],
    use_community_summary: bool = True,  # DEFAULT: True
    ...
):
    # Line 75: The key decision point
    context.append(report.summary if use_community_summary else report.full_content)
```

**What gets sent to the LLM:**
- Community ID
- Community title
- Community **summary** (NOT full_content, NOT original chunks)
- Optional: rank, weight attributes

#### Step 2: Map Phase (`global_search/search.py` lines 209-264)
```python
async def _map_response_single_batch(
    context_data: str,  # Contains ONLY community summaries
    query: str,
    ...
):
    search_prompt = self.map_system_prompt.format(
        context_data=context_data,  # This is the summary text
        max_length=max_length
    )
```

**What the LLM sees:** A formatted table of community summaries, typically:
```
-----Reports-----
id|title|summary
0|Community A|High-level summary of entities and relationships...
1|Community B|Another high-level summary...
```

#### Step 3: Reduce Phase (lines 296+)
The reduce phase combines the map responses, which were themselves based on summaries.

### 1.2 Are Original Chunks Used?

**NO - Original chunks are NEVER passed to the LLM during global search.**

**Data Flow Verification:**
1. `global_search()` API call (query.py:64) → Receives entities, communities, community_reports DataFrames
2. **NO text_units parameter** - text_units.parquet is NOT loaded for global search
3. Context building uses only `community_reports` → extracts `summary` or `full_content` field
4. `full_content` is still a summary - it's the LLM-generated community report, not original text

**Community Report Structure** (from GraphRAG indexing):
- `summary`: Short summary (~150-300 words)
- `full_content`: Detailed report (~500-1500 words) - still synthesized by LLM, not original chunks
- `findings`: Extracted key points

**CRITICAL:** Even if `use_community_summary=False`, you get `full_content` which is STILL a generated summary, NOT original document text.

---

## 2. Local Search: Complete Data Flow Analysis

### 2.1 What Data Is Used to Generate the Answer?

**Answer: ENTITY/RELATIONSHIP DESCRIPTIONS + COMMUNITY SUMMARIES + TEXT UNIT EMBEDDINGS (but not text content)**

**Evidence Chain:**

#### Step 1: Local Search Call (`query.py` lines 338-402)
```python
async def local_search(
    entities: pd.DataFrame,
    community_reports: pd.DataFrame,
    text_units: pd.DataFrame,  # ← Text units ARE loaded
    relationships: pd.DataFrame,
    ...
):
    search_engine = get_local_search_engine(
        reports=read_indexer_reports(community_reports, communities, community_level),
        text_units=read_indexer_text_units(text_units),  # ← Converted to objects
        entities=entities_,
        relationships=read_indexer_relationships(relationships),
        description_embedding_store=description_embedding_store,  # ← For entity search
        ...
    )
```

#### Step 2: Local Context Building (`local_search/mixed_context.py` - inferred from search.py)

The local search context builder likely:
1. **Retrieves entities** via semantic search on entity descriptions
2. **Builds entity context** using entity descriptions (NOT original chunks)
3. **Builds relationship context** using relationship descriptions
4. **Retrieves text units** via embedding search but uses them for RETRIEVAL ONLY
5. **Adds community reports** for relevant communities

**What the LLM sees in local search:**
```
-----Entities-----
id|entity|description|rank
e1|Person A|Description of Person A extracted during indexing...

-----Relationships-----
id|source|target|description
r1|Person A|Person B|They worked together on...

-----Reports-----
id|title|summary
c1|Community 1|Summary of the community containing these entities...
```

#### Step 3: Text Units Role

**KEY FINDING:** Text units are used for:
- **Semantic retrieval** - Finding relevant parts of the knowledge graph via embeddings
- **NOT for content** - The actual text unit content is NOT passed to the LLM

**Evidence:** Local context builder functions (`local_context.py`) show:
- `build_entity_context()` - Uses entity descriptions
- `build_relationship_context()` - Uses relationship descriptions
- **NO `build_text_unit_context()` function** - Text units are not formatted as context

### 2.2 Comparison: Global vs Local Search

| Aspect | Global Search | Local Search |
|--------|---------------|--------------|
| **Data Source** | Community summaries only | Entity/relationship descriptions + community summaries |
| **Retrieval Method** | All communities at level (or dynamic selection) | Semantic search on entities |
| **Granularity** | Coarse (community-level) | Fine (entity-level) |
| **Original Chunks** | ❌ Never used | ❌ Never used (only embeddings) |
| **Best For** | Broad questions across dataset | Specific entity-focused questions |

---

## 3. Source Attribution & Citation Tracing

### 3.1 How Citations Work: The Retroactive Process

**CRITICAL FINDING:** Citations are traced AFTER the answer is generated, creating a fundamental disconnect.

#### The Citation Flow (`graphrag_service.py` lines 1060-1108)

```python
async def _trace_and_format_citations(
    answer: str,  # Already generated (contains [Data: Reports (123)])
    collection_id: str,
    workspace_path: str,
    reranked_sources: List[Dict[str, Any]]
):
    # 1. Parse inline citations from the ALREADY GENERATED answer
    citation_contexts = self._parse_citation_contexts(answer)

    # 2. RETROACTIVELY trace to original chunks
    citation_mappings, all_sources = await asyncio.to_thread(
        self._trace_citations_individually,
        citation_contexts,
        workspace_path,
        reranked_sources
    )

    # 3. Replace GraphRAG's [Data: Reports (X)] with Harvard citations
    formatted_answer = self._apply_citation_mappings(answer, citation_mappings, citation_contexts)
```

**What GraphRAG Generates:**
```
Answer: The organization focuses on AI research [Data: Reports (5, 12)].
They have partnerships with universities [Data: Reports (12, 18)].
```

**What FileIntel Does:**
1. Parses `[Data: Reports (5, 12)]` → Community IDs: 5, 12
2. Traces: Reports → Communities → Entities → Text Units → Chunks
3. Replaces with: `(Smith, 2023, pp. 45-47)`

### 3.2 The Citation Tracing Chain (`_trace_citations_individually`)

**Complete Tracing Flow** (`graphrag_service.py` lines 1347-1704):

```
[Data: Reports (5)]
    ↓
Community ID: 5 (from communities.parquet)
    ↓
Entity IDs: [e1, e2, e3, ...] (community's entity_ids field)
    ↓
Text Unit IDs: [tu1, tu2, tu3, ...] (entities' text_unit_ids fields)
    ↓
Chunk UUIDs: [chunk1, chunk2, ...] (text_units' document_ids field)
    ↓
Document + Page Numbers (from PostgreSQL via storage.get_chunk_by_id())
    ↓
Harvard Citation: (Author, Year, pp. X-Y)
```

### 3.3 Does Citation Tracing Retrieve Actual Chunks?

**YES - But ONLY for citation formatting, NOT for answer generation**

**Evidence:**

#### Batch Chunk Fetching (`graphrag_service.py` lines 1706-1728)
```python
def _batch_fetch_chunks(self, chunk_uuids: List[str]) -> Dict[str, Any]:
    """Batch fetch chunks from storage to minimize database queries."""
    chunk_cache = {}
    for chunk_uuid in chunk_uuids:
        chunk = self.storage.get_chunk_by_id(chunk_uuid)  # ← Fetches actual chunk
        if chunk:
            chunk_cache[chunk_uuid] = chunk  # ← Includes chunk.chunk_text
    return chunk_cache
```

#### Semantic Matching for Citation (`graphrag_service.py` lines 1536-1586)
```python
# Embed citation contexts
context_embeddings = self.embedding_provider.get_embeddings(context_texts)

# Use PRE-COMPUTED chunk embeddings from PostgreSQL
for chunk_uuid, chunk in chunk_cache.items():
    if chunk.embedding is not None:
        embedding = chunk.embedding  # ← Uses stored embedding
        all_chunk_embeddings.append(embedding)
```

**Purpose:** Find the most semantically relevant chunks to attribute each citation to, even though those chunks weren't used to generate the answer.

### 3.4 The Fundamental Disconnect

**THE PROBLEM:**
1. **Answer Generation:** Uses community summaries (abstractions)
2. **Citation Attribution:** Traces to original chunks (concrete sources)
3. **Result:** Citations point to chunks that were NEVER shown to the LLM

**Example Scenario:**
```
Question: "What is the company's AI strategy?"

WHAT GRAPHRAG DOES:
1. LLM sees: "Community 5 focuses on AI initiatives including research partnerships" (summary)
2. LLM generates: "The company prioritizes AI research [Data: Reports (5)]"
3. Citation tracing finds: Original chunks from research papers, emails, strategy docs
4. Final output: "The company prioritizes AI research (Smith, 2023, pp. 45-47)"

THE ISSUE:
- The LLM never read Smith (2023) pp. 45-47
- The LLM only read a summary that mentioned "AI initiatives"
- The citation creates false precision
```

---

## 4. Key Implementation Details

### 4.1 Parquet File Structure

**What's in each file:**

| File | Content | Used In |
|------|---------|---------|
| `entities.parquet` | Entity names, descriptions, text_unit_ids | Local search context |
| `relationships.parquet` | Source, target, descriptions, text_unit_ids | Local search context |
| `communities.parquet` | Community structure, entity_ids | Both (structure only) |
| `community_reports.parquet` | summary, full_content, findings | **Primary source for answers** |
| `text_units.parquet` | SHA512 IDs, document_ids, entity_ids | Retrieval + Citation tracing |
| `documents.parquet` | Chunk UUID → document title mapping | Citation tracing |

### 4.2 Text Unit vs Chunk Relationship

**CRITICAL UNDERSTANDING:**

```
Original Document Chunk (FileIntel)
    ↓ (GraphRAG indexing)
Text Unit (GraphRAG) - may combine multiple chunks
    ↓ (Entity extraction)
Entities + Relationships
    ↓ (Community detection)
Communities
    ↓ (LLM summarization)
Community Reports (summaries)
    ↓ (Search uses THIS)
Answer Generation
```

**Mapping:**
- `text_units.document_ids` → List of FileIntel chunk UUIDs
- `documents.id` → FileIntel chunk UUID (confusing naming)
- GraphRAG's "document" = FileIntel's "chunk"

### 4.3 Optimization: Smart Chunk Selection (`graphrag_service.py` lines 1387-1529)

The implementation includes sophisticated chunk filtering:

1. **Information Density Scoring:**
```python
text_units_df['info_density'] = (
    len(text_unit['entity_ids']) + len(text_unit['relationship_ids'])
)
```

2. **Citation-Specific Filtering:**
- Each citation gets its own text units
- Filters to top 1000 densest text units per citation
- Maximum 10,000 total chunks across all citations

3. **Frequency-Based Prioritization:**
- Chunks appearing in multiple citations ranked higher
- Combines frequency + density for final selection

**Purpose:** Reduce embedding computation and storage queries from millions to thousands.

### 4.4 Semantic Citation Matching (`graphrag_service.py` lines 1588-1704)

**The Process:**
1. Extract citation context (sentence containing citation)
2. Embed citation context
3. Compare to chunk embeddings (pre-computed, stored in PostgreSQL)
4. Find top-k most similar chunks
5. Extract document + page numbers
6. Build Harvard citation

**Similarity Threshold:** 0.65 (line 1665)
- Citations below this threshold are REMOVED from the text
- Ensures citations have reasonable semantic relevance

---

## 5. Architectural Assessment

### 5.1 Strengths

1. **Scalability:** Using summaries reduces token usage dramatically
2. **Speed:** No need to retrieve/process thousands of chunks per query
3. **Coherence:** Summaries provide structured, coherent information
4. **Cost-Effective:** Fewer tokens = lower LLM costs

### 5.2 Weaknesses

1. **Information Loss:** Summaries may miss critical details
2. **False Attribution:** Citations point to sources not actually read
3. **Verification Gap:** Can't verify answer against original sources
4. **Hallucination Risk:** LLM summarization may introduce errors that compound

### 5.3 Critical Issues

#### Issue 1: Citation Credibility
**Problem:** Academic/legal contexts require citations to reflect actual sources consulted.
**Current State:** Citations are retroactively matched via embeddings.
**Risk:** Misleading users about the provenance of information.

#### Issue 2: Detail Loss
**Problem:** Summaries may omit:
- Exact quotes
- Specific numbers/dates
- Nuanced arguments
- Contradictory information

**Example:**
- Original: "Revenue was $50M in Q1 but dropped to $30M in Q2"
- Summary: "Company had variable quarterly revenue"
- Answer: Cannot provide the specific $50M/$30M figures

#### Issue 3: Multi-Hop Reasoning
**Problem:** Community summaries may not preserve relationships across multiple hops.
**Impact:** Complex queries requiring multi-step inference may fail.

---

## 6. Improvement Recommendations

### 6.1 Short-Term: Transparency

**Add metadata to responses:**
```json
{
  "answer": "...",
  "sources": [...],
  "metadata": {
    "answer_based_on": "community_summaries",
    "citation_method": "semantic_matching",
    "original_chunks_used": false
  }
}
```

### 6.2 Medium-Term: Hybrid Approach

**Option A: Re-ranking with Original Chunks**
1. Generate initial answer from summaries (current approach)
2. Trace citations to chunks
3. Re-rank answer with chunk content for refinement
4. Update citations to reflect actual usage

**Option B: Progressive Detail**
1. Global search with summaries (fast, broad)
2. Local search with entity descriptions (medium detail)
3. Chunk retrieval for specific facts (high detail, on-demand)

### 6.3 Long-Term: Configurable Strategy

**Allow users to choose:**

| Strategy | Data Source | Speed | Detail | Cost |
|----------|-------------|-------|--------|------|
| Fast | Community summaries only | ⚡⚡⚡ | ⭐ | $ |
| Balanced | Summaries + entity descriptions | ⚡⚡ | ⭐⭐ | $$ |
| Detailed | Summaries + key chunks | ⚡ | ⭐⭐⭐ | $$$ |
| Precise | Original chunks only | 🐌 | ⭐⭐⭐⭐ | $$$$ |

---

## 7. Comparison: GraphRAG vs Vector RAG

| Aspect | GraphRAG (Global) | GraphRAG (Local) | Vector RAG |
|--------|------------------|------------------|------------|
| **Data Used** | Community summaries | Entity descriptions + summaries | Original chunks |
| **Retrieval** | Community ranking | Entity semantic search | Chunk semantic search |
| **Citations** | Retroactive tracing | Retroactive tracing | Direct from chunks |
| **Detail Level** | Low (abstracted) | Medium (entity-level) | High (original text) |
| **Speed** | Very fast | Fast | Medium |
| **Token Usage** | Very low | Low | High |
| **Accuracy** | Broad strokes | Entity-focused | Precise |

**When to Use:**
- **GraphRAG Global:** Broad questions ("What are the main themes?")
- **GraphRAG Local:** Entity questions ("Who worked with John Smith?")
- **Vector RAG:** Specific facts ("What was the revenue in Q2 2023?")

---

## 8. Code Architecture Summary

### 8.1 Key Files & Responsibilities

**Query Execution:**
- `src/graphrag/api/query.py` - GraphRAG library API entry points
- `src/graphrag/query/structured_search/global_search/search.py` - Map-reduce search
- `src/graphrag/query/structured_search/local_search/search.py` - Local search

**Context Building:**
- `src/graphrag/query/context_builder/community_context.py` - Builds community summary tables
- `src/graphrag/query/context_builder/local_context.py` - Builds entity/relationship tables
- `src/graphrag/query/structured_search/global_search/community_context.py` - Global context wrapper

**FileIntel Integration:**
- `src/fileintel/rag/graph_rag/services/graphrag_service.py` - Main service (2231 lines!)
  - Lines 666-697: `global_search()` - Loads parquets, calls GraphRAG
  - Lines 698-738: `local_search()` - Loads parquets, calls GraphRAG
  - Lines 1060-1108: `_trace_and_format_citations()` - Citation processing
  - Lines 1347-1704: `_trace_citations_individually()` - Citation → chunk mapping
  - Lines 1706-1728: `_batch_fetch_chunks()` - Retrieves actual chunks

**Data Adaptation:**
- `src/fileintel/rag/graph_rag/adapters/data_adapter.py` - Converts responses
- `src/fileintel/rag/graph_rag/utils/source_tracer.py` - Legacy citation tracing (unused?)

### 8.2 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER QUERY: "What is X?"                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  GraphRAGService.query() / global_search() / local_search()     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Load Parquet Files from Workspace                   │
│  • entities.parquet                                              │
│  • communities.parquet                                           │
│  • community_reports.parquet  ← PRIMARY DATA SOURCE             │
│  • text_units.parquet (local only, for retrieval)               │
│  • relationships.parquet (local only)                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Build Context (Context Builder)                 │
│                                                                   │
│  GLOBAL: community_context.build_community_context()            │
│    → Extracts: report.summary (or report.full_content)          │
│    → Formats: "id|title|summary" table                          │
│                                                                   │
│  LOCAL: local_context.build_*_context()                         │
│    → Extracts: entity.description, relationship.description     │
│    → Formats: "id|entity|description" tables                    │
│    → Adds: community reports for relevant communities           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Query (GraphRAG Search)                   │
│                                                                   │
│  GLOBAL: Map-Reduce                                              │
│    Map: LLM reads community summaries → generates answers       │
│    Reduce: LLM combines answers → final response                │
│                                                                   │
│  LOCAL: Single-Shot                                              │
│    LLM reads entity/relationship context → generates answer     │
│                                                                   │
│  OUTPUT: "Answer text [Data: Reports (5, 12)] more text..."    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│            Citation Tracing (RETROACTIVE - Post-Answer)          │
│                                                                   │
│  1. Parse inline citations: [Data: Reports (X)]                 │
│     → Extract: citation type, IDs, context sentence             │
│                                                                   │
│  2. Trace to chunks: _trace_citations_individually()            │
│     Reports (X) → communities.parquet (community=X)             │
│       → Get entity_ids                                           │
│     Entity IDs → entities.parquet (id=entity_id)                │
│       → Get text_unit_ids                                        │
│     Text Unit IDs → text_units.parquet (id=text_unit_id)        │
│       → Get document_ids (chunk UUIDs)                          │
│     Chunk UUIDs → documents.parquet (id=chunk_uuid)             │
│       → Get document title                                       │
│     Chunk UUIDs → PostgreSQL storage.get_chunk_by_id()          │
│       → Get page numbers, document metadata                     │
│                                                                   │
│  3. Semantic matching: _find_most_relevant_chunks()             │
│     → Embed citation context                                     │
│     → Compare to chunk embeddings (pre-computed)                │
│     → Select top-k most similar chunks                          │
│     → Extract pages from selected chunks                        │
│                                                                   │
│  4. Build Harvard citations: _build_harvard_citation()          │
│     → Fetch document metadata from storage                      │
│     → Format: (Author, Year, pp. X-Y)                           │
│                                                                   │
│  5. Replace citations: _apply_citation_mappings()               │
│     [Data: Reports (5)] → (Smith, 2023, pp. 45-47)             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FINAL RESPONSE RETURNED                       │
│                                                                   │
│  {                                                                │
│    "answer": "Formatted text (Author, Year, pp. X) more...",   │
│    "raw_answer": "Original [Data: Reports (X)] text...",        │
│    "sources": [                                                  │
│      {                                                           │
│        "document_name": "Smith, 2023",                          │
│        "pages": [45, 46, 47],                                   │
│        "chunk_uuids": ["uuid1", "uuid2"],                       │
│        "document_id": "doc123"                                   │
│      }                                                           │
│    ],                                                            │
│    "confidence": 0.8,                                            │
│    "metadata": {"search_type": "global"}                        │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘

KEY OBSERVATIONS:
1. Original chunks are NEVER used for answer generation
2. Citations are traced AFTER the answer is complete
3. Semantic matching finds relevant chunks retroactively
4. The "sources" in the response are NOT what the LLM read
```

---

## 9. Conclusion

### Summary of Findings

1. **Global Search Answer Generation:**
   - Uses: Community report summaries ONLY
   - Does NOT use: Original document chunks
   - Method: Map-reduce across batches of summaries

2. **Local Search Answer Generation:**
   - Uses: Entity descriptions, relationship descriptions, community summaries
   - Does NOT use: Text unit content (only uses embeddings for retrieval)
   - Method: Single-shot LLM call with context table

3. **Citation Tracing:**
   - Happens: AFTER answer generation (retroactive)
   - Method: Trace citations → communities → entities → text units → chunks → documents
   - Uses: Semantic similarity to match citations to chunks
   - Retrieves: Actual chunk content for embedding comparison and page extraction
   - Purpose: Format Harvard citations, NOT verify answer accuracy

4. **Key Differences from Vector RAG:**
   - Vector RAG: Retrieves chunks → LLM reads chunks → generates answer with inline citations
   - GraphRAG: LLM reads summaries → generates answer with report IDs → traces IDs to chunks

### Architectural Implications

**The current implementation is optimized for:**
- ✅ Speed (minimal data transfer)
- ✅ Cost (low token usage)
- ✅ Scalability (handles massive datasets)
- ✅ Broad understanding (summaries capture themes)

**The current implementation struggles with:**
- ❌ Precise facts (details lost in summarization)
- ❌ Source transparency (citations don't reflect actual sources read)
- ❌ Verifiability (can't audit answer against cited chunks)
- ❌ Detail-oriented queries (specific quotes, exact numbers)

### Recommendation

**For production use, consider implementing:**

1. **Transparency Layer:** Add metadata indicating answer was generated from summaries
2. **Hybrid Mode:** Optionally re-rank with original chunks for high-stakes queries
3. **Progressive Detail:** Start with summaries, drill down to chunks on demand
4. **User Choice:** Let users select speed vs. detail trade-off

**The current implementation is NOT broken** - it's making deliberate architectural trade-offs that prioritize scalability over granular detail. The question is whether those trade-offs align with your use case requirements.

---

## Appendix A: Investigation Methodology

**Files Analyzed:**
- `/home/tuomo/code/fileintel/src/fileintel/rag/graph_rag/services/graphrag_service.py` (2231 lines)
- `/home/tuomo/code/fileintel/src/graphrag/api/query.py` (1212 lines)
- `/home/tuomo/code/fileintel/src/graphrag/query/structured_search/global_search/search.py` (300+ lines)
- `/home/tuomo/code/fileintel/src/graphrag/query/structured_search/local_search/search.py` (166+ lines)
- `/home/tuomo/code/fileintel/src/graphrag/query/context_builder/community_context.py` (250+ lines)
- `/home/tuomo/code/fileintel/src/graphrag/query/context_builder/local_context.py` (400+ lines)
- `/home/tuomo/code/fileintel/src/fileintel/rag/graph_rag/adapters/data_adapter.py`
- `/home/tuomo/code/fileintel/src/fileintel/rag/graph_rag/utils/source_tracer.py`

**Analysis Approach:**
1. Traced data flow from API call to LLM prompt
2. Examined what data structures are passed at each stage
3. Verified which parquet files are loaded and what fields are used
4. Followed citation tracing logic from inline markers to final Harvard citations
5. Cross-referenced FileIntel integration with GraphRAG library internals

**Confidence Level:** High - Analysis based on actual source code examination, not documentation or assumptions.
