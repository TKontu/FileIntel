# GraphRAG Analysis - Quick Reference Summary

## TL;DR - The Critical Truth

**GraphRAG answers are based on SUMMARIES, not original chunks. Citations are traced back AFTER the answer is generated.**

---

## What Actually Happens

### Global Search
```
Question → Community Summaries (200-500 words each) → LLM → Answer [Data: Reports (5)]
                                                                    ↓
                                                          Trace Report 5 → Chunks
                                                                    ↓
                                                          Replace with (Smith, 2023, pp. 45)
```

**The LLM NEVER reads Smith (2023) pp. 45** - it only reads a summary about that community.

### Local Search
```
Question → Entity Descriptions + Relationship Summaries → LLM → Answer [Data: Entities (23)]
                                                                          ↓
                                                                Trace Entity 23 → Chunks
                                                                          ↓
                                                                Replace with (Jones, 2024, p. 12)
```

**The LLM NEVER reads Jones (2024) p. 12** - it only reads extracted entity descriptions.

---

## The Data Flow (Simplified)

### What GraphRAG Indexes
1. **Original Chunks** → Entity Extraction → **Entities** (with descriptions)
2. **Entities** → Relationship Extraction → **Relationships** (with descriptions)
3. **Entities** → Community Detection → **Communities**
4. **Communities** → LLM Summarization → **Community Reports** (summaries)

### What GraphRAG Searches

**Global Search Uses:**
- ✅ Community report summaries
- ❌ NOT original chunks
- ❌ NOT entity descriptions
- ❌ NOT text units

**Local Search Uses:**
- ✅ Entity descriptions
- ✅ Relationship descriptions
- ✅ Community report summaries
- ❌ NOT original chunks
- ❌ NOT text unit content (only embeddings for retrieval)

### What Citation Tracing Does (Retroactive)

1. Parse `[Data: Reports (5)]` from generated answer
2. Lookup: Report 5 → Community → Entities → Text Units → Chunk UUIDs
3. Fetch chunks from PostgreSQL
4. Use semantic similarity to find most relevant chunks
5. Extract page numbers and document metadata
6. Build Harvard citation: `(Author, Year, pp. X-Y)`
7. Replace inline marker with citation

**Key Point:** This happens AFTER the answer is complete. The chunks found here were NOT used to generate the answer.

---

## The Architectural Trade-offs

### Why Use Summaries?

**Advantages:**
- 🚀 10-100x faster (less data to process)
- 💰 10-100x cheaper (fewer LLM tokens)
- 📈 Scales to millions of documents
- 🎯 Captures high-level themes well

**Disadvantages:**
- ❌ Loses specific details (exact quotes, numbers)
- ❌ Can't verify citations (didn't read the cited source)
- ❌ Information loss through multi-stage summarization
- ❌ May miss nuance and contradictions

### When to Use GraphRAG vs Vector RAG

| Use Case | Best Method | Why |
|----------|-------------|-----|
| "What are the main themes in the research?" | GraphRAG Global | Needs broad synthesis |
| "Who collaborated with Dr. Smith?" | GraphRAG Local | Entity-relationship focused |
| "What was the exact revenue figure for Q2?" | Vector RAG | Needs precise fact from original text |
| "List all mentions of 'climate change'" | Vector RAG | Needs exact quotes |
| "Summarize all product feedback" | GraphRAG Global | Needs thematic synthesis |
| "Find contradictions about project timeline" | Vector RAG | Needs detailed comparison |

---

## Code References (Key Files)

### Answer Generation
- **Global:** `src/graphrag/query/context_builder/community_context.py:75`
  ```python
  context.append(report.summary if use_community_summary else report.full_content)
  ```
  → This line determines what the LLM sees (summary, not chunks)

- **Local:** `src/graphrag/query/context_builder/local_context.py:30-88`
  ```python
  def build_entity_context(...):
      # Uses entity.description, NOT original text
  ```

### Citation Tracing
- **Main Logic:** `src/fileintel/rag/graph_rag/services/graphrag_service.py:1347-1704`
  ```python
  def _trace_citations_individually(...):
      # Reports → Communities → Entities → Text Units → Chunks
  ```

### Data Loading
- **Global Search:** `src/fileintel/rag/graph_rag/services/graphrag_service.py:666-697`
  - Loads: entities, communities, **community_reports**
  - Does NOT load: text_units (not needed)

- **Local Search:** `src/fileintel/rag/graph_rag/services/graphrag_service.py:698-738`
  - Loads: entities, communities, community_reports, **text_units**, relationships
  - text_units used for: retrieval via embeddings (NOT content)

---

## Quick Verification Commands

### See What Data GraphRAG Actually Uses

```bash
# Check what community_reports.parquet contains
python -c "import pandas as pd; df = pd.read_parquet('/path/to/output/community_reports.parquet'); print(df[['community', 'title', 'summary']].head())"

# Check if summaries are used (vs full_content)
grep -n "use_community_summary" src/graphrag/query/context_builder/community_context.py

# Verify text_units are NOT used in context building
grep -r "build_text_unit_context" src/graphrag/query/context_builder/
# (Should return nothing - this function doesn't exist)
```

### See Citation Tracing Flow

```bash
# Find the citation tracing logic
grep -n "_trace_citations_individually" src/fileintel/rag/graph_rag/services/graphrag_service.py

# See what chunks are fetched
grep -n "_batch_fetch_chunks" src/fileintel/rag/graph_rag/services/graphrag_service.py
```

---

## Recommendations

### For Current Users

1. **Understand the limitations:**
   - Don't expect exact quotes
   - Citations are retroactive, not literal
   - Works best for thematic questions

2. **Use the right tool:**
   - GraphRAG: Broad questions, themes, relationships
   - Vector RAG: Specific facts, quotes, numbers

3. **Validate important facts:**
   - Cross-check specific claims against original sources
   - Use citations as a starting point, not ground truth

### For Developers

1. **Add transparency:**
   ```json
   {
     "answer": "...",
     "metadata": {
       "answer_based_on": "community_summaries",
       "original_chunks_consulted": false,
       "citation_method": "retroactive_semantic_matching"
     }
   }
   ```

2. **Consider hybrid approach:**
   - Phase 1: GraphRAG for broad answer
   - Phase 2: Retrieve cited chunks
   - Phase 3: Refine answer with actual chunk content
   - Result: Speed + accuracy

3. **Implement progressive detail:**
   - Quick answer from summaries (2s)
   - Detailed answer with chunks (10s)
   - Let user choose speed vs. detail

---

## Common Misconceptions

### ❌ Myth 1: "GraphRAG cites the sources it read"
**Reality:** GraphRAG traces citations back to sources AFTER generating the answer. The cited chunks were not necessarily "read" by the LLM.

### ❌ Myth 2: "Local search uses the original text units"
**Reality:** Local search uses text unit EMBEDDINGS for retrieval, but the LLM only sees entity and relationship DESCRIPTIONS, not the original text unit content.

### ❌ Myth 3: "Citations prove the answer is grounded in those sources"
**Reality:** Citations are semantically matched to the citation context using embeddings. A similarity threshold of 0.65 is used - citations below this are removed.

### ❌ Myth 4: "GraphRAG is just like Vector RAG with better organization"
**Reality:** Fundamental architectural difference:
- Vector RAG: LLM reads chunks → answers
- GraphRAG: LLM reads summaries → answers (then traces to chunks for citations)

---

## Impact on Different Use Cases

### ✅ Good Fit for GraphRAG
- Academic literature review (themes and trends)
- Corporate knowledge synthesis (what does the organization think about X?)
- Relationship mapping (who works with whom?)
- Trend analysis (how has sentiment changed over time?)
- Exploratory research (what are the main topics?)

### ⚠️ Marginal Fit for GraphRAG
- Technical documentation (if answers need specific procedures)
- Legal research (may need exact language)
- Financial analysis (if specific numbers are critical)
- Compliance checking (may need verbatim policy text)

### ❌ Poor Fit for GraphRAG
- Quote extraction ("Find all mentions of X")
- Exact fact lookup ("What was the revenue on March 15, 2023?")
- Contract analysis (exact terms matter)
- Code documentation (exact syntax matters)
- Debugging (need exact error messages)

**For these cases, use Vector RAG instead.**

---

## Bottom Line

**GraphRAG is a brilliant architecture for scalable, thematic analysis of large corpora.**

**It achieves this by trading detailed precision for broad synthesis.**

**The key is knowing what you're getting:**
- Fast, broad, thematic answers based on multi-layer summarization
- Retroactive citations that point to semantically relevant chunks
- NOT a system for precise fact retrieval or verbatim quote extraction

**Use it for what it's good at, and combine it with Vector RAG for what it's not.**

---

## Related Documentation

- Full Analysis: `/home/tuomo/code/fileintel/GRAPHRAG_IMPLEMENTATION_ANALYSIS.md`
- GraphRAG Official Docs: https://microsoft.github.io/graphrag/
- FileIntel Service: `src/fileintel/rag/graph_rag/services/graphrag_service.py`
