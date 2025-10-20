# GraphRAG Source Auditability Analysis

## Problem Statement

**Question:** As GraphRAG cannot directly connect entities and relationships to original source material, how can source truthfulness auditability be implemented?

## Current GraphRAG Data Model

### What GraphRAG Stores

GraphRAG creates several parquet files during indexing:

1. **`entities.parquet`**
   - Extracted entities (people, organizations, concepts)
   - Columns: `id`, `name`, `type`, `description`, `text_unit_ids`, `graph_embedding`
   - **Key field:** `text_unit_ids` - array of source chunk IDs

2. **`relationships.parquet`**
   - Relationships between entities
   - Columns: `source`, `target`, `description`, `weight`, `text_unit_ids`
   - **Key field:** `text_unit_ids` - array of source chunk IDs

3. **`text_units.parquet`** (THE CRUCIAL LINK)
   - The original chunks used for extraction
   - Columns: `id`, `text`, `n_tokens`, `document_ids`, `entity_ids`, `relationship_ids`
   - **Key field:** `id` - maps to `text_unit_ids` in entities/relationships
   - **Key field:** `document_ids` - links back to original documents

4. **`communities.parquet`**
   - Hierarchical entity communities
   - Columns: `id`, `level`, `title`, `entity_ids`, `relationship_ids`

5. **`community_reports.parquet`**
   - LLM-generated summaries of communities
   - Columns: `id`, `community`, `level`, `title`, `summary`, `findings`, `rank`

### Current Limitation

**The problem:** When GraphRAG returns a response with citations like `[Data: Reports (5)]`, it references:
- Community reports (aggregated summaries)
- NOT the original source chunks or documents

This makes it **impossible to verify which specific pages/documents** support a claim.

## GraphRAG Citation Format

Current GraphRAG responses include inline citations:

```markdown
Agile methodologies are central to modern product development [Data: Reports (5)].
Scrum is a popular framework [Data: Entities (1034)].
Several companies use Agile-Stage-Gate [Data: Relationships (2372)].
```

**What these mean:**
- `[Data: Reports (5)]` = Community report #5 was used
- `[Data: Entities (1034)]` = Entity #1034 was referenced
- `[Data: Relationships (2372)]` = Relationship #2372 was used

**What's missing:**
- Which original document/page this came from
- Which specific chunk(s) support this claim
- How to verify the original source

## Solution Options

### Option 1: Add Text Units to Query Response ⭐ (Recommended)

**Implementation:**

1. **During indexing:** Ensure text_units.parquet preserves FileIntel metadata
   ```python
   # data_adapter.py - adapt_documents()
   record = {
       "id": chunk.id,  # FileIntel chunk UUID
       "text": chunk.chunk_text,
       "document_ids": [chunk.document_id],  # Link to original document
       "page_number": chunk.chunk_metadata.get("page_number"),
       "original_filename": chunk.document.original_filename,
   }
   ```

2. **During query:** Load text_units alongside other parquet files
   ```python
   # parquet_loader.py
   files_to_load = {
       "entities": "entities.parquet",
       "communities": "communities.parquet",
       "community_reports": "community_reports.parquet",
       "text_units": "text_units.parquet",  # ADD THIS
       "relationships": "relationships.parquet",  # ADD THIS
   }
   ```

3. **After GraphRAG search:** Map citations back to source chunks
   ```python
   # graphrag_service.py - after global_search()
   def extract_source_chunks(context, text_units_df):
       """Extract original source chunks from GraphRAG context."""
       sources = []

       # Get community reports used
       if "reports" in context:
           for report in context["reports"]:
               # Find entities in this community
               community_entities = get_entities_for_community(report.id)

               # For each entity, get source text units
               for entity in community_entities:
                   text_unit_ids = entity.get("text_unit_ids", [])

                   # Look up text units
                   for unit_id in text_unit_ids:
                       text_unit = text_units_df[text_units_df.id == unit_id]
                       if not text_unit.empty:
                           sources.append({
                               "chunk_id": unit_id,
                               "document": text_unit.original_filename,
                               "page": text_unit.page_number,
                               "text": text_unit.text[:200],  # Preview
                               "entity": entity.name,
                           })

       return deduplicate_sources(sources)
   ```

4. **Display in CLI:**
   ```python
   # graphrag.py - after answer display
   if sources:
       cli_handler.console.print("\n[bold blue]Source Documents:[/bold blue]")
       for source in sources[:10]:
           cli_handler.console.print(
               f"  • {source['document']}, p. {source['page']} "
               f"[dim](Entity: {source['entity']})[/dim]"
           )
   ```

**Pros:**
- ✅ Direct traceability to source documents
- ✅ Page-level citations
- ✅ Preserves GraphRAG's reasoning while adding auditability
- ✅ Relatively simple to implement

**Cons:**
- ❌ May return many source chunks (needs deduplication)
- ❌ Doesn't show *which specific claim* maps to which source

---

### Option 2: Hybrid GraphRAG + Vector RAG

**Approach:** Use GraphRAG for reasoning, then verify with Vector RAG

1. **Run GraphRAG query** to get high-level answer with entity reasoning
2. **Extract key claims** from GraphRAG response
3. **Run Vector RAG queries** for each claim to find supporting chunks
4. **Merge results:** GraphRAG answer + Vector RAG source citations

**Implementation:**
```python
async def auditable_graphrag_query(question, collection_id):
    # Step 1: GraphRAG for comprehensive answer
    graphrag_result = await graphrag_service.global_query(collection_id, question)
    answer = graphrag_result["answer"]

    # Step 2: Extract key claims (simple: split by sentence)
    claims = extract_claims(answer)

    # Step 3: Verify each claim with Vector RAG
    verified_sources = []
    for claim in claims[:5]:  # Top 5 claims
        vector_result = await vector_service.query(claim, collection_id, max_results=2)
        verified_sources.extend(vector_result["sources"])

    # Step 4: Return merged result
    return {
        "answer": answer,
        "sources": deduplicate_sources(verified_sources),
        "metadata": {
            "reasoning_method": "graphrag",
            "verification_method": "vector_rag"
        }
    }
```

**Pros:**
- ✅ Best of both worlds: GraphRAG reasoning + Vector RAG citations
- ✅ High confidence in source accuracy
- ✅ Works with existing Vector RAG citation system

**Cons:**
- ❌ More expensive (multiple LLM calls)
- ❌ Slower response time
- ❌ May cite sources that weren't actually used by GraphRAG

---

### Option 3: Parse GraphRAG Inline Citations

**Approach:** Extract `[Data: Reports (X)]` citations and trace back to sources

1. **Parse inline citations** from GraphRAG response
   ```python
   import re

   def parse_graphrag_citations(response_text):
       # Match: [Data: Reports (5)]
       pattern = r'\[Data: (Reports|Entities|Relationships) \((\d+(?:,\s*\d+)*)\)\]'
       matches = re.findall(pattern, response_text)

       citations = []
       for match in matches:
           citation_type = match[0]  # "Reports", "Entities", "Relationships"
           ids = [int(x.strip()) for x in match[1].split(',')]
           citations.append({"type": citation_type, "ids": ids})

       return citations
   ```

2. **Map citations to source chunks**
   ```python
   def resolve_citations(citations, context_dataframes):
       sources = []

       for citation in citations:
           if citation["type"] == "Reports":
               # Look up community reports
               reports = context_dataframes["community_reports"]
               for report_id in citation["ids"]:
                   report = reports[reports.id == report_id]

                   # Get entities in this community
                   community = get_community(report.community)
                   entity_ids = community.entity_ids

                   # Get text units for these entities
                   entities = context_dataframes["entities"]
                   entity_rows = entities[entities.id.isin(entity_ids)]

                   for _, entity in entity_rows.iterrows():
                       text_unit_ids = entity.text_unit_ids
                       # Look up source chunks
                       sources.extend(get_chunks_by_ids(text_unit_ids))

       return sources
   ```

3. **Display with inline citation links**
   ```python
   # Show answer with clickable citations
   cli_handler.console.print(answer)

   # Show mapped sources
   cli_handler.console.print("\n[bold blue]Source Evidence:[/bold blue]")
   for i, source in enumerate(sources, 1):
       cli_handler.console.print(
           f"  [{i}] {source.document}, p. {source.page} "
           f"[dim]- supports: {source.supporting_claim}[/dim]"
       )
   ```

**Pros:**
- ✅ Preserves GraphRAG's inline citation style
- ✅ Direct mapping from claim to source
- ✅ Minimal performance overhead

**Cons:**
- ❌ Complex implementation (citation parsing + multi-level lookup)
- ❌ Depends on GraphRAG citation format (may change)
- ❌ May not work for all GraphRAG search types (local vs global)

---

### Option 4: Store Provenance During Indexing

**Approach:** Enhance GraphRAG indexing to preserve full provenance chain

1. **Extend entities.parquet with source metadata:**
   ```python
   # During entity extraction, add:
   {
       "id": entity_id,
       "name": "Agile Methodology",
       "description": "...",
       "source_chunks": [
           {"chunk_id": "uuid1", "document": "Cooper2018.pdf", "page": 17},
           {"chunk_id": "uuid2", "document": "Cooper2018.pdf", "page": 18},
       ],
       "extraction_confidence": 0.95
   }
   ```

2. **Store in PostgreSQL alongside GraphRAG index:**
   ```sql
   CREATE TABLE graphrag_entity_sources (
       entity_id TEXT,
       chunk_id UUID REFERENCES document_chunks(id),
       extraction_confidence FLOAT,
       PRIMARY KEY (entity_id, chunk_id)
   );
   ```

3. **Query with full provenance:**
   ```python
   # When GraphRAG uses entity X, look up its sources
   entity_sources = storage.get_entity_sources(entity_id)

   for source in entity_sources:
       chunk = storage.get_chunk(source.chunk_id)
       print(f"{chunk.document.filename}, p. {chunk.page_number}")
   ```

**Pros:**
- ✅ Complete audit trail
- ✅ Stores provenance at indexing time (no runtime overhead)
- ✅ Can track extraction confidence

**Cons:**
- ❌ Requires modifying GraphRAG indexing pipeline
- ❌ Increases storage requirements
- ❌ More complex database schema

---

## Recommended Implementation Plan

### Phase 1: Quick Win (Option 1 - Partial)

**Goal:** Show which source documents were used (not claim-level)

1. Load `text_units.parquet` during query
2. Extract document/page metadata from text units
3. Display unique source documents at bottom of response

**Effort:** 1-2 hours
**Value:** Immediate auditability improvement

### Phase 2: Claim-Level Citations (Option 3)

**Goal:** Map specific claims to specific sources

1. Parse inline citations from GraphRAG response
2. Resolve citations → communities → entities → text units → chunks
3. Display sources inline with claims

**Effort:** 4-6 hours
**Value:** High - enables full auditability

### Phase 3: Hybrid Verification (Option 2)

**Goal:** Verify GraphRAG claims with Vector RAG

1. Extract key claims from GraphRAG answer
2. Query Vector RAG for supporting evidence
3. Show verified vs unverified claims

**Effort:** 2-3 hours (using existing Vector RAG)
**Value:** Highest confidence in accuracy

## Example: Auditable GraphRAG Response

### Before (Current):
```
Answer:

Agile methodologies are central to modern product development [Data: Reports (5)].
```

### After (Phase 2):
```
Answer:

Agile methodologies are central to modern product development [1,2,3].

Sources:
  [1] Cooper, R. (2018). "Agile-Stage-Gate Hybrids", p. 17
  [2] Cooper, R. (2018). "Agile-Stage-Gate Hybrids", p. 18
  [3] Schwaber, K. (2020). "Scrum Guide", p. 5
```

### After (Phase 3 - Hybrid):
```
Answer:

Agile methodologies are central to modern product development [1✓,2✓,3✓].

✓ = Verified by Vector RAG

Sources:
  [1] Cooper, R. (2018). "Agile-Stage-Gate Hybrids", p. 17
      "Agile methodologies have become increasingly central to NPD processes..."
  [2] Cooper, R. (2018). "Agile-Stage-Gate Hybrids", p. 18
      "The adoption of Agile across sectors demonstrates its importance..."
  [3] Schwaber, K. (2020). "Scrum Guide", p. 5
      "Scrum is the most widely used Agile framework for product development..."
```

## Technical Details

### Data Flow for Source Tracing

```
User Query
    ↓
GraphRAG Global Search
    ↓
Community Reports Selected (based on relevance)
    ↓
Communities → Entity IDs
    ↓
Entities → Text Unit IDs
    ↓
Text Units → Chunk IDs (FileIntel UUIDs)
    ↓
Chunks → Document + Page Number
    ↓
Format Citations
    ↓
Display to User
```

### Key Files to Modify

1. **`parquet_loader.py`**
   - Add `text_units` and `relationships` to loaded files

2. **`graphrag_service.py`**
   - Add `extract_source_chunks()` method
   - Modify `global_query()` and `local_query()` to include sources

3. **`data_adapter.py`**
   - Preserve FileIntel metadata (page_number, document name) in text units

4. **`graphrag.py` (CLI)**
   - Display source documents with page numbers

## Conclusion

**Recommendation:** Implement **Option 1** (Phase 1) first for immediate value, then add **Option 3** (Phase 2) for full claim-level auditability.

This provides:
- ✅ Traceable sources (document + page)
- ✅ Claim-level citations
- ✅ Backward compatible with existing GraphRAG
- ✅ Minimal performance impact

**Optional:** Add **Option 2** (Phase 3) for high-stakes queries where verification is critical.
