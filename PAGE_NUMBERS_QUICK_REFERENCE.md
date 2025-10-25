# Quick Reference: Page Number Handling Comparison

## Where Page Numbers Are Handled

### Vector RAG - Complete Chain

| Step | File | Lines | What Happens |
|------|------|-------|--------------|
| 1. Retrieve Chunks | `/home/tuomo/code/fileintel/src/fileintel/storage/vector_search_storage.py` | 94-115 | SQL SELECT includes `c.chunk_metadata` with page_number |
| 2. Package Chunks | `/home/tuomo/code/fileintel/src/fileintel/storage/vector_search_storage.py` | 120-136 | `chunk_data["metadata"] = row.chunk_metadata or {}` includes page_number |
| 3. Format Citations | `/home/tuomo/code/fileintel/src/fileintel/rag/vector_rag/services/vector_rag_service.py` | 116-145 | `format_in_text_citation(chunk)` called for each chunk |
| 4. Extract Page # | `/home/tuomo/code/fileintel/src/fileintel/citation/citation_formatter.py` | 50-88 | `page_number = chunk_metadata.get("page_number")` |
| 5. Format Citation | `/home/tuomo/code/fileintel/src/fileintel/citation/citation_formatter.py` | 70-71 | Returns `(Author, Year, p. {page_number})` |
| 6. Build LLM Context | `/home/tuomo/code/fileintel/src/fileintel/llm_integration/unified_provider.py` | 374-391 | `source_info = format_in_text_citation(chunk)` |
| 7. Instruct LLM | `/home/tuomo/code/fileintel/src/fileintel/llm_integration/unified_provider.py` | 430-439 | Prompt: "When citing sources, include specific page numbers when available..." |
| 8. Return Sources | `/home/tuomo/code/fileintel/src/fileintel/rag/vector_rag/services/vector_rag_service.py` | 129-145 | `in_text_citation` field populated in source_data |

### GraphRAG - Incomplete Chain

| Step | File | Lines | Status |
|------|------|-------|--------|
| 1. Retrieve via GraphRAG | `/home/tuomo/code/fileintel/src/fileintel/rag/graph_rag/services/graphrag_service.py` | 199-209 | GraphRAG's internal search (no direct chunk access) |
| 2. LLM Generates Answer | GraphRAG library | - | Uses GraphRAG's own LLM (no page numbers provided) |
| 3. Return Response | `/home/tuomo/code/fileintel/src/fileintel/rag/graph_rag/services/graphrag_service.py` | 66-72 | `sources = raw_response.get("context", {}).get("data", [])` |
| 4. PROBLEM: Sources Empty | `/home/tuomo/code/fileintel/src/fileintel/rag/graph_rag/services/graphrag_service.py` | 264 | `"sources": []` - EMPTY LIST |
| 5. Source Tracing Exists | `/home/tuomo/code/fileintel/src/fileintel/rag/graph_rag/utils/source_tracer.py` | 14-41, 146-194 | Functions exist but NOT CALLED in query() |
| 6. Page Extraction Ready | `/home/tuomo/code/fileintel/src/fileintel/rag/graph_rag/utils/source_tracer.py` | 267-270, 146-194 | Can extract page_number and format it |
| 7. NOT Returned to User | - | - | Source tracing not integrated into query response |

## Key Code Snippets

### Vector RAG: Getting Page Numbers from Chunks

**File**: `/home/tuomo/code/fileintel/src/fileintel/storage/vector_search_storage.py`, Lines 94-115

```python
query = text(
    f"""
    SELECT
        c.id as chunk_id,
        c.chunk_text,
        c.position,
        c.chunk_metadata,  # <-- Contains page_number
        d.filename,
        d.original_filename,
        d.id as document_id,
        d.document_metadata,
        1 - (c.embedding <=> CAST(:query_embedding AS vector)) as similarity
    FROM document_chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE d.collection_id = :collection_id
```

### Vector RAG: Formatting In-Text Citations with Page Numbers

**File**: `/home/tuomo/code/fileintel/src/fileintel/citation/citation_formatter.py`, Lines 50-88

```python
def format_in_text_citation(self, chunk: Dict[str, Any]) -> str:
    """Format an in-text citation for Harvard style with page number."""
    try:
        document_metadata = chunk.get("document_metadata", {})
        chunk_metadata = chunk.get("metadata", chunk.get("chunk_metadata", {}))
        page_number = chunk_metadata.get("page_number")  # <-- Extract page number
        
        if self._has_citation_metadata(document_metadata):
            author_surname = self._extract_author_surname(document_metadata)
            year = self._extract_year(document_metadata)
            
            # Build citation with page number if available
            if author_surname and year and page_number:
                return f"({author_surname}, {year}, p. {page_number})"  # <-- PAGE INCLUDED
            elif author_surname and year:
                return f"({author_surname}, {year})"
```

### Vector RAG: Building LLM Context with Citations

**File**: `/home/tuomo/code/fileintel/src/fileintel/llm_integration/unified_provider.py`, Lines 374-391

```python
context_parts = []
for i, chunk in enumerate(context_chunks[:8], 1):
    chunk_text = (
        chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
    )
    
    # Use in-text citation format (with page numbers) for context
    if isinstance(chunk, dict):
        try:
            from fileintel.citation import format_in_text_citation
            # Use in-text citation so LLM sees the page-specific format to use
            source_info = format_in_text_citation(chunk)  # <-- Page numbers here!
        except ImportError:
            source_info = chunk.get("original_filename", chunk.get("filename", f"Source {i}"))
    else:
        source_info = f"Source {i}"
    
    context_parts.append(f"[{source_info}]: {chunk_text}")
```

### Vector RAG: LLM Prompt Instruction for Page Numbers

**File**: `/home/tuomo/code/fileintel/src/fileintel/llm_integration/unified_provider.py`, Lines 430-439

```python
return f"""...[RAG prompt]...

Please provide your answer based on the sources above. If the sources don't contain sufficient information to fully answer the question, indicate what information is available and what might be missing.

IMPORTANT: When citing sources, include specific page numbers when available. Use the format (Author, Year, p. X) for page-specific claims. Cite sources to support your key points."""
```

### GraphRAG: Empty Sources in Response

**File**: `/home/tuomo/code/fileintel/src/fileintel/rag/graph_rag/services/graphrag_service.py`, Lines 45-82

```python
async def query(self, query: str, collection_id: str) -> Dict[str, Any]:
    """Standard query interface for the orchestrator."""
    
    raw_response = await self.global_search(query, collection_id)
    
    # Extract response from dict
    if isinstance(raw_response, dict):
        answer = raw_response.get("response", str(raw_response))
        sources = raw_response.get("context", {}).get("data", [])  # <-- Usually empty
    
    return {
        "answer": answer,
        "sources": sources,  # <-- Empty list
        "confidence": confidence,
        "metadata": {"search_type": "global"},
    }
```

### GraphRAG: Source Tracer Ready But Not Used

**File**: `/home/tuomo/code/fileintel/src/fileintel/rag/graph_rag/utils/source_tracer.py`, Lines 267-270

```python
if chunk:
    # Extract metadata
    page_number = None
    if chunk.chunk_metadata:
        page_number = chunk.chunk_metadata.get("page_number")  # <-- Gets page_number
    
    sources.append({
        "chunk_id": str(chunk.id),
        "document": document_name,
        "page_number": page_number,  # <-- Includes page_number
        "text_preview": chunk.chunk_text[:200] if chunk.chunk_text else "",
    })
```

**But this function is NOT called from the GraphRAG query() method!**

## Summary

### Why Vector RAG Works:
1. ✓ Chunks retrieved directly with chunk_metadata.page_number
2. ✓ Citation formatter extracts page_number
3. ✓ LLM sees citations with page numbers in context
4. ✓ LLM is instructed to use page numbers
5. ✓ Sources returned with in_text_citation field

### Why GraphRAG Doesn't:
1. ✗ GraphRAG's internal LLM generates answer (no page context)
2. ✗ Sources list returned empty (line 264)
3. ✗ source_tracer.py exists but not integrated
4. ✗ Page extraction logic available but unused
5. ✗ No post-processing to add page numbers to response

