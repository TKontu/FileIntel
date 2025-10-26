# Investigation: Page Numbers in Vector RAG vs GraphRAG Query Results

## Executive Summary

Page numbers are **correctly extracted and passed** in Vector RAG query results, but the issue is that:

1. **Vector RAG**: Page numbers ARE included in chunk metadata and in-text citations
2. **GraphRAG**: Page numbers are extracted through source tracing but NOT included in the LLM-generated answer text
3. **Root Cause**: The difference in HOW page numbers are presented to users:
   - Vector RAG: Formats citations with page numbers in the **sources list** alongside answers
   - GraphRAG: Returns GraphRAG's answer first, then would need separate source tracing to get page numbers

## Detailed Findings

### 1. Vector RAG Query Implementation

**File**: `/home/tuomo/code/fileintel/src/fileintel/rag/vector_rag/services/vector_rag_service.py`

#### How Vector RAG Retrieves Chunks with Page Numbers

**Lines 96-102** - Chunk retrieval with metadata:
```python
similar_chunks = self.storage.find_relevant_chunks_in_collection(
    collection_id=collection_id,
    query_embedding=query_embedding,
    limit=top_k,
    similarity_threshold=min_similarity,
)
```

**Storage Implementation** - `/home/tuomo/code/fileintel/src/fileintel/storage/vector_search_storage.py`

**Lines 94-115** - SQL Query retrieves chunk_metadata (which includes page_number):
```python
query = text(
    f"""
    SELECT
        c.id as chunk_id,
        c.chunk_text,
        c.position,
        c.chunk_metadata,  # <-- INCLUDES page_number
        d.filename,
        d.original_filename,
        d.id as document_id,
        d.document_metadata,
        1 - (c.embedding <=> CAST(:query_embedding AS vector)) as similarity
    FROM document_chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE d.collection_id = :collection_id
        AND c.embedding IS NOT NULL
        AND 1 - (c.embedding <=> CAST(:query_embedding AS vector)) >= :similarity_threshold
    ORDER BY c.embedding <=> CAST(:query_embedding AS vector)
    LIMIT :limit
    """
)
```

**Lines 120-136** - Data structure includes chunk_metadata:
```python
chunk_data = {
    "chunk_id": row.chunk_id,
    "text": self.base._clean_text(row.chunk_text),
    "metadata": row.chunk_metadata or {},  # <-- page_number is here
    "filename": row.filename,
    "original_filename": row.original_filename,
    "document_id": row.document_id,
    "document_metadata": row.document_metadata or {},
    "similarity": float(row.similarity),
}
```

#### How Vector RAG Formats Citations with Page Numbers

**Vector RAG Service** - Lines 116-145:
```python
# Format sources using enhanced citation formatting
sources = []
for chunk in similar_chunks:
    from fileintel.citation import format_source_reference, format_in_text_citation
    citation = format_source_reference(chunk)
    in_text_citation = format_in_text_citation(chunk)
    
    source_data = {
        "document_id": chunk["document_id"],
        "chunk_id": chunk["chunk_id"],
        "filename": chunk["original_filename"],
        "citation": citation,
        "in_text_citation": in_text_citation,  # <-- Page-aware citation
        "chunk_metadata": chunk.get("metadata", chunk.get("chunk_metadata", {})),
    }
    sources.append(source_data)
```

#### Citation Generation with Page Numbers

**File**: `/home/tuomo/code/fileintel/src/fileintel/citation/citation_formatter.py`

**Lines 50-88** - format_in_text_citation method:
```python
def format_in_text_citation(self, chunk: Dict[str, Any]) -> str:
    """Format in-text citation with page number."""
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

#### LLM Context Building with In-Text Citations

**File**: `/home/tuomo/code/fileintel/src/fileintel/llm_integration/unified_provider.py`

**Lines 349-403** - generate_rag_response method:
```python
def generate_rag_response(self, query: str, context_chunks: list, ...):
    context_parts = []
    for i, chunk in enumerate(context_chunks[:8], 1):
        # Use in-text citation format (with page numbers) for context
        if isinstance(chunk, dict):
            try:
                from fileintel.citation import format_in_text_citation
                # Use in-text citation so LLM sees the page-specific format to use
                source_info = format_in_text_citation(chunk)  # <-- Page number included here!
            except ImportError:
                source_info = chunk.get("original_filename", ...)
        
        context_parts.append(f"[{source_info}]: {chunk_text}")
```

**Lines 430-439** - RAG Prompt explicitly asks for page numbers:
```python
return f"""{base_instruction} {specific_instruction}

Question: {query}

Retrieved Sources:
{context}

Please provide your answer based on the sources above. If the sources don't contain sufficient information to fully answer the question, indicate what information is available and what might be missing.

IMPORTANT: When citing sources, include specific page numbers when available. Use the format (Author, Year, p. X) for page-specific claims. Cite sources to support your key points."""
```

**CRITICAL**: The LLM is EXPLICITLY instructed (line 439) to use page numbers in the format "(Author, Year, p. X)"

### 2. GraphRAG Query Implementation

**File**: `/home/tuomo/code/fileintel/src/fileintel/rag/graph_rag/services/graphrag_service.py`

#### GraphRAG Query Flow

**Lines 45-82** - query() method:
```python
async def query(self, query: str, collection_id: str) -> Dict[str, Any]:
    raw_response = await self.global_search(query, collection_id)
    
    # Extract response from dict
    if isinstance(raw_response, dict):
        answer = raw_response.get("response", str(raw_response))
        sources = raw_response.get("context", {}).get("data", [])  # <-- Sources extracted here
    
    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
    }
```

**Lines 183-209** - global_search method:
```python
async def global_search(self, query: str, collection_id: str):
    # ... load parquet files ...
    
    result, context = await global_search(  # <-- GraphRAG's internal search
        config=graphrag_config,
        entities=dataframes["entities"],
        communities=dataframes["communities"],
        community_reports=dataframes["community_reports"],
        ...
    )
    return self.data_adapter.convert_response(result, context)
```

#### The Problem: GraphRAG's Response Structure

**Lines 250-270** - global_query wrapper:
```python
async def global_query(self, collection_id: str, query: str):
    raw_response = await self.global_search(query, collection_id)
    
    if isinstance(raw_response, dict):
        answer = raw_response.get("response", str(raw_response))
        context = raw_response.get("context", {})
        
        # Extract text_unit IDs for source tracing
        text_unit_ids = self._extract_text_unit_ids_from_context(context)
        
        return {
            "answer": answer,
            "sources": [],  # <-- SOURCES ARE EMPTY!
            "context": {},  # <-- Can't serialize DataFrames
            "metadata": {
                "search_type": "global",
                "text_unit_ids": list(text_unit_ids)
            }
        }
```

**KEY ISSUE**: GraphRAG returns an empty sources list. The answer is generated by GraphRAG's LLM, not by FileIntel's unified provider.

### 3. Source Tracing in GraphRAG

**File**: `/home/tuomo/code/fileintel/src/fileintel/rag/graph_rag/utils/source_tracer.py`

This file HAS functionality to extract page numbers, but it's NOT INTEGRATED into the query response:

**Lines 146-194** - _format_page_numbers():
```python
def _format_page_numbers(pages: set) -> str:
    """Format page numbers according to citation style."""
    if not pages:
        return None
    
    sorted_pages = sorted(pages)
    
    if len(sorted_pages) == 1:
        return f"p. {sorted_pages[0]}"
    
    # Check if pages are consecutive
    ranges = []
    # ... logic to format ranges ...
    
    return f"pp. {', '.join(ranges)}"
```

**Lines 219-288** - lookup_chunks():
```python
def lookup_chunks(text_unit_ids: Set[str], workspace_path: str, storage):
    """Look up FileIntel chunks by text unit IDs."""
    
    for chunk_uuid in chunk_uuids:
        chunk = storage.get_chunk_by_id(str(chunk_uuid))
        
        if chunk:
            # Extract metadata
            page_number = None
            if chunk.chunk_metadata:
                page_number = chunk.chunk_metadata.get("page_number")  # <-- Gets page_number
            
            sources.append({
                "chunk_id": str(chunk.id),
                "document": document_name,
                "page_number": page_number,  # <-- Includes page_number
                "text_preview": chunk.chunk_text[:200],
            })
```

**CRITICAL**: The source_tracer.py module CAN extract page numbers from chunks, but this functionality is NOT called from the GraphRAG query flow.

### 4. Comparison: Vector RAG vs GraphRAG Citation Handling

| Aspect | Vector RAG | GraphRAG |
|--------|-----------|----------|
| **Page Number Source** | chunk_metadata.page_number | chunk_metadata.page_number |
| **Where Extracted** | In vector_rag_service.py (lines 116-145) | In source_tracer.py (NOT integrated) |
| **Citation Format** | format_in_text_citation() | N/A (not used) |
| **LLM Prompt** | Includes page numbers in context + explicit instruction | N/A (GraphRAG's own LLM) |
| **Sources in Response** | Populated with citations | Empty list |
| **Page Number Presentation** | In in_text_citation field of sources | Not included |
| **Answer Generation** | FileIntel's LLM (unified_provider) | GraphRAG's internal LLM |

### 5. The Core Difference in Architecture

#### Vector RAG Flow:
1. Retrieve chunks → chunks have chunk_metadata.page_number
2. Format in-text citations with page numbers
3. Build LLM context: "[Citation with page]: chunk text"
4. LLM generates answer while seeing page numbers
5. Return answer + sources (with page numbers in citations)

#### GraphRAG Flow:
1. Retrieve text units via knowledge graph
2. GraphRAG's internal LLM generates answer (page numbers NOT provided to it)
3. Extract text_unit_ids from GraphRAG's response
4. Return answer + empty sources list
5. Page numbers are extractable via source_tracer.py but NOT integrated

### 6. Citation Prompt Template

**File**: `/home/tuomo/code/fileintel/prompts/templates/citation_generation/harvard_style.md`

Shows expected format including page numbers:
```
- In-text: single page: (Smith 2013, p.45)
- In-text: consecutive pages: (Smith 2013, pp.16-17)
- In-text: non-consecutive: (Taylor 2015, pp.30,35)
```

This template is used for citation generation but NOT currently used in the main query flow.

## Root Cause Analysis

**Vector RAG Page Numbers WORK Because:**
1. Chunks are retrieved with chunk_metadata containing page_number ✓
2. Citation formatter extracts page_number from chunk_metadata ✓
3. In-text citations are formatted with page numbers ✓
4. LLM context includes formatted citations with page numbers ✓
5. LLM is explicitly instructed to use page numbers in responses ✓
6. Sources list includes in_text_citation field with page numbers ✓

**GraphRAG Page Numbers DON'T Appear Because:**
1. Chunks are never directly retrieved in query flow ✗
2. GraphRAG's internal LLM generates answer without page number context ✗
3. Sources list is empty (page numbers not extracted/formatted) ✗
4. source_tracer.py has functionality but is NOT called during query ✗
5. Page numbers are only extractable post-hoc, not integrated into response ✗

## Recommendations

To fix GraphRAG page number presentation:

### Option 1: Integrate Source Tracing into GraphRAG Query
**File**: `/home/tuomo/code/fileintel/src/fileintel/rag/graph_rag/services/graphrag_service.py`

Modify query() method (lines 45-82) to:
1. Extract text_unit_ids from context
2. Call source_tracer.extract_source_chunks()
3. Format page numbers using _format_page_numbers()
4. Populate sources list with formatted sources

### Option 2: Provide Page Numbers to GraphRAG's LLM
Add page number context to GraphRAG's index/query process, but this requires modifying GraphRAG's protected library code.

### Option 3: Post-Process GraphRAG Answers
Extract page numbers from answer text and map them back to sources using source_tracer.

## Files Summary

### Vector RAG (Works Correctly)
- `/home/tuomo/code/fileintel/src/fileintel/rag/vector_rag/services/vector_rag_service.py` (lines 96-145)
- `/home/tuomo/code/fileintel/src/fileintel/storage/vector_search_storage.py` (lines 94-136)
- `/home/tuomo/code/fileintel/src/fileintel/citation/citation_formatter.py` (lines 50-88)
- `/home/tuomo/code/fileintel/src/fileintel/llm_integration/unified_provider.py` (lines 349-439)

### GraphRAG (Needs Integration)
- `/home/tuomo/code/fileintel/src/fileintel/rag/graph_rag/services/graphrag_service.py` (lines 45-82, 183-209)
- `/home/tuomo/code/fileintel/src/fileintel/rag/graph_rag/utils/source_tracer.py` (lines 14-41, 146-194)

