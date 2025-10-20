# Citation Page Number Implementation Plan

## Executive Summary

**Current State:** Citation infrastructure exists but doesn't include page numbers in in-text citations.
**Goal:** Add page numbers to in-text citations: `(Cooper, 2018, p. 17)` instead of `(Cooper, 2018)`
**Approach:** Minimal modification to existing `CitationFormatter` class - single method change.

## Analysis: Existing Implementation

### What Already Works ✓

1. **CitationFormatter class** (`src/fileintel/citation/citation_formatter.py`)
   - Extracts authors, year from document metadata
   - Formats Harvard-style citations
   - Has three formatting methods: `format_source_reference()`, `format_in_text_citation()`, `format_full_citation()`

2. **Vector RAG integration** (`src/fileintel/rag/vector_rag/services/vector_rag_service.py`)
   - Lines 121-127: Already imports and uses `CitationFormatter`
   - Adds `citation` and `in_text_citation` fields to each source
   - Returns page metadata in chunks: `chunk_metadata` includes `page_number`

3. **Page number tracking**
   - `DocumentChunk.chunk_metadata` includes `page_number` field
   - Vector search returns this metadata in `chunk['metadata']['page_number']`
   - Data is available but not used in citation formatting

### What's Missing ✗

**Single missing piece:** `format_in_text_citation()` doesn't include page number from `chunk['metadata']`

Current output:
```python
format_in_text_citation(chunk)  # Returns: (Cooper, 2018)
```

Needed output:
```python
format_in_text_citation(chunk)  # Returns: (Cooper, 2018, p. 17)
```

## Architecture Review

### Quality Assessment

**Good practices found:**
- ✓ Single responsibility: `CitationFormatter` only formats citations
- ✓ Clear separation: Citation logic isolated from RAG logic
- ✓ Module-level convenience functions for common use cases
- ✓ Graceful degradation: Falls back to filename if metadata missing
- ✓ Already integrated: RAG service uses it consistently

**No issues found:**
- No dead code
- No unnecessary abstractions
- No wrapper-around-wrapper patterns
- No circular dependencies
- Clear naming throughout

**Integration points:**
1. `VectorRAGService._generate_answer()` → `UnifiedLLMProvider.generate_rag_response()`
2. `UnifiedLLMProvider.generate_rag_response()` → Uses `format_source_reference()` for context
3. `VectorRAGService.query()` → Uses both `format_source_reference()` and `format_in_text_citation()` for sources

### No New Components Needed

The architecture is clean and complete. We only need to:
1. Modify one method (`format_in_text_citation`)
2. Update one prompt instruction

## Implementation Plan

### Change 1: Update `format_in_text_citation()` Method

**File:** `src/fileintel/citation/citation_formatter.py`
**Lines:** 50-81
**Action:** Add page number extraction and formatting

**Before:**
```python
def format_in_text_citation(self, chunk: Dict[str, Any]) -> str:
    """Format an in-text citation for Harvard style."""
    try:
        document_metadata = chunk.get("document_metadata", {})

        if self._has_citation_metadata(document_metadata):
            author = self._extract_primary_author(document_metadata)
            year = self._extract_year(document_metadata)
            if author and year:
                return f"({author}, {year})"
            elif author:
                return f"({author})"

        # Fallback to simplified filename
        filename = chunk.get("original_filename", chunk.get("filename", "Unknown"))
        if "." in filename:
            filename = filename.rsplit(".", 1)[0]
        return f"({filename})"
```

**After:**
```python
def format_in_text_citation(self, chunk: Dict[str, Any]) -> str:
    """
    Format an in-text citation for Harvard style with page number.

    Returns:
        Citation string like "(Author, Year, p. 15)" or "(Author, Year)" if no page
    """
    try:
        document_metadata = chunk.get("document_metadata", {})
        chunk_metadata = chunk.get("metadata", chunk.get("chunk_metadata", {}))
        page_number = chunk_metadata.get("page_number")

        if self._has_citation_metadata(document_metadata):
            author = self._extract_primary_author(document_metadata)
            year = self._extract_year(document_metadata)

            # Build citation with page number if available
            if author and year and page_number:
                return f"({author}, {year}, p. {page_number})"
            elif author and year:
                return f"({author}, {year})"
            elif author:
                return f"({author})"

        # Fallback to simplified filename
        filename = chunk.get("original_filename", chunk.get("filename", "Unknown"))
        if "." in filename:
            filename = filename.rsplit(".", 1)[0]
        return f"({filename})"
```

**Changes:**
- Added 3 lines to extract page_number from chunk metadata
- Modified 1 line to include page number in citation format
- Added docstring clarification
- Total: 5-line change

### Change 2: Update RAG Prompt Instructions

**File:** `src/fileintel/llm_integration/unified_provider.py`
**Method:** `_build_rag_prompt()` (line 404-436)
**Action:** Update instruction to use page-specific citations

**Before (line 436):**
```python
Please provide your answer based on the sources above. If the sources don't contain sufficient information to fully answer the question, indicate what information is available and what might be missing. Always cite which sources support your key points.
```

**After:**
```python
Please provide your answer based on the sources above. If the sources don't contain sufficient information to fully answer the question, indicate what information is available and what might be missing.

IMPORTANT: When citing sources, include specific page numbers when available. Use the format (Author, Year, p. X) for page-specific claims. Cite sources to support your key points.
```

**Changes:**
- Modified final instruction paragraph
- Total: 2-line change

## Testing Strategy

### Unit Tests

**File:** `tests/unit/test_citation_formatter.py` (create if doesn't exist)

```python
def test_format_in_text_citation_with_page():
    """Test in-text citation includes page number."""
    chunk = {
        "document_metadata": {
            "authors": ["Robert Cooper"],
            "publication_date": "2018-03-07",
            "llm_extracted": True
        },
        "metadata": {
            "page_number": 17
        }
    }

    formatter = CitationFormatter()
    result = formatter.format_in_text_citation(chunk)

    assert result == "(Cooper, 2018, p. 17)"

def test_format_in_text_citation_without_page():
    """Test in-text citation works without page number."""
    chunk = {
        "document_metadata": {
            "authors": ["Robert Cooper"],
            "publication_date": "2018-03-07",
            "llm_extracted": True
        },
        "metadata": {}  # No page number
    }

    formatter = CitationFormatter()
    result = formatter.format_in_text_citation(chunk)

    assert result == "(Cooper, 2018)"  # Graceful degradation
```

### Integration Test

**Verify end-to-end:**
1. Run vector RAG query
2. Check that returned sources have `in_text_citation` field with page numbers
3. Verify LLM response uses page-specific citations

## Deployment Checklist

1. ✓ No database changes needed (page_number already in metadata)
2. ✓ No migration needed (data already exists)
3. ✓ Backward compatible (works with or without page numbers)
4. ✓ No API contract changes (fields already exist)
5. ✓ No configuration changes needed

## Risk Assessment

### Risks: NONE

**Why this is safe:**
- Only modifying citation formatting logic
- Graceful degradation: works without page numbers
- No changes to data storage or retrieval
- No new dependencies
- No performance impact
- Existing tests should still pass

### Rollback Plan

If issues arise, revert the 5-line change to `format_in_text_citation()`. That's it.

## Summary

**Total changes required:**
- 1 method modification (5 lines)
- 1 prompt update (2 lines)
- 2 unit tests (new)

**Total files touched:** 2
**New files created:** 0
**New dependencies:** 0
**New abstractions:** 0

**Justification:** This change directly solves the stated problem with minimal code changes to existing, well-architected components.

## Implementation Steps

1. Modify `format_in_text_citation()` method
2. Update `_build_rag_prompt()` instruction
3. Add unit tests
4. Test with actual RAG query
5. Done

**Estimated time:** 15 minutes
