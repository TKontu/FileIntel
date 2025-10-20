# Robust Chunking System Implementation Plan

## Executive Summary

**Problem**: Current chunking system has **data loss** - validation layer silently drops chunks that exceed token limits. Evidence from logs shows chunks being dropped with messages like "Dropping oversized chunk | chunk_index=3186 | Token count: 20315/450". Additionally, 13 oversized chunks (451-553 tokens) are stored in the database.

**Root Causes**:
- Severely corrupt PDF extractions (backwards text, artifacts, no word spacing)
- Book indexes and statistical reference tables not filtered before chunking
- No content-specific chunking strategies applied
- Validation layer drops content instead of handling it
- MinerU structure metadata discarded (system flattens to plain text)

**Solution**:
1. **Phase 0**: Detect and filter corrupt/non-content before chunking (CRITICAL - prevents data loss)
2. **Phases 1-5**: Integrate type-aware chunking with statistical heuristics and validation

**Impact**: Eliminate silent data loss, handle diverse content types appropriately, preserve all retrievable content with full transparency.

---

## Current State Analysis

### What Works
- ✅ MinerU provides `layout_type` (text/table/image) and `semantic_type` (header/prose/toc/etc)
- ✅ `type_aware_chunking.py` exists with handlers for different content types
- ✅ Validation layer prevents embedding crashes by catching oversized chunks
- ✅ Stored chunks (13,398) are mostly within acceptable size ranges

### What's Broken
- ❌ **CRITICAL**: Validation layer drops chunks silently (data loss)
- ❌ `document_tasks.py` flattens elements to text, discarding MinerU metadata
- ❌ `type_aware_chunking.py` is never called (dead code)
- ❌ No corrupt content detection before chunking
- ❌ Book indexes and statistical tables not filtered
- ❌ No content-specific strategies are applied
- ❌ No validation testing framework
- ❌ No monitoring or alerting for data loss

### Evidence from Logs

**Stored Oversized Chunks** (in database):
- 13 chunks: 451-553 tokens
- Content: Bullet lists (5), tables (3), citation-heavy prose (3), structured sections (2)

**Dropped Chunks** (from celery logs):
- Evidence of chunks being dropped: "Dropping oversized chunk | chunk_index=X"
- Examples found:
  - Chunk 3186: 20,315 tokens - Severely corrupt PDF (backwards text, artifacts)
  - Chunk 3187: 3,449 tokens - Statistical reference table
  - Chunk 3188: 3,380 tokens - Statistical reference table
  - Chunks 2648-2652: 7,125-10,075 tokens - Book indexes
  - More examples throughout logs

**Content Types Being Dropped**:
1. **Severely corrupt PDF extractions** - Backwards/mirrored text, no word spacing, PDF artifacts
2. **Book indexes** - Dense page references, should be filtered earlier
3. **Statistical reference tables** - Appendix content with minimal prose value

---

## Implementation Phases

### Phase 0: Prevent Data Loss - Corrupt Content Detection (CRITICAL - 3-6 hours)

**Goal**: Stop dropping content silently. Detect corrupt/non-content elements before chunking and handle them appropriately.

**Priority**: **HIGHEST** - This phase prevents data loss and must be completed first.

**Architecture**: Keep all filtering code in `document_tasks.py` (single use location, no need for separate module)

#### Task 0.1: Add corruption detection to document_tasks.py

**File**: `src/fileintel/tasks/document_tasks.py`

**Add at module level** (after imports, before functions):

```python
# Corruption detection thresholds
CORRUPTION_THRESHOLDS = {
    'max_pdf_artifacts': 20,        # More than 20 (cid:X) indicates corrupt extraction
    'min_stat_table_tokens': 2000,  # Statistical tables are typically large appendix content
    'max_avg_word_length': 15,      # Normal English ~5-6 chars, >15 indicates no word spacing
    'max_table_lines': 500,         # Corrupt tables render one value per line
    'book_index_page_density': 0.3, # Index pages have 30%+ page number references
    'book_index_comma_density': 0.05, # Index pages use commas to separate page refs
    'extremely_large_element': 10000  # >10k tokens almost certainly corrupt
}
```

**Add detection functions** (private, focused, single responsibility):

```python
def _has_excessive_pdf_artifacts(text: str) -> bool:
    """Check for PDF extraction artifacts like (cid:X) placeholders."""
    return text.count('(cid:') > CORRUPTION_THRESHOLDS['max_pdf_artifacts']


def _is_statistical_reference_table(text: str) -> bool:
    """Detect statistical reference tables in appendices."""
    if not re.search(r'TABLE [IVX]+', text, re.IGNORECASE):
        return False

    stat_terms = ['Critical Values', 'Degrees of Freedom', 'Percentage Points', 'Distribution']
    term_count = sum(1 for term in stat_terms if term in text)

    from fileintel.document_processing.type_aware_chunking import estimate_tokens
    tokens = estimate_tokens(text)

    return term_count >= 2 and tokens > CORRUPTION_THRESHOLDS['min_stat_table_tokens']


def _has_missing_word_boundaries(text: str) -> bool:
    """Detect text extracted without word spacing."""
    words = text.split()
    if len(words) < 10:
        return False

    avg_length = sum(len(w) for w in words) / len(words)
    return avg_length > CORRUPTION_THRESHOLDS['max_avg_word_length']


def _is_corrupt_table_extraction(text: str) -> bool:
    """Detect tables rendered as one number per line."""
    lines = [l for l in text.split('\n') if l.strip()]
    if len(lines) <= CORRUPTION_THRESHOLDS['max_table_lines']:
        return False

    avg_line_length = sum(len(l) for l in lines) / len(lines)
    return avg_line_length < 15


def _is_book_index(text: str) -> bool:
    """Detect book index pages with page number references."""
    if len(text) < 1000:
        return False

    # Count page number references
    page_numbers = len(re.findall(r'\b\d{1,4}\b', text))
    words = text.split()

    page_density = page_numbers / len(words)
    comma_density = text.count(',') / len(text)

    if page_density <= CORRUPTION_THRESHOLDS['book_index_page_density']:
        return False
    if comma_density <= CORRUPTION_THRESHOLDS['book_index_comma_density']:
        return False

    # Additional check: short average line length
    lines = [l for l in text.split('\n') if l.strip()]
    avg_line_len = sum(len(l) for l in lines) / max(len(lines), 1)
    return avg_line_len < 80


def _should_filter_element(element: TextElement) -> Tuple[bool, Optional[str]]:
    """
    Determine if element should be filtered before chunking.

    Single responsibility: make filtering decision based on all checks.

    Returns: (should_filter, reason)
    """
    if not element or not element.text:
        return (True, 'empty_element')

    # Check 1: MinerU semantic type (trust MinerU if available)
    semantic_type = element.metadata.get('semantic_type') if element.metadata else None
    if semantic_type in ['toc', 'lof', 'lot']:
        return (True, f'semantic_type_{semantic_type}')

    # Check 2: Corruption patterns (early returns for clarity)
    text = element.text

    if _has_excessive_pdf_artifacts(text):
        return (True, 'excessive_pdf_artifacts')

    if _is_statistical_reference_table(text):
        return (True, 'statistical_reference_table')

    if _has_missing_word_boundaries(text):
        return (True, 'no_word_boundaries')

    if _is_corrupt_table_extraction(text):
        return (True, 'corrupt_table_extraction')

    if _is_book_index(text):
        return (True, 'book_index')

    # Check 3: Extremely large (likely corrupt)
    from fileintel.document_processing.type_aware_chunking import estimate_tokens
    if estimate_tokens(text) > CORRUPTION_THRESHOLDS['extremely_large_element']:
        return (True, 'extremely_large_element')

    return (False, None)
```

#### Task 0.2: Integrate filtering with separation of concerns

**File**: `src/fileintel/tasks/document_tasks.py`

**Add filtering function** (pure logic, no I/O):

```python
def _filter_elements(elements: List[TextElement]) -> Tuple[List[TextElement], List[Dict]]:
    """
    Filter corrupt/non-content elements.

    Pure function - no I/O, easily testable.

    Returns: (clean_elements, filtered_metadata)
    """
    clean = []
    filtered = []

    for idx, element in enumerate(elements):
        try:
            should_filter, reason = _should_filter_element(element)

            if should_filter:
                from fileintel.document_processing.type_aware_chunking import estimate_tokens

                logger.warning(
                    f"Filtering element {idx}: {reason} | "
                    f"{estimate_tokens(element.text)} tokens | "
                    f"preview: {element.text[:80]}..."
                )

                filtered.append({
                    'index': idx,
                    'reason': reason,
                    'token_count': estimate_tokens(element.text),
                    'char_count': len(element.text),
                    'preview': element.text[:500]
                })
            else:
                clean.append(element)

        except Exception as e:
            logger.error(f"Filter error on element {idx}: {e}")
            # Fail open - keep element if filtering crashes (prevent data loss)
            clean.append(element)

    return clean, filtered
```

**Add storage function** (separate I/O concern):

```python
def _store_filtering_results(
    document_id: str,
    total_elements: int,
    filtered_metadata: List[Dict],
    storage
) -> None:
    """
    Persist filtering results for transparency.

    Separate function for storage I/O - doesn't affect filtering logic.
    """
    if not filtered_metadata:
        return

    try:
        storage.store_document_structure(
            document_id=document_id,
            structure_type='filtered_content',
            data={
                'filtered_count': len(filtered_metadata),
                'total_elements': total_elements,
                'items': filtered_metadata[:50]  # Limit storage
            }
        )
    except Exception as e:
        logger.error(f"Failed to store filtering metadata: {e}")
        # Don't fail the whole process if metadata storage fails
```

**Add backwards-compatible read function**:

```python
def read_document_with_elements(file_path: str) -> Tuple[str, List[Dict], Dict, List[TextElement]]:
    """
    Read document preserving structured elements.

    Returns: (combined_text, page_mappings, metadata, elements)
    """
    # ... existing read_document_content implementation ...
    # Just rename the function and ensure it returns elements
    return combined_text, page_mappings, metadata, elements

# Keep old function for backwards compatibility
def read_document_content(file_path: str) -> Tuple[str, List[Dict], Dict]:
    """BACKWARDS COMPATIBLE: Read without elements. Use read_document_with_elements() for new code."""
    text, mappings, meta, _ = read_document_with_elements(file_path)
    return text, mappings, meta
```

**Modify `process_document` function** (around line ~407):

```python
# Use new function that preserves elements
content, page_mappings, doc_metadata, elements = read_document_with_elements(file_path)

logger.info(f"Extracted {len(content)} characters with {len(elements)} elements")

# Filter elements (pure logic)
clean_elements, filtered_metadata = _filter_elements(elements)

logger.info(
    f"Filtering: {len(clean_elements)}/{len(elements)} elements kept, "
    f"{len(filtered_metadata)} filtered"
)

# Store filtering results (separate I/O) - do this AFTER document creation
# when we have actual_document_id available
if filtered_metadata:
    filter_reasons = {}
    for item in filtered_metadata:
        reason = item['reason']
        filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
    logger.info(f"Filter reasons: {filter_reasons}")

# Continue with chunking using clean_elements
# (modify chunking call in next section)
```

**Later in process_document** (after document creation, around line ~490):

```python
# After document is created and we have actual_document_id
_store_filtering_results(actual_document_id, len(elements), filtered_metadata, storage)
```

#### Task 0.3: Change validation to flag instead of drop

**File**: Search codebase for where validation drops chunks

```bash
# Find the validation code
grep -r "Dropping oversized chunk" src/
```

**Current behavior** (find and change):
```python
# CURRENT - drops silently
if token_count > max_tokens:
    logger.error(f"Dropping oversized chunk: {token_count} tokens")
    continue  # DROPS THE CHUNK - DATA LOSS
```

**New behavior**:
```python
# NEW - flag and store
if token_count > max_tokens:
    # Structure metadata clearly
    chunk['metadata']['size_validation'] = {
        'oversized': True,
        'token_count': token_count,
        'max_allowed': max_tokens,
        'overage': token_count - max_tokens,
        'skip_embedding': True  # Don't attempt to embed this chunk
    }

    if token_count > max_tokens * 2:  # >900 tokens
        logger.error(
            f"CRITICALLY oversized chunk: {token_count}/{max_tokens} tokens | "
            f"This should not happen after Phase 0 filtering - investigate!"
        )
        chunk['metadata']['size_validation']['critically_oversized'] = True
    else:
        logger.warning(f"Oversized chunk: {token_count}/{max_tokens} tokens")

    # ALWAYS STORE - never drop silently
    # Downstream embedding can skip based on 'skip_embedding' flag

# Continue to store the chunk, not continue/skip
```

#### Task 0.4: Add transparency endpoint (optional)

**File**: `src/fileintel/api/routes/documents.py` (add to existing routes, don't create new file)

```python
@router.get("/documents/{document_id}/filtered-content")
def get_filtered_content(document_id: str):
    """
    Show what content was filtered from a document and why.

    Provides transparency into filtering decisions.
    """
    storage = get_shared_storage()
    try:
        structures = storage.get_document_structures(
            document_id=document_id,
            structure_type='filtered_content'
        )

        if not structures:
            return {
                'document_id': document_id,
                'has_filtered_content': False
            }

        data = structures[0].data if structures else {}

        # Aggregate filter reasons
        filter_reasons = {}
        for item in data.get('items', []):
            reason = item['reason']
            filter_reasons[reason] = filter_reasons.get(reason, 0) + 1

        return {
            'document_id': document_id,
            'has_filtered_content': True,
            'filtered_count': data.get('filtered_count', 0),
            'total_elements': data.get('total_elements', 0),
            'filter_rate': data.get('filtered_count', 0) / max(data.get('total_elements', 1), 1),
            'filter_reasons': filter_reasons,
            'sample_items': data.get('items', [])[:5]  # First 5 for preview
        }
    finally:
        storage.close()
```

**Expected Outcomes**:
- ✅ Corrupt content detected before chunking
- ✅ Book indexes and statistical tables filtered
- ✅ **Zero silent data loss** - all content either chunked or logged as filtered
- ✅ Filtering transparent and auditable via API (optional)
- ✅ Critically oversized chunks flagged, not dropped

**Code Quality Improvements**:
- ✅ All code in single file (no unnecessary module)
- ✅ Constants for thresholds (easy to tune)
- ✅ Separation of concerns (filtering vs storage)
- ✅ Backwards compatible (no breaking changes)
- ✅ Error handling (fail open to prevent data loss)
- ✅ Each function has single responsibility

**Risk**: Low - Adds filtering layer without breaking existing functionality

**Testing** (run these in `process_document` integration tests):
```python
# Test 1: Corrupt content detection
from fileintel.tasks.document_tasks import _should_filter_element
from fileintel.document_processing.elements import TextElement

test_cases = [
    # (text, should_filter, expected_reason)
    ("Normal paragraph text here.", False, None),
    ("(cid:2) " * 50, True, 'excessive_pdf_artifacts'),
    ("Job rotation, 20, 21, 212\n" * 100, True, 'book_index'),
    ("TABLE VI Critical Values Degrees of Freedom " + "x" * 8000, True, 'statistical_reference_table'),
]

for text, expected_filter, expected_reason in test_cases:
    element = TextElement(text, {})
    should_filter, reason = _should_filter_element(element)
    assert should_filter == expected_filter, f"Failed on: {text[:50]}"
    if expected_reason:
        assert reason == expected_reason

# Test 2: Filtering preserves clean content
elements = [
    TextElement("Clean text", {}),
    TextElement("(cid:2)" * 50, {}),  # Should be filtered
    TextElement("More clean text", {})
]

clean, filtered = _filter_elements(elements)
assert len(clean) == 2
assert len(filtered) == 1
assert filtered[0]['reason'] == 'excessive_pdf_artifacts'

# Test 3: Error handling doesn't lose data
def _should_filter_element_that_crashes(element):
    raise ValueError("Test error")

# Monkey patch for test
original_filter = _should_filter_element
_should_filter_element = _should_filter_element_that_crashes

clean, filtered = _filter_elements([TextElement("text", {})])
assert len(clean) == 1  # Element preserved despite error

_should_filter_element = original_filter  # Restore
```

---

### Phase 1: Activate Existing Infrastructure (IMMEDIATE - 2-4 hours)

**Goal**: Use MinerU's existing metadata by integrating `type_aware_chunking.py`

#### Task 1.1: Modify `document_tasks.py` to preserve elements

**File**: `src/fileintel/tasks/document_tasks.py`

**Changes**:

```python
# Line ~407: Add elements to return tuple
def read_document_content(file_path: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any], List[TextElement]]:
    """
    Returns:
        Tuple of (raw_text_content, page_mappings, document_metadata, elements)
    """
    # ... existing code ...

    # Line ~140: Return elements instead of discarding them
    combined_text = " ".join(text_parts)
    return combined_text, page_mappings, metadata, elements  # ADD elements
```

#### Task 1.2: Integrate type-aware chunking

**File**: `src/fileintel/tasks/document_tasks.py`

**Changes**:

```python
# Line ~407: Update function call
content, page_mappings, doc_metadata, elements = read_document_content(file_path)

# Line ~414-422: Replace with type-aware chunking
from fileintel.document_processing.type_aware_chunking import chunk_elements_by_type

# Try type-aware chunking first
if elements and len(elements) > 0:
    logger.info(f"Using type-aware chunking for {len(elements)} elements")
    chunks = chunk_elements_by_type(elements, max_tokens=450)
    logger.info(f"Type-aware chunking created {len(chunks)} chunks")
else:
    # Fallback to traditional text-based chunking
    logger.warning("No elements available, falling back to text-based chunking")
    chunker = TextChunker()
    if chunker.enable_two_tier:
        chunks, full_chunking_result = clean_and_chunk_text(
            content, page_mappings=page_mappings, return_full_result=True
        )
    else:
        chunks = clean_and_chunk_text(content, page_mappings=page_mappings)
        full_chunking_result = None
```

#### Task 1.3: Add logging for debugging

**File**: `src/fileintel/tasks/document_tasks.py`

**Changes**:

```python
# After chunking, log statistics
content_types = {}
for chunk in chunks:
    ctype = chunk.get('metadata', {}).get('content_type', 'unknown')
    content_types[ctype] = content_types.get(ctype, 0) + 1

logger.info(f"Chunk distribution by content type: {content_types}")

oversized = [c for c in chunks if estimate_tokens(c['text']) > 450]
if oversized:
    logger.warning(f"Found {len(oversized)} oversized chunks after type-aware chunking")
    for chunk in oversized:
        logger.warning(
            f"Oversized: {estimate_tokens(chunk['text'])} tokens | "
            f"type={chunk.get('metadata', {}).get('chunk_strategy', 'unknown')}"
        )
```

**Expected Outcome**:
- Type-aware chunking activates for documents with MinerU processing
- Tables identified by `layout_type='table'` handled differently
- Logging shows content type distribution

**Risk**: Low - fallback to existing chunking if elements unavailable

---

### Phase 2: Add Statistical Heuristics (SHORT-TERM - 1-2 days)

**Goal**: Handle documents where MinerU metadata is incomplete/missing

#### Task 2.1: Create content analysis module

**New File**: `src/fileintel/document_processing/content_classifier.py`

```python
"""Statistical content analysis without hardcoded patterns."""

import re
import numpy as np
from typing import Dict, List
from .elements import TextElement


def analyze_text_statistics(text: str) -> Dict[str, float]:
    """Extract statistical features from text for classification."""

    lines = [l for l in text.split('\n') if l.strip()]
    if not lines:
        return {}

    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        sentences = [text]

    # Line-level statistics
    line_lengths = [len(l) for l in lines]

    # Sentence-level statistics
    sentence_lengths = [len(s) for s in sentences]

    return {
        # Line patterns (bullets/lists have short, varied lines)
        'line_count': len(lines),
        'avg_line_length': np.mean(line_lengths),
        'line_length_std': np.std(line_lengths),
        'short_lines_ratio': sum(1 for l in line_lengths if l < 80) / len(lines),

        # Sentence patterns (citations have very long sentences)
        'sentence_count': len(sentences),
        'avg_sentence_length': np.mean(sentence_lengths),
        'sentence_length_std': np.std(sentence_lengths),
        'long_sentences_ratio': sum(1 for s in sentence_lengths if s > 200) / len(sentences),

        # Quote/citation indicators
        'quote_count': text.count('"') + text.count('"') + text.count('"'),
        'quote_density': (text.count('"') + text.count('"')) / max(len(text), 1),

        # Structure indicators
        'newline_density': text.count('\n') / max(len(text), 1),
        'has_section_numbers': bool(re.search(r'\d+\.\d+\.\d+', text)),
        'bullet_like_lines': sum(1 for l in lines if re.match(r'^\s*[•\-\*\d]+\.?\s', l))
    }


def classify_by_heuristics(text: str, stats: Dict = None) -> str:
    """
    Classify content type using statistical heuristics.

    Returns: 'bullet_list', 'citation_heavy', 'structured_sections', or 'prose'
    """
    if stats is None:
        stats = analyze_text_statistics(text)

    # Bullet list detection
    # - Many short lines with high variance
    # - High bullet-like line ratio
    if (stats.get('short_lines_ratio', 0) > 0.6 and
        stats.get('line_length_std', 0) > 50 and
        stats.get('bullet_like_lines', 0) / max(stats.get('line_count', 1), 1) > 0.4):
        return 'bullet_list'

    # Citation-heavy prose detection
    # - High quote density
    # - Long average sentence length
    # - Few sentences but high word count
    if (stats.get('quote_density', 0) > 0.008 and
        stats.get('avg_sentence_length', 0) > 150):
        return 'citation_heavy'

    # Structured sections detection
    # - Has section numbering
    # - Multiple newlines (section breaks)
    if (stats.get('has_section_numbers') and
        stats.get('newline_density', 0) > 0.03):
        return 'structured_sections'

    return 'prose'


def enrich_element_metadata(element: TextElement) -> TextElement:
    """
    Add statistical classification to element metadata if not already present.

    Priority:
    1. Trust MinerU metadata (layout_type, semantic_type) if present
    2. Add statistical classification as fallback
    """
    metadata = element.metadata or {}

    # Skip if already has reliable metadata
    if metadata.get('layout_type') or metadata.get('semantic_type'):
        metadata['classification_source'] = 'mineru'
        element.metadata = metadata
        return element

    # Add statistical classification
    stats = analyze_text_statistics(element.text)
    content_type = classify_by_heuristics(element.text, stats)

    metadata.update({
        'classification_source': 'statistical',
        'heuristic_type': content_type,
        'stats': {k: v for k, v in stats.items() if isinstance(v, (int, float, bool))}
    })

    element.metadata = metadata
    return element
```

#### Task 2.2: Integrate into chunking pipeline

**File**: `src/fileintel/document_processing/type_aware_chunking.py`

**Changes**:

```python
from .content_classifier import enrich_element_metadata

def chunk_element_by_type(
    element: TextElement,
    max_tokens: int = 450,
    chunker = None
) -> List[Dict[str, Any]]:
    """Chunk with fallback to statistical classification."""

    # Enrich metadata if needed
    element = enrich_element_metadata(element)

    # Check MinerU metadata first
    semantic_type = element.metadata.get('semantic_type', 'prose')
    layout_type = element.metadata.get('layout_type', 'text')

    # Fallback to heuristic classification
    if layout_type == 'text' and semantic_type == 'prose':
        heuristic_type = element.metadata.get('heuristic_type')
        if heuristic_type:
            logger.debug(f"Using heuristic classification: {heuristic_type}")
            # Map heuristic types to handlers
            if heuristic_type == 'bullet_list':
                return _chunk_bullet_list(element, max_tokens, chunker)
            elif heuristic_type == 'citation_heavy':
                return _chunk_citation_prose(element, max_tokens, chunker)
            elif heuristic_type == 'structured_sections':
                return _chunk_structured_sections(element, max_tokens, chunker)

    # Existing type-aware logic
    if layout_type == 'table':
        chunks = _chunk_table(element, max_tokens)
    # ... rest of existing code ...
```

**Expected Outcome**:
- Documents without MinerU metadata get statistical classification
- Heuristics catch bullet lists, citations, structured sections
- Metadata shows classification source (mineru vs statistical)

**Risk**: Medium - heuristic thresholds need tuning on real data

---

### Phase 3: Implement Specialized Chunkers (SHORT-TERM - 2-3 days)

**Goal**: Add content-specific chunking strategies

#### Task 3.1: Bullet list chunker

**File**: `src/fileintel/document_processing/type_aware_chunking.py`

**Add new function**:

```python
def _chunk_bullet_list(element: TextElement, max_tokens: int, chunker) -> List[Dict]:
    """
    Split bullet lists at semantic boundaries.

    Strategy:
    - Group bullets under section headers
    - Keep nested bullets with parents
    - Split when group exceeds token limit
    """
    text = element.text
    lines = text.split('\n')

    # Simple grouping: split at empty lines or major headers
    groups = []
    current_group = []
    current_tokens = 0

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            # Empty line - possible group boundary
            if current_group and current_tokens > max_tokens * 0.8:
                groups.append('\n'.join(current_group))
                current_group = []
                current_tokens = 0
            continue

        line_tokens = estimate_tokens(line)

        # If adding this line exceeds limit significantly, start new group
        if current_tokens + line_tokens > max_tokens * 1.1 and current_group:
            groups.append('\n'.join(current_group))
            current_group = [line]
            current_tokens = line_tokens
        else:
            current_group.append(line)
            current_tokens += line_tokens

    # Add final group
    if current_group:
        groups.append('\n'.join(current_group))

    # Convert to chunks
    chunks = []
    for i, group_text in enumerate(groups):
        tokens = estimate_tokens(group_text)
        chunks.append({
            'text': group_text,
            'metadata': {
                **element.metadata,
                'chunk_strategy': 'bullet_group_split',
                'content_type': 'bullet_list',
                'group_index': i,
                'token_count': tokens,
                'within_limit': tokens <= max_tokens
            }
        })

    logger.info(f"Split bullet list into {len(chunks)} groups")
    return chunks
```

#### Task 3.2: Citation-heavy prose chunker

**File**: `src/fileintel/document_processing/type_aware_chunking.py`

```python
def _chunk_citation_prose(element: TextElement, max_tokens: int, chunker) -> List[Dict]:
    """
    Split citation-heavy text at quote boundaries.

    Strategy:
    - Treat quoted passages as atomic units
    - Split between quotes when possible
    - Respect sentence boundaries within non-quoted text
    """
    text = element.text

    # Split at sentence boundaries (existing logic is fine)
    # But log that it's citation-heavy for monitoring
    chunks = _chunk_text(element, max_tokens, chunker)

    for chunk in chunks:
        chunk['metadata']['content_type'] = 'citation_heavy'
        chunk['metadata']['chunk_strategy'] = 'citation_aware_sentence'

    return chunks
```

#### Task 3.3: Progressive fallback splitter

**File**: `src/fileintel/document_processing/type_aware_chunking.py`

**Enhance `_chunk_text` function**:

```python
def _chunk_text(element: TextElement, max_tokens: int, chunker = None) -> List[Dict[str, Any]]:
    """
    Chunk text using progressive fallback strategy.

    Try increasingly aggressive splits until we get chunks under limit.
    """
    if not element.text:
        return []

    text_tokens = estimate_tokens(element.text)

    if text_tokens <= max_tokens:
        return [{
            'text': element.text,
            'metadata': {
                **element.metadata,
                'chunk_strategy': 'single_element',
                'token_count': text_tokens
            }
        }]

    # Progressive fallback splitting
    split_strategies = [
        ('\n\n', 'paragraph', 1.0),      # Double newline
        ('\n', 'line', 1.1),              # Single newline (allow 10% overage)
        ('. ', 'sentence', 1.15),         # Sentence (allow 15% overage)
        (', ', 'clause', 1.2),            # Clause (allow 20% overage)
    ]

    for delimiter, strategy_name, max_overage_factor in split_strategies:
        parts = element.text.split(delimiter)
        if len(parts) <= 1:
            continue  # Can't split with this delimiter

        chunks = []
        current_chunk = []
        current_tokens = 0

        for part in parts:
            part_tokens = estimate_tokens(part)

            # Start new chunk if needed
            if current_tokens + part_tokens > max_tokens and current_chunk:
                chunk_text = delimiter.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        **element.metadata,
                        'chunk_strategy': f'split_at_{strategy_name}',
                        'token_count': current_tokens
                    }
                })
                current_chunk = [part]
                current_tokens = part_tokens
            else:
                current_chunk.append(part)
                current_tokens += part_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = delimiter.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    **element.metadata,
                    'chunk_strategy': f'split_at_{strategy_name}',
                    'token_count': current_tokens
                }
            })

        # Check if this strategy worked
        max_chunk_tokens = max(c['metadata']['token_count'] for c in chunks)
        if max_chunk_tokens <= max_tokens * max_overage_factor:
            logger.info(
                f"Split using {strategy_name} boundaries: "
                f"{len(chunks)} chunks, max={max_chunk_tokens} tokens"
            )
            return chunks

    # Last resort: hard truncate
    logger.warning(f"No clean split found, truncating to {max_tokens} tokens")
    return [{
        'text': element.text[:max_tokens * 4],  # Rough char estimate
        'metadata': {
            **element.metadata,
            'chunk_strategy': 'truncated',
            'token_count': max_tokens,
            'truncated': True,
            'original_tokens': text_tokens
        }
    }]
```

**Expected Outcome**:
- Bullet lists split at natural group boundaries
- Long prose tries paragraph → sentence → clause splitting
- Citation-heavy text handled with logging
- No chunks exceed 20% overage (540 tokens max)

**Risk**: Low - fallbacks ensure something always works

---

### Phase 4: Validation & Testing (MEDIUM-TERM - 2-3 days)

**Goal**: Validate improvements with real data before full deployment

#### Task 4.1: Create test framework

**New File**: `tests/integration/test_chunking_improvements.py`

```python
"""Integration tests for chunking improvements."""

import pytest
from fileintel.celery_config import get_shared_storage
from fileintel.document_processing.type_aware_chunking import chunk_elements_by_type
from fileintel.document_processing.elements import TextElement


def get_oversized_chunks_from_db(limit=50):
    """Get sample of current oversized chunks for testing."""
    storage = get_shared_storage()
    try:
        # Query for chunks >450 tokens (1800 chars)
        with storage.engine.connect() as conn:
            result = conn.execute(text('''
                SELECT
                    chunk_text,
                    chunk_metadata,
                    LENGTH(chunk_text) / 4 as est_tokens
                FROM document_chunks
                WHERE LENGTH(chunk_text) > 1800
                ORDER BY LENGTH(chunk_text) DESC
                LIMIT :limit
            '''), {'limit': limit})

            return [{'text': r.chunk_text, 'metadata': r.chunk_metadata, 'tokens': r.est_tokens}
                    for r in result]
    finally:
        storage.close()


def test_chunking_on_oversized_sample():
    """Test new chunking on known oversized chunks."""

    oversized_chunks = get_oversized_chunks_from_db(limit=13)  # Our 13 known cases
    assert len(oversized_chunks) == 13, "Should have 13 test cases"

    results = {
        'improved': 0,
        'same': 0,
        'worse': 0,
        'details': []
    }

    for original in oversized_chunks:
        # Create TextElement from chunk
        element = TextElement(original['text'], original['metadata'])

        # Apply new chunking
        new_chunks = chunk_elements_by_type([element], max_tokens=450)

        # Measure improvement
        original_max_tokens = original['tokens']
        new_max_tokens = max(estimate_tokens(c['text']) for c in new_chunks)

        if new_max_tokens <= 450:
            results['improved'] += 1
            outcome = 'FIXED'
        elif new_max_tokens < original_max_tokens:
            results['improved'] += 1
            outcome = 'BETTER'
        elif new_max_tokens == original_max_tokens:
            results['same'] += 1
            outcome = 'SAME'
        else:
            results['worse'] += 1
            outcome = 'WORSE'

        results['details'].append({
            'original_tokens': original_max_tokens,
            'new_tokens': new_max_tokens,
            'new_chunk_count': len(new_chunks),
            'outcome': outcome,
            'preview': original['text'][:100]
        })

    # Print summary
    print(f"\n=== Chunking Test Results ===")
    print(f"Improved: {results['improved']}/13")
    print(f"Same: {results['same']}/13")
    print(f"Worse: {results['worse']}/13")

    # Require at least 70% improvement
    assert results['improved'] >= 9, "Should improve at least 9/13 cases"
    assert results['worse'] == 0, "Should not make any cases worse"


def test_new_chunking_quality():
    """Test that new chunking maintains quality standards."""

    # Test with diverse content
    test_cases = [
        # Bullet list
        TextElement(
            "• First item\n• Second item\n• Third item\n" * 50,
            {'layout_type': 'text'}
        ),
        # Table (should allow oversized)
        TextElement(
            "TABLE 1.1 Comparison\nCol1 | Col2\n" + "Row data\n" * 100,
            {'layout_type': 'table'}
        ),
        # Normal prose
        TextElement(
            "This is a normal paragraph. " * 100,
            {'semantic_type': 'prose'}
        )
    ]

    for element in test_cases:
        chunks = chunk_element_by_type(element, max_tokens=450)

        # Check metadata presence
        for chunk in chunks:
            assert 'metadata' in chunk
            assert 'chunk_strategy' in chunk['metadata']
            assert 'token_count' in chunk['metadata']

        # Check that non-table chunks respect limit (or are close)
        if element.metadata.get('layout_type') != 'table':
            for chunk in chunks:
                tokens = chunk['metadata']['token_count']
                assert tokens <= 540, f"Non-table chunk too large: {tokens} tokens"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

#### Task 4.2: Validation script for manual review

**New File**: `scripts/validate_chunking.py`

```python
"""Manual validation tool for chunking improvements."""

from fileintel.celery_config import get_shared_storage
from fileintel.document_processing.type_aware_chunking import chunk_elements_by_type
from fileintel.document_processing.elements import TextElement
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


def compare_chunking_strategies():
    """Interactive comparison of old vs new chunking."""

    console = Console()
    storage = get_shared_storage()

    try:
        # Get oversized chunks
        with storage.engine.connect() as conn:
            result = conn.execute(text('''
                SELECT chunk_text, chunk_metadata, LENGTH(chunk_text) / 4 as tokens
                FROM document_chunks
                WHERE LENGTH(chunk_text) > 1800
                ORDER BY LENGTH(chunk_text) DESC
                LIMIT 13
            '''))

            oversized = list(result)

        for idx, chunk in enumerate(oversized, 1):
            console.print(f"\n{'='*80}")
            console.print(f"[bold]Chunk {idx}/13[/bold] - Original: {chunk.tokens:.0f} tokens")
            console.print(f"{'='*80}\n")

            # Show original
            console.print(Panel(
                chunk.chunk_text[:500] + "..." if len(chunk.chunk_text) > 500 else chunk.chunk_text,
                title="Original Chunk",
                border_style="red"
            ))

            # Try new chunking
            element = TextElement(chunk.chunk_text, chunk.chunk_metadata or {})
            new_chunks = chunk_elements_by_type([element], max_tokens=450)

            # Show results
            table = Table(title="New Chunking Results")
            table.add_column("Chunk #", style="cyan")
            table.add_column("Tokens", justify="right")
            table.add_column("Strategy", style="green")
            table.add_column("Preview", style="dim")

            for i, nc in enumerate(new_chunks, 1):
                tokens = nc['metadata'].get('token_count', 0)
                strategy = nc['metadata'].get('chunk_strategy', 'unknown')
                preview = nc['text'][:80]
                table.add_row(str(i), str(tokens), strategy, preview)

            console.print(table)

            # Ask for feedback
            response = console.input("\n[yellow]Is this improvement good? (y/n/q):[/yellow] ")
            if response.lower() == 'q':
                break
            elif response.lower() == 'y':
                console.print("[green]✓ Marked as good[/green]")
            else:
                console.print("[red]✗ Needs more work[/red]")

    finally:
        storage.close()


if __name__ == '__main__':
    compare_chunking_strategies()
```

**Expected Outcome**:
- Automated test validates 9/13 improvements
- Manual validation tool for human review
- Baseline established for regression testing

**Risk**: Low - tests don't affect production

---

### Phase 5: Monitoring & Analytics (MEDIUM-TERM - 1-2 days)

**Goal**: Visibility into chunking quality over time

#### Task 5.1: Add chunking metrics

**File**: `src/fileintel/tasks/document_tasks.py`

**Add after chunking (line ~422)**:

```python
def log_chunking_metrics(chunks: List[Dict], document_id: str, file_path: str):
    """Log detailed chunking metrics for monitoring."""

    total_chunks = len(chunks)
    by_strategy = {}
    by_content_type = {}
    token_distribution = {'0-300': 0, '301-400': 0, '401-450': 0, '451-500': 0, '500+': 0}
    oversized = []

    for chunk in chunks:
        meta = chunk.get('metadata', {})

        # Count by strategy
        strategy = meta.get('chunk_strategy', 'unknown')
        by_strategy[strategy] = by_strategy.get(strategy, 0) + 1

        # Count by content type
        ctype = meta.get('content_type', 'unknown')
        by_content_type[ctype] = by_content_type.get(ctype, 0) + 1

        # Token distribution
        tokens = meta.get('token_count', estimate_tokens(chunk['text']))
        if tokens <= 300:
            token_distribution['0-300'] += 1
        elif tokens <= 400:
            token_distribution['301-400'] += 1
        elif tokens <= 450:
            token_distribution['401-450'] += 1
        elif tokens <= 500:
            token_distribution['451-500'] += 1
        else:
            token_distribution['500+'] += 1

        # Track oversized
        if tokens > 450:
            oversized.append({
                'tokens': tokens,
                'strategy': strategy,
                'content_type': ctype,
                'preview': chunk['text'][:100]
            })

    # Log summary
    logger.info(
        f"Chunking metrics for {document_id} | "
        f"total={total_chunks} | "
        f"oversized={len(oversized)} ({len(oversized)/total_chunks*100:.1f}%)"
    )
    logger.info(f"By strategy: {by_strategy}")
    logger.info(f"By content type: {by_content_type}")
    logger.info(f"Token distribution: {token_distribution}")

    # Warn about oversized
    if oversized:
        logger.warning(f"Found {len(oversized)} oversized chunks:")
        for ov in oversized[:5]:  # Log first 5
            logger.warning(
                f"  {ov['tokens']} tokens | {ov['strategy']} | "
                f"{ov['content_type']} | {ov['preview'][:60]}..."
            )

    # Alert if too many oversized
    if len(oversized) / total_chunks > 0.05:  # >5%
        logger.error(
            f"HIGH OVERSIZED RATE: {len(oversized)}/{total_chunks} "
            f"({len(oversized)/total_chunks*100:.1f}%) in {file_path}"
        )

# Call after chunking
log_chunking_metrics(chunks, actual_document_id, file_path)
```

#### Task 5.2: Add dashboard metrics

**File**: `src/fileintel/api/routes/metrics.py` (if exists) or create new endpoint

```python
@router.get("/chunking-quality")
def get_chunking_quality_metrics():
    """Get system-wide chunking quality metrics."""

    storage = get_shared_storage()
    try:
        with storage.engine.connect() as conn:
            # Overall statistics
            result = conn.execute(text('''
                SELECT
                    COUNT(*) as total_chunks,
                    AVG(LENGTH(chunk_text)) as avg_length,
                    COUNT(CASE WHEN LENGTH(chunk_text) > 1800 THEN 1 END) as oversized_count,
                    MAX(LENGTH(chunk_text)) as max_length
                FROM document_chunks
            '''))
            stats = result.fetchone()

            # By content type
            result = conn.execute(text('''
                SELECT
                    chunk_metadata->>'content_type' as content_type,
                    COUNT(*) as count,
                    AVG(LENGTH(chunk_text)) as avg_length
                FROM document_chunks
                WHERE chunk_metadata->>'content_type' IS NOT NULL
                GROUP BY content_type
                ORDER BY count DESC
            '''))
            by_type = [dict(r) for r in result]

            return {
                'total_chunks': stats.total_chunks,
                'avg_chunk_length': stats.avg_length,
                'oversized_count': stats.oversized_count,
                'oversized_rate': stats.oversized_count / stats.total_chunks,
                'max_chunk_length': stats.max_length,
                'by_content_type': by_type
            }
    finally:
        storage.close()
```

**Expected Outcome**:
- Real-time visibility into chunking quality
- Alert if oversized rate exceeds threshold
- Track improvements over time

**Risk**: Low - monitoring only

---

## Success Criteria

### Critical (Phase 0 complete) - MUST HAVE
- ✅ **Zero silent data loss** - No chunks dropped without logging
- ✅ Corrupt content detected and filtered transparently
- ✅ Book indexes and statistical tables filtered before chunking
- ✅ All filtered content logged and auditable via API
- ✅ Validation layer flags oversized chunks instead of dropping

### Minimum Viable (Phases 0-1 complete)
- ✅ Type-aware chunking integrated and working
- ✅ Uses MinerU metadata when available
- ✅ Fallback to text-based chunking works
- ✅ No regression in existing functionality
- ✅ Transparent filtering prevents data loss

### Target (Phases 0-3 complete)
- ✅ All retrievable content preserved (chunked or filtered with reason)
- ✅ Oversized stored chunks reduced to ≤5
- ✅ All oversized chunks are semantically justified (tables, complex content)
- ✅ No chunks exceed 550 tokens (current max: 553)
- ✅ Content-specific strategies applied to 80%+ of chunks

### Excellent (All phases complete)
- ✅ Oversized chunks <3
- ✅ Automated tests validate improvements
- ✅ Monitoring dashboard shows quality and filtering metrics
- ✅ Documentation for tuning heuristics
- ✅ User transparency: can view filtered content per document

---

## Risk Mitigation

### Risk 1: Breaking existing functionality
**Mitigation**:
- Fallback to existing chunking if elements unavailable
- Feature flag to enable/disable type-aware chunking
- A/B test on subset of documents before full rollout

### Risk 2: Heuristics fail on edge cases
**Mitigation**:
- Conservative thresholds (require high confidence)
- Validation testing on diverse documents
- Logging for manual review of failures
- Progressive fallback ensures something always works

### Risk 3: Performance degradation
**Mitigation**:
- Statistical analysis is fast (< 1ms per chunk)
- Only applied when MinerU metadata missing
- Benchmark before/after
- Optimize hot paths if needed

---

## Timeline Estimate

| Phase | Duration | Dependencies | Priority |
|-------|----------|--------------|----------|
| **Phase 0: Corrupt Content Detection** | **3-6 hours** | **None** | **CRITICAL** |
| Phase 1: Integration | 2-4 hours | Phase 0 complete | High |
| Phase 2: Heuristics | 1-2 days | Phase 1 complete | Medium |
| Phase 3: Specialized chunkers | 2-3 days | Phase 2 complete | Medium |
| Phase 4: Validation | 2-3 days | Phase 3 complete | Medium |
| Phase 5: Monitoring | 1-2 days | Phase 4 complete | Low |
| **Total** | **8-14 days** | Sequential | - |

**Timeline Notes**:
- **Phase 0 is CRITICAL** - Must be completed first to prevent data loss
- **Aggressive timeline** (parallel work after Phase 0): 6-8 days
- **Conservative timeline** (careful validation): 12-16 days
- **Minimum viable** (Phases 0-1 only): 0.5-1 day

---

## Next Steps

1. **Review this plan** with team
2. **Create feature branch**: `feature/prevent-data-loss`
3. **START WITH PHASE 0** (CRITICAL - prevents data loss)
   - Implement corrupt content detection
   - Change validation to flag instead of drop
   - Add filtering transparency
4. **Proceed to Phase 1** (activate type-aware chunking)
5. **Validate results** at each phase before proceeding
6. **Monitor filtering statistics** to tune detection thresholds
7. **Iterate based on real data**

---

## Appendix: Testing Data

### Known Oversized Chunks for Testing

```python
TEST_CASES = [
    # Chunk 2275 (553 tokens) - Dense bullet list
    {
        'id': 'fc5458cb-535c-45dc-8712-c21ef389d852',
        'tokens': 553,
        'type': 'bullet_list',
        'expected_improvement': 'split into 3 chunks ~190 tokens each'
    },
    # Chunk 1728 (533 tokens) - Citation-heavy prose
    {
        'id': 'c3b9bea7-53b2-4bf8-b0c1-e4f524d57328',
        'tokens': 533,
        'type': 'citation_heavy',
        'expected_improvement': 'split at quote boundaries ~265 tokens each'
    },
    # Chunk 1377 (481 tokens) - Table
    {
        'id': 'c8aa3dcf-3787-4945-9127-63cfc806e4f2',
        'tokens': 481,
        'type': 'table',
        'expected_improvement': 'keep as single chunk (acceptable)'
    },
    # ... all 13 cases documented
]
```

---

## Questions for Review

1. Should tables be allowed up to 600 tokens? Or strictly 450?
2. What's the acceptable oversized rate? Currently 0.1%, targeting 0.04%
3. Should we A/B test this before full rollout?
4. Who reviews the validation results before deployment?
5. What's the rollback plan if issues arise?
