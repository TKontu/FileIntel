# MinerU Structure Utilization Analysis

**Date:** 2025-10-18
**Purpose:** Analyze how fileintel currently uses MinerU output structure and propose improvements for handling TOC/LOF content

---

## Executive Summary

**Current State:** Fileintel **discards** most of MinerU's rich structure metadata by concatenating all elements into page-level text blobs.

**Problem:** TOC/LOF elements get mixed with prose, causing chunking failures because their type information is lost.

**Solution:** Preserve element-level type metadata through the pipeline and use it for type-aware chunking/filtering.

**Implementation Status:** ❌ **NOT YET IMPLEMENTED** - This document contains analysis and proposed solutions only.

---

## What Has Been Completed

✅ **Analysis and Design Work:**
1. Root cause analysis of TOC/LOF chunking failures (see `toc_lof_chunking_issue.md`)
2. Architecture analysis of current vs proposed element handling
3. RAG relevance assessment of different element types
4. Detailed implementation plan with 4 phases
5. Code examples for proposed changes

❌ **Implementation Work:**
- No code changes have been made yet
- The implementation roadmap below shows PROPOSED tasks
- All checkboxes in the roadmap are unchecked (- [ ])

**Next Step:** Review proposed plan and decide whether to proceed with implementation.

---

## Current Architecture

### What MinerU Provides

**content_list.json structure:**

```json
[
  {
    "type": "text",
    "text": "This is regular prose content.",
    "page_idx": 1,
    "bbox": [x, y, width, height]
  },
  {
    "type": "text",
    "text": "6.1 Overview...........19\n6.2 Velocity...........19\n...",
    "page_idx": 2,
    "bbox": [x, y, width, height]
  },
  {
    "type": "table",
    "text": "",
    "table_body": "<table>...</table>",
    "table_caption": ["Table 1: Results"],
    "page_idx": 3,
    "bbox": [x, y, width, height]
  }
]
```

**Rich metadata per element:**
- `type`: text, table, image, header, footer, list, page_number, etc.
- `bbox`: Precise coordinates
- `page_idx`: Exact page location
- Special fields: `table_body`, `table_caption`, etc.

---

### How Fileintel Currently Uses It

**Code flow in `mineru_selfhosted.py:_create_elements_from_json()`:**

```python
# Line 553-566: Group by page and THROW AWAY type information
elements_by_page = {}
for item in content_list:
    page_idx = item.get('page_idx', 0)
    if page_idx not in elements_by_page:
        elements_by_page[page_idx] = []

    element_info = {
        'text': item.get('text', ''),
        'type': item.get('type', 'text'),  # ← Preserved here
        'bbox': item.get('bbox', [])        # ← Preserved here
    }
    elements_by_page[page_idx].append(element_info)

# Line 573-595: CONCATENATE ALL TEXT, losing per-element types
page_text_parts = []
element_types = {}  # Only keep TYPE COUNTS

for elem_info in page_elements:
    text = elem_info['text'].strip()
    if text:
        page_text_parts.append(text)  # ← All text merged!

    # Count types (but don't preserve which text came from which type)
    elem_type = elem_info['type']
    element_types[elem_type] = element_types.get(elem_type, 0) + 1

page_text = '\n'.join(page_text_parts)  # ← ONE BIG STRING

# Line 598-606: Store only type COUNTS in metadata
metadata = {
    'source': str(file_path),
    'page_number': page_idx + 1,
    'element_count': total_elements,
    'element_types': element_types,  # ← {'text': 5, 'table': 1}
    # No way to know which text came from which type!
}
```

**Result:** One `TextElement` per page with:
- `text`: ALL text from ALL elements concatenated
- `metadata.element_types`: Just counts like `{'text': 10, 'table': 2}`
- **No way to know which part of the text is TOC, which is prose, which is table**

---

### What Happens Next

**Chunking (`document_tasks.py:clean_and_chunk_text()`):**

```python
# Line 109: Get elements from processor
elements, metadata = processor.read(path)

# Line 116: Concatenate ALL element text
for elem in elements:
    full_text += elem.text + "\n"

# Line 161: Chunk the concatenated text
chunker = TextChunker()
chunking_result = chunker.chunk_text_adaptive(full_text, page_mappings)
```

**At this point:**
- All structure is lost
- Chunker sees one giant string
- No idea which parts are TOC, prose, tables, etc.
- Sentence-based splitter fails on TOC (no sentence boundaries)

---

## Information Loss Analysis

### What MinerU Gives Us

**Per element:**

| Field | Available? | Currently Used? | Purpose |
|-------|-----------|----------------|---------|
| `type` | ✅ Yes | ⚠️ Partially (counts only) | Identify element purpose |
| `text` | ✅ Yes | ✅ Yes | Content |
| `bbox` | ✅ Yes | ⚠️ Stored but unused | Position on page |
| `page_idx` | ✅ Yes | ✅ Yes | Page mapping |
| `table_body` | ✅ Yes (tables) | ❌ No | HTML table structure |
| `table_caption` | ✅ Yes (tables) | ❌ No | Table titles |
| Special content | ✅ Yes (varies) | ❌ No | Type-specific data |

**What we COULD know but DON'T:**
- Which text blocks are TOC/LOF
- Which text blocks are headers/footers
- Which text blocks are page numbers
- Exact table structure (HTML)
- Element ordering within page
- Element spatial relationships

---

### What Gets Lost

**Example page with mixed content:**

```
MinerU provides (simplified):
[
  {type: "header", text: "Chapter 1"},
  {type: "text", text: "This is prose..."},
  {type: "text", text: "1.1 Introduction.....5\n1.2 Methods.....12"},  ← TOC!
  {type: "text", text: "More prose..."},
  {type: "footer", text: "Page 1"}
]
```

**Fileintel creates:**
```
TextElement(
  text="Chapter 1\nThis is prose...\n1.1 Introduction.....5\n1.2 Methods.....12\nMore prose...\nPage 1",
  metadata={'element_types': {'header': 1, 'text': 3, 'footer': 1}}
)
```

**Problems:**
1. Can't identify TOC within the text
2. Header and footer mixed with content
3. No way to filter by element type
4. Chunker treats everything equally

---

## RAG Relevance Analysis

### Is TOC/LOF Relevant for RAG?

**Arguments FOR keeping TOC/LOF:**

1. **Navigation context**
   - User asks "What sections are in this document?"
   - Answer: Extract from TOC

2. **Structure understanding**
   - User asks "Does this paper cover topic X?"
   - Answer: Check TOC for relevant section titles

3. **Cross-references**
   - User asks "Which section discusses Y?"
   - Answer: TOC maps topics to page numbers

4. **Figure/Table discovery**
   - User asks "Are there any tables about Z?"
   - Answer: Check List of Tables

**Arguments AGAINST keeping TOC/LOF:**

1. **Duplication**
   - TOC entries duplicate actual section headers in content
   - Wastes embedding space

2. **Low semantic value**
   - "6.1 Overview.......19" has minimal semantic content
   - Just a pointer to actual content

3. **Format noise**
   - Dots and page numbers don't help semantic search
   - "Introduction.....5" vs "Introduction" has same meaning

4. **Chunking difficulty**
   - As we've seen, causes oversized chunks
   - Requires special handling

**Verdict:**

**Moderately relevant BUT:**
- Value is in section/figure TITLES, not page numbers or formatting
- Better to extract titles structurally than embed TOC text
- For RAG: **Exclude TOC/LOF from vector embeddings**, extract structure separately

---

### RAG Relevance by Element Type

| Element Type | RAG Value | Recommendation |
|-------------|-----------|----------------|
| **text** (prose) | ⭐⭐⭐⭐⭐ High | ✅ Embed |
| **header** | ⭐⭐⭐⭐ High | ✅ Embed (structure context) |
| **list** | ⭐⭐⭐⭐ High | ✅ Embed |
| **table** (text) | ⭐⭐⭐ Medium | ⚠️ Embed caption, skip table body |
| **image** (caption) | ⭐⭐⭐ Medium | ⚠️ Embed caption only |
| **footer** | ⭐ Low | ❌ Skip (repetitive "Page X") |
| **page_number** | ⭐ Low | ❌ Skip (noise) |
| **TOC/LOF** | ⭐⭐ Low | ❌ Skip, extract structure |
| **ref_text** (bibliography) | ⭐⭐ Low | ⚠️ Optionally skip |

**Key insight:** Not all text is equally valuable for RAG. Element type should guide inclusion/exclusion.

---

## Proposed Architecture

### Design Goals

1. **Preserve element-level type information** through entire pipeline
2. **Type-aware chunking** based on element characteristics
3. **Selective embedding** - only embed RAG-relevant content
4. **Structural metadata extraction** for TOC/LOF/headers
5. **Backward compatibility** with existing data model

---

### Proposed Flow

```
MinerU content_list
        ↓
    ┌───────────────────────────────────┐
    │ 1. Parse & Classify Elements      │
    │    - Standard: text, header, list │
    │    - Special: TOC, LOF, footer    │
    └───────────────────────────────────┘
        ↓
    ┌───────────────────────────────────┐
    │ 2. Extract Structure (don't chunk)│
    │    - TOC → structured metadata    │
    │    - LOF → figure list            │
    │    - Headers → document outline   │
    └───────────────────────────────────┘
        ↓
    ┌───────────────────────────────────┐
    │ 3. Filter Elements                │
    │    - Skip: page_number, footer    │
    │    - Keep: text, header, list     │
    │    - Special: table (caption only)│
    └───────────────────────────────────┘
        ↓
    ┌───────────────────────────────────┐
    │ 4. Type-Aware Chunking            │
    │    - Prose: sentence-based        │
    │    - Tables: keep as single chunk │
    │    - Lists: keep items together   │
    │    - Headers: chunk with context  │
    └───────────────────────────────────┘
        ↓
    ┌───────────────────────────────────┐
    │ 5. Generate Embeddings            │
    │    (Only for filtered chunks)     │
    └───────────────────────────────────┘
```

---

### Implementation Approach

#### Phase 1: Element-Level Preservation

**Current (mineru_selfhosted.py:560-566):**

```python
# Preserves type but then discards it
element_info = {
    'text': item.get('text', ''),
    'type': item.get('type', 'text'),  # Preserved...
    'bbox': item.get('bbox', [])
}
# ...but then concatenates all text, losing type association
```

**Proposed:**

```python
# Create TextElement PER content_list item, not per page
def _create_elements_from_json(self, json_data, ...) -> List[TextElement]:
    text_elements = []

    for item in content_list:
        # Detect special types
        elem_type = item.get('type', 'text')
        text = item.get('text', '')

        # Enhanced type detection
        if elem_type == 'text' and text:
            # Check if it's actually TOC/LOF
            is_toc, toc_type = is_toc_or_lof(text)
            if is_toc:
                elem_type = toc_type  # Change to 'toc', 'lof', or 'lot'

        # Build metadata with full type information
        metadata = {
            'source': str(file_path),
            'page_number': item.get('page_idx', 0) + 1,
            'element_type': elem_type,  # ← Preserved at element level!
            'element_index': item.get('index', 0),
            'bbox': item.get('bbox', []),
            'has_table_body': 'table_body' in item,
            'table_caption': item.get('table_caption', [])
        }

        # Handle special cases
        if elem_type == 'table' and not text:
            # Tables often have no .text, only table_body
            text = ' '.join(item.get('table_caption', []))

        text_elements.append(TextElement(text=text, metadata=metadata))

    return text_elements
```

**Benefits:**
- Each `TextElement` has its own type
- Can filter/process by type later
- Chunker can see element types

---

#### Phase 2: Type-Based Filtering

**Add filtering before chunking:**

```python
# In document_tasks.py or new filtering module

def filter_elements_for_rag(elements: List[TextElement]) -> Tuple[List[TextElement], Dict]:
    """
    Filter elements based on RAG relevance.

    Returns:
        (filtered_elements, extracted_structure)
    """
    filtered = []
    toc_entries = []
    lof_entries = []
    headers = []

    for elem in elements:
        elem_type = elem.metadata.get('element_type', 'text')

        # Skip non-RAG-relevant types
        if elem_type in ['page_number', 'footer']:
            continue

        # Extract structure from TOC/LOF (don't chunk)
        if elem_type in ['toc', 'lof', 'lot']:
            if elem_type == 'toc':
                toc_entries.extend(parse_toc_entries(elem.text))
            elif elem_type in ['lof', 'lot']:
                lof_entries.extend(parse_figure_table_list(elem.text))
            continue  # Don't include in filtered elements

        # Extract headers as structure
        if elem_type == 'header':
            headers.append({
                'text': elem.text,
                'page': elem.metadata.get('page_number'),
                'level': detect_header_level(elem.text)
            })
            # Include headers in RAG (they provide context)
            filtered.append(elem)

        # Include everything else
        else:
            filtered.append(elem)

    structure = {
        'table_of_contents': toc_entries,
        'list_of_figures': lof_entries,
        'headers': headers
    }

    return filtered, structure
```

**Benefits:**
- TOC/LOF excluded from embeddings
- But structural information preserved
- Can query structure separately
- Reduced embedding costs

---

#### Phase 3: Type-Aware Chunking

**Modify chunker to respect element types:**

```python
# In chunking.py

def chunk_elements_by_type(
    elements: List[TextElement],
    max_tokens: int = 450
) -> List[Dict]:
    """
    Chunk elements using type-aware strategies.
    """
    chunks = []

    for elem in elements:
        elem_type = elem.metadata.get('element_type', 'text')

        if elem_type == 'table':
            # Tables: keep as single chunk if possible
            if estimate_tokens(elem.text) <= max_tokens:
                chunks.append({
                    'text': elem.text,
                    'metadata': elem.metadata,
                    'chunk_strategy': 'table_whole'
                })
            else:
                # Table too large: extract caption only
                caption = ' '.join(elem.metadata.get('table_caption', []))
                chunks.append({
                    'text': caption,
                    'metadata': {**elem.metadata, 'table_body_skipped': True},
                    'chunk_strategy': 'table_caption_only'
                })

        elif elem_type == 'list':
            # Lists: try to keep items together
            list_chunks = chunk_preserving_list_items(elem.text, max_tokens)
            for chunk_text in list_chunks:
                chunks.append({
                    'text': chunk_text,
                    'metadata': elem.metadata,
                    'chunk_strategy': 'list_aware'
                })

        elif elem_type in ['text', 'header']:
            # Prose: sentence-based chunking
            prose_chunks = chunk_text_sentence_based(elem.text, max_tokens)
            for chunk_text in prose_chunks:
                chunks.append({
                    'text': chunk_text,
                    'metadata': elem.metadata,
                    'chunk_strategy': 'sentence_based'
                })

        # Other types handled similarly...

    return chunks
```

**Benefits:**
- Tables handled appropriately
- Lists preserved better
- Prose chunked semantically
- TOC never reaches this stage (filtered out)

---

#### Phase 4: Structured Metadata Storage

**Database schema additions:**

```sql
-- New table for document structure
CREATE TABLE document_structure (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id),
    structure_type VARCHAR(50), -- 'toc', 'lof', 'lot', 'headers'
    data JSONB,                 -- Structured entries
    created_at TIMESTAMP
);

-- Example data for TOC
{
    "entries": [
        {"section": "1.1", "title": "Introduction", "page": 5},
        {"section": "1.2", "title": "Background", "page": 8},
        ...
    ]
}
```

**Query capabilities:**

```python
# Get document structure
structure = storage.get_document_structure(document_id)

# Navigate by section
sections = structure['toc']['entries']
intro_section = next(s for s in sections if 'Introduction' in s['title'])

# Find figures about topic
relevant_figures = [
    f for f in structure['lof']['entries']
    if 'architecture' in f['title'].lower()
]
```

**Benefits:**
- Queryable document structure
- Navigation by section
- Filter by structural elements
- Enable advanced UI features

---

## Migration Strategy

### Backward Compatibility

**Issue:** Existing data has page-level concatenated text.

**Solution:** Dual-mode support:

```python
def _create_elements_from_json(self, json_data, ...) -> List[TextElement]:
    # Check config flag
    if self.config.document_processing.use_element_level_types:
        # NEW: Element-level preservation
        return self._create_elements_element_level(json_data, ...)
    else:
        # OLD: Page-level concatenation (current behavior)
        return self._create_elements_page_level(json_data, ...)
```

**Rollout:**
1. Implement new flow with feature flag OFF
2. Test on new documents
3. Gradually enable for new uploads
4. Optionally reprocess old documents

---

### Config Changes

**Add to config.yaml:**

```yaml
document_processing:
  # MinerU structure utilization
  use_element_level_types: true  # NEW: Preserve element-level types

  # Element filtering for RAG
  element_filters:
    exclude_types: ['page_number', 'footer']  # Skip these types
    extract_structure_types: ['toc', 'lof', 'lot']  # Extract, don't embed
    embed_types: ['text', 'header', 'list']  # Embed these

  # Type-aware chunking
  chunking:
    use_type_aware: true
    table_strategy: 'caption_only'  # or 'whole_if_small', 'split'
    list_strategy: 'preserve_items'
    toc_strategy: 'extract_structure'
```

---

## Implementation Roadmap

**⚠️ STATUS: PLANNED - NOT YET IMPLEMENTED**

These phases represent the PROPOSED implementation plan. Only the analysis and design work has been completed. The actual code changes described below have NOT been implemented yet.

---

### Phase 1: Element-Level Preservation (1 week) - **NOT IMPLEMENTED**

**Tasks:**
1. - [ ] Modify `_create_elements_from_json()` to create one TextElement per content_list item
2. - [ ] Add element_type to TextElement metadata
3. - [ ] Add TOC/LOF detection function
4. - [ ] Add feature flag `use_element_level_types`
5. - [ ] Test with sample documents
6. - [ ] Verify backward compatibility

**Deliverables:**
- Modified `mineru_selfhosted.py`
- Element-level type preservation working
- Tests passing

---

### Phase 2: Type-Based Filtering (3 days) - **NOT IMPLEMENTED**

**Tasks:**
1. - [ ] Implement `filter_elements_for_rag()`
2. - [ ] Add TOC/LOF structure extraction
3. - [ ] Integrate filtering before chunking
4. - [ ] Add config for element filters
5. - [ ] Test filtering accuracy

**Deliverables:**
- `element_filter.py` module
- TOC/LOF extraction working
- Config-driven filtering

---

### Phase 3: Type-Aware Chunking (1 week) - **NOT IMPLEMENTED**

**Tasks:**
1. - [ ] Implement `chunk_elements_by_type()`
2. - [ ] Add table-specific chunking
3. - [ ] Add list-preserving chunking
4. - [ ] Integrate with existing chunker
5. - [ ] Add chunking strategy metadata
6. - [ ] Test on diverse documents

**Deliverables:**
- Modified `chunking.py`
- Type-aware strategies working
- Zero oversized chunks from TOC/LOF

---

### Phase 4: Structured Storage (2 weeks) - **NOT IMPLEMENTED**

**Tasks:**
1. - [ ] Design document_structure schema
2. - [ ] Implement storage layer
3. - [ ] Add structure extraction to workflow
4. - [ ] Create query API for structure
5. - [ ] Add UI for structure navigation
6. - [ ] Write migration for existing docs

**Deliverables:**
- Database schema
- Storage implementation
- Query API
- Documentation

---

## Expected Benefits

### Immediate (Phase 1-2)

1. **Zero chunking failures from TOC/LOF**
   - Current: 1-2 oversized chunks per document with TOC
   - After: 0 failures

2. **Reduced embedding costs**
   - Skip page numbers, footers, TOC formatting
   - Estimate: 10-15% reduction in chunks

3. **Better chunk quality**
   - No page numbers mixed with content
   - No TOC dots cluttering text
   - Cleaner semantic units

### Medium-term (Phase 3)

4. **Table handling**
   - Tables kept as single chunks (if small)
   - Large tables: caption only
   - No mid-table chunk breaks

5. **List preservation**
   - Related list items stay together
   - Better semantic coherence

6. **Header context**
   - Headers can inform chunk boundaries
   - Section-aware chunking possible

### Long-term (Phase 4)

7. **Document navigation**
   - Query by section: "What's in Section 3?"
   - Find figures: "Show all figures about X"
   - Jump to page: "Go to the methodology section"

8. **Advanced RAG queries**
   - "Which section discusses topic Y?" → Check structure
   - "Are there tables about Z?" → Query List of Tables
   - "What's the document outline?" → Return TOC

9. **Analytics**
   - Track which sections get queried most
   - Identify documents by structure
   - Understand content organization

---

## Risks & Mitigations

### Risk 1: Breaking Changes

**Risk:** New element-level approach breaks existing code

**Mitigation:**
- Feature flag for gradual rollout
- Dual-mode support (page-level and element-level)
- Extensive testing before enable
- Rollback plan

### Risk 2: Performance Impact

**Risk:** More elements = slower processing

**Mitigation:**
- Batch processing where possible
- Optimize filtering logic
- Monitor processing times
- Set performance budgets

### Risk 3: Type Detection Accuracy

**Risk:** TOC/LOF detection has false positives/negatives

**Mitigation:**
- High-precision patterns (95%+ accuracy)
- Manual review of first 100 docs
- Logging for monitoring
- Feedback mechanism

### Risk 4: Storage Growth

**Risk:** Storing structure separately increases DB size

**Mitigation:**
- Structure is small (few KB per doc)
- Optional (can disable if not needed)
- Compression for JSON storage
- Archive old structure data

---

## Success Metrics

### Technical Metrics

| Metric | Before | Target | Measurement |
|--------|--------|--------|-------------|
| Oversized chunks (TOC/LOF) | 1-2 per doc | 0 | Chunking logs |
| Total chunks per doc | Baseline | -10% to -15% | Chunk counts |
| Processing time | Baseline | <+5% | Task duration |
| TOC/LOF detection accuracy | N/A | >95% | Manual review |

### Quality Metrics

| Metric | Before | Target | Measurement |
|--------|--------|--------|-------------|
| Chunk coherence | Baseline | +20% | Semantic similarity |
| Embedding quality | Baseline | +10% | Retrieval accuracy |
| Structure extraction | N/A | >90% | Manual review |
| User satisfaction | Baseline | +15% | Survey |

---

## Conclusion

**Current problem:** Fileintel discards MinerU's rich element-level type information, causing TOC/LOF chunking failures and sub-optimal RAG quality.

**Root cause:** Architecture concatenates all text per page, losing type association.

**Solution:** Preserve element types, filter non-RAG content (TOC/LOF/footers), extract structure, and chunk type-appropriately.

**Impact:**
- Immediate: Fixes TOC/LOF chunking failures
- Medium-term: Better chunk quality, lower costs
- Long-term: Advanced document navigation and structure-aware RAG

**Effort:** 3-4 weeks for full implementation (Phases 1-4)

**Recommendation:** Implement Phases 1-2 immediately (1.5 weeks) to fix TOC/LOF issue, then evaluate if Phases 3-4 are needed based on results.

---

**Status:** Ready for implementation - detailed plan provided.
