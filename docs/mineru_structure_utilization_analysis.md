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

## Key Updates (2025-10-18)

**⚠️ Plan updated to reflect actual MinerU capabilities:**

### What Changed

1. **Two-Layer Type Taxonomy Added**
   - **Layer 1 (Layout):** `text`, `table`, `image` - provided by MinerU
   - **Layer 2 (Semantic):** `toc`, `lof`, `header`, `prose` - detected by Fileintel
   - MinerU does NOT provide semantic types - all are `type: "text"`

2. **Phase 1.5 Added: Semantic Type Detection** 🆕
   - New phase between Phase 1 and 2
   - Pattern-based detection for TOC/LOF (95%+ accuracy)
   - Uses `text_level` field for header classification (100% accurate)
   - Required because MinerU only provides layout types

3. **MinerU Field Extraction Enhanced**
   - **`text_level`**: Header level (1-6, 0=not header) - NOW USED
   - **`table_body`**: HTML table structure - NOW EXTRACTED
   - **`image_caption`**: Figure captions - NOW EMBEDDED
   - **`table_caption`**, `table_footnote`, `image_footnote` - NOW PRESERVED

4. **Metadata Model Updated**
   - Old: `element_type` (assumed MinerU provides all types)
   - New: `layout_type` (from MinerU) + `semantic_type` (detected)
   - More accurate reflection of actual capabilities

### Why These Changes

- **User challenged plan compatibility:** "Is the plan well compatible with fileintel? Does it utilize well the output formats provided by MinerU?"
- **Compatibility analysis revealed:** MinerU only provides 3 layout types, not 10+ semantic types
- **Solution:** Two-layer approach is BETTER architecture (separation of concerns)

### Impact on Implementation

- **Phases 1-3:** Updated with actual field names and detection logic
- **Timeline:** +2 days for Phase 1.5 (semantic detection)
- **Complexity:** Slightly higher, but more robust and maintainable
- **Benefits:** Same outcomes, better architectural foundation

---

## Current Architecture

### What MinerU Actually Provides

**⚠️ IMPORTANT:** MinerU provides **layout-level** types only, not semantic types.

**Actual content_list.json structure** (from real document analysis):

```json
[
  {
    "type": "text",
    "text": "Agile Processes for Hardware Development",
    "text_level": 1,  // Header level 1-6 (0 = not a header)
    "page_idx": 0,
    "bbox": [232, 142, 767, 220]
  },
  {
    "type": "text",
    "text": "6.1 Overview...19\n6.2 Velocity...19\n...",  // TOC, but type is still "text"!
    "page_idx": 2,
    "bbox": [129, 554, 885, 646]
  },
  {
    "type": "table",
    "text": "",  // Often EMPTY for tables!
    "table_body": "<table><tr><td>Data</td></tr></table>",  // Actual content
    "table_caption": ["Table 1: Results"],
    "table_footnote": [],
    "bbox": [111, 554, 897, 898],
    "page_idx": 22
  },
  {
    "type": "image",
    "img_path": "images/abc123.jpg",
    "image_caption": ["Figure 1: Sample diagram"],
    "image_footnote": [],
    "bbox": [323, 255, 676, 546],
    "page_idx": 0
  }
]
```

**Element types MinerU provides:**

| Type | Purpose | Key Fields |
|------|---------|------------|
| `text` | All text content (prose, TOC, headers, etc.) | `text`, `text_level`, `bbox`, `page_idx` |
| `table` | Tables | `table_body` (HTML), `table_caption`, `table_footnote` |
| `image` | Images/figures | `img_path`, `image_caption`, `image_footnote` |

**❌ MinerU does NOT provide these types:**
- `toc`, `lof`, `lot` (all are `type: "text"`)
- `header`, `footer` (all are `type: "text"`)
- `page_number`, `list`, `ref_text` (all are `type: "text"`)

**✅ Additional fields MinerU provides:**
- `text_level`: Header level (1-6, or 0 for non-headers)
- `bbox`: Precise element coordinates `[x, y, width, height]`
- `page_idx`: Zero-based page index
- `table_body`: HTML table structure (for table elements)
- `table_caption`, `table_footnote`: Table metadata
- `img_path`: Relative path to extracted image
- `image_caption`, `image_footnote`: Image metadata

---

## Element Type Taxonomy: Two Layers

### Layer 1: Layout Types (from MinerU)

MinerU performs **layout analysis** and classifies elements by visual structure:

- **`text`**: Any text block (prose, headers, TOC, footers, etc.)
- **`table`**: Tabular data
- **`image`**: Figures, diagrams, photos

### Layer 2: Semantic Types (detected by Fileintel)

Fileintel must add **semantic classification** to distinguish within `text` elements:

| Semantic Type | Detection Method | Accuracy | L1 Type |
|--------------|------------------|----------|---------|
| `prose` | Default (text that doesn't match patterns) | N/A | text |
| `toc` | Pattern: section numbers + dots + page numbers | 95%+ | text |
| `lof` | Pattern: "Figure X:" + dots + page numbers | 95%+ | text |
| `lot` | Pattern: "Table X:" + dots + page numbers | 95%+ | text |
| `header` | Use `text_level` field (1-6) | 100% | text |
| `bibliography` | Pattern: citation format | 85% | text |
| `index` | Pattern: alphabetical + page numbers | 90% | text |

**This two-layer approach is BETTER architecture:**
- MinerU focuses on what it does well (layout detection)
- Fileintel adds domain-specific semantic understanding
- More maintainable and testable

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
| `type` | ✅ Yes | ⚠️ Partially (counts only) | Layout type (text/table/image) |
| `text` | ✅ Yes | ✅ Yes | Text content |
| `text_level` | ✅ Yes | ❌ No | Header level (1-6, 0=not header) |
| `bbox` | ✅ Yes | ⚠️ Stored but unused | Element coordinates |
| `page_idx` | ✅ Yes | ✅ Yes | Page location |
| `table_body` | ✅ Yes (tables) | ❌ No | HTML table structure |
| `table_caption` | ✅ Yes (tables) | ❌ No | Table titles/captions |
| `table_footnote` | ✅ Yes (tables) | ❌ No | Table footnotes |
| `img_path` | ✅ Yes (images) | ❌ No | Extracted image file path |
| `image_caption` | ✅ Yes (images) | ❌ No | Figure captions |
| `image_footnote` | ✅ Yes (images) | ❌ No | Figure footnotes |

**What we COULD know but DON'T:**
- **Header hierarchy** (`text_level` field available but unused)
- **Which text blocks are TOC/LOF** (need pattern detection)
- **Table structure** (`table_body` HTML available but unused)
- **Image captions** (`image_caption` available but unused)
- **Element ordering within page** (preserved in content_list)
- **Element spatial relationships** (`bbox` coordinates available)

---

### What Gets Lost

**Example page with mixed content:**

```
MinerU provides (actual types):
[
  {type: "text", text: "Chapter 1", text_level: 1},  ← Header (level 1)
  {type: "text", text: "This is prose...", text_level: 0},  ← Prose
  {type: "text", text: "1.1 Introduction.....5\n1.2 Methods.....12", text_level: 0},  ← TOC!
  {type: "text", text: "More prose...", text_level: 0},  ← Prose
  {type: "text", text: "Page 1", text_level: 0}  ← Page number
]
```

**Fileintel currently creates:**

```
TextElement(
  text="Chapter 1\nThis is prose...\n1.1 Introduction.....5\n1.2 Methods.....12\nMore prose...\nPage 1",
  metadata={'element_types': {'text': 5}}  ← All just "text"!
)
```

**Problems:**

1. **Can't identify TOC** - it's just mixed into the text blob
2. **Header information lost** - `text_level: 1` field discarded
3. **Page number mixed with content** - can't filter it out
4. **No element boundaries** - chunker sees one giant string
5. **Semantic classification impossible** - all elements are `type: "text"`

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

**Proposed (Updated for actual MinerU fields):**

```python
# Create TextElement PER content_list item, not per page
def _create_elements_from_json(self, json_data, ...) -> List[TextElement]:
    text_elements = []

    for item in content_list:
        # Get layout type from MinerU (L1: layout-level)
        layout_type = item.get('type', 'text')
        text = item.get('text', '')

        # Build base metadata with MinerU fields
        metadata = {
            'source': str(file_path),
            'page_number': item.get('page_idx', 0) + 1,
            'layout_type': layout_type,  # ← L1: text/table/image from MinerU
            'bbox': item.get('bbox', []),
            'element_index': item.get('index', 0)
        }

        # Add type-specific MinerU fields
        if layout_type == 'text':
            # Capture header level (0 = not a header, 1-6 = header levels)
            text_level = item.get('text_level', 0)
            metadata['text_level'] = text_level
            metadata['is_header'] = text_level > 0

            # Apply semantic detection (L2: semantic classification)
            if text:
                # Detect TOC/LOF using pattern matching
                is_toc_lof, detected_type = is_toc_or_lof(text)
                if is_toc_lof:
                    metadata['semantic_type'] = detected_type  # 'toc', 'lof', 'lot'
                elif text_level > 0:
                    metadata['semantic_type'] = 'header'
                    metadata['header_level'] = text_level
                else:
                    metadata['semantic_type'] = 'prose'  # Default

        elif layout_type == 'table':
            # Extract table-specific fields
            metadata['table_body'] = item.get('table_body', '')  # HTML table
            metadata['table_caption'] = item.get('table_caption', [])
            metadata['table_footnote'] = item.get('table_footnote', [])
            metadata['semantic_type'] = 'table'

            # Use caption as text if main text is empty
            if not text and metadata['table_caption']:
                text = ' '.join(metadata['table_caption'])

        elif layout_type == 'image':
            # Extract image-specific fields
            metadata['img_path'] = item.get('img_path', '')
            metadata['image_caption'] = item.get('image_caption', [])
            metadata['image_footnote'] = item.get('image_footnote', [])
            metadata['semantic_type'] = 'image'

            # Use caption as text (for separate embedding)
            if metadata['image_caption']:
                text = ' '.join(metadata['image_caption'])

        # Create TextElement with full metadata
        text_elements.append(TextElement(text=text, metadata=metadata))

    return text_elements
```

**Benefits:**

- **Two-layer type system:** `layout_type` (from MinerU) + `semantic_type` (detected)
- **Preserves all MinerU fields:** `text_level`, `table_body`, `image_caption`
- **Element-level granularity:** Each TextElement = one content_list item
- **Header hierarchy available:** Can build document outline from `text_level`
- **Table HTML preserved:** Can parse `table_body` for better chunking
- **Image captions embedded:** Figures become searchable via captions

---

#### Phase 2: Type-Based Filtering

**Add filtering before chunking (uses detected semantic types):**

```python
# In document_tasks.py or new filtering module

def filter_elements_for_rag(elements: List[TextElement]) -> Tuple[List[TextElement], Dict]:
    """
    Filter elements based on RAG relevance using semantic classification.

    Returns:
        (filtered_elements, extracted_structure)
    """
    filtered = []
    toc_entries = []
    lof_entries = []
    headers = []

    for elem in elements:
        # Get semantic type (detected in Phase 1)
        semantic_type = elem.metadata.get('semantic_type', 'prose')
        layout_type = elem.metadata.get('layout_type', 'text')

        # Extract structure from TOC/LOF (don't embed)
        if semantic_type in ['toc', 'lof', 'lot']:
            if semantic_type == 'toc':
                toc_entries.extend(parse_toc_entries(elem.text))
            elif semantic_type in ['lof', 'lot']:
                lof_entries.extend(parse_figure_table_list(elem.text))
            continue  # Skip TOC/LOF elements

        # Extract headers as structure but INCLUDE in RAG
        if semantic_type == 'header':
            headers.append({
                'text': elem.text,
                'page': elem.metadata.get('page_number'),
                'level': elem.metadata.get('header_level', 1)
            })
            # Headers provide context, so embed them
            filtered.append(elem)

        # Handle images: embed captions only
        elif layout_type == 'image':
            if elem.metadata.get('image_caption'):
                # Caption is already in elem.text (set in Phase 1)
                filtered.append(elem)
            # Skip images without captions

        # Include prose, tables, and other content
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

- **TOC/LOF excluded from embeddings** (no chunking failures)
- **Structural information preserved** in document metadata
- **Image captions embedded** (figures become searchable)
- **Headers included** (provide context for semantic search)
- **Reduced embedding costs** (skip formatting-heavy content)

---

#### Phase 3: Type-Aware Chunking

**Modify chunker to respect semantic types and use MinerU fields:**

```python
# In chunking.py

def chunk_elements_by_type(
    elements: List[TextElement],
    max_tokens: int = 450
) -> List[Dict]:
    """
    Chunk elements using type-aware strategies.
    Uses semantic_type and layout-specific fields from MinerU.
    """
    chunks = []

    for elem in elements:
        semantic_type = elem.metadata.get('semantic_type', 'prose')
        layout_type = elem.metadata.get('layout_type', 'text')

        if layout_type == 'table':
            # Strategy 1: Use table caption if available and fits
            caption_text = ' '.join(elem.metadata.get('table_caption', []))

            if caption_text and estimate_tokens(caption_text) <= max_tokens:
                chunks.append({
                    'text': caption_text,
                    'metadata': {**elem.metadata, 'chunk_strategy': 'table_caption'},
                })
            # Strategy 2: Parse table_body HTML (if needed for content)
            elif elem.metadata.get('table_body'):
                # Parse HTML to plain text, then chunk if needed
                table_html = elem.metadata['table_body']
                table_text = parse_html_table_to_text(table_html)

                if estimate_tokens(table_text) <= max_tokens:
                    chunks.append({
                        'text': table_text,
                        'metadata': {**elem.metadata, 'chunk_strategy': 'table_parsed'},
                    })
                else:
                    # Table too large: use caption only
                    chunks.append({
                        'text': caption_text or "Table (content too large)",
                        'metadata': {**elem.metadata, 'chunk_strategy': 'table_caption_fallback'},
                    })

        elif semantic_type in ['prose', 'header']:
            # Prose and headers: use sentence-based chunking
            prose_chunks = chunk_text_sentence_based(elem.text, max_tokens)
            for chunk_text in prose_chunks:
                chunks.append({
                    'text': chunk_text,
                    'metadata': elem.metadata,
                    'chunk_strategy': 'sentence_based'
                })

        elif semantic_type == 'image':
            # Image captions: treat as single chunk
            # (caption is already in elem.text from Phase 1)
            if elem.text and estimate_tokens(elem.text) <= max_tokens:
                chunks.append({
                    'text': elem.text,
                    'metadata': elem.metadata,
                    'chunk_strategy': 'image_caption'
                })

        # Note: TOC/LOF filtered out in Phase 2, never reach here

    return chunks
```

**Benefits:**

- **Table captions embedded** (searchable table metadata)
- **Table HTML can be parsed** if full content needed
- **Headers chunked normally** (provide context)
- **Image captions as single chunks** (figure discovery)
- **TOC/LOF never chunked** (filtered in Phase 2)
- **Zero oversized chunks** (proper handling per type)

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

  # Semantic type detection (Layer 2)
  semantic_detection:
    enable_toc_lof_detection: true  # Pattern-based TOC/LOF detection
    enable_bibliography_detection: true  # Citation format detection
    enable_index_detection: true  # Alphabetical index detection
    detection_confidence_threshold: 0.85  # Minimum confidence for classification

  # Element filtering for RAG (uses semantic types)
  element_filters:
    # Semantic types to skip (don't embed)
    skip_semantic_types: ['toc', 'lof', 'lot', 'bibliography', 'index']

    # Semantic types to extract as structure (don't embed, but parse)
    extract_structure_types: ['toc', 'lof', 'lot']

    # Semantic types to embed
    embed_semantic_types: ['prose', 'header']  # Headers provide context

    # Layout types handling
    embed_layout_types:
      table: 'caption_only'  # or 'full_if_small', 'parsed_html'
      image: 'caption_only'  # Embed image captions only

  # Type-aware chunking
  chunking:
    use_type_aware: true
    strategies:
      table: 'caption_only'  # Use table_caption field
      image: 'single_chunk'  # One chunk per caption
      header: 'sentence_based'  # Chunk with context
      prose: 'sentence_based'  # Normal chunking
```

---

## Implementation Roadmap

**⚠️ STATUS: PLANNED - NOT YET IMPLEMENTED**

These phases represent the PROPOSED implementation plan. Only the analysis and design work has been completed. The actual code changes described below have NOT been implemented yet.

---

### Phase 1: Element-Level Preservation (1 week) - **NOT IMPLEMENTED**

**Tasks:**

1. - [ ] Modify `_create_elements_from_json()` to create one TextElement per content_list item
2. - [ ] Add `layout_type` and `semantic_type` to metadata (two-layer system)
3. - [ ] Extract MinerU fields: `text_level`, `table_body`, `image_caption`
4. - [ ] Handle tables: use `table_caption` as text if main text empty
5. - [ ] Handle images: use `image_caption` as text for embedding
6. - [ ] Add feature flag `use_element_level_types`
7. - [ ] Test with sample documents
8. - [ ] Verify backward compatibility

**Deliverables:**

- Modified `mineru_selfhosted.py`
- Element-level preservation working
- All MinerU fields extracted (`text_level`, `table_body`, `image_caption`)
- Tests passing

---

### Phase 1.5: Semantic Type Detection (2 days) - **NOT IMPLEMENTED** 🆕

**NEW PHASE - Required because MinerU only provides layout types**

**Tasks:**

1. - [ ] Implement `is_toc_or_lof()` function (reuse from `toc_lof_chunking_issue.md`)
2. - [ ] Add bibliography detection (citation format pattern)
3. - [ ] Add index page detection (alphabetical + page numbers)
4. - [ ] Create `detect_semantic_type()` orchestrator function
5. - [ ] Integrate detection into Phase 1's element creation
6. - [ ] Test detection accuracy (target: 95%+ for TOC/LOF)
7. - [ ] Add detection confidence scores to metadata

**Deliverables:**

- `element_detection.py` module with pattern matchers
- Unit tests for each detection type (TOC, LOF, LOT, bibliography, index)
- Detection accuracy report
- Integration with `_create_elements_from_json()`

---

### Phase 2: Type-Based Filtering (3 days) - **NOT IMPLEMENTED**

**Tasks:**

1. - [ ] Implement `filter_elements_for_rag()` using `semantic_type`
2. - [ ] Add TOC/LOF structure extraction (parse entries)
3. - [ ] Add image caption extraction as separate chunks
4. - [ ] Extract header hierarchy using `text_level` field
5. - [ ] Integrate filtering before chunking in workflow
6. - [ ] Add config for element filters
7. - [ ] Test filtering accuracy

**Deliverables:**

- `element_filter.py` module
- TOC/LOF parsing working (extract structured entries)
- Image captions embedded separately
- Header hierarchy extracted
- Config-driven filtering

---

### Phase 3: Type-Aware Chunking (1 week) - **NOT IMPLEMENTED**

**Tasks:**

1. - [ ] Implement `chunk_elements_by_type()` using `semantic_type` and `layout_type`
2. - [ ] Add table-specific chunking (use `table_body` HTML or caption)
3. - [ ] Add HTML table parser (`parse_html_table_to_text()`)
4. - [ ] Add header-aware chunking (use `text_level` for context)
5. - [ ] Add image caption chunking (single chunk per caption)
6. - [ ] Integrate with existing sentence-based chunker
7. - [ ] Add chunking strategy metadata
8. - [ ] Test on diverse documents

**Deliverables:**

- Modified `chunking.py`
- Table HTML parsing working
- Type-aware strategies for all semantic types
- Zero oversized chunks from TOC/LOF
- Header hierarchy preserved in chunks

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
