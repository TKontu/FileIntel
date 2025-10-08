# Markdown Metadata Enhancement

## Overview

Successfully implemented markdown header extraction and metadata enrichment for MinerU-processed PDFs. This enhancement adds structural context to chunks without polluting embeddings with markdown syntax.

## Implementation Summary

### Key Features

1. **Markdown Header Extraction**: Extracts H1-H6 headers from MinerU markdown output with regex pattern matching
2. **Perfect Page Mapping**: Maps headers to exact pages using `page_idx` from MinerU's `content_list.json`
3. **Hierarchical Context**: Tracks section titles and hierarchical paths through the document
4. **Complete Metadata Flow**: Preserves header information from extraction → elements → chunks → storage

### Architecture

The enhancement follows this flow:

```
MinerU Output (markdown + JSON)
    ↓
Header Extraction (_extract_markdown_headers)
    ↓
Page Mapping (_map_headers_to_pages using content_list.json)
    ↓
TextElement Metadata (section_title, section_path, markdown_headers)
    ↓
Page Mappings (document_tasks.py)
    ↓
Sentence Page Info (_find_sentence_pages in chunking.py)
    ↓
Chunk Page Info (aggregated in create_vector/graph_chunks)
    ↓
Storage Metadata (preserved in DocumentChunk.chunk_metadata)
```

### Files Modified

#### 1. `src/fileintel/document_processing/processors/mineru_selfhosted.py`

**Added Methods**:
- `_extract_markdown_headers()` - Extracts headers using regex `r'^(#{1,6})\s+(.+)`
- `_map_headers_to_pages()` - Maps extracted headers to page numbers using content_list.json

**Updated Methods**:
- `_create_elements_from_json()` - Calls header extraction and adds to TextElement metadata

**New Metadata Fields in TextElements**:
- `markdown_headers` - List of header objects with level, text, type
- `section_title` - Primary section title (highest level header on page)
- `section_path` - Hierarchical path (e.g., "Document > Introduction > Methods")

#### 2. `src/fileintel/tasks/document_tasks.py`

**Updated Page Mappings** (lines 122-131):
Added extraction of new metadata fields from element metadata:
- `section_title`
- `section_path`
- `markdown_headers`

**Updated Vector Chunk Conversion** (lines 175-202):
Expanded metadata preservation to include all enhanced fields from page_info

**Updated Traditional Chunking** (lines 228-296):
Added collection and aggregation of enhanced metadata in traditional chunking path

#### 3. `src/fileintel/document_processing/chunking.py`

**Updated `_find_sentence_pages()`** (lines 146-216):
Enhanced to collect metadata from overlapping pages:
- `extraction_methods` - List of extraction methods used
- `section_title` - Primary section title
- `section_path` - Hierarchical section path
- `markdown_headers` - Deduplicated list of headers

**Updated `create_vector_chunks_from_sentences()`** (lines 251-316):
Added aggregation of enhanced metadata from sentences:
- Collects all metadata from sentence page_info
- Deduplicates headers by text
- Uses first/primary values for section_title and section_path

**Updated `create_graph_chunks_from_vector_chunks()`** (lines 365-430):
Same enhancements as vector chunks for graph chunk metadata

## Metadata Structure

### In TextElement
```python
metadata = {
    'source': '/path/to/file.pdf',
    'page_number': 1,
    'extraction_method': 'mineru_selfhosted_json',
    'format': 'structured_json',
    'element_count': 15,
    'element_types': {'text': 12, 'title': 3},
    'has_coordinates': True,
    'coordinate_coverage': 0.95,
    # Enhanced fields:
    'markdown_headers': [
        {'level': 1, 'text': 'Introduction', 'type': 'h1', 'line_number': 0},
        {'level': 2, 'text': 'Background', 'type': 'h2', 'line_number': 5}
    ],
    'section_title': 'Introduction',
    'section_path': 'Document > Introduction'
}
```

### In Chunk Metadata
```python
chunk_metadata = {
    'position': 0,
    'chunk_type': 'vector',
    'pages': [1, 2],
    'page_range': '1-2',
    'token_count': 350,
    'sentence_count': 4,
    # Enhanced fields:
    'extraction_methods': ['mineru_selfhosted_json'],
    'section_title': 'Introduction',
    'section_path': 'Document > Introduction',
    'markdown_headers': [
        {'level': 1, 'text': 'Introduction', 'type': 'h1'},
        {'level': 2, 'text': 'Background', 'type': 'h2'}
    ]
}
```

## Testing

### Test Suite: `scripts/test_markdown_metadata.py`

Three comprehensive tests validate the implementation:

1. **Header Extraction Test**: Validates regex extraction of H1-H6 headers
2. **Header to Page Mapping Test**: Validates correlation with content_list.json
3. **Chunk Metadata Flow Test**: Validates end-to-end metadata preservation

**All tests passing** ✓

### Test Results
```
✓ PASS: Header Extraction
✓ PASS: Header to Page Mapping
✓ PASS: Chunk Metadata Flow

Total: 3/3 tests passed
```

## Benefits

### For RAG Retrieval
- **Better Context**: Chunks include document structure information
- **Section-Aware Search**: Can filter/boost by section titles
- **Hierarchical Understanding**: Full document path preserved

### For GraphRAG
- **Relationship Context**: Headers help identify entity relationships within sections
- **Community Detection**: Section boundaries aid in community identification
- **Entity Resolution**: Section titles provide disambiguation context

### For Embeddings
- **Clean Text**: Plain text for embeddings (no markdown syntax)
- **Rich Metadata**: Structure preserved separately in metadata
- **Token Efficiency**: No wasted tokens on markdown formatting

## Backward Compatibility

✓ Fully backward compatible with existing code:
- Optional fields (won't break if missing)
- Traditional chunking path updated
- Two-tier chunking path updated
- Storage layer already configured to preserve all metadata

## Usage Example

```python
from fileintel.document_processing.processors.mineru_selfhosted import MinerUSelfHostedProcessor

processor = MinerUSelfHostedProcessor()
elements, metadata = processor.read(Path("document.pdf"))

# TextElements now have enhanced metadata
for elem in elements:
    if 'section_title' in elem.metadata:
        print(f"Section: {elem.metadata['section_title']}")
    if 'markdown_headers' in elem.metadata:
        for header in elem.metadata['markdown_headers']:
            print(f"  [{header['type']}] {header['text']}")

# Chunks will automatically inherit this metadata
from fileintel.tasks.document_tasks import clean_and_chunk_text, combine_elements

text, page_mappings = combine_elements(elements)
chunks = clean_and_chunk_text(text, page_mappings=page_mappings)

# Access enhanced metadata in chunks
for chunk in chunks:
    metadata = chunk['metadata']
    if 'section_title' in metadata:
        print(f"Chunk from section: {metadata['section_title']}")
```

## Future Enhancements

Potential future improvements:
- Extract table captions and figure references from markdown
- Add formula/equation metadata from MinerU's formula extraction
- Link chunks to specific images via MinerU's image references
- Build table of contents from header hierarchy
