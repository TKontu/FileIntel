# Page-Aware Chunking Implementation Plan

## Current Architecture Issues
- PDF processor extracts page-level text but loses boundaries during concatenation
- Post-processing text mapping is fragile and inaccurate
- No intelligent boundary handling for semantic coherence

## New Architecture: Page-Aware Chunking at Source

### 1. Core Design Principles
- **Clean separation**: PDF processor handles page extraction, chunker handles semantic chunking
- **Source-level awareness**: Chunks inherit page numbers from creation, no post-mapping
- **Intelligent boundaries**: Preserve semantic coherence while maintaining citation accuracy
- **Simple metadata**: Use `pages: [5]` or `pages: [5, 6]` for multi-page content

### 2. Page Numbering Strategy
**Problem**: Books/publications often have:
- Cover pages, table of contents, prefaces (roman numerals: i, ii, iii)
- Content pages (arabic numerals: 1, 2, 3)
- Different numbering schemes in same document

**Solution**: Coordinate-based header/footer extraction
```json
{
  "physical_page": 15,      // Actual PDF page (always sequential)
  "logical_page": "vii",    // Printed page number from header/footer
  "header_text": "Chapter 2: Introduction",
  "footer_text": "vii"     // Where page number was found
}
```

**Page Assignment Priority**:
1. Use `logical_page` if found in header/footer
2. Fall back to `physical_page` (PDF page number)
3. Keep it simple - no complex content detection

### 3. Implementation Components

#### A. Enhanced PDF Processor
**File**: `src/fileintel/document_processing/processors/traditional_pdf.py`

**Changes**:
- Modify `_extract_text_traditional()` to extract header/footer text
- Use coordinate-based extraction (top 50px, bottom 50px regions)
- Apply simple regex patterns to find page numbers
- Return enhanced page metadata per element

**Page Number Detection**:
```python
def extract_page_numbers(self, pdf):
    """Extract page numbers from headers/footers using coordinates."""
    page_mapping = {}

    for i, page in enumerate(pdf.pages):
        physical_page = i + 1

        # Extract header (top 50px) and footer (bottom 50px)
        header_text = page.within_bbox((0, 0, page.width, 50)).extract_text() or ""
        footer_text = page.within_bbox((0, page.height-50, page.width, page.height)).extract_text() or ""

        # Look for page number patterns
        logical_page = self.find_page_number(header_text + " " + footer_text)

        page_mapping[physical_page] = {
            "logical_page": logical_page or str(physical_page),
            "header": header_text.strip(),
            "footer": footer_text.strip()
        }

    return page_mapping

def find_page_number(self, text):
    """Find page number in header/footer text using simple patterns."""
    import re
    patterns = [
        r'\b([ivxlcdm]+)\b',           # Roman numerals: i, ii, iii, iv, v
        r'\b(\d+)\b',                  # Arabic numbers: 1, 2, 3
        r'Page\s+(\d+)',               # "Page 1", "Page 2"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None
```

#### B. Page-Aware TextChunker
**File**: `src/fileintel/document_processing/chunking.py`

**New Method**: `chunk_page_elements(elements: List[DocumentElement])`
- Process elements with page context preserved
- Handle page boundaries intelligently
- Return chunks with accurate page assignments

**Intelligent Boundary Logic**:
1. **Sentence-level**: If sentence spans pages, include all pages
2. **Paragraph-level**: Keep paragraphs together, assign to all involved pages
3. **Semantic coherence**: Don't split concepts mid-thought
4. **Reference preservation**: Keep citations with context

#### C. Updated Document Tasks
**File**: `src/fileintel/tasks/document_tasks.py`

**Changes**:
- `read_document_content()` returns page-aware elements (not concatenated text)
- `clean_and_chunk_text()` becomes `clean_and_chunk_elements()`
- Process elements preserving page context throughout

### 4. Boundary Handling Strategies

#### Intelligent Spanning Rules:
- **Complete entities**: Keep sentences/paragraphs/concepts complete
- **Multi-page assignment**: Assign all involved pages to spanning content
- **Context preservation**: Include sufficient context for understanding

#### Boundary Types:
```json
{
  "boundary_type": "complete",      // Chunk fits entirely within pages
  "boundary_type": "sentence_span", // Sentence spans multiple pages
  "boundary_type": "concept_span"   // Logical concept spans multiple pages
}
```

### 5. Metadata Structure

#### Simplified Chunk Metadata:
```json
{
  "position": 12,
  "pages": [4, 5],                    // Logical pages (from headers/footers)
  "physical_pages": [18, 19],         // Actual PDF pages (fallback)
  "header_text": "Chapter 2",        // Context from headers
  "footer_text": "4",                // Where page number was found
  "boundary_type": "paragraph_span", // How boundary was handled
  "extraction_method": "traditional"
}
```

#### Citation Support:
- Single page: "Found on page 4"
- Multi-page: "Found on pages 4-5"
- Roman numerals: "Found on page vii" (from header/footer extraction)
- Physical fallback: "Found on PDF page 18" (when no logical page found)

### 6. Implementation Steps

#### Phase 1: Coordinate-Based Page Detection
1. Modify PDF processor to extract header/footer regions using coordinates
2. Apply simple regex patterns to find page numbers (roman/arabic)
3. Build physical_page → logical_page mapping
4. Test with various PDF types (academic, books, reports)

#### Phase 2: Page-Aware Chunking
1. Create `chunk_page_elements()` method in TextChunker
2. Implement intelligent boundary handling for page spans
3. Preserve page context throughout chunking process
4. Test semantic coherence vs citation accuracy

#### Phase 3: Integration and Testing
1. Update document tasks to use page-aware processing
2. Modify chunk storage to handle enhanced metadata
3. Test end-to-end with real documents
4. Validate citation accuracy and semantic quality

#### Phase 4: EPUB/MOBI Support
1. Extend chapter-aware chunking for ebooks (chapters → page equivalents)
2. Handle chapter boundaries intelligently
3. Map chapters to citation-friendly metadata

### 7. Test Cases

#### Document Types:
- Academic papers (simple 1,2,3 numbering)
- Books (roman preface + arabic content)
- Technical manuals (section-based numbering)
- Legal documents (complex numbering schemes)

#### Boundary Scenarios:
- Sentences spanning pages
- Paragraphs spanning pages
- Mathematical formulas across pages
- Tables and figures with references
- Lists and enumerations

#### Expected Outcomes:
- 95%+ chunks have accurate page assignments
- No semantic concepts split inappropriately
- Citations are precise and reliable
- Processing handles various numbering schemes

### 8. Backwards Compatibility

#### Migration Strategy:
- Keep existing `chunk_text()` method for backward compatibility
- New `chunk_page_elements()` for enhanced functionality
- Document tasks detect and use appropriate method
- Gradual migration of existing workflows

#### Fallback Behavior:
- If page detection fails, use physical page numbers
- If boundary detection fails, use simple sentence boundaries
- Always preserve some page information, even if incomplete

---

## Next Steps
1. Review plan for architectural soundness
2. Implement Phase 1: Enhanced page detection
3. Test page numbering detection with sample documents
4. Proceed with page-aware chunking implementation