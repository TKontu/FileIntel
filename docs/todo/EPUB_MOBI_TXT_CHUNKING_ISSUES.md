# EPUB, MOBI, and TXT File Processing Issues

**Status:** Bug Identified - Not Fixed
**Priority:** Medium
**Impact:** Chapter metadata lost for ebooks, citations incomplete
**Affected Files:** `src/fileintel/document_processing/chunking.py`

---

## Summary

The chunking system has a critical bug that **drops all chapter/section metadata** for EPUB, MOBI, and plain text files. This occurs because the metadata collection logic only executes when `page_number` exists, which is only true for PDF files.

**Result:** Citations and search results for ebooks are incomplete and lack chapter information.

---

## Issue #1: Chapter Metadata Lost During Chunking

### Root Cause

**File:** `src/fileintel/document_processing/chunking.py`
**Lines:** 165-186

```python
for page_info in page_mappings:
    page_start = page_info.get('start_pos', 0)
    page_end = page_info.get('end_pos', 0)
    page_number = page_info.get('page_number')

    # ❌ BUG: This condition excludes all non-PDF files
    if page_number and sentence_start < page_end and sentence_end > page_start:
        pages_involved.add(page_number)

        # Collect metadata from this page
        if page_info.get('extraction_method'):
            extraction_methods.append(page_info['extraction_method'])

        if page_info.get('section_title'):
            section_titles.append(page_info['section_title'])

        if page_info.get('section_path'):
            section_paths.append(page_info['section_path'])

        if page_info.get('markdown_headers'):
            all_headers.extend(page_info['markdown_headers'])
```

### Why This Fails

| File Type | Metadata Available | Bug Behavior |
|-----------|-------------------|--------------|
| **PDF** | `page_number: 15` | ✅ Condition passes → metadata collected |
| **EPUB** | `chapter: "Chapter 3"` | ❌ `page_number` is `None` → condition fails → **metadata LOST** |
| **MOBI** | `part: "part0003.html"` | ❌ `page_number` is `None` → condition fails → **metadata LOST** |
| **TXT** | `source: "notes.txt"` | ❌ `page_number` is `None` → condition fails → **metadata LOST** |
| **MD** | `source: "README.md"` | ❌ `page_number` is `None` → condition fails → **metadata LOST** |

### Impact

1. **Incomplete Citations**
   - PDF: `(Davis, 2019, p. 15)` ✓
   - EPUB: `(Davis, 2019)` ✗ (should show chapter)
   - Expected: `(Davis, 2019, Chapter 3)`

2. **Search Results Lack Context**
   - Users can't see which chapter content came from
   - No way to filter by chapter/section
   - Poor user experience for ebook citations

3. **Metadata Loss Cascade**
   - `section_title` not collected
   - `section_path` not collected
   - `markdown_headers` not collected
   - `extraction_method` not tracked

### Proposed Fix

Replace the `if page_number` condition with position-based checking:

```python
# Check if sentence overlaps with this page/section (regardless of page_number)
if sentence_start < page_end and sentence_end > page_start:
    # Collect page number if available (PDFs)
    if page_number:
        pages_involved.add(page_number)

    # Collect chapter info if available (EPUBs)
    chapter = page_info.get('chapter')
    if chapter:
        chapters_involved.add(chapter)

    # Collect part info if available (MOBIs)
    part = page_info.get('part')
    if part:
        parts_involved.add(part)

    # Always collect other metadata
    if page_info.get('extraction_method'):
        extraction_methods.append(page_info['extraction_method'])

    if page_info.get('section_title'):
        section_titles.append(page_info['section_title'])

    # ... etc
```

---

## Issue #2: MOBI Metadata Not Extracted

### Root Cause

**File:** `src/fileintel/document_processing/processors/mobi_processor.py`
**Line:** 38

```python
elements = []
doc_metadata = {}  # ❌ Empty - no metadata extraction implemented
```

### Comparison

| Processor | Metadata Extraction |
|-----------|---------------------|
| **PDF** | ✅ Full (via MinerU or traditional extraction) |
| **EPUB** | ✅ Full (title, authors, publisher, date, language, ISBN) |
| **MOBI** | ❌ **Not implemented** |
| **TXT** | ⚠️ N/A (no standard metadata) |

### EPUB Metadata Implementation (for reference)

```python
# From epub_processor.py:41-66
doc_metadata = {
    "title": book.get_metadata("DC", "title")[0][0],
    "authors": [author[0] for author in book.get_metadata("DC", "creator")],
    "publisher": book.get_metadata("DC", "publisher")[0][0],
    "publication_date": book.get_metadata("DC", "date")[0][0],
    "language": book.get_metadata("DC", "language")[0][0],
    "identifier": book.get_metadata("DC", "identifier")[0][0],
}
```

### Impact

- MOBI files can't be properly cited (no author/title/year)
- Fallback to filename only: `(book-file.mobi)` instead of `(Author, Year)`
- Metadata extraction task won't find anything
- Citations incomplete

### Proposed Fix

The `mobi` library may not provide easy metadata access. Options:

1. **Extract from unpacked OPF file** (MOBI metadata is in OPF XML)
2. **Use calibre's ebook-meta tool** (if available)
3. **Document limitation** and recommend converting MOBI to EPUB first

---

## Issue #3: TXT/MD Structure Not Preserved

### Root Cause

**File:** `src/fileintel/document_processing/processors/text_processor.py`
**Lines:** 24-28

```python
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()
elements = [TextElement(text=text, metadata={"source": str(file_path)})]
return elements, {}  # ❌ No structure detection
```

### Current Behavior

- Entire file read as single blob of text
- No markdown header detection
- No section parsing
- No structure metadata

### Example

**Input (notes.md):**
```markdown
# Chapter 1: Introduction

This is the introduction.

# Chapter 2: Methodology

This is the methodology.
```

**Current Processing:**
- 1 giant TextElement
- No chapter boundaries detected
- Citations show filename only

**Desired Processing:**
- 2 TextElements (one per chapter)
- Each with `section_title` metadata
- Citations can reference sections

### Impact

- Markdown files lose all structure
- Can't cite specific sections
- Poor chunking boundaries (may split mid-section)
- No header-based search filtering

### Proposed Fix

Add markdown structure detection:

```python
def read(self, file_path: Path, adapter: logging.LoggerAdapter = None):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Detect if markdown
    if file_path.suffix.lower() == '.md':
        elements = self._parse_markdown_structure(text, file_path)
    else:
        # Plain text - single element
        elements = [TextElement(text=text, metadata={"source": str(file_path)})]

    return elements, {}

def _parse_markdown_structure(self, text: str, file_path: Path):
    """Split markdown by headers (# Header)"""
    import re

    # Find all markdown headers
    header_pattern = r'^(#{1,6})\s+(.+)$'
    sections = []
    current_section = {"level": 0, "title": "Introduction", "content": []}

    for line in text.split('\n'):
        header_match = re.match(header_pattern, line)
        if header_match:
            # Save previous section
            if current_section["content"]:
                sections.append(current_section)

            # Start new section
            level = len(header_match.group(1))
            title = header_match.group(2)
            current_section = {"level": level, "title": title, "content": []}
        else:
            current_section["content"].append(line)

    # Save final section
    if current_section["content"]:
        sections.append(current_section)

    # Convert to TextElements
    elements = []
    for section in sections:
        text = '\n'.join(section["content"])
        if text.strip():
            metadata = {
                "source": str(file_path),
                "section_title": section["title"],
                "section_level": section["level"]
            }
            elements.append(TextElement(text=text, metadata=metadata))

    return elements
```

---

## Testing Status

### ✅ Processors Work
- EPUB reading: ✓ (tested with ebooklib)
- MOBI reading: ✓ (tested with mobi library)
- TXT reading: ✓ (UTF-8 encoding works)
- MD reading: ✓ (same as TXT currently)

### ❌ Chunking Fails
- EPUB chunks: Missing chapter info
- MOBI chunks: Missing part info
- TXT chunks: Missing section info
- MD chunks: Missing header info

### ❌ Citations Incomplete
- EPUB: No chapter reference
- MOBI: No metadata at all
- TXT/MD: Filename only

---

## Recommendation

### Priority 1: Fix Chunking Bug (High Impact)
**Effort:** 2-3 hours
**Benefit:** Fixes all ebook citations immediately

1. Update `chunking.py:165-186` to handle non-page metadata
2. Add chapter/part/section tracking alongside pages
3. Update chunk metadata to include chapter/section fields
4. Test with EPUB/MOBI/TXT files

### Priority 2: Add MOBI Metadata (Medium Impact)
**Effort:** 3-4 hours
**Benefit:** Proper citations for MOBI files

1. Extract metadata from unpacked OPF file
2. Match EPUB metadata structure
3. Test with various MOBI formats

### Priority 3: Add Markdown Structure (Low-Medium Impact)
**Effort:** 4-5 hours
**Benefit:** Better chunking and citations for documentation

1. Implement markdown header parser
2. Create sections from headers
3. Add section metadata to elements
4. Test with various markdown formats

---

## Related Files

- `src/fileintel/document_processing/chunking.py` - Main bug location
- `src/fileintel/document_processing/processors/epub_processor.py` - EPUB handling
- `src/fileintel/document_processing/processors/mobi_processor.py` - MOBI handling (needs metadata)
- `src/fileintel/document_processing/processors/text_processor.py` - TXT/MD handling (needs structure)
- `src/fileintel/citation/citation_formatter.py` - Citation formatting (works, but gets no chapter data)

---

## Questions for User

1. **Fix Priority:** Should we fix the chunking bug immediately, or document and defer?
2. **MOBI Support:** Is MOBI important, or should we recommend converting to EPUB?
3. **Markdown Structure:** Do you use markdown files that need section-based citations?
4. **Citation Format:** How should chapter citations look?
   - `(Davis, 2019, Chapter 3)` ?
   - `(Davis, 2019, Ch. 3)` ?
   - `(Davis, 2019, "Chapter Title")` ?

---

**Last Updated:** 2025-11-14
**Discovered By:** Code review during TXT/MD format addition
