# Chunking Issues Investigation

## Problem Statement

Chunks frequently exceed token limits (450 for vector RAG) causing embedding failures and data loss:

- **990 tokens**: Table data (Pugh Diagram) - should be ~200 tokens
- **8584 tokens**: Index section - should be split into ~20 chunks
- **10130 tokens**: Index section - should be split into ~23 chunks

### Example Failed Chunk

```
520 Automotive Product Development: A System Engineering Implementation
4.52
ELBAT          ← "TABLE" backwards
setubirttA     ← "Attributes" backwards
elciheV        ← "Vehicle" backwards
fo
margaiD
hguP
```

**Issues**:
1. Text is reversed (tables read right-to-left)
2. Table structure lost (cells concatenated)
3. No sentence boundaries
4. Treated as continuous prose

## Root Cause Analysis

### Current Flow

```
MinerU JSON Output
├─ content_list: [
│   {text: "...", type: "table", bbox: [...], page_idx: 0},
│   {text: "...", type: "text", bbox: [...], page_idx: 0},
│   {text: "...", type: "title", bbox: [...], page_idx: 1}
│  ]
└─ Structured metadata (types, coordinates, pages)

                    ↓ mineru_selfhosted.py:325

Elements Combined (LOSES STRUCTURE)
├─ page_text_parts.append(elem_info['text'])
└─ page_text = '\n'.join(page_text_parts)  ← ALL types treated equally!

                    ↓ document_tasks.py:136

Single Text Blob
└─ combined_text = " ".join(text_parts)  ← One big string, no structure

                    ↓ chunking.py:834

Sentence-Based Chunker
└─ Tries to split by sentences
    ├─ Works for prose ✅
    └─ FAILS for tables, indexes, lists ❌
```

### MinerU Provides Rich Metadata

From `mineru_selfhosted.py:316-327`:

```python
element_info = {
    'text': item.get('text', ''),
    'type': item.get('type', 'text'),  # ← AVAILABLE but IGNORED!
    'bbox': item.get('bbox', [])        # ← AVAILABLE but IGNORED!
}
```

**Available Types** (from MinerU content_list):
- `text` - Normal paragraphs
- `title` - Section headers
- `table` - Table content
- `image` - Image captions
- `list` - List items
- `equation` - Mathematical equations
- `footer` - Page footers
- `header` - Page headers

### Where Structure is Lost

**Line 325** (`mineru_selfhosted.py`):
```python
# ❌ Concatenates all element types equally
page_text_parts.append(text)
```

**Line 328**:
```python
# ❌ Joins with newlines, no type markers
page_text = '\n'.join(page_text_parts)
```

**Line 136** (`document_tasks.py`):
```python
# ❌ Further flattens into single string
combined_text = " ".join(text_parts)
```

## Specific Problems

### Problem 1: Tables Treated as Prose

**Current Behavior**:
```
# Table cells concatenated:
"520 Automotive Product Development\n4.52\nELBAT\nsetubirttA\nelciheV..."

# Chunker tries to split by sentences:
# - No sentence boundaries in table!
# - All data becomes one 990-token chunk
# - Exceeds 450 token limit
```

**What Should Happen**:
```
# Table recognized and chunked separately:
Chunk 1 (table header): "Table 4.52: Vehicle Attributes Pugh Diagram"
Chunk 2 (table row 1): "Package: GMC Acadia +2, Honda Pilot +2, Ford Explorer +1"
Chunk 3 (table row 2): "Ergonomics: GMC Acadia +4, Honda Pilot +1, Ford Explorer +3"
...
```

### Problem 2: Indexes Treated as Prose

**Current Behavior**:
```
# Index entries concatenated:
"Index\nA to lower levels, 153, 154\nfor systems design requirements\nAACN, see Advanced automatic crash\ndevelopment, 155-156..."

# No natural sentence breaks → 8584 token chunk!
```

**What Should Happen**:
```
# Index chunked by entries or letter sections:
Chunk 1 (A section): "A to lower levels, 153, 154\nAACN, see Advanced..."
Chunk 2 (B section): "Braking system, see ABS..."
...
```

### Problem 3: Reversed Text in Tables

**Current Behavior**:
```
"ELBAT" instead of "TABLE"
"elciheV" instead of "Vehicle"
```

**Root Cause**: MinerU extracts table columns right-to-left or cells in reverse order

**Fix Needed**: Proper table parsing or text reversal detection

## Available Solutions

### Solution 1: Structure-Aware Chunking (Recommended)

Use MinerU's type information to chunk differently based on element type:

```python
def chunk_by_structure(elements: List[Dict]) -> List[Chunk]:
    chunks = []

    for elem in elements:
        elem_type = elem['type']
        text = elem['text']

        if elem_type == 'table':
            # Split tables by rows or logical sections
            table_chunks = chunk_table(text, max_tokens=450)
            chunks.extend(table_chunks)

        elif elem_type == 'list':
            # Split lists by items
            list_chunks = chunk_list(text, max_tokens=450)
            chunks.extend(list_chunks)

        elif elem_type == 'text':
            # Use sentence-based chunking for prose
            text_chunks = chunk_sentences(text, max_tokens=450)
            chunks.extend(text_chunks)

        elif elem_type == 'title':
            # Keep titles with following content
            chunks.append(Chunk(text, type='header'))

        # Special handling for indexes, equations, etc.

    return chunks
```

**Pros**:
- Respects document structure
- Prevents oversized chunks from tables/indexes
- Better semantic coherence

**Cons**:
- Requires refactoring
- Need table parsing logic
- More complex

### Solution 2: Hard Token Limit with Force Split

Quick fix: Force split any chunk exceeding limit, regardless of structure:

```python
def force_split_oversized(chunks: List[str], max_tokens: int) -> List[str]:
    result = []
    for chunk in chunks:
        if count_tokens(chunk) > max_tokens:
            # Split at word boundaries until under limit
            words = chunk.split()
            current = []
            current_tokens = 0

            for word in words:
                word_tokens = count_tokens(word)
                if current_tokens + word_tokens > max_tokens:
                    result.append(' '.join(current))
                    current = [word]
                    current_tokens = word_tokens
                else:
                    current.append(word)
                    current_tokens += word_tokens

            if current:
                result.append(' '.join(current))
        else:
            result.append(chunk)

    return result
```

**Pros**:
- Prevents data loss
- Simple implementation
- No structure changes needed

**Cons**:
- Breaks semantic coherence
- Tables still garbled
- Indexes fragmented incorrectly

### Solution 3: Detect and Handle Special Structures

Add heuristics to detect tables/indexes/lists in plain text:

```python
def detect_structure_type(text: str) -> str:
    # Detect tables
    if has_table_patterns(text):
        return 'table'

    # Detect indexes
    if text.startswith('Index') or has_index_patterns(text):
        return 'index'

    # Detect lists
    if has_list_markers(text):
        return 'list'

    return 'text'

def has_table_patterns(text: str) -> bool:
    # High ratio of single words per line
    lines = text.split('\n')
    short_lines = sum(1 for line in lines if len(line.split()) < 3)
    return short_lines / len(lines) > 0.7 if lines else False
```

**Pros**:
- Works without MinerU metadata changes
- Can improve gradually

**Cons**:
- Heuristics unreliable
- Misses edge cases
- Still loses some structure

## Recommendations

### Phase 1: Immediate Fix (Prevent Data Loss)

1. ✅ **Add hard token limit check** before embedding:
   ```python
   # In chunking.py, add force-split for oversized chunks
   if token_count > max_tokens:
       return force_split_at_word_boundaries(chunk, max_tokens)
   ```

2. ✅ **Add structure type detection** for tables/indexes:
   ```python
   # Detect reversed text (more single-char words than normal)
   # Detect table structure (short lines, minimal prose)
   # Detect index structure (starts with "Index", alphabetical)
   ```

### Phase 2: Structure-Aware Chunking (Optimal)

3. **Pass element type through to chunker**:
   ```python
   # document_tasks.py:116
   for elem in elements:
       text_parts.append({
           'text': elem.text,
           'type': elem.metadata.get('type', 'text')  # ← Add type!
       })
   ```

4. **Implement type-specific chunking**:
   ```python
   # chunking.py
   class StructureAwareChunker:
       def chunk(self, elements: List[Dict]) -> List[Chunk]:
           chunks = []
           for elem in elements:
               if elem['type'] == 'table':
                   chunks.extend(self.chunk_table(elem['text']))
               elif elem['type'] == 'text':
                   chunks.extend(self.chunk_prose(elem['text']))
               # ... other types
           return chunks
   ```

### Phase 3: Table/Index Parsing

5. **Add table parser**:
   - Detect table structure from MinerU bbox data
   - Parse rows/columns correctly
   - Generate row-based chunks with context

6. **Add index parser**:
   - Split by alphabetical sections
   - Group related entries
   - Maintain cross-references

## Implementation Plan

### Week 1: Emergency Fix
- [x] Improved error logging (DONE)
- [ ] Hard token limit with force-split
- [ ] Structure detection heuristics

### Week 2: Structure Preservation
- [ ] Pass element types from MinerU to chunker
- [ ] Implement type-specific chunking strategies
- [ ] Add table row detection

### Week 3: Advanced Parsing
- [ ] Table parser using bbox data
- [ ] Index section parser
- [ ] List item chunking

## Testing Strategy

1. **Create test documents**:
   - PDF with large tables
   - PDF with index sections
   - PDF with normal prose
   - Mixed content PDF

2. **Measure improvements**:
   - % of chunks within token limit (target: 100%)
   - Semantic coherence scores
   - Embedding quality metrics

3. **Edge cases**:
   - Rotated tables
   - Multi-column layouts
   - Nested structures

## Related Files

- `src/fileintel/document_processing/processors/mineru_selfhosted.py:316-328` - Element creation
- `src/fileintel/tasks/document_tasks.py:116-136` - Text concatenation
- `src/fileintel/document_processing/chunking.py:834-885` - Sentence-based chunking

## References

- MinerU documentation on content types
- Token counting in `chunking.py:657-674`
- Chunk validation in `chunking.py:753-771`
