# Improved Error Logging for Debugging

## Overview

Enhanced error logging throughout the indexing pipeline to provide complete context for debugging issues, especially for oversized chunks and embedding failures.

## Changes Made

### 1. Chunking Module (`src/fileintel/document_processing/chunking.py`)

#### Enhanced `_check_token_safety()` Method
- **Added Parameters**: `document_id` and `chunk_index` for context
- **Improved Text Preview**: Shows 500 characters instead of 100
- **Structured Logging**: Uses pipe-separated format for easy parsing
- **Detailed Information**:
  - Document ID
  - Chunk index
  - Token count vs limit (with overage amount)
  - Text length in characters
  - Full chunk text preview

**Example Log Output**:
```
CRITICAL: Vector RAG | document_id=abc123 | chunk_index=5 | Token count: 5167/450 (exceeds limit by 4717) | Text length: 21000 chars | This will cause embedding failures. | Full chunk text:
Safety : Designed for a 5-star National Highway Trafic Safety Administration
(NHTSA) safety rating w...
```

#### Updated `_validate_chunks_against_token_limit()` Method
- Passes document_id through to safety checks
- Enhanced "Dropping oversized chunk" logging with full context
- Shows 500-character preview of dropped chunks

#### Updated `_chunk_by_sentences()` Method
- Accepts optional `document_id` parameter
- Passes context through to validation methods

#### Updated `chunk_text()` Method
- Accepts optional `document_id` parameter
- Logs document context in all messages
- Enhanced oversized chunk logging with:
  - Document ID
  - Chunk index
  - Token count with limit
  - Text length
  - 500-character preview

#### Updated `chunk_text_for_graphrag()` Method
- Accepts optional `document_id` parameter
- Passes document context to safety checks

### 2. LLM Tasks Module (`src/fileintel/tasks/llm_tasks.py`)

#### Enhanced `generate_and_store_chunk_embedding()` Error Handling
- Retrieves chunk and document information on error
- Logs complete context:
  - Chunk ID
  - Document ID
  - Chunk index (position)
  - Text length
  - Error message
  - 500-character text preview
- Includes fallback logging if chunk details unavailable

**Example Log Output**:
```
Error generating and storing embedding | chunk_id=xyz789 | document_id=abc123 | chunk_index=5 | text_length=21000 chars | error=Token limit exceeded | chunk_text:
Safety : Designed for a 5-star National Highway Trafic Safety Administration...
```

### 3. Embedding Provider (`src/fileintel/llm_integration/embedding_provider.py`)

#### Enhanced `_truncate_text()` Method
- Accepts optional `chunk_context` parameter for logging
- Shows 500-character preview instead of 200
- Structured logging with pipe-separated format
- Includes token overage calculation

**Example Log Output**:
```
EMERGENCY TRUNCATION | document_id=abc123 | Token count: 5167/450 (exceeds by 4717) | Text length: 21000 chars | This indicates a bug in the text chunking system | Full text:
Safety : Designed for a 5-star National Highway Trafic Safety Administration...
```

#### Enhanced `get_embeddings()` Oversized Text Logging
- Shows 500-character preview for all oversized texts
- Structured logging format
- Includes dual tokenizer analysis (OpenAI + BERT)
- Full text content for debugging

#### Enhanced Individual Text Processing Error Logging
- Shows complete error context for each failed text
- Includes token count, character count, and error message
- 500-character text preview
- Better formatted emergency truncation messages

## Benefits

### 1. Complete Context
Every error now includes:
- **Which document** has the issue (document_id)
- **Which chunk** has the issue (chunk_id or chunk_index)
- **What the content is** (500-char preview or full text)
- **How severe** the issue is (token count vs limit)

### 2. Easier Debugging
- Can immediately identify problematic documents
- Can inspect actual chunk content causing issues
- Can track down why chunking is failing for specific content
- Can correlate errors across different processing stages

### 3. Better Visibility
- Warnings show approaching limits (90% threshold)
- Errors show exact overage amounts
- Critical errors show full context for investigation
- All logs use consistent structured format

## Log Format

All enhanced logs use a pipe-separated format for easy parsing:

```
LEVEL: context_type | key1=value1 | key2=value2 | ... | message | text:
<actual_text_content>
```

This format makes it easy to:
- Parse logs programmatically
- Filter by specific fields
- Extract text content for analysis
- Track issues across processing pipeline

## Usage Examples

### Finding Oversized Chunks
```bash
# Find all oversized chunks with document IDs
grep "CRITICAL.*exceeds.*limit" logs/celery.log | grep "document_id="

# Extract text from oversized chunks
grep -A 10 "CRITICAL.*exceeds.*limit" logs/celery.log
```

### Tracking Embedding Failures
```bash
# Find all embedding errors with context
grep "Error generating and storing embedding" logs/celery.log

# Find which documents are causing the most issues
grep "Error generating and storing embedding" logs/celery.log | grep -o "document_id=[^ ]*" | sort | uniq -c
```

### Identifying Chunking Issues
```bash
# Find all emergency truncations
grep "EMERGENCY TRUNCATION" logs/celery.log

# See which documents need chunking fixes
grep "EMERGENCY TRUNCATION" logs/celery.log | grep -o "document_id=[^ ]*"
```

## Next Steps

To fully utilize these improvements, consider:

1. **Update document_tasks.py** to pass document_id through to chunking methods
2. **Add document_id to embedding batch processing** for batch-level context
3. **Create monitoring dashboard** to track error patterns
4. **Set up alerts** for critical chunking failures
5. **Analyze error patterns** to identify systemic chunking issues

## Example: Debugging Oversized Chunks

With these improvements, when you see:
```
CRITICAL: Vector RAG chunk has 5167 tokens, exceeds 450 limit
```

You now also get:
- **Document ID**: `abc123-def456-ghi789`
- **Chunk Index**: 5 (6th chunk in document)
- **Full Text Preview**: First 500 characters
- **Exact Overage**: 4717 tokens over limit
- **Text Length**: 21000 characters

This allows you to:
1. Find the specific document in the database
2. Examine the source file
3. Understand why this content didn't chunk properly
4. Fix the underlying issue (e.g., no sentence boundaries, table data, etc.)
