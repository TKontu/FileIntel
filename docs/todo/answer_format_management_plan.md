# Answer Format Management System - Implementation Plan

**Version:** 2.0 (UPDATED)
**Date:** 2025-11-08
**Status:** Ready for Implementation
**Updates:** GraphRAG integration simplified based on detailed analysis

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [What Changed in v2.0](#what-changed-in-v20)
- [Current State Analysis](#current-state-analysis)
- [Design Goals](#design-goals)
- [Architecture Design](#architecture-design)
- [Implementation Plan](#implementation-plan)
- [Detailed File Changes](#detailed-file-changes-summary)
- [Implementation Order](#implementation-order-recommended)
- [Rollout Strategy](#rollout-strategy)
- [Risks & Mitigations](#risks--mitigations)
- [Success Metrics](#success-metrics)
- [Future Enhancements](#future-enhancements)

---

## Executive Summary

This document outlines a comprehensive plan to implement answer format management for FileIntel's RAG (Retrieval-Augmented Generation) systems. The primary goal is to enable users to request specific answer formats (single paragraph, table, list, essay, etc.) at query time via API parameter, addressing the current issue where LLM responses default to bullet-point lists.

**Key Benefits:**
- User control over answer structure
- Single paragraph responses with headlines
- Flexible format options (table, JSON, markdown, essay, list)
- Backward compatible with existing queries
- Reuses existing prompt infrastructure

**Estimated Implementation Time:** 11-16 hours across 5 sprints (unchanged)

**Key Improvement in v2.0:** GraphRAG integration is now **simpler and lower-risk** than originally planned, requiring only ~45 lines of code via string enhancement instead of template refactoring.

---

## What Changed in v2.0

### Major Updates

1. **GraphRAG Integration Simplified** ✨
   - **Old Approach:** Template injection (invasive)
   - **New Approach:** String enhancement of `response_type` variable
   - **Risk Level:** Very Low (was Medium)
   - **Code Changes:** ~45 lines (was ~100 lines)
   - **GraphRAG Source Changes:** Zero (was "minimal")

2. **response_type Variable Discovery**
   - Identified as THE injection point for GraphRAG
   - Currently set to `"text"` in FileIntel
   - Flows unchanged through entire GraphRAG stack
   - Can be safely enhanced with format templates

3. **Microsoft Code Preservation**
   - Zero modifications to `src/graphrag/prompts/query/*.py`
   - Respects copyright and licensing
   - No conflicts with future graphrag upgrades
   - Professional and maintainable approach

4. **Implementation Time Redistribution**
   - Sprint 4 (GraphRAG): Reduced from 2-3 hours to 2 hours
   - Sprint 2 (Vector RAG): Increased from 3-4 hours to 4 hours (more thorough testing)
   - Overall time: Still 11-16 hours

### Documentation Updates

**New Analysis Documents:**
- `docs/graphrag_prompt_analysis.md` - Complete GraphRAG analysis
- `docs/GRAPHRAG_ANALYSIS_COMPLETE.md` - GraphRAG summary
- `docs/README_PROMPT_ANALYSIS.md` - Master documentation index

**Updated Sections in This Document:**
- Current State Analysis (GraphRAG section)
- Architecture Design (GraphRAG integration)
- Phase 4 Implementation (simplified approach)
- Detailed File Changes (reduced GraphRAG changes)
- Risks & Mitigations (GraphRAG risk downgraded)

---

## Current State Analysis

### What Works Well

1. ✅ **Dynamic prompt infrastructure exists** (`src/fileintel/prompt_management/simple_prompts.py`)
   - `load_prompt_components()` - loads all `.md` files from a directory
   - `compose_prompt()` - renders Jinja2 templates
   - `render_template()` - handles variable substitution

2. ✅ **Answer format templates created** (7 formats in `prompts/examples/`)
   - `answer_format_table.md` - Table format with JSON schema
   - `answer_format_markdown.md` - Markdown content in JSON
   - `answer_format_list.md` - List format
   - `answer_format_json.md` - Flexible JSON structure
   - `answer_format_essay.md` - Essay with title/sections
   - `answer_format_single_paragraph.md` - Single paragraph under headline (NEW)
   - `answer_format.md` - Base format

3. ✅ **Jinja2 templating** supports variable substitution with `{{ variable }}` syntax

4. ✅ **Metadata extraction** already uses this pattern successfully
   - Located in `src/fileintel/document_processing/metadata_extractor.py`
   - Lines 72-85 demonstrate the template loading pattern

5. ✅ **Vector RAG templates created** (NEW in v2.0)
   - All hardcoded prompts extracted to `.md` templates
   - Located in `prompts/templates/vector_rag/`
   - Character-for-character match with current behavior

6. ✅ **GraphRAG integration point identified** (NEW in v2.0)
   - `response_type` variable is THE injection point
   - Currently set to `"text"` in FileIntel
   - Safe to enhance with format templates
   - Zero GraphRAG source modifications needed

### Current Problems

1. ❌ **Vector RAG**: Hardcoded prompts in `src/fileintel/llm_integration/unified_provider.py:424-466`
   - `_build_rag_prompt()` method uses string concatenation
   - Templates created but not yet used by code
   - Query-type instructions hardcoded

2. ⚠️ **GraphRAG**: Hardcoded prompts in `src/graphrag/prompts/query/*.py` (UPDATED)
   - System prompts defined as Python string constants (Microsoft copyright)
   - **NOT using `.md` template files** - and we won't modify them
   - **Solution:** Enhance `response_type` parameter instead of modifying prompts
   - **Risk:** Very Low (parameter enhancement only)

3. ❌ **No user control**: Answer format not configurable at query time
   - No API parameter for format preference
   - No mechanism to inject format instructions

4. ❌ **Bullet point bias**: LLMs default to lists without format guidance
   - Current prompts don't specify answer structure
   - Results in inconsistent formatting

5. ❌ **No API parameter**: `QueryRequest` model doesn't accept format preference
   - Located in `src/fileintel/api/routes/query.py:26-32`

### Current Template Usage Example

The **metadata extraction** system demonstrates the pattern:

```
prompts/templates/metadata_extraction/
├── prompt.md           # Main template with {{ answer_format }}
├── instruction.md      # Task instructions
├── question.md         # What to extract
├── answer_format.md    # JSON schema for metadata
└── embedding_reference.md
```

Loading pattern (`src/fileintel/document_processing/metadata_extractor.py:72-85`):
```python
components = load_prompt_components(task_dir)
context = {**components, "document_text": combined_text}
prompt = compose_prompt(components["prompt"], context, max_length)
```

---

## Design Goals

### Primary Goal

**Enable users to request specific answer formats** (single paragraph, table, list, essay, etc.) **at query time** via API parameter.

### Secondary Goals

1. **Maintain backward compatibility** - default format preserves current behavior
2. **Work with both Vector RAG and Graph RAG** - unified approach
3. **Reuse existing prompt infrastructure** - leverage `simple_prompts.py`
4. **Support custom formats per collection/project** - extensible design
5. **Minimal performance overhead** - cache loaded templates
6. **Zero GraphRAG source modifications** - respect Microsoft copyright (NEW)

### Non-Goals (Future Work)

- Collection-level default formats (Phase 2)
- Custom user-uploaded format templates (Phase 2)
- Format auto-detection based on query analysis (Phase 3)
- Multi-format responses (Phase 3)

---

## Architecture Design

### 1. Answer Format Options

**Available Formats:**

| Format Name | Description | Output Structure |
|-------------|-------------|------------------|
| `default` | Current behavior (no specific format constraint) | Plain text with citations |
| `single_paragraph` | Single paragraph under one headline | JSON: `{"headline": "...", "paragraph": "..."}` |
| `list` | Simple list of items | JSON: `{"response": ["item1", "item2", ...]}` |
| `table` | Structured table with headers and rows | JSON: `{"headers": [...], "rows": [[...]]}` |
| `json` | Flexible JSON structure | JSON object (structure varies by query) |
| `markdown` | Rich markdown content | JSON: `{"content": "# Markdown text"}` |
| `essay` | Structured essay with sections | JSON: `{"title": "...", "sections": [{...}]}` |

### 2. Data Flow Architecture

```
User Query Request
    ↓
API: QueryRequest (with answer_format parameter)
    ↓
Query Orchestrator
    ↓
┌─────────────────┬──────────────────┐
│   Vector RAG    │    Graph RAG     │
│                 │                  │
│  Load format    │  Load format     │
│  template       │  template        │
│     ↓           │      ↓           │
│  Build prompt   │  Enhance         │
│  with template  │  response_type   │  ← UPDATED: Simplified approach
│     ↓           │      ↓           │
│  Generate       │  Generate        │
│  answer         │  answer          │
└─────────────────┴──────────────────┘
    ↓
Return formatted answer
```

### 3. Component Architecture

```
┌─────────────────────────────────────────┐
│      AnswerFormatManager                │
│  - get_format_template(name)            │
│  - list_available_formats()             │
│  - validate_format(name)                │
└─────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│   Prompt Template System                │
│   (simple_prompts.py)                   │
│  - load_prompt_components(dir)          │
│  - compose_prompt(template, context)    │
└─────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│   RAG Services                          │
│  - Vector RAG: _build_rag_prompt()      │
│  - Graph RAG: _build_response_type()    │  ← UPDATED: New helper method
└─────────────────────────────────────────┘
```

### 4. Configuration Approach

**Option A: Query-Time Parameter (RECOMMENDED - Phase 1)**
- User specifies format per query
- Flexible, no global state
- Easy to A/B test formats
- Implemented via API parameter

**Option B: Collection-Level Default (Future - Phase 2)**
- Set default format per collection
- Can be overridden per query
- Good for consistent project workflows
- Stored in collection metadata

**Decision: Implement Option A first, add Option B later**

### 5. Template Integration Points

#### Vector RAG Integration

**Location**: `src/fileintel/llm_integration/unified_provider.py:424-466`

**Method**: `_build_rag_prompt(query, context, query_type, answer_format)`

**Approach**: Refactor to use template files instead of hardcoded strings

**Template Directory**: `prompts/templates/vector_rag/`

**Integration Point**: Between context and citation rules

```
Question: {query}
Retrieved Sources: {context}

👉 INSERT ANSWER FORMAT HERE 👈

Citation Requirements: ...
```

#### GraphRAG Integration (UPDATED v2.0)

**Location**: `src/fileintel/rag/graph_rag/services/graphrag_service.py:563-636`

**Method**: `global_search()` and `local_search()`

**Approach**: Enhance `response_type` string parameter (NOT template injection)

**No Template Files Needed**: GraphRAG prompts remain in Python files

**Integration Point**: `response_type` parameter

```python
# Current
response_type = "text"

# Enhanced
if answer_format != "default":
    format_template = load_format_template(answer_format)
    response_type = f"text\n\n{format_template}"

# Pass to GraphRAG
global_search(..., response_type=response_type, ...)
```

**Why This Works:**
- GraphRAG prompts include: `---Target response length and format---\n{response_type}`
- LLM sees enhanced format instructions
- Zero modifications to Microsoft prompt files
- Simple, safe, and maintainable

---

## Implementation Plan

### Phase 1: Core Infrastructure (Foundation)

**Estimated Time:** 2-3 hours

#### 1.1 Extend API Models

**File:** `src/fileintel/api/routes/query.py`

**Current:**
```python
class QueryRequest(BaseModel):
    question: str
    search_type: Optional[str] = "adaptive"
    max_results: Optional[int] = 5
    include_sources: Optional[bool] = True
    query_mode: Optional[str] = "sync"
```

**Updated:**
```python
class QueryRequest(BaseModel):
    question: str
    search_type: Optional[str] = "adaptive"
    max_results: Optional[int] = 5
    include_sources: Optional[bool] = True
    query_mode: Optional[str] = "sync"
    answer_format: Optional[str] = "default"  # NEW PARAMETER
```

**Validation:**
```python
@model_validator(mode='after')
def validate_answer_format(self):
    valid_formats = ["default", "single_paragraph", "list", "table",
                     "json", "markdown", "essay"]
    if self.answer_format not in valid_formats:
        raise ValueError(
            f"Invalid answer_format: {self.answer_format}. "
            f"Valid options: {', '.join(valid_formats)}"
        )
    return self
```

#### 1.2 Create Format Manager

**File:** `src/fileintel/prompt_management/format_manager.py` (NEW)

**Complete implementation in original plan sections 1.2 (lines 117-270)**

*(Implementation details preserved from original plan)*

#### 1.3 Update Prompt Management Module

**File:** `src/fileintel/prompt_management/__init__.py`

Add export:
```python
from .format_manager import AnswerFormatManager

__all__ = [
    "load_prompt_template",
    "load_prompt_components",
    "compose_prompt",
    "AnswerFormatManager",  # NEW
]
```

---

### Phase 2: Vector RAG Integration

**Estimated Time:** 3-4 hours → **UPDATED: 4 hours** (more thorough testing)

*(Vector RAG implementation unchanged from original plan - see sections 2.1-2.3)*

**No changes to Vector RAG approach** - template refactoring remains the same.

---

### Phase 3: Format Integration

**Estimated Time:** 2-3 hours

*(Format integration unchanged from original plan - see original section)*

---

### Phase 4: Graph RAG Integration (COMPLETELY UPDATED)

**Estimated Time:** 2-3 hours → **UPDATED: 2 hours** (simplified approach)

#### 4.1 Implementation Approach (UPDATED)

**Old Approach (v1.0):** Template injection into GraphRAG system prompts
- ❌ Requires modifying Microsoft code
- ❌ Higher risk of breaking graphrag upgrades
- ❌ More complex implementation

**New Approach (v2.0):** String enhancement of `response_type` parameter
- ✅ Zero GraphRAG source code changes
- ✅ Very low risk
- ✅ Simple implementation
- ✅ Respects Microsoft copyright

#### 4.2 Add Format Manager to GraphRAGService

**File:** `src/fileintel/rag/graph_rag/services/graphrag_service.py`

**Add to `__init__()` method:**

```python
from fileintel.prompt_management import AnswerFormatManager
from pathlib import Path

class GraphRAGService:
    def __init__(self, ...):
        # ... existing initialization ...

        # Initialize format manager
        formats_dir = Path(__file__).parent.parent.parent.parent / "prompts" / "examples"
        self.format_manager = AnswerFormatManager(formats_dir)
        logger.info("GraphRAGService initialized with format manager")
```

#### 4.3 Create Helper Method (NEW)

**File:** `src/fileintel/rag/graph_rag/services/graphrag_service.py`

**Add new helper method:**

```python
def _build_response_type(
    self,
    base_type: str = "text",
    answer_format: str = "default"
) -> str:
    """
    Build response_type string with optional format template.

    Args:
        base_type: Base response type description
        answer_format: Desired answer format template name

    Returns:
        Enhanced response_type string
    """
    if answer_format == "default":
        return base_type

    try:
        format_template = self.format_manager.get_format_template(answer_format)
        enhanced = f"{base_type}\n\n{format_template}"
        logger.debug(
            f"Built response_type: base={base_type}, "
            f"format={answer_format}, total_len={len(enhanced)}"
        )
        return enhanced
    except (ValueError, IOError) as e:
        logger.warning(
            f"Failed to load format template '{answer_format}': {e}. "
            "Using base response type only."
        )
        return base_type
```

#### 4.4 Update global_search() Method

**File:** `src/fileintel/rag/graph_rag/services/graphrag_service.py:563-594`

**Current signature:**
```python
async def global_search(self, query: str, collection_id: str):
```

**Updated signature:**
```python
async def global_search(
    self,
    query: str,
    collection_id: str,
    answer_format: str = "default"  # NEW PARAMETER
):
```

**Update implementation:**
```python
async def global_search(
    self,
    query: str,
    collection_id: str,
    answer_format: str = "default"  # NEW
):
    """Performs a global search on the GraphRAG index."""
    # ... existing code to load dataframes ...

    # Build enhanced response_type
    response_type = self._build_response_type("text", answer_format)

    graphrag_config = await self._get_cached_config(collection_id)

    # Pass enhanced response_type to GraphRAG
    result, context = await global_search(
        config=graphrag_config,
        entities=dataframes["entities"],
        communities=dataframes["communities"],
        community_reports=dataframes["community_reports"],
        community_level=self.settings.graphrag.community_levels,
        dynamic_community_selection=True,
        response_type=response_type,  # ENHANCED
        query=query,
    )

    self._sync_graphrag_logger_level()
    return self.data_adapter.convert_response(result, context)
```

#### 4.5 Update local_search() Method

**File:** `src/fileintel/rag/graph_rag/services/graphrag_service.py:595-636`

**Current signature:**
```python
async def local_search(
    self, query: str, collection_id: str, community: str, MOCK_ARGM=None
):
```

**Updated signature:**
```python
async def local_search(
    self,
    query: str,
    collection_id: str,
    community: str,
    answer_format: str = "default",  # NEW PARAMETER
    MOCK_ARGM=None
):
```

**Update implementation:**
```python
async def local_search(
    self,
    query: str,
    collection_id: str,
    community: str,
    answer_format: str = "default",  # NEW
    MOCK_ARGM=None
):
    """Performs a local search within a specific community."""
    # ... existing code to load dataframes ...

    # Build enhanced response_type
    response_type = self._build_response_type("text", answer_format)

    graphrag_config = await self._get_cached_config(collection_id)

    # Pass enhanced response_type to GraphRAG
    result, context = await local_search(
        config=graphrag_config,
        entities=dataframes["entities"],
        communities=dataframes["communities"],
        community_reports=dataframes["community_reports"],
        text_units=dataframes.get("text_units"),
        relationships=dataframes.get("relationships"),
        covariates=covariates,
        community_level=self.settings.graphrag.community_levels,
        response_type=response_type,  # ENHANCED
        query=query,
    )

    self._sync_graphrag_logger_level()
    return self.data_adapter.convert_response(result, context)
```

#### 4.6 Update query() Method

**File:** `src/fileintel/rag/graph_rag/services/graphrag_service.py:79-181`

**Add `answer_format` parameter and pass to search methods:**

```python
async def query(
    self,
    query: str,
    collection_id: str,
    search_type: str = "global",
    answer_format: str = "default"  # NEW PARAMETER
) -> Dict[str, Any]:
    """
    Standard query interface with citation tracing.

    Args:
        query: The search query
        collection_id: Collection to search
        search_type: "global" or "local" (default: "global")
        answer_format: Desired answer format (default: "default")

    Returns:
        Dict with answer, sources, confidence, and metadata
    """
    # ... existing validation code ...

    # Route to appropriate search method with answer_format
    if search_type == "local":
        raw_response = await self.local_search(
            query,
            collection_id,
            community="",
            answer_format=answer_format  # PASS FORMAT
        )
    else:
        raw_response = await self.global_search(
            query,
            collection_id,
            answer_format=answer_format  # PASS FORMAT
        )

    # ... rest of method unchanged (citation tracing, etc.) ...
```

**Summary of GraphRAG Changes:**
- Add `format_manager` to `__init__()`: ~5 lines
- Add `_build_response_type()` helper: ~20 lines
- Update `global_search()`: ~5 lines
- Update `local_search()`: ~5 lines
- Update `query()`: ~5 lines
- **Total: ~40 lines of code**

---

### Phase 5: Query Orchestrator Integration

**Estimated Time:** 2-3 hours

*(Query orchestrator unchanged from original plan)*

---

### Phase 6: Testing & Documentation

**Estimated Time:** 2-3 hours

#### 6.1 Unit Tests

**New Test for GraphRAG:**

```python
# tests/test_graphrag_format_manager.py

def test_build_response_type_default():
    """Test default response type (no format)."""
    service = GraphRAGService(...)
    response_type = service._build_response_type("text", "default")
    assert response_type == "text"

def test_build_response_type_with_format():
    """Test response type with format template."""
    service = GraphRAGService(...)
    response_type = service._build_response_type("text", "single_paragraph")

    assert "text" in response_type
    assert "paragraph" in response_type
    assert len(response_type) > len("text")

def test_build_response_type_invalid_format():
    """Test response type with invalid format (should fallback)."""
    service = GraphRAGService(...)
    response_type = service._build_response_type("text", "nonexistent")

    # Should fallback to base type
    assert response_type == "text"
```

*(Other tests unchanged from original plan)*

---

## Detailed File Changes Summary (UPDATED)

### Files to CREATE (9 new files - UNCHANGED)

1. `src/fileintel/prompt_management/format_manager.py`
2-9. `prompts/templates/vector_rag/*.md` (8 files)

### Files to MODIFY (5 files - UPDATED COUNTS)

1. **`src/fileintel/api/routes/query.py`**
   - Add `answer_format` param to `QueryRequest` (v1.0: unchanged)
   - **Lines changed:** ~15

2. **`src/fileintel/llm_integration/unified_provider.py`**
   - Refactor `_build_rag_prompt()` to use templates (v1.0: unchanged)
   - **Lines changed:** ~100

3. **`src/fileintel/rag/graph_rag/services/graphrag_service.py`** (UPDATED)
   - Add `format_manager` to `__init__()` (v2.0: new)
   - Add `_build_response_type()` helper (v2.0: new)
   - Update `global_search()` signature and implementation (v2.0: simplified)
   - Update `local_search()` signature and implementation (v2.0: simplified)
   - Update `query()` to pass format (v2.0: simplified)
   - **Lines changed:** ~40 (v1.0: ~60)

4. **`src/fileintel/rag/query_orchestrator.py`**
   - Add `answer_format` parameter to all query methods (v1.0: unchanged)
   - **Lines changed:** ~30

5. **`src/fileintel/rag/vector_rag/services/vector_rag_service.py`**
   - Add `answer_format` parameter (v1.0: unchanged)
   - **Lines changed:** ~20

6. **`src/fileintel/prompt_management/__init__.py`**
   - Add `AnswerFormatManager` to exports (v1.0: unchanged)
   - **Lines changed:** ~3

**Total Lines Changed: ~208 (v1.0: ~228)**
**GraphRAG-specific: ~40 lines (v1.0: ~60 lines)**

---

## Implementation Order (Recommended)

### Sprint 1: Foundation (2-3 hours) - UNCHANGED

*(Original plan preserved)*

---

### Sprint 2: Vector RAG Templates (3-4 hours) → **UPDATED: 4 hours**

*(Original plan preserved, add 30 min for additional testing)*

---

### Sprint 3: Format Integration (2-3 hours) - UNCHANGED

*(Original plan preserved)*

---

### Sprint 4: GraphRAG Integration (2-3 hours) → **UPDATED: 2 hours** (SIMPLIFIED)

**Goal:** Add answer format support to GraphRAG via response_type enhancement

**Tasks:**

1. **Add format manager to GraphRAGService (20 min)**
   - Add import and initialization in `__init__()`
   - Test format manager loads correctly
   - Estimated: 20 minutes

2. **Create _build_response_type() helper (30 min)**
   - Implement helper method
   - Add error handling and logging
   - Write unit tests
   - Estimated: 30 minutes

3. **Update global_search() (20 min)**
   - Add `answer_format` parameter
   - Call `_build_response_type()`
   - Pass enhanced response_type to GraphRAG API
   - Estimated: 20 minutes

4. **Update local_search() (20 min)**
   - Same changes as global_search()
   - Estimated: 20 minutes

5. **Update query() method (15 min)**
   - Add `answer_format` parameter
   - Pass to search method calls
   - Estimated: 15 minutes

6. **Testing (30 min)**
   - Test `answer_format="default"` → identical output
   - Test with `single_paragraph` format
   - Verify citations still work
   - Check citation tracing compatibility
   - Estimated: 30 minutes

**Validation:**
- ✅ `answer_format="default"` produces identical output
- ✅ Format templates work with global and local search
- ✅ Citations preserved: `[Data: Reports (ids)]`
- ✅ Citation tracing `_trace_and_format_citations()` still works
- ✅ No errors in logs

---

### Sprint 5: Testing & Polish (2-3 hours) - UNCHANGED

*(Original plan preserved)*

---

**Total Estimated Time: 11-16 hours (unchanged)**

---

## Rollout Strategy

*(Original plan preserved - unchanged)*

---

## Risks & Mitigations (UPDATED)

### High Impact Risks

| Risk | Impact | Probability | Mitigation | v2.0 Update |
|------|--------|-------------|------------|-------------|
| **Breaking changes to existing queries** | High | Low | Use "default" format for backward compatibility; extensive testing | Unchanged |
| **LLM doesn't follow format instructions** | High | Medium | Add format validation; retry with stronger instructions; use JSON mode where possible | Unchanged |
| **Template rendering errors** | High | Low | Add fallback to original hardcoded prompts; comprehensive error handling | Unchanged |

### Medium Impact Risks

| Risk | Impact | Probability | Mitigation | v2.0 Update |
|------|--------|-------------|------------|-------------|
| **GraphRAG prompt conflicts** | Medium | Medium → **Low** | ~~Test thoroughly; may need to refine injection~~ **String enhancement is safe** | ✅ **Risk reduced** |
| **Inconsistent format compliance** | Medium | Medium | Tune prompts; use few-shot examples; implement validation | Unchanged |
| **User confusion about formats** | Medium | Low | Clear documentation; good examples; sensible defaults | Unchanged |

### Low Impact Risks

| Risk | Impact | Probability | Mitigation | v2.0 Update |
|------|--------|-------------|------------|-------------|
| **Performance overhead** | Low | Low | Cache loaded templates; minimal template rendering cost; monitor performance | Unchanged |
| **Token limit issues** | Low | Low | Truncate format instructions if needed; prioritize context over format | Unchanged |
| **Template file management** | Low | Low | Clear directory structure; documentation for adding formats | Unchanged |

### GraphRAG-Specific Risks (NEW in v2.0)

| Risk | Impact | Mitigation |
|------|--------|------------|
| **response_type enhancement breaks prompt rendering** | Medium | Test with `answer_format="default"` first; verify LLM still sees format instructions |
| **Microsoft prompt updates** | Low | Our code doesn't touch their files; safe from upstream changes |
| **Token budget exceeded** | Low | Format templates are concise (~150 tokens); context dominates (8000 tokens) |

---

## Success Metrics

*(Original plan preserved - unchanged)*

---

## Future Enhancements

*(Original plan preserved - unchanged)*

---

## Appendix

### A. File Structure Overview (UPDATED)

```
fileintel/
├── docs/
│   ├── answer_format_management_plan.md        # THIS FILE (v2.0 UPDATED)
│   ├── existing_prompts_backup.md              # Complete backup
│   ├── prompt_structure_analysis.md            # Vector RAG analysis
│   ├── graphrag_prompt_analysis.md             # GraphRAG analysis (NEW)
│   ├── PROMPT_DOCUMENTATION_COMPLETE.md        # Vector summary
│   ├── GRAPHRAG_ANALYSIS_COMPLETE.md           # GraphRAG summary (NEW)
│   └── README_PROMPT_ANALYSIS.md               # Master index (NEW)
├── prompts/
│   ├── examples/
│   │   ├── answer_format_table.md
│   │   ├── answer_format_markdown.md
│   │   ├── answer_format_list.md
│   │   ├── answer_format_json.md
│   │   ├── answer_format_essay.md
│   │   ├── answer_format_single_paragraph.md   # NEW
│   │   └── answer_format.md
│   └── templates/
│       └── vector_rag/                          # NEW DIRECTORY
│           ├── prompt.md                        # NEW
│           ├── base_instruction.md              # NEW
│           ├── citation_rules.md                # NEW
│           └── query_type_instructions/         # NEW DIRECTORY
│               ├── factual.md                   # NEW
│               ├── analytical.md                # NEW
│               ├── summarization.md             # NEW
│               ├── comparison.md                # NEW
│               └── general.md                   # NEW
├── src/
│   └── fileintel/
│       ├── api/
│       │   └── routes/
│       │       └── query.py                     # MODIFY
│       ├── llm_integration/
│       │   └── unified_provider.py              # MODIFY
│       ├── prompt_management/
│       │   ├── __init__.py                      # MODIFY
│       │   ├── format_manager.py                # NEW
│       │   └── simple_prompts.py
│       └── rag/
│           ├── graph_rag/
│           │   └── services/
│           │       └── graphrag_service.py      # MODIFY (v2.0: SIMPLIFIED)
│           ├── vector_rag/
│           │   └── services/
│           │       └── vector_rag_service.py    # MODIFY
│           └── query_orchestrator.py            # MODIFY
└── tests/
    ├── integration/
    │   └── test_query_with_formats.py           # NEW
    ├── test_format_manager.py                   # NEW
    ├── test_graphrag_format_manager.py          # NEW (v2.0)
    └── test_rag_with_formats.py                 # NEW
```

### B. Configuration Example

*(Original plan preserved - unchanged)*

### C. Example API Usage

*(Original plan preserved - unchanged)*

---

## Document Control

**Version History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-08 | System | Initial comprehensive plan |
| 2.0 | 2025-11-08 | System | GraphRAG integration simplified based on detailed analysis |
|     |            |        | - Updated GraphRAG approach (string enhancement) |
|     |            |        | - Reduced GraphRAG code changes (~40 lines vs ~60) |
|     |            |        | - Clarified Microsoft code preservation |
|     |            |        | - Added new analysis document references |
|     |            |        | - Updated risks (GraphRAG risk reduced) |
|     |            |        | - Added GraphRAG-specific unit tests |

**Reviewers:**
- [ ] Technical Lead
- [ ] Product Manager
- [ ] QA Lead

**Approval:**
- [ ] Approved for implementation
- [ ] Budget allocated
- [ ] Sprint planned

---

**End of Document**
