# GraphRAG Prompt System - Comprehensive Analysis

**Date:** 2025-11-08
**Purpose:** Complete analysis of GraphRAG prompt system, variable flow, and answer format integration strategy
**Status:** Reference Document

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [GraphRAG Architecture Overview](#graphrag-architecture-overview)
- [Prompt Flow Analysis](#prompt-flow-analysis)
- [response_type Variable Deep Dive](#response_type-variable-deep-dive)
- [Search Method Details](#search-method-details)
- [Answer Format Integration Strategy](#answer-format-integration-strategy)
- [Safety and Preservation Requirements](#safety-and-preservation-requirements)

---

## Executive Summary

### Key Findings

1. **response_type is the ONLY user-controllable variable** in GraphRAG prompts that affects answer format
2. **Default value:** `"multiple paragraphs"` (hardcoded in `search.py:57`)
3. **Current injection point:** FileIntel passes `response_type="text"` (line 586, 628 in `graphrag_service.py`)
4. **Safe integration point:** Enhance `response_type` string BEFORE passing to GraphRAG engine
5. **No template files:** GraphRAG uses hardcoded Python string constants (not `.md` files)

### Integration Approach

**RECOMMENDED:** String concatenation enhancement

```python
# Current
response_type = "text"

# With answer format
if answer_format != "default":
    format_template = load_format_template(answer_format)
    response_type = f"text\n\n{format_template}"

# Pass enhanced response_type to GraphRAG
result, context = await global_search(..., response_type=response_type, ...)
```

**Risk Level:** ✅ VERY LOW
- Only modifies one string variable
- No changes to GraphRAG source code
- Easy to test and validate
- Simple rollback if needed

---

## GraphRAG Architecture Overview

### Component Stack

```
┌─────────────────────────────────────────────────────────┐
│  FileIntel GraphRAGService                              │
│  src/fileintel/rag/graph_rag/services/graphrag_service.py │
│  - query() method                                       │
│  - Calls global_search() or local_search()            │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  GraphRAG API Layer                                     │
│  src/graphrag/api/query.py                             │
│  - global_search(response_type, ...)                   │
│  - local_search(response_type, ...)                    │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  GraphRAG Search Engine                                 │
│  src/graphrag/query/structured_search/*/search.py      │
│  - GlobalSearch.__init__(response_type="multiple paragraphs") │
│  - Stores self.response_type                            │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  Prompt Template Rendering                              │
│  - Map Phase: MAP_SYSTEM_PROMPT.format(...)            │
│  - Reduce Phase: REDUCE_SYSTEM_PROMPT.format(          │
│      response_type=self.response_type,                  │
│      max_length=self.reduce_max_length                  │
│  )                                                      │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  LLM Generation                                         │
│  - System prompt with response_type injected            │
│  - Generates answer matching requested format           │
└─────────────────────────────────────────────────────────┘
```

### Key Insight

**response_type flows through the entire stack without modification**

```python
# FileIntel (current)
global_search(..., response_type="text", ...)
    ↓
# GraphRAG API
GlobalSearch(response_type="text")  # stored as self.response_type
    ↓
# Reduce Phase
REDUCE_SYSTEM_PROMPT.format(response_type="text", ...)
    ↓
# Final Prompt sent to LLM contains:
# "---Target response length and format---"
# "text"
```

---

## Prompt Flow Analysis

### Global Search (Map-Reduce Pattern)

#### Phase 1: Map Phase

**Purpose:** Extract key points from community reports

**Prompt:** `MAP_SYSTEM_PROMPT`
**File:** `src/graphrag/prompts/query/global_search_map_system_prompt.py`

**Variables Used:**
- `{context_data}` - Community reports data
- `{max_length}` - Token limit (default: 1000)

**No response_type variable in map phase!**

**Output:** JSON with scored key points
```json
{
  "points": [
    {"description": "...[Data: Reports (ids)]", "score": 85},
    {"description": "...[Data: Reports (ids)]", "score": 72}
  ]
}
```

**Invocation:**
```python
# src/graphrag/query/structured_search/global_search/search.py:220-222
search_prompt = self.map_system_prompt.format(
    context_data=context_data,
    max_length=max_length
)
```

#### Phase 2: Reduce Phase

**Purpose:** Synthesize map results into final answer

**Prompt:** `REDUCE_SYSTEM_PROMPT`
**File:** `src/graphrag/prompts/query/global_search_reduce_system_prompt.py`

**Variables Used:**
- `{report_data}` - Analyst reports from map phase (ranked by importance)
- `{response_type}` - ✨ **THIS IS WHERE WE INJECT!**
- `{max_length}` - Token limit (default: 2000)

**Invocation:**
```python
# src/graphrag/query/structured_search/global_search/search.py:373-376
search_prompt = self.reduce_system_prompt.format(
    report_data=text_data,
    response_type=self.response_type,  # <--- INJECTION POINT
    max_length=self.reduce_max_length,
)
```

**Output:** Final answer with `[Data: Reports (ids)]` citations

---

### Local Search

**Purpose:** Entity-relationship based local context search

**Prompt:** `LOCAL_SEARCH_SYSTEM_PROMPT`
**File:** `src/graphrag/prompts/query/local_search_system_prompt.py`

**Variables Used:**
- `{context_data}` - Tables with entities, relationships, sources
- `{response_type}` - ✨ **INJECTION POINT**

**No max_length variable in local search!**

**Invocation:**
```python
# Similar pattern to global reduce, but in local_search engine
search_prompt = self.system_prompt.format(
    context_data=context_data,
    response_type=self.response_type  # <--- INJECTION POINT
)
```

**Output:** Answer with `[Data: Sources (ids), Entities (ids), ...]` citations

---

### Basic Search

**Purpose:** Simple search without graph structure

**Prompt:** `BASIC_SEARCH_SYSTEM_PROMPT`
**File:** `src/graphrag/prompts/query/basic_search_system_prompt.py`

**Variables Used:**
- `{context_data}` - Source data tables
- `{response_type}` - ✨ **INJECTION POINT**

**Invocation:** Same pattern as local search

**Output:** Answer with `[Data: Sources (ids)]` citations

---

## response_type Variable Deep Dive

### Default Value

**Location:** `src/graphrag/query/structured_search/global_search/search.py:57`

```python
class GlobalSearch(BaseSearch[GlobalContextBuilder]):
    def __init__(
        self,
        ...
        response_type: str = "multiple paragraphs",  # <--- DEFAULT
        ...
    ):
```

**Same default** in `LocalSearch` and `BasicSearch`

### FileIntel Override

**Location:** `src/fileintel/rag/graph_rag/services/graphrag_service.py`

**Global Search Call (Line 579-588):**
```python
result, context = await global_search(
    config=graphrag_config,
    entities=dataframes["entities"],
    communities=dataframes["communities"],
    community_reports=dataframes["community_reports"],
    community_level=self.settings.graphrag.community_levels,
    dynamic_community_selection=True,
    response_type="text",  # <--- FILEINTEL OVERRIDE
    query=query,
)
```

**Local Search Call (Line 619-630):**
```python
result, context = await local_search(
    config=graphrag_config,
    entities=dataframes["entities"],
    communities=dataframes["communities"],
    community_reports=dataframes["community_reports"],
    text_units=dataframes.get("text_units"),
    relationships=dataframes.get("relationships"),
    covariates=covariates,
    community_level=self.settings.graphrag.community_levels,
    response_type="text",  # <--- FILELINTEL OVERRIDE
    query=query,
)
```

**Why "text"?**
- More concise than "multiple paragraphs"
- Allows flexibility in response structure
- Compatible with citation tracing

### How response_type is Used in Prompts

**Global Search Reduce Prompt:**
```
---Goal---

Generate a response of the target length and format that responds to the user's question...

Limit your response length to {max_length} words.

---Target response length and format---

{response_type}      <--- INJECTED HERE


---Analyst Reports---

{report_data}
```

**Local Search Prompt:**
```
---Goal---

Generate a response of the target length and format that responds to the user's question...

---Target response length and format---

{response_type}      <--- INJECTED HERE


---Data tables---

{context_data}
```

**Current Rendered Example:**
```
---Target response length and format---

text

Add sections and commentary to the response as appropriate...
```

**With Answer Format Enhancement:**
```
---Target response length and format---

text

# Answer Format: Single Paragraph

Please provide your answer as a **single, cohesive paragraph** under one clear headline.

Your response **must** be a JSON object with two keys:
1. `"headline"`: A string containing a clear, concise headline...
2. `"paragraph"`: A string containing the complete answer...

Add sections and commentary to the response as appropriate...
```

---

## Search Method Details

### Global Search

**Use Case:** Broad questions about entire dataset

**Method:** Map-Reduce
1. **Map:** Each community report → scored key points (parallel)
2. **Reduce:** Combine all key points → final answer

**Prompt Templates:**
- Map: `MAP_SYSTEM_PROMPT` (no response_type)
- Reduce: `REDUCE_SYSTEM_PROMPT` (has response_type)

**Variables Injected:**
- Map: `context_data`, `max_length`
- Reduce: `report_data`, `response_type`, `max_length`

**Citation Format:** `[Data: Reports (ids)]`

**Max Tokens:**
- Map: 1000 words per batch
- Reduce: 2000 words final answer

**JSON Mode:** Enabled for map phase (forces JSON output)

---

### Local Search

**Use Case:** Specific questions about entities/relationships

**Method:** Direct query on entity subgraph

**Prompt Template:** `LOCAL_SEARCH_SYSTEM_PROMPT`

**Variables Injected:**
- `context_data` - Entities, relationships, sources, etc.
- `response_type` - Format description

**Citation Format:** `[Data: Sources (ids), Entities (ids), Relationships (ids)]`

**No max_length variable** (relies on LLM's natural length control)

**No JSON mode** (free-form text response)

---

### Basic Search

**Use Case:** Simple search without graph structure

**Method:** Direct query on sources

**Prompt Template:** `BASIC_SEARCH_SYSTEM_PROMPT`

**Variables Injected:**
- `context_data` - Source documents
- `response_type` - Format description

**Citation Format:** `[Data: Sources (ids)]`

**Similar to local search but simpler**

---

## Answer Format Integration Strategy

### Recommended Approach: String Enhancement

**Pros:**
- ✅ No GraphRAG source code changes
- ✅ No new template files in graphrag module
- ✅ Simple string concatenation
- ✅ Easy to test
- ✅ Easy to rollback
- ✅ Preserves Microsoft prompt structure

**Cons:**
- ⚠️ Slightly less clean than template system
- ⚠️ Format instructions mixed with response_type

**Implementation:**

```python
# src/fileintel/rag/graph_rag/services/graphrag_service.py

class GraphRAGService:
    def __init__(self, ...):
        # ... existing init ...

        # Add format manager
        from fileintel.prompt_management import AnswerFormatManager
        formats_dir = Path(__file__).parent.parent.parent.parent / "prompts" / "examples"
        self.format_manager = AnswerFormatManager(formats_dir)

    async def global_search(
        self,
        query: str,
        collection_id: str,
        answer_format: str = "default"  # NEW PARAMETER
    ):
        """Performs a global search on the GraphRAG index."""
        # ... existing code to load dataframes ...

        # Build enhanced response_type
        base_response_type = "text"
        enhanced_response_type = base_response_type

        if answer_format != "default":
            try:
                format_template = self.format_manager.get_format_template(answer_format)
                enhanced_response_type = f"{base_response_type}\n\n{format_template}"
                logger.debug(
                    f"Enhanced response_type with format '{answer_format}' "
                    f"(added {len(format_template)} chars)"
                )
            except (ValueError, IOError) as e:
                logger.warning(
                    f"Failed to load format template '{answer_format}': {e}. "
                    "Using default response type."
                )

        # Pass enhanced response_type to GraphRAG
        result, context = await global_search(
            config=graphrag_config,
            entities=dataframes["entities"],
            communities=dataframes["communities"],
            community_reports=dataframes["community_reports"],
            community_level=self.settings.graphrag.community_levels,
            dynamic_community_selection=True,
            response_type=enhanced_response_type,  # ENHANCED
            query=query,
        )

        return self.data_adapter.convert_response(result, context)

    async def local_search(
        self,
        query: str,
        collection_id: str,
        community: str,
        answer_format: str = "default",  # NEW PARAMETER
        MOCK_ARGM=None
    ):
        """Performs a local search within a specific community."""
        # ... existing code to load dataframes ...

        # Build enhanced response_type (same pattern as global_search)
        base_response_type = "text"
        enhanced_response_type = base_response_type

        if answer_format != "default":
            try:
                format_template = self.format_manager.get_format_template(answer_format)
                enhanced_response_type = f"{base_response_type}\n\n{format_template}"
                logger.debug(
                    f"Enhanced response_type with format '{answer_format}' "
                    f"(added {len(format_template)} chars)"
                )
            except (ValueError, IOError) as e:
                logger.warning(
                    f"Failed to load format template '{answer_format}': {e}. "
                    "Using default response type."
                )

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
            response_type=enhanced_response_type,  # ENHANCED
            query=query,
        )

        return self.data_adapter.convert_response(result, context)
```

**Changes Required:**
1. Add `format_manager` to `GraphRAGService.__init__()`
2. Add `answer_format` parameter to `global_search()` and `local_search()`
3. Enhance `response_type` string before passing to GraphRAG API
4. Add logging for debugging

**Lines Changed:** ~30 lines total

---

### Alternative Approach: Helper Method

**Cleaner code organization:**

```python
class GraphRAGService:
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

    async def global_search(self, query, collection_id, answer_format="default"):
        # ... load dataframes ...

        response_type = self._build_response_type("text", answer_format)

        result, context = await global_search(
            ...,
            response_type=response_type,
            ...
        )
```

**Pros:**
- ✅ Cleaner separation of concerns
- ✅ Reusable across search methods
- ✅ Easier to test
- ✅ Better logging/debugging

---

### Token Budget Considerations

**Current Prompt Sizes:**

| Component | Tokens (approx) |
|-----------|----------------|
| Global Map System Prompt | ~450 |
| Global Reduce System Prompt | ~550 |
| Local Search System Prompt | ~400 |
| Community Reports (context) | 8000 max |
| Map Key Points (reduce input) | 8000 max |

**Answer Format Template Sizes:**

| Format | Tokens (approx) |
|--------|----------------|
| single_paragraph | ~150 |
| table | ~120 |
| list | ~100 |
| json | ~80 |
| essay | ~140 |
| markdown | ~110 |

**Impact Analysis:**

Adding format template to `response_type`:
- **Global Reduce:** 550 → ~700 tokens (+27%)
- **Local Search:** 400 → ~550 tokens (+38%)

**Context Budget Impact:**
- Prompts use ~5% more tokens
- Context (reports/data) dominates token usage (8000 tokens)
- Overall impact: < 2% of total query tokens

**Mitigation:** If tokens become tight, can truncate format template or reduce context slightly

---

## Safety and Preservation Requirements

### What MUST NOT Change

1. **Citation Format**
   - `[Data: Reports (ids)]` for global search
   - `[Data: Sources (ids), Entities (ids), ...]` for local/basic
   - "Max 5 record ids" rule
   - "+more" indicator

2. **Microsoft Prompt Files**
   - DO NOT modify files in `src/graphrag/prompts/query/*.py`
   - These are copyrighted Microsoft code
   - Changes would complicate graphrag upgrades

3. **GraphRAG Engine Behavior**
   - Map-reduce pattern must work as-is
   - JSON mode in map phase preserved
   - Dynamic community selection unchanged
   - Importance scoring preserved

4. **Citation Tracing**
   - FileIntel's post-processing must still work
   - `_trace_and_format_citations()` must parse output
   - Harvard citation replacement must succeed

### What CAN Change

1. ✅ **response_type value** - It's a string parameter, we can enhance it
2. ✅ **FileIntel service layer** - Our code, we control it
3. ✅ **Format template files** - New files, no impact on existing

### Validation Requirements

Before deploying, verify:

**Format Compliance:**
- [ ] Format templates don't conflict with citation requirements
- [ ] All formats include citation compatibility notes
- [ ] JSON formats include citation fields

**GraphRAG Compatibility:**
- [ ] `response_type` enhancement doesn't break prompt rendering
- [ ] LLM still produces `[Data: Reports (ids)]` citations
- [ ] Citation tracing still parses output correctly

**Backward Compatibility:**
- [ ] `answer_format="default"` produces identical output
- [ ] No regression in answer quality
- [ ] Citation accuracy maintained

**Token Budget:**
- [ ] Total prompt size < 16K tokens
- [ ] Context not truncated excessively
- [ ] Map phase still fits in context window

---

## Testing Strategy

### Unit Tests

**Test: Format Template Loading**
```python
def test_format_manager_loads_templates():
    manager = AnswerFormatManager("prompts/examples")
    template = manager.get_format_template("single_paragraph")
    assert "paragraph" in template.lower()
    assert "headline" in template.lower()
```

**Test: Response Type Building**
```python
def test_build_response_type_with_format():
    service = GraphRAGService(...)
    response_type = service._build_response_type("text", "single_paragraph")
    assert "text" in response_type
    assert "paragraph" in response_type
```

**Test: Response Type Building Default**
```python
def test_build_response_type_default():
    service = GraphRAGService(...)
    response_type = service._build_response_type("text", "default")
    assert response_type == "text"
```

### Integration Tests

**Test: Global Search with Format**
```python
async def test_global_search_with_single_paragraph_format():
    service = GraphRAGService(...)
    result = await service.global_search(
        query="What is machine learning?",
        collection_id="test-collection",
        answer_format="single_paragraph"
    )

    # Verify answer exists
    assert "answer" in result

    # Verify citations present
    assert "[Data: Reports (" in result["raw_answer"]

    # Note: Can't verify JSON structure since citation tracing
    # converts to Harvard format, but can verify no errors
```

**Test: Local Search with Format**
```python
async def test_local_search_with_table_format():
    service = GraphRAGService(...)
    result = await service.local_search(
        query="Compare X and Y",
        collection_id="test-collection",
        community="",
        answer_format="table"
    )

    assert "answer" in result
    assert "[Data:" in result["raw_answer"]  # Citations present
```

### End-to-End Tests

**Test: API Query with Format**
```python
def test_query_api_with_answer_format():
    response = client.post(
        "/api/v2/collections/test/query",
        json={
            "question": "What is AI?",
            "search_type": "graph",
            "answer_format": "single_paragraph"
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data["data"]
    assert "sources" in data["data"]
```

**Test: Citation Tracing Compatibility**
```python
async def test_citation_tracing_with_formatted_answer():
    """Verify citation tracing works with formatted answers."""
    service = GraphRAGService(...)

    # Get answer with format
    result = await service.query(
        query="Test query",
        collection_id="test",
        search_type="global",
        answer_format="single_paragraph"
    )

    # Verify Harvard citations in final answer
    assert "(" in result["answer"] and ")" in result["answer"]

    # Verify sources traced
    assert len(result["sources"]) > 0

    # Verify raw answer has GraphRAG citations
    assert "[Data: Reports (" in result["raw_answer"]
```

---

## Prompt Template Files Structure (Proposed)

**NOT RECOMMENDED** but documented for completeness:

If we wanted to convert GraphRAG to use `.md` templates (Phase 2+):

```
prompts/templates/graphrag/
├── global_search/
│   ├── map_system_prompt.md
│   └── reduce_system_prompt.md
├── local_search/
│   └── system_prompt.md
└── basic_search/
    └── system_prompt.md
```

**Reasons to defer:**
1. Requires modifying Microsoft code
2. Higher risk of breaking graphrag upgrades
3. String enhancement approach is sufficient
4. Can revisit in Phase 2 if needed

---

## Variable Reference Table

### Global Search

| Variable | Phase | Type | Source | Purpose |
|----------|-------|------|--------|---------|
| `context_data` | Map | String | Community reports | Data for analysis |
| `max_length` | Map | Integer | Config (1000) | Token limit |
| `report_data` | Reduce | String | Map results | Analyst reports |
| `response_type` | Reduce | String | Parameter | **Format control** |
| `max_length` | Reduce | Integer | Config (2000) | Token limit |

### Local Search

| Variable | Type | Source | Purpose |
|----------|------|--------|---------|
| `context_data` | String | Entities/relationships/sources | Data tables |
| `response_type` | String | Parameter | **Format control** |

### Basic Search

| Variable | Type | Source | Purpose |
|----------|------|--------|---------|
| `context_data` | String | Source documents | Data tables |
| `response_type` | String | Parameter | **Format control** |

---

## Migration Path from Current to Enhanced

### Phase 1: Add Format Manager (Sprint 4)

1. Add `AnswerFormatManager` to `GraphRAGService.__init__()`
2. Add `answer_format` parameter to `query()` method
3. Create `_build_response_type()` helper method
4. Add logging for debugging

**Estimated Time:** 1 hour

### Phase 2: Update Search Methods (Sprint 4)

1. Add `answer_format` to `global_search()` signature
2. Add `answer_format` to `local_search()` signature
3. Call `_build_response_type()` in both methods
4. Pass enhanced `response_type` to GraphRAG API

**Estimated Time:** 1 hour

### Phase 3: Wire Through Stack (Sprint 4)

1. Update `query()` method to accept `answer_format`
2. Pass `answer_format` to search method calls
3. Update `global_query()` and `local_query()` wrappers

**Estimated Time:** 30 minutes

### Phase 4: Testing (Sprint 5)

1. Unit tests for `_build_response_type()`
2. Integration tests for search methods
3. End-to-end API tests
4. Citation tracing validation

**Estimated Time:** 1.5 hours

**Total: ~4 hours for GraphRAG integration**

---

## Conclusion

### Summary of Findings

1. **response_type is the perfect injection point** for answer format templates
2. **No GraphRAG source modifications needed** - string enhancement approach works
3. **Very low risk** - only enhances one parameter value
4. **Easy to test and validate** - clear before/after comparison
5. **Preserves all existing functionality** - citations, scoring, map-reduce, etc.

### Recommended Next Steps

1. ✅ **Documentation complete** - This document
2. ⏭️ **Implement AnswerFormatManager** - Sprint 1 (already planned)
3. ⏭️ **Refactor Vector RAG** - Sprint 2 (already planned)
4. ⏭️ **GraphRAG Integration** - Sprint 4 (use string enhancement approach)
5. ⏭️ **Testing** - Sprint 5 (verify citation tracing compatibility)

### Success Criteria

- ✅ `answer_format="default"` produces identical GraphRAG output
- ✅ Format templates work with all search types (global/local/basic)
- ✅ Citations preserved: `[Data: Reports (ids)]` → Harvard format
- ✅ Citation tracing `_trace_and_format_citations()` still works
- ✅ No performance degradation
- ✅ Token budget within limits

---

**Document Status:** Complete ✅

**Next Action:** Ready for Sprint 4 - GraphRAG Integration

---

**End of Analysis**
