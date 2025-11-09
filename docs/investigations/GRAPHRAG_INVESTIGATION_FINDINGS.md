# Microsoft GraphRAG Library: Summary vs. Original Chunks Analysis

## Executive Summary

After a thorough investigation of the Microsoft GraphRAG source code, **the library is definitively designed to use community report summaries for answer generation in global and local search modes, not original chunks**. This is a fundamental architectural decision, not just a default setting.

---

## Key Findings

### 1. Global Search Implementation

**File:** `/src/graphrag/query/structured_search/global_search/search.py` and `community_context.py`

#### Data Flow:
1. **Context Building**: The `GlobalCommunityContext.build_context()` method calls `build_community_context()` with a parameter `use_community_summary: bool = True`
2. **Default Behavior**: In the factory configuration (`factory.py`), global search explicitly sets `use_community_summary: False` - **but wait, this is misleading**
3. **Actual Implementation**: Looking at `community_context.py` line 75:
```python
context.append(report.summary if use_community_summary else report.full_content)
```

**Critical Finding**: When `use_community_summary=False`, it uses `report.full_content` (the full community report), **NOT original text chunks**. Original chunks (TextUnits) are never passed to the LLM in global search.

#### What Gets Sent to LLM:
- CSV table with community report summaries OR full community report content
- Never original text units/chunks
- Format: `id | title | attributes | summary/full_content | rank`

#### System Prompt (MAP phase):
```
"You are a helpful assistant responding to questions about data in the tables provided.
Generate a response consisting of a list of key points that responds to the user's question, 
summarizing all relevant information in the input data tables."
```

The prompt explicitly says "data in the tables provided" - referring to community reports, not raw chunks.

---

### 2. Local Search Implementation

**File:** `/src/graphrag/query/structured_search/local_search/mixed_context.py`

#### Data Flow:
The local search actually DOES support text units, with configurable proportions:

```python
def build_context(
    self,
    ...
    text_unit_prop: float = 0.5,
    community_prop: float = 0.25,
    use_community_summary: bool = False,
    ...
) -> ContextBuilderResult:
```

#### What Gets Included:
1. **Community Context**: Community reports (controlled by `community_prop`)
   - Uses `use_community_summary=False` by default (uses full_content)
   - Still not original chunks - these are LLM-generated summaries/full reports

2. **Text Units**: Original text chunks are included via `_build_text_unit_context()`
   - Uses actual `text_unit.text` (line 345-352)
   - Proportion controlled by `text_unit_prop` parameter (default 0.5 = 50% of context)
   
3. **Entity/Relationship/Covariate Context**: Structured metadata
   - Not original text, but extracted information

#### Local Search Prompt:
```
"Points supported by data should list their data references as follows:
[Data: Sources (record ids), Reports (record ids), Entities (record ids), etc.]"
```

**Key Insight**: Local search is the ONLY mode that includes original text chunks (Sources), and it does so by design with configurable proportions.

---

### 3. DRIFT Search (New Alternative)

**File:** `/src/graphrag/query/structured_search/drift_search/drift_context.py` and `primer.py`

DRIFT search uses the `LocalSearchMixedContext` builder, inheriting its text unit support.

**Line 63 in primer.py**:
```python
template = secrets.choice(self.reports).full_content
```

Interesting: DRIFT's query expansion phase uses `full_content`, not summaries, to create the hypothetical answer template.

---

### 4. Data Model Structure

**CommunityReport** (`community_report.py`):
```python
@dataclass
class CommunityReport(Named):
    summary: str = ""
    """Summary of the report."""
    
    full_content: str = ""
    """Full content of the report."""
```

The data model has both `summary` and `full_content`, indicating that both are generated and available.

**TextUnit** (`text_unit.py`):
```python
@dataclass
class TextUnit(Identified):
    text: str
    """The text of the unit."""
```

TextUnits contain the original chunks.

---

### 5. Configuration & Parameters

**Factory Configuration** (`factory.py`):

For Global Search:
```python
"use_community_summary": False,  # Uses full_content instead
```

For Local Search:
```python
# No explicit setting - defaults apply
# Default: text_unit_prop=0.5, community_prop=0.25, use_community_summary=False
```

**Important**: The `use_community_summary` parameter **ONLY toggles between summary and full_content of community reports**. It does NOT add or remove original text chunks.

---

### 6. Retrieved Functions That Use TextUnits

**Only in local search context building** (`source_context.py`):

```python
def build_text_unit_context(
    text_units: list[TextUnit],
    ...
) -> tuple[str, dict[str, pd.DataFrame]]:
    """Prepare text-unit data table as context data for system prompt."""
    for unit in text_units:
        new_context = [
            unit.short_id,
            unit.text,  # <- Original text
            ...
        ]
```

This function is ONLY called from `LocalSearchMixedContext._build_text_unit_context()`.

**Not called from**:
- Global search
- Dynamic community selection
- DRIFT search (though DRIFT uses LocalSearchMixedContext)

---

### 7. Architectural Design Intent

#### Evidence from Code Comments and Structure:

1. **Hierarchical Abstraction**: The architecture uses:
   - Text Units (raw chunks) at the bottom
   - Entities, Relationships (extracted/structured)
   - Community Reports (summarized at hierarchical levels)
   - The design goes UP the abstraction ladder, not down

2. **Efficiency Design**: 
   - Global search processes summaries in parallel (map-reduce)
   - Local search balances efficiency with detail (configurable proportions)
   - This suggests intentional trade-offs between comprehensiveness and efficiency

3. **Query Routing Pattern**:
   - Global search: best for broad, high-level questions
   - Local search: best for detailed, entity-specific questions
   - This makes sense if global search uses summaries and local search includes chunks

4. **No "Use Chunks" Flag**: 
   - There is NO global parameter like `--use_original_chunks` 
   - No configuration to switch global search to use text units
   - This suggests it's not intended as a toggleable feature

---

## Summary: Design Philosophy

### Global Search (Summary-Only)
- **Purpose**: Answer broad questions across entire knowledge base
- **Data Source**: Community report summaries (or full_content)
- **Pattern**: Map-Reduce over summaries
- **Reasoning**: Efficient, scalable, good for overview questions
- **Original Chunks**: Never used
- **Cost**: 1-2 LLM calls (map + reduce), minimal tokens

### Local Search (Mixed Context)
- **Purpose**: Answer detailed questions about specific entities
- **Data Source**: 
  - 50% context: original text units (chunks)
  - 25% context: community report content
  - 25% context: entity/relationship metadata
- **Pattern**: Single LLM call with rich context
- **Reasoning**: More comprehensive for local questions
- **Original Chunks**: Always included (50% of context)
- **Cost**: 1 LLM call, more tokens per call

### DRIFT Search (Hybrid)
- **Purpose**: Decompose complex queries with primer phase
- **Data Source**: Inherits from LocalSearchMixedContext
- **Reasoning**: Gets benefits of both hierarchy and chunks
- **Original Chunks**: Included via local context builder

---

## Answers to Key Questions

### Q1: Does Microsoft's GraphRAG library ever use original chunk text for answer generation?

**A1**: Only in LOCAL SEARCH mode (and DRIFT which inherits from it). Global search never uses original chunks.

### Q2: Is the summary-only approach a fundamental design decision or just the default?

**A2**: It's a FUNDAMENTAL DESIGN DECISION for global search. There is no built-in parameter to switch it. The architecture is built around hierarchical summarization.

### Q3: Are there any built-in options to use original chunks instead of summaries?

**A3**: 
- For Global Search: **NO** - there is no option to use original chunks
- For Local Search: **YES** - text units are automatically included with configurable proportions
- You could modify `community_context.py` to override this behavior, but it would require code changes

### Q4: What's the architectural reasoning behind the current approach?

**A4**:
1. **Scalability**: Summaries reduce token consumption for large knowledge bases
2. **Efficiency**: Global search can process many communities in parallel
3. **Quality**: Hierarchical abstraction preserves important information while filtering noise
4. **Flexibility**: Local search allows mixing levels of detail based on query needs
5. **Cost**: Fewer LLM calls and tokens for global queries

---

## Recommendations for Your Implementation

If you want to use original chunks:

1. **Use LOCAL SEARCH** instead of global search - it already includes text units
2. **Adjust proportions** via `text_unit_prop` parameter (0.0 to 1.0)
3. **Modify the code** if you need chunks in global search (advanced)
   - Update `build_community_context()` to optionally include text units
   - Would need to handle token limits and context management

---

## Code References

Key files examined:
- `/src/graphrag/query/structured_search/global_search/search.py` (Lines 145-147)
- `/src/graphrag/query/structured_search/global_search/community_context.py` (Lines 103-120)
- `/src/graphrag/query/structured_search/local_search/mixed_context.py` (Lines 91-222)
- `/src/graphrag/query/context_builder/community_context.py` (Lines 24-79)
- `/src/graphrag/query/context_builder/source_context.py` (Lines 21-79)
- `/src/graphrag/data_model/community_report.py`
- `/src/graphrag/data_model/text_unit.py`
- `/src/graphrag/query/factory.py` (Configuration defaults)
