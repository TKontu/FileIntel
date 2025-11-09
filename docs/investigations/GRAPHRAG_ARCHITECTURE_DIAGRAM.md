# GraphRAG Architecture: Data Flow Diagram

## Query Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                           USER QUERY                                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │ Query Type?     │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼─────┐      ┌─────▼──────┐     ┌─────▼──────┐
    │  GLOBAL  │      │   LOCAL    │     │   DRIFT    │
    │  SEARCH  │      │  SEARCH    │     │   SEARCH   │
    └────┬─────┘      └─────┬──────┘     └─────┬──────┘
         │                  │                  │
         │                  │                  │
    ┌────▼──────────────────▼──────────────────▼──────┐
    │         CONTEXT BUILDING PHASE                  │
    └────┬───────────────────────────────────────────┘
         │
         │   ┌──────────────────────────────────────────────────┐
         │   │  GLOBAL SEARCH CONTEXT BUILDER                  │
         │   │  ▼ GlobalCommunityContext.build_context()       │
         │   │  ▼ build_community_context(                     │
         │   │       use_community_summary=False,              │
         │   │       use_community_summary=False <- Uses       │
         │   │    )                                            │
         │   │                                                  │
         │   │  RESULT: CSV Table Format                       │
         │   │  ┌────────────────────────────────────────┐    │
         │   │  │ id │ title │ rank │ FULL_CONTENT     │    │
         │   │  │ 1  │ Comm1 │ 0.9  │ "Summary of..."  │    │
         │   │  │ 2  │ Comm2 │ 0.7  │ "Summary of..."  │    │
         │   │  └────────────────────────────────────────┘    │
         │   │                                                  │
         │   │  ✓ Includes: Community Reports                  │
         │   │  ✗ Does NOT include: Text Units/Chunks         │
         │   └──────────────────────────────────────────────────┘
         │
         │   ┌──────────────────────────────────────────────────┐
         │   │  LOCAL SEARCH CONTEXT BUILDER                   │
         │   │  ▼ LocalSearchMixedContext.build_context()      │
         │   │                                                  │
         │   │  Three Components Combined:                     │
         │   │                                                  │
         │   │  1. Community Context (25%)                    │
         │   │     ├─ CommunityReport.full_content            │
         │   │     └─ Format: CSV with full report             │
         │   │                                                  │
         │   │  2. Text Units (50%) ← ORIGINAL CHUNKS!         │
         │   │     ├─ TextUnit.text                           │
         │   │     └─ Format: CSV with original text           │
         │   │                                                  │
         │   │  3. Metadata (25%)                              │
         │   │     ├─ Entities, Relationships, Covariates      │
         │   │     └─ Format: CSV tables                       │
         │   │                                                  │
         │   │  RESULT: Rich Context Matrix                    │
         │   │  ┌────────────────────────────────────────┐    │
         │   │  │ Reports                                │    │
         │   │  │ ├─ id │ title │ content              │    │
         │   │  ├────────────────────────────────────────┤    │
         │   │  │ Sources (TEXT UNITS)                  │    │
         │   │  │ ├─ id │ text (ORIGINAL CHUNK)        │    │
         │   │  ├────────────────────────────────────────┤    │
         │   │  │ Entities                               │    │
         │   │  │ ├─ id │ name │ description           │    │
         │   │  ├────────────────────────────────────────┤    │
         │   │  │ Relationships                          │    │
         │   │  │ ├─ source │ target │ description      │    │
         │   │  └────────────────────────────────────────┘    │
         │   │                                                  │
         │   │  ✓ Includes: Text Units, Reports, Metadata     │
         │   │  ✓ Text units are ALWAYS included              │
         │   │  ✓ Proportions are configurable               │
         │   └──────────────────────────────────────────────────┘
         │
         │   ┌──────────────────────────────────────────────────┐
         │   │  DRIFT SEARCH CONTEXT BUILDER                   │
         │   │  ▼ DRIFTSearchContextBuilder.build_context()    │
         │   │                                                  │
         │   │  Inherits from: LocalSearchMixedContext          │
         │   │  Additional Phase: Primer Query Expansion        │
         │   │  ├─ Uses: CommunityReport.full_content          │
         │   │  └─ Purpose: Generate hypothetical answer       │
         │   │                                                  │
         │   │  Final Context: Same as Local Search            │
         │   │  ✓ Includes: Text Units, Reports, Metadata      │
         │   └──────────────────────────────────────────────────┘
         │
         │
    ┌────▼──────────────────────────────────────────────┐
    │           LLM INFERENCE PHASE                     │
    └────┬───────────────────────────────────────────────┘
         │
         │   ┌──────────────────────────────────────────┐
         │   │ GLOBAL SEARCH (Map-Reduce)              │
         │   │                                          │
         │   │ MAP Phase:                              │
         │   │ ├─ Process each community batch         │
         │   │ ├─ Extract key points from summaries    │
         │   │ └─ 1 LLM call per batch (parallel)      │
         │   │                                          │
         │   │ REDUCE Phase:                           │
         │   │ ├─ Combine all key points               │
         │   │ ├─ Synthesize final answer              │
         │   │ └─ 1 final LLM call                      │
         │   │                                          │
         │   │ Total Calls: 2-N (batch + 1 reduce)     │
         │   └──────────────────────────────────────────┘
         │
         │   ┌──────────────────────────────────────────┐
         │   │ LOCAL SEARCH (Single Call)              │
         │   │                                          │
         │   │ Process rich context:                   │
         │   │ ├─ Entity mapping                       │
         │   │ ├─ Relationship analysis                │
         │   │ ├─ Text unit ranking                    │
         │   │ └─ All in one LLM call                  │
         │   │                                          │
         │   │ Total Calls: 1                          │
         │   └──────────────────────────────────────────┘
         │
         │   ┌──────────────────────────────────────────┐
         │   │ DRIFT SEARCH (Decomposed + Local)       │
         │   │                                          │
         │   │ Primer Phase:                           │
         │   │ ├─ Decompose query into subqueries      │
         │   │ └─ N LLM calls for decomposition        │
         │   │                                          │
         │   │ Local Phase:                            │
         │   │ ├─ Process with local context           │
         │   │ └─ 1 LLM call for local search          │
         │   │                                          │
         │   │ Total Calls: N + 1                      │
         │   └──────────────────────────────────────────┘
         │
         │
    ┌────▼─────────────────────────────────────────────┐
    │         RESPONSE ASSEMBLY & RETURN               │
    └──────────────────────────────────────────────────┘
```

---

## Data Source Comparison

```
                    ┌─────────────────────┐
                    │   RAW KNOWLEDGE     │
                    │   BASE DOCUMENTS    │
                    │   (Original Text)   │
                    └──────────┬──────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
           ┌────▼────┐  ┌─────▼────┐  ┌─────▼────┐
           │Text     │  │ Entities │  │Relations │
           │Units    │  │          │  │          │
           │(Chunks) │  │ Extracted│  │Extracted │
           └────┬────┘  └─────┬────┘  └─────┬────┘
                │             │            │
                │   ┌─────────┼────────┐   │
                │   │                  │   │
           ┌────▼───▼──────────────────▼───┴──────┐
           │   COMMUNITY REPORTS                  │
           │   (LLM-generated summaries)          │
           │   ├─ summary (short)                 │
           │   └─ full_content (longer)           │
           └────┬───────────────────────────────┘
                │
        ┌───────┴──────────┬──────────────┐
        │                  │              │
   ┌────▼─────┐    ┌─────▼──────┐  ┌───▼──────┐
   │ GLOBAL   │    │   LOCAL    │  │  DRIFT   │
   │ SEARCH   │    │   SEARCH   │  │  SEARCH  │
   └──────────┘    └────────────┘  └──────────┘
        │                │              │
        │                │              │
   Uses Reports      Uses Reports +  Uses Reports +
   ONLY              Text Units      Text Units
        │                │
   No access to      Direct access
   original text      to original
   chunks            chunks
```

---

## Key Code Paths

### Global Search Data Flow
```
Query
  ↓
GlobalSearch.search()
  ↓
GlobalCommunityContext.build_context()
  ↓
build_community_context(use_community_summary=False)
  ↓
← Returns CSV with FULL_CONTENT only (NOT text units)
  ↓
LLM receives community reports in table format
  ↓
LLM extracts key points from reports
  ↓
Combine and synthesize final answer
```

**Data passed to LLM**: Community report summaries/full_content in CSV table
**Text chunks passed**: NONE

### Local Search Data Flow
```
Query
  ↓
LocalSearch.search()
  ↓
LocalSearchMixedContext.build_context()
  ↓
├─ _build_community_context() → Reports (25%)
├─ _build_local_context() → Entities/Relationships (25%)
└─ _build_text_unit_context() → TEXT UNITS (50%) ← ORIGINAL CHUNKS!
  ↓
← Returns combined context with all three components
  ↓
LLM receives rich context including original text
  ↓
LLM synthesizes answer using all available information
```

**Data passed to LLM**: Reports + Entities/Relationships + TEXT UNITS
**Text chunks passed**: YES, 50% of context by default

---

## Configuration Parameters

### Global Search
```python
GlobalSearch(
    # Only affects community reports
    use_community_summary: bool = False  # False → use full_content
                                        # True → use summary
)
```
**No parameter to include text units**

### Local Search
```python
LocalSearchMixedContext.build_context(
    text_unit_prop: float = 0.5,      # % of context for text units
    community_prop: float = 0.25,     # % of context for reports
    # (remaining ~25% for entities/relationships/covariates)
    
    use_community_summary: bool = False  # False → use full_content
                                         # True → use summary
)
```
**Text units are ALWAYS included when using local search**

### DRIFT Search
```python
DRIFTSearchContextBuilder(
    # Uses LocalSearchMixedContext under the hood
    # Inherits all text unit support
)
```

---

## Summary Table

| Aspect | Global Search | Local Search | DRIFT Search |
|--------|---------------|--------------|--------------|
| **Primary Data** | Community reports | Reports + Chunks + Metadata | Reports + Chunks + Metadata |
| **Text Units Used** | NO | YES (50% default) | YES (via local context) |
| **LLM Calls** | 2-N | 1 | N+1 |
| **Use Case** | Broad overview | Entity-specific detail | Complex decomposed queries |
| **Context Size** | Minimal (summaries) | Large (mixed) | Large (mixed) |
| **Quality Focus** | Breadth | Depth | Structured depth |
| **Configuration** | Limited (summary toggle) | Flexible (proportions) | Flexible (inherits local) |

