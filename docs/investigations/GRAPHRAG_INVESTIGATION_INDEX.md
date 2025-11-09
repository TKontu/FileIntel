# GraphRAG Investigation: Complete Documentation Index

## Overview

This directory contains a comprehensive investigation of the Microsoft GraphRAG library's design, specifically examining whether it uses original chunks or summaries for answer generation.

**Key Finding**: GraphRAG is **summary-focused by design**. Only Local Search and DRIFT Search include original text chunks.

---

## Document Guide

### 1. GRAPHRAG_INVESTIGATION_FINDINGS.md
**Primary Investigation Report** - Start here

Contains:
- Executive summary
- Global search implementation details
- Local search implementation details
- DRIFT search overview
- Data model structure
- Configuration parameters
- Architectural design intent
- Answer to all 4 key questions

**Best for**: Understanding WHAT the library does and WHY

---

### 2. GRAPHRAG_ARCHITECTURE_DIAGRAM.md
**Visual & Structural Reference**

Contains:
- Query processing pipeline diagram
- Data source comparison hierarchy
- Code flow paths (Global vs Local vs DRIFT)
- Configuration parameters reference
- Summary comparison table

**Best for**: Visual learners and quick reference

---

### 3. GRAPHRAG_IMPLEMENTATION_IMPLICATIONS.md
**Practical Guidance for Your Project**

Contains:
- Current state analysis
- Key design decisions explained
- Recommendations for FileIntel
- Query-type routing strategy
- Configuration examples
- Token cost analysis
- Decision matrix
- Testing strategy

**Best for**: Implementation planning and decision-making

---

## Quick Answers

### Q: Does Microsoft's GraphRAG use original chunks?

**A**: 
- **Global Search**: NO - uses summaries only
- **Local Search**: YES - includes chunks (50% of context)
- **DRIFT Search**: YES - includes chunks (inherits from Local)

### Q: Can I configure Global Search to use chunks?

**A**: NO - this is a fundamental design choice, not a configuration option.

### Q: Which mode should I use?

**A**: 
| Use Case | Mode |
|----------|------|
| Broad overview | Global |
| Entity details | Local |
| With sources | Local |
| Complex queries | DRIFT |

### Q: Where are the original chunks used?

**A**: Only in:
1. `LocalSearchMixedContext._build_text_unit_context()` 
2. `build_text_unit_context()` in source_context.py
3. Results are proportionally mixed with reports/metadata

---

## Key Code References

### Files That Process Chunks
- `/src/graphrag/query/context_builder/source_context.py` - Builds text unit context
- `/src/graphrag/query/structured_search/local_search/mixed_context.py` - Uses text units
- `/src/graphrag/query/input/retrieval/text_units.py` - Converts units to DataFrame

### Files That DON'T Use Chunks
- `/src/graphrag/query/structured_search/global_search/search.py` - No text units
- `/src/graphrag/query/structured_search/global_search/community_context.py` - No text units
- `/src/graphrag/query/context_builder/dynamic_community_selection.py` - No text units

### Configuration
- `/src/graphrag/query/factory.py` - Default configurations

### Data Models
- `/src/graphrag/data_model/community_report.py` - Has `summary` and `full_content`
- `/src/graphrag/data_model/text_unit.py` - Has `text` (original chunks)

---

## Investigation Methodology

This investigation examined:

1. **Code Analysis**
   - Traced data flow through search modes
   - Examined context building functions
   - Checked LLM prompt construction
   - Reviewed data model definitions

2. **Configuration Analysis**
   - Searched for toggles/flags for chunk usage
   - Examined factory defaults
   - Checked parameter documentation

3. **Architectural Analysis**
   - Identified hierarchical design
   - Analyzed separation of concerns
   - Examined optimization choices

4. **Evidence Collection**
   - Direct code references
   - Configuration inspection
   - System prompt examination
   - Data model inspection

---

## Key Findings Summary

### Global Search (Summary-Only)
```
Uses: CommunityReport.full_content or .summary
Does NOT use: TextUnit.text (original chunks)
Pattern: Map-reduce over summaries
Cost: Efficient, scales to large KB
Best for: Broad overview questions
```

### Local Search (Mixed Context)
```
Uses: TextUnit.text (50%) + Reports (25%) + Metadata (25%)
Proportions: Configurable via text_unit_prop parameter
Pattern: Single LLM call with rich context
Cost: Higher tokens, but still manageable
Best for: Entity-specific detail questions
```

### DRIFT Search (Hybrid)
```
Uses: Same as Local Search (chunks included)
Pattern: Query decomposition + local context
Cost: Moderate (primer + local)
Best for: Complex multi-faceted queries
```

---

## Design Philosophy

Microsoft's GraphRAG implements a **hierarchical abstraction strategy**:

```
Raw Documents
    ↓
Text Units (Chunks)
    ↓
Entities, Relationships (Structured)
    ↓
Community Reports (Summaries)
    ↓
Global Search (Summaries) OR Local Search (Mixed)
```

**Trade-offs**:
- **Global Search**: Breadth over depth, efficiency over detail
- **Local Search**: Depth over breadth, traceability over coverage
- **DRIFT Search**: Complexity handling with hierarchical awareness

---

## Recommendations

### For FileIntel Project

1. **Use Local Search** if you need original chunks
2. **Use Global Search** for broad summaries
3. **Route queries** based on type (broad vs. detailed)
4. **Adjust proportions** in Local Search if needed
5. **Don't try to modify** Global Search for chunks

### If You Want Both

Option A: Local Search with `text_unit_prop=0.7`
```python
# 70% original chunks, 20% reports, 10% metadata
```

Option B: Call both and merge results
```python
# Global for context, Local for details
```

Option C: Accept the architectural boundary
```python
# Each search mode has a purpose
# Use the right tool for the right question
```

---

## File Locations

All investigation documents are in:
```
/home/tuomo/code/fileintel/
├── GRAPHRAG_INVESTIGATION_INDEX.md (this file)
├── GRAPHRAG_INVESTIGATION_FINDINGS.md (main report)
├── GRAPHRAG_ARCHITECTURE_DIAGRAM.md (visual reference)
└── GRAPHRAG_IMPLEMENTATION_IMPLICATIONS.md (practical guide)
```

---

## Next Steps

1. **Read** GRAPHRAG_INVESTIGATION_FINDINGS.md for detailed analysis
2. **Review** GRAPHRAG_ARCHITECTURE_DIAGRAM.md for visual understanding
3. **Plan** using GRAPHRAG_IMPLEMENTATION_IMPLICATIONS.md
4. **Decide** which search mode(s) to use
5. **Implement** query routing if using multiple modes

---

## Questions Answered

- [x] Does GraphRAG use original chunks?
- [x] Where are chunks used/not used?
- [x] Can you toggle between chunks and summaries?
- [x] What's the architectural reasoning?
- [x] How to best use the library?
- [x] What are the limitations?
- [x] Token costs comparison?

---

## Investigation Completeness

This investigation covered:
- [x] Global Search implementation (100%)
- [x] Local Search implementation (100%)
- [x] DRIFT Search implementation (100%)
- [x] Context builders (100%)
- [x] Data models (100%)
- [x] Configuration system (100%)
- [x] System prompts (100%)
- [x] Factory defaults (100%)
- [x] Code references (complete)

Not covered (not in scope):
- Indexing phase (focused on query phase)
- Embedding models
- Vector store implementations
- LLM provider specifics

---

## Document Versions

- Investigation Date: 2025-11-09
- GraphRAG Library: Microsoft (as implemented in /src/graphrag/)
- Scope: Query phase data usage
- Completeness: Comprehensive (all major code paths examined)

