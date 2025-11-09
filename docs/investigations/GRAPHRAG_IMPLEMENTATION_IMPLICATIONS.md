# Implementation Implications: GraphRAG Design Choices

## Overview

This document outlines the practical implications of GraphRAG's architectural decisions regarding data usage for your implementation.

---

## Current State Analysis

### What the Library Does

1. **Global Search**: Processes summaries of communities in parallel
   - Designed for broad, overview-level questions
   - Minimal token usage
   - Fast execution

2. **Local Search**: Combines summaries, chunks, and structured metadata
   - Designed for detailed, entity-specific questions  
   - Richer context
   - Single LLM call

3. **DRIFT Search**: Hybrid approach with query decomposition
   - Designed for complex multi-faceted queries
   - Primer phase + local search phase
   - Medium complexity

---

## Key Design Decisions & Why They Matter

### Decision 1: Global Search Uses Summaries Only

**What This Means**:
- Global search never accesses `text_unit.text` (original chunks)
- LLM only sees `community_report.full_content` or `community_report.summary`
- No way to configure this without modifying source code

**Why Microsoft Did This**:
1. **Scalability**: Can process hundreds of communities efficiently
2. **Cost**: Fewer tokens needed for large knowledge bases
3. **Speed**: Parallel processing of summary batches
4. **Quality**: Summaries filter out noise while preserving key info

**Impact for Your Use Case**:
- If you have a large knowledge base, use Global Search
- If you need chunk-level detail, use Local Search
- If you want both, you need to call both and merge results

### Decision 2: Local Search Includes Text Units by Default

**What This Means**:
- Local search ALWAYS includes original text chunks
- Chunks occupy ~50% of the context window by default
- This is configurable via `text_unit_prop` parameter

**Why Microsoft Did This**:
1. **Detail**: Local search targets specific entities, needs chunk-level info
2. **Traceability**: Chunks allow exact source references
3. **Flexibility**: Users can adjust proportions per query

**Impact for Your Use Case**:
- If you want chunks in your results, use Local Search
- You can adjust `text_unit_prop` from 0.0 (no chunks) to 1.0 (only chunks)
- Good for entity-focused questions and detailed analysis

### Decision 3: Hierarchical Abstraction (No Downward Access)

**What This Means**:
- Global search can't access entities/relationships directly
- Local search can't skip summaries to use only chunks
- Architecture goes UP (chunks → entities → communities) not DOWN

**Why Microsoft Did This**:
1. **Separation of Concerns**: Different search modes for different needs
2. **Optimization**: Each mode is optimized for its use case
3. **Predictability**: Behavior is deterministic and well-defined

**Impact for Your Use Case**:
- You can't "mix and match" behaviors easily
- Must choose appropriate search mode per query type
- Consider implementing query routing logic

---

## Recommendations for Your Implementation

### For FileIntel Project

Based on the investigation, here are recommendations:

#### 1. Implement Query-Type Routing

```python
def route_query(query: str) -> SearchMode:
    """Route query to appropriate search mode."""
    
    # Broad questions → Global Search
    if is_broad_question(query):
        return SearchMode.GLOBAL
    
    # Entity/Detail questions → Local Search  
    elif is_entity_question(query):
        return SearchMode.LOCAL
    
    # Complex multi-faceted → DRIFT Search
    elif is_complex_question(query):
        return SearchMode.DRIFT
```

#### 2. Global Search Best For:
- "What are the main findings in this dataset?"
- "Summarize the key results across all documents"
- "What patterns exist globally?"
- "Overall trends and observations"

#### 3. Local Search Best For:
- "What is known about entity X?"
- "How are entities X and Y related?"
- "Details about specific topics"
- "Evidence-based answers with sources"

#### 4. If You Need Both Summaries and Chunks

Option A: Use Local Search with adjusted proportions
```python
context = local_search.build_context(
    query=query,
    text_unit_prop=0.7,      # 70% chunks
    community_prop=0.2,      # 20% reports
    # 10% metadata
)
```

Option B: Call both search modes and merge
```python
global_result = global_search.search(query)      # Summaries
local_result = local_search.search(query)        # Chunks + summaries

# Combine results with global providing context,
# local providing details
combined = merge_results(global_result, local_result)
```

Option C: Modify source code (advanced)
```python
# Update build_community_context() to optionally include
# text_units alongside community reports
# NOT RECOMMENDED without thorough testing
```

#### 5. Configuration Recommendations

```python
# For broad overview queries
global_search_config = {
    "use_community_summary": False,  # Use full_content for more detail
    "response_type": "multiple paragraphs",
    "allow_general_knowledge": True,  # Broader context
}

# For detailed entity queries
local_search_config = {
    "text_unit_prop": 0.5,           # Balance chunks and reports
    "community_prop": 0.25,
    "top_k_mapped_entities": 10,
    "use_community_summary": False,  # Use full_content
}

# For complex queries
drift_search_config = {
    "text_unit_prop": 0.5,           # Same as local
    "community_prop": 0.25,
    "primer_folds": 3,               # Decompose into 3 parts
}
```

---

## Token Cost Analysis

### Estimated Tokens Per Search Mode

#### Global Search (1000 communities)
```
MAP phase:
  ├─ 10 batches × 50 communities each
  ├─ Avg 500 tokens per community report
  ├─ Total MAP: 10 × (50 × 500 + prompt) ≈ 250K tokens
  └─ 10 LLM calls

REDUCE phase:
  ├─ Combine 10 outputs
  ├─ Avg 5K tokens per output
  └─ Total REDUCE: 50K tokens + 1 LLM call

Total: ~300K tokens, 11 LLM calls
```

#### Local Search (Entity-focused)
```
Entity mapping:
  ├─ 1 embedding call
  └─ 100 tokens

Context building:
  ├─ 50 community tokens (25% of 8K max)
  ├─ 400 text unit tokens (50% of 8K max)
  ├─ 150 entity tokens (25% of 8K max)
  └─ Total context: ~600 tokens

LLM call:
  ├─ Prompt + context: ~700 tokens
  ├─ Response: ~500 tokens
  └─ Total: ~1.2K tokens, 1 LLM call

Total: ~1.3K tokens, 1-2 LLM calls
```

#### DRIFT Search (Complex)
```
Query Expansion:
  ├─ 1 embedding call

Primer Phase:
  ├─ 3 decomposition calls
  ├─ ~1K tokens each
  └─ Total: 3K tokens

Local Phase:
  ├─ Same as Local Search
  └─ 1.2K tokens

Total: ~4.2K tokens, 5 LLM calls
```

**Cost Ratio**: DRIFT (4.2K) < Local (1.3K) < Global (300K)

Note: Global search costs increase with knowledge base size; Local/DRIFT stay constant.

---

## Practical Decision Matrix

| Scenario | Recommended | Reasoning |
|----------|------------|-----------|
| "Summarize everything" | Global | Designed exactly for this |
| "Tell me about X" | Local | Chunk access needed |
| "Compare X and Y" | Local + Global | Use both perspectives |
| "Complex analysis" | DRIFT | Query decomposition helps |
| "First-time exploration" | Global first, then Local | Start broad, drill down |
| "Production QA" | Local | More reliable, cheaper |
| "Large knowledge base" | Global | Cost-effective at scale |
| "Need exact sources" | Local | Chunks provide exact references |

---

## Known Limitations

### Global Search Limitations
- Cannot cite exact original text
- Loses granular details
- Poor for specific fact queries
- May miss nuanced information

### Local Search Limitations
- Query must map to existing entities
- Entity coverage limits scope
- Smaller view of knowledge base
- May miss broader patterns

### DRIFT Limitations
- More complex query handling
- Higher token cost than Local
- More points of failure
- Query decomposition errors compound

---

## Testing Strategy

### For Global Search
```python
def test_global_search():
    queries = [
        "What are the major themes?",
        "Summarize the findings",
        "What patterns emerge?",
    ]
    # Should succeed for broad questions
    # May fail for specific facts
```

### For Local Search
```python
def test_local_search():
    queries = [
        "Tell me about John Smith",
        "What is the relationship between X and Y?",
        "What evidence supports this?",
    ]
    # Should succeed for entity questions
    # May fail if entities not in knowledge base
```

### For Hybrid Approach
```python
def test_hybrid():
    query = "Analyze the data"
    
    # 1. Get broad context with Global
    global_context = global_search(query)
    
    # 2. Get detailed context with Local
    local_context = local_search(query)
    
    # 3. Merge and verify consistency
    assert are_consistent(global_context, local_context)
```

---

## Conclusion

Microsoft's design is intentional and well-reasoned:

1. **Global Search** = Efficiency and breadth (summaries only)
2. **Local Search** = Detail and traceability (chunks included)
3. **DRIFT Search** = Complexity handling (chunk-aware decomposition)

For FileIntel, the key question is:
- Do you need chunk-level traceability? → Use Local Search
- Do you need broad summaries? → Use Global Search
- Do you need both? → Implement hybrid routing

The architecture doesn't support "chunks in global search" by design. Trying to force this would require significant modifications and testing.

