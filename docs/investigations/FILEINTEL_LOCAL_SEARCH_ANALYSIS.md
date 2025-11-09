# FileIntel Local Search Analysis vs Microsoft GraphRAG

**Date:** 2025-11-09
**Analysis:** Comparison of FileIntel's local search implementation against Microsoft's GraphRAG local search pattern

---

## Executive Summary

### Current Status: ✅ **MOSTLY COMPLIANT** with Microsoft's pattern

FileIntel's `--type local` implementation **correctly uses** Microsoft's `graphrag.api.query.local_search()` function and passes all required parameters. However, it's **missing key configuration options** that control the balance between chunks, summaries, and entity descriptions.

### Key Findings

| Aspect | FileIntel | Microsoft GraphRAG | Status |
|--------|-----------|-------------------|---------|
| **Uses official API** | ✅ Yes (`from graphrag.api.query import local_search`) | ✅ Official | ✅ Correct |
| **Passes required params** | ✅ All 10 required params | ✅ All passed | ✅ Correct |
| **Uses text_units** | ✅ Yes (fixed in parquet_loader.py) | ✅ Required | ✅ Correct |
| **Uses relationships** | ✅ Yes (fixed in parquet_loader.py) | ✅ Required | ✅ Correct |
| **Configurable proportions** | ❌ **MISSING** | ✅ `text_unit_prop`, `community_prop` | ⚠️ **GAP** |
| **Exposed in config** | ❌ Not in GraphRAGSettings | ✅ In LocalSearchConfig | ⚠️ **GAP** |
| **Customizable per query** | ❌ Uses defaults only | ✅ Can override | ⚠️ **GAP** |

---

## 1. Parameter Comparison

### Microsoft GraphRAG `local_search()` Signature

```python
async def local_search(
    config: GraphRagConfig,              # ✅ FileIntel passes
    entities: pd.DataFrame,              # ✅ FileIntel passes
    communities: pd.DataFrame,           # ✅ FileIntel passes
    community_reports: pd.DataFrame,     # ✅ FileIntel passes
    text_units: pd.DataFrame,            # ✅ FileIntel passes (FIXED)
    relationships: pd.DataFrame,         # ✅ FileIntel passes (FIXED)
    covariates: pd.DataFrame | None,     # ✅ FileIntel passes (optional)
    community_level: int,                # ✅ FileIntel passes
    response_type: str,                  # ✅ FileIntel passes
    query: str,                          # ✅ FileIntel passes
    callbacks: list[QueryCallbacks] | None = None,  # ⚠️ FileIntel doesn't use
    verbose: bool = False                # ⚠️ FileIntel doesn't use
) -> tuple[str, dict]:
```

### FileIntel's Implementation

**File:** `src/fileintel/rag/graph_rag/services/graphrag_service.py:722-733`

```python
result, context = await local_search(
    config=graphrag_config,                          # ✅ Passed
    entities=dataframes["entities"],                 # ✅ Passed
    communities=dataframes["communities"],           # ✅ Passed
    community_reports=dataframes["community_reports"], # ✅ Passed
    text_units=dataframes.get("text_units"),        # ✅ Passed (NOW LOADED)
    relationships=dataframes.get("relationships"),   # ✅ Passed (NOW LOADED)
    covariates=covariates,                          # ✅ Passed (optional)
    community_level=self.settings.graphrag.community_levels,  # ✅ Passed
    response_type="text",                           # ✅ Passed
    query=query,                                    # ✅ Passed
)
```

**Conclusion:** ✅ All required parameters are correctly passed.

---

## 2. Missing Configuration: Context Proportions

### Microsoft's Local Search Context Building

Microsoft's local search builds context from **3 sources** with configurable proportions:

```python
# File: src/graphrag/config/defaults.py:309-311
text_unit_prop: float = 0.5      # 50% tokens for original chunks
community_prop: float = 0.15     # 15% tokens for community summaries
# Remaining 35% = entity descriptions + relationships + covariates
```

**How it works:**
```
Total Context = 8000 tokens (default max_context_tokens)

text_unit_prop = 0.5  → 4000 tokens for text_units.text (original chunks)
community_prop = 0.15 → 1200 tokens for community_reports.summary
local_prop = 0.35     → 2800 tokens for entities + relationships + covariates
```

### FileIntel's Current Behavior

FileIntel uses **Microsoft's defaults** because it doesn't override these settings:

```python
# FileIntel implicitly uses:
text_unit_prop = 0.5   # 50% chunks (Microsoft default)
community_prop = 0.15  # 15% summaries (Microsoft default)
```

**This means FileIntel's local search DOES use chunks!** ✅

**But:** Users cannot customize the proportions.

---

## 3. Configuration Gap Analysis

### Microsoft's LocalSearchConfig

**File:** `src/graphrag/config/models/local_search_config.py:26-33`

```python
class LocalSearchConfig:
    text_unit_prop: float = 0.5      # Configurable
    community_prop: float = 0.15     # Configurable
    top_k_entities: int = 10         # Configurable
    top_k_relationships: int = 10    # Configurable
    max_tokens: int = 8000          # Configurable
    # ... more params
```

### FileIntel's GraphRAGSettings

**File:** `src/fileintel/core/config.py:120-167`

```python
class GraphRAGSettings(BaseModel):
    llm_model: str = "gemma3-12b-awq"
    embedding_model: str = "bge-large-en"
    community_levels: int = 5
    max_cluster_size: int = 50
    # ... more indexing params

    # ❌ MISSING: text_unit_prop
    # ❌ MISSING: community_prop
    # ❌ MISSING: top_k_entities
    # ❌ MISSING: top_k_relationships
    # ❌ MISSING: max_context_tokens
```

**Gap:** FileIntel's config doesn't expose local search tuning parameters.

---

## 4. Impact of Missing Configuration

### What Users Cannot Do Currently

1. **Cannot increase chunk usage** for detail-oriented queries:
   ```python
   # WANT: Use 80% chunks, 10% summaries for precise answers
   text_unit_prop=0.8, community_prop=0.1
   # CURRENT: Stuck at 50% chunks, 15% summaries
   ```

2. **Cannot reduce chunk usage** for broad queries:
   ```python
   # WANT: Use 20% chunks, 50% summaries for thematic answers
   text_unit_prop=0.2, community_prop=0.5
   # CURRENT: Stuck at 50% chunks, 15% summaries
   ```

3. **Cannot tune entity/relationship retrieval**:
   ```python
   # WANT: Retrieve top-20 entities instead of top-10
   top_k_entities=20
   # CURRENT: Stuck at default (10)
   ```

4. **Cannot adjust context size**:
   ```python
   # WANT: Use 12,000 tokens for complex queries
   max_context_tokens=12000
   # CURRENT: Stuck at default (8,000)
   ```

### What Works Fine Currently

✅ **Default behavior (50% chunks) is already good** for most queries
✅ **Text units ARE being used** (chunks are included)
✅ **Microsoft's defaults are reasonable** for general-purpose queries

---

## 5. Recommended Enhancements

### Priority 1: Add Config Options (High Impact, Low Effort)

**File:** `src/fileintel/core/config.py`

```python
class GraphRAGSettings(BaseModel):
    # ... existing fields ...

    # Local Search Configuration
    local_search_text_unit_prop: float = Field(
        default=0.5,
        description="Proportion of context tokens for original chunks (0.0-1.0). "
                    "Higher = more detailed, chunk-based answers. Default: 0.5 (50%)"
    )
    local_search_community_prop: float = Field(
        default=0.15,
        description="Proportion of context tokens for community summaries (0.0-1.0). "
                    "Higher = more thematic, summary-based answers. Default: 0.15 (15%)"
    )
    local_search_top_k_entities: int = Field(
        default=10,
        description="Number of top entities to retrieve for local search. Default: 10"
    )
    local_search_top_k_relationships: int = Field(
        default=10,
        description="Number of top relationships to retrieve. Default: 10"
    )
    local_search_max_context_tokens: int = Field(
        default=12000,
        description="Maximum tokens for local search context. Default: 12000"
    )
```

**Validation:**
```python
@model_validator(mode='after')
def validate_proportions(self):
    """Ensure text_unit_prop + community_prop <= 1.0"""
    total = self.local_search_text_unit_prop + self.local_search_community_prop
    if total > 1.0:
        raise ValueError(
            f"text_unit_prop ({self.local_search_text_unit_prop}) + "
            f"community_prop ({self.local_search_community_prop}) = {total} > 1.0. "
            "Sum must be <= 1.0 to leave room for entity/relationship context."
        )
    return self
```

### Priority 2: Wire Config to GraphRAG (Medium Effort)

**File:** `src/fileintel/rag/graph_rag/adapters/config_adapter.py`

Update `GraphRAGConfigAdapter.adapt_config()` to include local search settings:

```python
def adapt_config(self, settings: Settings, collection_id: str, base_path: str):
    # ... existing code ...

    # Add local search configuration
    config_dict["local_search"] = {
        "text_unit_prop": settings.graphrag.local_search_text_unit_prop,
        "community_prop": settings.graphrag.local_search_community_prop,
        "top_k_entities": settings.graphrag.local_search_top_k_entities,
        "top_k_relationships": settings.graphrag.local_search_top_k_relationships,
        "max_tokens": settings.graphrag.local_search_max_context_tokens,
    }

    return GraphRagConfig.from_dict(config_dict)
```

### Priority 3: Per-Query Overrides (Low Priority, Nice-to-Have)

**File:** `src/fileintel/api/routes/query.py`

Add optional query parameters:

```python
class QueryRequest(BaseModel):
    question: str
    search_type: str = "adaptive"
    # ... existing fields ...

    # Local Search Overrides (optional)
    text_unit_prop: Optional[float] = None
    community_prop: Optional[float] = None
    top_k_entities: Optional[int] = None
```

**File:** `src/fileintel/tasks/graphrag_tasks.py`

Pass overrides to GraphRAG service:

```python
@app.task
def query_graph_local(
    self,
    query: str,
    collection_id: str,
    text_unit_prop: Optional[float] = None,  # NEW
    community_prop: Optional[float] = None,  # NEW
    **kwargs
):
    # Override config if provided
    if text_unit_prop is not None or community_prop is not None:
        # Create temporary config with overrides
        # ... implementation ...
```

---

## 6. Use Cases for Different Proportions

### High Chunks (80/10) - Precision Queries

**Config:**
```yaml
text_unit_prop: 0.8
community_prop: 0.1
```

**Best for:**
- "What specific data does the study present about X?"
- "Quote the exact definition of Y from the document"
- "What are the numerical results for experiment Z?"

**Example:**
```bash
fileintel query collection thesis \
  "What are the exact completion rates reported in the study?" \
  --type local \
  --text-unit-prop 0.8 \
  --community-prop 0.1
```

### Balanced (50/25) - Default

**Config:**
```yaml
text_unit_prop: 0.5
community_prop: 0.25
```

**Best for:**
- General questions
- Mix of detail and context
- Most queries (good default)

### High Summaries (20/60) - Thematic Queries

**Config:**
```yaml
text_unit_prop: 0.2
community_prop: 0.6
```

**Best for:**
- "What are the main themes in this collection?"
- "Summarize the overall approach to project management"
- "What topics are covered across all documents?"

**Example:**
```bash
fileintel query collection thesis \
  "What are the main themes in project management literature?" \
  --type local \
  --text-unit-prop 0.2 \
  --community-prop 0.6
```

---

## 7. Testing Plan

### Test 1: Verify Current Behavior (Baseline)

**Query:** "What does the study say about time management?"

```bash
# Test current implementation
fileintel query collection thesis_collection \
  "What does the study say about time management?" \
  --type local
```

**Expected:**
- Uses 50% chunks (default)
- Returns detailed answer with chunk content
- Citations point to specific pages

### Test 2: High-Chunk Configuration (After Enhancement)

**Query:** Same question, but prefer chunks over summaries

```bash
# After implementing config options
fileintel query collection thesis_collection \
  "What does the study say about time management?" \
  --type local \
  --text-unit-prop 0.8 \
  --community-prop 0.1
```

**Expected:**
- Uses 80% chunks
- More detailed, quote-heavy answer
- More specific citations

### Test 3: High-Summary Configuration (After Enhancement)

**Query:** Same question, but prefer summaries

```bash
fileintel query collection thesis_collection \
  "What does the study say about time management?" \
  --type local \
  --text-unit-prop 0.2 \
  --community-prop 0.6
```

**Expected:**
- Uses 60% summaries
- More thematic, high-level answer
- Fewer specific quotes, more synthesis

### Test 4: Compare with Global Search

```bash
# Global (summaries only)
fileintel query collection thesis_collection \
  "What does the study say about time management?" \
  --type global

# Local (50% chunks)
fileintel query collection thesis_collection \
  "What does the study say about time management?" \
  --type local

# Local high-chunks (80% chunks) - after enhancement
fileintel query collection thesis_collection \
  "What does the study say about time management?" \
  --type local \
  --text-unit-prop 0.8
```

**Expected differences:**
- Global: Fastest, most abstract
- Local (default): Balanced, some detail
- Local (high-chunks): Slowest, most detailed

---

## 8. Implementation Checklist

### Phase 1: Configuration (Week 1)
- [ ] Add `local_search_text_unit_prop` to GraphRAGSettings
- [ ] Add `local_search_community_prop` to GraphRAGSettings
- [ ] Add `local_search_top_k_entities` to GraphRAGSettings
- [ ] Add `local_search_top_k_relationships` to GraphRAGSettings
- [ ] Add `local_search_max_context_tokens` to GraphRAGSettings
- [ ] Add validation for proportion sum <= 1.0
- [ ] Update settings.yaml with new defaults
- [ ] Test config loading

### Phase 2: Config Adapter (Week 1)
- [ ] Update GraphRAGConfigAdapter.adapt_config()
- [ ] Wire settings to GraphRagConfig.local_search
- [ ] Verify GraphRagConfig object structure
- [ ] Test config propagation to Microsoft's local_search()

### Phase 3: Documentation (Week 1)
- [ ] Document new config options in README
- [ ] Add examples for different use cases
- [ ] Update API documentation
- [ ] Create migration guide for existing users

### Phase 4: Optional Enhancements (Week 2)
- [ ] Add CLI flags for per-query overrides
- [ ] Add API query parameters for overrides
- [ ] Update Celery tasks to accept overrides
- [ ] Add validation for override values

### Phase 5: Testing (Week 2)
- [ ] Run baseline tests (current behavior)
- [ ] Test high-chunk configuration
- [ ] Test high-summary configuration
- [ ] Compare with global search
- [ ] Performance testing (latency, cost)

---

## 9. Example Configuration File

**File:** `settings.yaml`

```yaml
graphrag:
  # Indexing Configuration
  llm_model: "gemma3-12b-awq"
  embedding_model: "bge-large-en"
  community_levels: 5
  max_cluster_size: 50

  # Local Search Configuration
  local_search_text_unit_prop: 0.5    # 50% chunks (default)
  local_search_community_prop: 0.15   # 15% summaries (default)
  local_search_top_k_entities: 10
  local_search_top_k_relationships: 10
  local_search_max_context_tokens: 12000

  # Global Search Configuration
  # (uses community summaries only, no configuration needed)
```

**For detail-oriented research:**
```yaml
graphrag:
  local_search_text_unit_prop: 0.8    # 80% chunks
  local_search_community_prop: 0.1    # 10% summaries
  local_search_top_k_entities: 20     # More entities
  local_search_max_context_tokens: 15000  # Larger context
```

**For thematic analysis:**
```yaml
graphrag:
  local_search_text_unit_prop: 0.2    # 20% chunks
  local_search_community_prop: 0.6    # 60% summaries
  local_search_top_k_entities: 5      # Fewer entities (broader)
  local_search_max_context_tokens: 10000
```

---

## 10. Comparison: Current vs Enhanced

| Feature | Current FileIntel | Enhanced FileIntel | Benefit |
|---------|------------------|-------------------|---------|
| **Chunks in local search** | ✅ 50% (default) | ✅ 0-90% (configurable) | ⭐ User control |
| **Summaries in local search** | ✅ 15% (default) | ✅ 0-90% (configurable) | ⭐ User control |
| **Entity retrieval** | ✅ Top-10 (default) | ✅ 1-100 (configurable) | ⭐ Precision tuning |
| **Context size** | ✅ 8k tokens (default) | ✅ 1k-20k tokens | ⭐ Complexity handling |
| **Per-query tuning** | ❌ Not possible | ✅ API/CLI overrides | ⭐ Query-specific optimization |
| **Use case optimization** | ❌ One-size-fits-all | ✅ Precision vs Thematic modes | ⭐ Better results |

---

## 11. Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| **Breaking changes** | High | Low | Keep defaults identical to Microsoft's |
| **Invalid proportions** | Medium | Medium | Add validation (sum <= 1.0) |
| **Performance regression** | Low | Low | Defaults unchanged, opt-in tuning |
| **Configuration complexity** | Low | Medium | Good documentation + examples |
| **User confusion** | Medium | Medium | Clear naming, helpful defaults |

---

## 12. Conclusion

### Current State: ✅ **GOOD FOUNDATION**

FileIntel's local search implementation:
- ✅ Uses Microsoft's official API correctly
- ✅ Passes all required parameters
- ✅ Already includes chunks in answers (50% by default)
- ✅ Works well with reasonable defaults

### Enhancement Opportunity: ⭐ **HIGH VALUE, LOW EFFORT**

Adding configuration options would:
- ⭐ Enable precision vs thematic trade-off
- ⭐ Support different query types optimally
- ⭐ Match Microsoft's full feature set
- ⭐ Maintain backward compatibility (same defaults)

### Recommendation: **IMPLEMENT ENHANCEMENTS**

**Why:**
1. **Low risk:** Defaults unchanged, purely additive
2. **High value:** Enables advanced use cases
3. **Easy implementation:** Just config wiring, no algorithm changes
4. **Aligned with Microsoft:** Matches their design pattern

**Timeline:** 1-2 weeks for full implementation + testing

**Effort:** ~20-30 hours (mostly config + testing)

---

## 13. Next Steps

1. **Review this analysis** and approve enhancement plan
2. **Implement Phase 1** (configuration options in settings)
3. **Test baseline** to verify current 50/15 behavior
4. **Implement Phase 2** (wire config to GraphRAG)
5. **Test enhanced** behavior with different proportions
6. **Document** new features with examples
7. **Deploy** and gather user feedback

**Priority:** Medium-High (enhances existing feature, not critical bug)

**Dependencies:** None (purely additive changes)
