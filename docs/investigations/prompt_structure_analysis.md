# Prompt Structure Analysis - Answer Format Integration Strategy

**Date:** 2025-11-08
**Purpose:** Analyze current prompt structure and determine safe integration points for answer format templates
**Reference:** See `existing_prompts_backup.md` for complete prompt backups

---

## Executive Summary

The current RAG prompts are well-structured and working effectively. This analysis identifies **safe injection points** for answer format templates that will:
- ✅ Preserve all existing functionality
- ✅ Maintain citation accuracy
- ✅ Keep query classification logic
- ✅ Only modify the output structure (not the content quality)

**Key Finding:** Both Vector RAG and GraphRAG have clearly defined injection points where format templates can be inserted without disrupting the existing prompt logic.

---

## Vector RAG Prompt Structure

### Current Prompt Assembly

```
┌─────────────────────────────────────────────────────────────┐
│ Base Instruction                                             │
│ "Based on the following retrieved documents..."             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Query-Type Specific Instruction                             │
│ (factual/analytical/summarization/comparison/general)       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Question                                                     │
│ "Question: {user_query}"                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Retrieved Sources                                            │
│ "Retrieved Sources:"                                         │
│ "[Citation 1]: Chunk 1"                                      │
│ "[Citation 2]: Chunk 2"                                      │
│ ...                                                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Guidance Text                                                │
│ "Please provide your answer based on the sources above..."  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Citation Requirements                                        │
│ "CRITICAL CITATION REQUIREMENTS:"                            │
│ - Harvard style format                                       │
│ - Preserve page numbers                                      │
│ - etc.                                                       │
└─────────────────────────────────────────────────────────────┘
```

### Safe Injection Point for Answer Format

**RECOMMENDED LOCATION:** Between "Retrieved Sources" and "Citation Requirements"

**Rationale:**
1. ✅ Answer format comes AFTER context (LLM has all information)
2. ✅ Answer format comes BEFORE citation rules (format instructions don't interfere with citation requirements)
3. ✅ Format can reference the sources provided above
4. ✅ Citation rules apply to formatted answer (both work together)

**Modified Structure:**

```
┌─────────────────────────────────────────────────────────────┐
│ Base Instruction + Query-Type Instruction                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Question: {user_query}                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Retrieved Sources: [Citations + Chunks]                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ ✨ ANSWER FORMAT TEMPLATE (NEW - OPTIONAL)                  │
│ "Please provide your answer as a single paragraph..."       │
│ or "Format your answer as a JSON table..." etc.             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Citation Requirements (UNCHANGED)                           │
│ "CRITICAL CITATION REQUIREMENTS: ..."                       │
└─────────────────────────────────────────────────────────────┘
```

### Template File Mapping

**Files Created:**
```
prompts/templates/vector_rag/
├── prompt.md                          # Main template
├── base_instruction.md                # Base RAG instruction
├── citation_rules.md                  # Citation requirements
└── query_type_instructions/
    ├── factual.md                     # Factual query instruction
    ├── analytical.md                  # Analytical query instruction
    ├── summarization.md               # Summarization query instruction
    ├── comparison.md                  # Comparison query instruction
    └── general.md                     # General query instruction
```

**Main Template (`prompt.md`):**
```jinja2
{{ base_instruction }} {{ query_type_instruction }}

Question: {{ query }}

Retrieved Sources:
{{ context }}

{{ answer_format }}

{{ citation_rules }}
```

**Variables:**
- `{{ base_instruction }}` - Loaded from `base_instruction.md`
- `{{ query_type_instruction }}` - Loaded from `query_type_instructions/{query_type}.md`
- `{{ query }}` - User's question (runtime)
- `{{ context }}` - Formatted chunks with citations (runtime)
- `{{ answer_format }}` - **NEW** - Format template or empty string (runtime)
- `{{ citation_rules }}` - Loaded from `citation_rules.md`

---

## GraphRAG Prompt Structure

### Current Prompt Structure

All GraphRAG prompts follow a similar pattern:

```
┌─────────────────────────────────────────────────────────────┐
│ Role                                                         │
│ "You are a helpful assistant..."                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Goal                                                         │
│ "Generate a response of the target length and format..."    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Instructions                                                 │
│ - Don't make things up                                       │
│ - Use data references                                        │
│ - Max 5 record IDs per reference                            │
│ - etc.                                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Target Response Length and Format                           │
│ "---Target response length and format---"                   │
│ "{response_type}"                                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Data Tables                                                  │
│ "{context_data}" or "{report_data}"                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Goal Repeated + Additional Instructions                     │
└─────────────────────────────────────────────────────────────┘
```

### Safe Injection Point for Answer Format

**RECOMMENDED LOCATION:** Inject into `{response_type}` variable

**Current Default:**
```python
response_type = "multiple paragraphs"
```

**With Format Template:**
```python
if answer_format != "default":
    format_template = format_manager.get_format_template(answer_format)
    response_type = f"{response_type}\n\n{format_template}"
```

**Example Result:**
```
---Target response length and format---

multiple paragraphs

# Answer Format: Single Paragraph

Please provide your answer as a **single, cohesive paragraph** under one clear headline.

Your response **must** be a JSON object with two keys:
1. `"headline"`: A string containing a clear, concise headline...
2. `"paragraph"`: A string containing the complete answer...
```

**Rationale:**
1. ✅ Uses existing variable injection point
2. ✅ Doesn't modify GraphRAG system prompt files (Microsoft copyright)
3. ✅ Format instructions come in logical location (with response type)
4. ✅ Minimal code changes required
5. ✅ Easy to test and validate

### Alternative Approaches (Not Recommended)

**Option B: Modify System Prompt Files**
- ❌ Requires changing Microsoft-copyrighted files
- ❌ More invasive changes
- ❌ Harder to maintain across graphrag updates
- ✅ Cleaner separation of concerns
- **Decision:** NOT RECOMMENDED for Phase 1

**Option C: Create New Prompt Files**
- ❌ Requires major refactoring of graphrag module
- ❌ Risk of breaking existing functionality
- ✅ Better long-term maintainability
- **Decision:** Consider for Phase 2+

---

## Answer Format Template Design

### Template Requirements

**Must Include:**
1. Clear description of desired format
2. JSON schema or structure example
3. Specific formatting rules
4. Example output

**Must NOT Include:**
- Instructions that conflict with citation requirements
- Instructions that override query-type logic
- Anything that limits content quality

### Template Structure Pattern

```markdown
# Answer Format: {Format Name}

Brief description of the format.

Your response **must** be a JSON object with the following structure:
- Field 1: Description
- Field 2: Description

### Example of Expected Format

```json
{
  "field1": "example value",
  "field2": "example value"
}
```

### Requirements

- Requirement 1
- Requirement 2
- etc.
```

### Citation Compatibility

**Critical:** All answer format templates must be compatible with citation requirements.

**Compatible Formats:**
- ✅ Single paragraph (citations inline)
- ✅ Markdown (citations inline)
- ✅ Essay (citations in sections)
- ✅ List (citations per item)

**Potentially Problematic:**
- ⚠️ Pure JSON without citation fields (need to add citation fields)
- ⚠️ Table (need column for citations or footnotes)

**Solution:** All JSON-based formats should include citation fields:

```json
{
  "content": "Main content with inline citations (Author, Year, p.X)",
  "sources": ["Citation 1", "Citation 2"]
}
```

---

## Variable Flow Analysis

### Vector RAG Variable Flow

```
User Request
    ↓
API: QueryRequest
    ↓
query_orchestrator.route_query(answer_format="single_paragraph")
    ↓
vector_rag_service.query(answer_format="single_paragraph")
    ↓
vector_rag_service._generate_answer(answer_format="single_paragraph")
    ↓
llm_provider.generate_rag_response(answer_format="single_paragraph")
    ↓
llm_provider._build_rag_prompt(answer_format="single_paragraph")
    ↓
[Load format template from format_manager]
    ↓
[Render prompt template with format]
    ↓
LLM generates formatted response
```

**Changes Required:**
1. Add `answer_format` parameter to `QueryRequest` API model
2. Add `answer_format` parameter to `route_query()`
3. Add `answer_format` parameter to `vector_rag_service.query()`
4. Add `answer_format` parameter to `_generate_answer()`
5. Add `answer_format` parameter to `generate_rag_response()`
6. Modify `_build_rag_prompt()` to load and inject format template

**Impact:** Low - simple parameter passing through existing chain

---

### GraphRAG Variable Flow

```
User Request
    ↓
API: QueryRequest
    ↓
query_orchestrator.route_query(answer_format="single_paragraph")
    ↓
graphrag_service.query(answer_format="single_paragraph")
    ↓
[Load format template]
    ↓
enhanced_response_type = f"{response_type}\n\n{format_template}"
    ↓
global_search() or local_search() or basic_search()
    ↓
[GraphRAG engine uses enhanced_response_type]
    ↓
LLM generates formatted response
    ↓
Citation tracing processes the formatted response
```

**Changes Required:**
1. Add `answer_format` parameter to `QueryRequest` API model
2. Add `answer_format` parameter to `route_query()`
3. Add `answer_format` parameter to `graphrag_service.query()`
4. Load format template and inject into `response_type`

**Impact:** Very Low - only modifies one variable before passing to graphrag engine

---

## Integration Strategy

### Phase 1: Minimal Viable Integration

**Goal:** Enable answer format control with minimal code changes

**Vector RAG:**
1. Extract existing prompts to template files ✅ DONE
2. Refactor `_build_rag_prompt()` to use templates
3. Add `answer_format` parameter injection

**GraphRAG:**
1. Add `answer_format` parameter to `query()` method
2. Inject format template into `response_type` variable

**API:**
1. Add `answer_format` to `QueryRequest` model
2. Pass through orchestrator

**Testing:**
1. Verify `answer_format="default"` produces identical results to current
2. Test each format with sample queries
3. Validate citation accuracy maintained

### Phase 2: Enhanced Features (Future)

- Format validation
- Collection-level format defaults
- Custom format upload
- Format auto-detection

---

## Risk Assessment

### Low Risk Areas ✅

1. **Template File Creation**
   - Risk: None (new files)
   - Impact: No effect until code uses them

2. **API Parameter Addition**
   - Risk: Low (optional parameter with default)
   - Impact: Backward compatible

3. **GraphRAG response_type Injection**
   - Risk: Low (modifies existing variable)
   - Impact: Easy to test and validate

### Medium Risk Areas ⚠️

1. **Vector RAG Prompt Refactoring**
   - Risk: Medium (changes core prompt logic)
   - Impact: Could affect answer quality
   - Mitigation: Keep fallback to hardcoded prompts; extensive testing

2. **Format-Citation Interaction**
   - Risk: Medium (formats might not preserve citations)
   - Impact: Loss of academic accuracy
   - Mitigation: Design all formats with citation compatibility; validate outputs

### High Risk Areas ❌

1. **Modifying GraphRAG System Prompts**
   - Risk: High (changes Microsoft code)
   - Impact: Could break graphrag engine
   - Mitigation: DON'T DO THIS - use injection approach instead

---

## Validation Checklist

Before deploying any changes, verify:

### Functional Tests
- [ ] `answer_format="default"` produces identical output to current
- [ ] All query types work with all formats
- [ ] Citations preserved in all formats
- [ ] Page numbers maintained
- [ ] No hallucinations introduced

### Format-Specific Tests
- [ ] Single paragraph format produces 1 paragraph
- [ ] Table format produces valid JSON table
- [ ] List format produces valid JSON list
- [ ] Essay format produces structured sections
- [ ] Markdown format is valid markdown

### Citation Tests
- [ ] Harvard citations present in output
- [ ] Page numbers preserved from chunks
- [ ] Citation format matches requirements
- [ ] Sources traceable to chunks

### Performance Tests
- [ ] Template loading < 10ms
- [ ] Prompt rendering < 5ms
- [ ] No memory leaks from template caching
- [ ] Overall query time increase < 50ms

---

## Recommended Next Steps

1. **Implement FormatManager** (Sprint 1)
   - Create `AnswerFormatManager` class
   - Load and cache format templates
   - Write unit tests

2. **Refactor Vector RAG** (Sprint 2)
   - Modify `_build_rag_prompt()` to use templates
   - Add fallback to hardcoded prompts
   - Test with `answer_format="default"`

3. **Integrate Formats** (Sprint 3)
   - Wire `answer_format` parameter through
   - Test each format individually
   - Validate citation preservation

4. **GraphRAG Integration** (Sprint 4)
   - Inject format into `response_type`
   - Test with all search types
   - Validate citation tracing works

5. **End-to-End Testing** (Sprint 5)
   - API integration tests
   - Format compliance validation
   - Performance benchmarking

---

## Conclusion

The current prompt architecture provides **safe, well-defined injection points** for answer format templates:

- **Vector RAG:** Between context and citation rules
- **GraphRAG:** Via `response_type` variable injection

Both approaches:
- ✅ Preserve existing functionality
- ✅ Maintain citation accuracy
- ✅ Require minimal code changes
- ✅ Are easy to test and validate
- ✅ Support rollback if needed

The template files have been created and match the exact current behavior. We can now proceed with implementation knowing we have a solid foundation and clear backup of the working system.

---

**Document Status:** Complete ✅

**Next Action:** Proceed with Sprint 1 - Implement AnswerFormatManager

---

**End of Analysis**
