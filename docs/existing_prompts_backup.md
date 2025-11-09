# Existing Prompts Backup - FileIntel RAG Systems

**Date:** 2025-11-08
**Purpose:** Complete backup and documentation of all existing prompts before implementing answer format management
**Status:** Reference Only - Do Not Modify Original Prompts

---

## Table of Contents

- [Vector RAG Prompts](#vector-rag-prompts)
- [GraphRAG Prompts](#graphrag-prompts)
- [Prompt Variables Reference](#prompt-variables-reference)
- [Context Formatting](#context-formatting)
- [Query Classification Logic](#query-classification-logic)

---

## Vector RAG Prompts

### Source Files
- **Main Prompt Builder:** `src/fileintel/llm_integration/unified_provider.py:424-466`
- **Query Classifier:** `src/fileintel/rag/vector_rag/services/vector_rag_service.py:280-357`
- **Context Formatter:** `src/fileintel/llm_integration/unified_provider.py:361-422`

---

### 1. Base Instruction

**Location:** `unified_provider.py:436`

```
Based on the following retrieved documents, answer the user's question accurately and comprehensively.
```

**Usage:** Always included at the start of every RAG prompt

---

### 2. Query-Type Specific Instructions

#### 2.1 Factual Queries

**Triggered by keywords:** `who`, `what`, `when`, `where`, `how many`, `which`, `specific`, `exactly`

**Location:** `unified_provider.py:438-439`

```
Focus on providing specific facts, dates, numbers, and concrete information. If exact information isn't available, clearly state what is known and what is uncertain.
```

#### 2.2 Analytical Queries

**Triggered by keywords:** `why`, `how`, `analyze`, `explain`, `relationship`, `impact`, `cause`, `effect`, `implications`

**Location:** `unified_provider.py:440-441`

```
Provide an analytical response that examines relationships, patterns, and implications. Use the sources to support your analysis and reasoning.
```

#### 2.3 Summarization Queries

**Triggered by keywords:** `summarize`, `summary`, `overview`, `main points`, `key points`, `outline`

**Location:** `unified_provider.py:442-443`

```
Provide a comprehensive summary that captures the key points and main themes from the sources. Organize the information logically.
```

#### 2.4 Comparison Queries

**Triggered by keywords:** `compare`, `contrast`, `difference`, `differences`, `similar`, `similarities`, `versus`, `vs`

**Location:** `unified_provider.py:444-445`

```
Compare and contrast the information from different sources. Highlight similarities, differences, and any conflicting information.
```

#### 2.5 General Queries

**Triggered by:** Default when no other keywords match

**Location:** `unified_provider.py:446-447`

```
Provide a clear and well-reasoned answer. Use evidence from the sources to support your response.
```

---

### 3. Complete Vector RAG Prompt Template

**Location:** `unified_provider.py:449-466`

**Template Structure:**

```
{base_instruction} {query_type_instruction}

Question: {query}

Retrieved Sources:
{context}

Please provide your answer based on the sources above. If the sources don't contain sufficient information to fully answer the question, indicate what information is available and what might be missing.

CRITICAL CITATION REQUIREMENTS:
- You MUST cite sources using the EXACT citation format shown in square brackets [like this] before each source text
- When a source citation includes a page number (e.g., "(Author, Year, p. X)"), you MUST include that page number in your citation
- Harvard style in-text citation format:
  * Single page: (Author Year, p.X)
  * Consecutive pages: (Author Year, pp.X-Y)
  * Non-consecutive pages: (Author Year, pp.X,Y)
- Example: If the source shows "[(Smith, 2023, p. 45)]: Some text", cite it as (Smith 2023, p.45) in your answer
- ALWAYS preserve page numbers from the source citations - they are critical for academic accuracy
```

---

### 4. Context Formatting

**Location:** `unified_provider.py:384-405`

**Format:** Each chunk is formatted as:
```
[{citation}]: {chunk_text}
```

**Citation Format:**
- Uses `format_in_text_citation(chunk)` from `fileintel.citation` module
- Falls back to `original_filename` or `filename` if citation unavailable
- Falls back to `Source {i}` if no metadata

**Example Context:**
```
[(Smith, 2023, p. 45)]: Machine learning is a subset of artificial intelligence...

[(Jones, 2024, pp. 12-15)]: Deep learning networks use multiple layers of neurons...

[(Brown, 2022, p. 8)]: Neural networks are inspired by biological brain structures...
```

**Limits:**
- Top 8 chunks only (`context_chunks[:8]`)
- Chunks concatenated with `\n\n` separator

---

### 5. LLM Parameters

**Location:** `unified_provider.py:417-422`

- **max_tokens:** 600 (default)
- **temperature:** 0.1 (low for consistent answers)
- **Model:** Configured in settings

---

## GraphRAG Prompts

### Source Files
- **Global Search Map:** `src/graphrag/prompts/query/global_search_map_system_prompt.py`
- **Global Search Reduce:** `src/graphrag/prompts/query/global_search_reduce_system_prompt.py`
- **Local Search:** `src/graphrag/prompts/query/local_search_system_prompt.py`
- **Basic Search:** `src/graphrag/prompts/query/basic_search_system_prompt.py`

---

### 1. Global Search Map Prompt

**File:** `global_search_map_system_prompt.py`
**Purpose:** First phase of global search - extract key points from community reports

**Complete Prompt:**

```
---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response should be JSON formatted as follows:
{
    "points": [
        {"description": "Description of point 1 [Data: Reports (report ids)]", "score": score_value},
        {"description": "Description of point 2 [Data: Reports (report ids)]", "score": score_value}
    ]
}

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Points supported by data should list the relevant reports as references as follows:
"This is an example sentence supported by data references [Data: Reports (report ids)]"

**Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 64, 46, 34, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data report in the provided tables.

Do not include information where the supporting evidence for it is not provided.

Limit your response length to {max_length} words.

---Data tables---

{context_data}

---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Points supported by data should list the relevant reports as references as follows:
"This is an example sentence supported by data references [Data: Reports (report ids)]"

**Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 64, 46, 34, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data report in the provided tables.

Do not include information where the supporting evidence for it is not provided.

Limit your response length to {max_length} words.

The response should be JSON formatted as follows:
{
    "points": [
        {"description": "Description of point 1 [Data: Reports (report ids)]", "score": score_value},
        {"description": "Description of point 2 [Data: Reports (report ids)]", "score": score_value}
    ]
}
```

**Variables:**
- `{max_length}` - Maximum response length in words
- `{context_data}` - Data tables containing community reports

**Output Format:** JSON with list of scored points

---

### 2. Global Search Reduce Prompt

**File:** `global_search_reduce_system_prompt.py`
**Purpose:** Second phase of global search - synthesize map results into final answer

**Complete Prompt:**

```
---Role---

You are a helpful assistant responding to questions about a dataset by synthesizing perspectives from multiple analysts.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

The response should also preserve all the data references previously included in the analysts' reports, but do not mention the roles of multiple analysts in the analysis process.

**Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 34, 46, 64, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.

Limit your response length to {max_length} words.

---Target response length and format---

{response_type}


---Analyst Reports---

{report_data}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

The response should also preserve all the data references previously included in the analysts' reports, but do not mention the roles of multiple analysts in the analysis process.

**Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 34, 46, 64, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.

Limit your response length to {max_length} words.

---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
```

**Variables:**
- `{max_length}` - Maximum response length in words
- `{response_type}` - Target response format (default: "multiple paragraphs")
- `{report_data}` - Analyst reports from map phase (ranked by importance)

**Output Format:** Markdown text with synthesized answer

**Fallback Response:**
```
I am sorry but I am unable to answer this question given the provided data.
```

---

### 3. Local Search Prompt

**File:** `local_search_system_prompt.py`
**Purpose:** Local search using entity relationships and nearby context

**Complete Prompt:**

```
---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.

If you don't know the answer, just say so. Do not make anything up.

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Sources (15, 16), Reports (1), Entities (5, 7); Relationships (23); Claims (2, 7, 34, 46, 64, +more)]."

where 15, 16, 1, 5, 7, 23, 2, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}


---Data tables---

{context_data}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.

If you don't know the answer, just say so. Do not make anything up.

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Sources (15, 16), Reports (1), Entities (5, 7); Relationships (23); Claims (2, 7, 34, 46, 64, +more)]."

where 15, 16, 1, 5, 7, 23, 2, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
```

**Variables:**
- `{response_type}` - Target response format (default: "multiple paragraphs")
- `{context_data}` - Data tables with entities, relationships, sources, etc.

**Output Format:** Markdown text

**Citation Format:** Multiple datasets per reference
- Example: `[Data: Sources (15, 16), Reports (1), Entities (5, 7)]`

---

### 4. Basic Search Prompt

**File:** `basic_search_system_prompt.py`
**Purpose:** Simple search without entity/relationship graph

**Complete Prompt:**

```
---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all relevant information in the input data tables appropriate for the response length and format.

You should use the data provided in the data tables below as the primary context for generating the response.

If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: Sources (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Sources (2, 7, 64, 46, 34, +more)]. He is also CEO of company X [Data: Sources (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the source id taken from the "source_id" column in the provided tables.

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}


---Data tables---

{context_data}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all relevant information in the input data appropriate for the response length and format.

You should use the data provided in the data tables below as the primary context for generating the response.

If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: Sources (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Sources (2, 7, 64, 46, 34, +more)]. He is also CEO of company X [Data: Sources (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the source id taken from the "source_id" column in the provided tables.

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
```

**Variables:**
- `{response_type}` - Target response format (default: "multiple paragraphs")
- `{context_data}` - Data tables with sources

**Output Format:** Markdown text

**Citation Format:** Sources only
- Example: `[Data: Sources (2, 7, 64, 46, 34, +more)]`

---

## Prompt Variables Reference

### Vector RAG Variables

| Variable | Type | Source | Description |
|----------|------|--------|-------------|
| `{base_instruction}` | String | Hardcoded | Base RAG instruction |
| `{query_type_instruction}` | String | Query classifier | Query-type specific instruction |
| `{query}` | String | User input | User's question |
| `{context}` | String | Chunk retrieval | Formatted retrieved chunks with citations |

### GraphRAG Variables

| Variable | Type | Source | Description |
|----------|------|--------|-------------|
| `{response_type}` | String | Config/default | Target response format description |
| `{max_length}` | Integer | Config/default | Maximum response length in words |
| `{context_data}` | String | GraphRAG engine | Data tables (entities, reports, etc.) |
| `{report_data}` | String | Map phase | Analyst reports from map phase |

---

## Context Formatting

### Vector RAG Context Format

**Location:** `unified_provider.py:384-405`

**Structure:**
```
[Citation 1]: Chunk text 1

[Citation 2]: Chunk text 2

[Citation 3]: Chunk text 3
```

**Citation Formats:**
1. **With full metadata:** `(Author, Year, p. 45)`
2. **Fallback to filename:** `filename.pdf`
3. **Final fallback:** `Source 1`

**Example:**
```
[(Smith, 2023, p. 45)]: Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses statistical techniques to give computer systems the ability to progressively improve their performance on a specific task.

[(Jones, 2024, pp. 12-15)]: Deep learning is a specialized form of machine learning that uses neural networks with multiple layers. These networks can automatically learn hierarchical representations of data, making them particularly effective for tasks like image recognition and natural language processing.

[(Brown, 2022, p. 8)]: Neural networks are computational models inspired by the structure of biological brains. They consist of interconnected nodes (neurons) organized in layers that process information through weighted connections.
```

**Processing:**
- Maximum 8 chunks (`context_chunks[:8]`)
- Each chunk separated by `\n\n`
- Citation extracted using `format_in_text_citation(chunk)` from citation module
- Page numbers preserved from chunk metadata

---

### GraphRAG Context Format

**Location:** GraphRAG engine (data table formatting)

**Structure:** Tabular data in text format

**Example (simplified):**
```
Entities:
| entity_id | name | type | description |
|-----------|------|------|-------------|
| 1 | Machine Learning | Concept | AI technique for learning from data |
| 2 | Neural Networks | Concept | Computational models inspired by brains |

Reports:
| report_id | content | entity_ids |
|-----------|---------|------------|
| 1 | Machine learning uses statistical methods... | [1] |
| 2 | Neural networks consist of layers... | [2] |
```

**Note:** Actual format generated by GraphRAG engine, varies by search type

---

## Query Classification Logic

**Location:** `vector_rag_service.py:280-357`

**Function:** `_classify_query_type(query: str) -> str`

### Classification Rules (in order of precedence)

1. **Factual Queries**
   - Keywords: `who`, `what`, `when`, `where`, `how many`, `which`, `specific`, `exactly`
   - Returns: `"factual"`
   - Instruction: Focus on specific facts, dates, numbers

2. **Analytical Queries**
   - Keywords: `why`, `how`, `analyze`, `explain`, `relationship`, `impact`, `cause`, `effect`, `implications`
   - Returns: `"analytical"`
   - Instruction: Examine relationships, patterns, implications

3. **Summarization Queries**
   - Keywords: `summarize`, `summary`, `overview`, `main points`, `key points`, `outline`
   - Returns: `"summarization"`
   - Instruction: Capture key points and main themes

4. **Comparison Queries**
   - Keywords: `compare`, `contrast`, `difference`, `differences`, `similar`, `similarities`, `versus`, `vs`
   - Returns: `"comparison"`
   - Instruction: Compare and contrast from different sources

5. **General Queries (Default)**
   - Triggered when: No keywords match
   - Returns: `"general"`
   - Instruction: Provide clear and well-reasoned answer

**Implementation:**
```python
def _classify_query_type(self, query: str) -> str:
    query_lower = query.lower()

    # Check for factual keywords
    if any(word in query_lower for word in ["who", "what", "when", "where", "how many", "which", "specific", "exactly"]):
        return "factual"

    # Check for analytical keywords
    elif any(word in query_lower for word in ["why", "how", "analyze", "explain", "relationship", "impact", "cause", "effect", "implications"]):
        return "analytical"

    # Check for summarization keywords
    elif any(word in query_lower for word in ["summarize", "summary", "overview", "main points", "key points", "outline"]):
        return "summarization"

    # Check for comparison keywords
    elif any(word in query_lower for word in ["compare", "contrast", "difference", "differences", "similar", "similarities", "versus", "vs"]):
        return "comparison"

    # Default to general
    else:
        return "general"
```

**Note:** Classification is case-insensitive (uses `query.lower()`)

---

## Important Implementation Notes

### Vector RAG
1. **DO NOT MODIFY** the citation requirements - they are critical for academic accuracy
2. **PRESERVE** the query classification logic - it's working well
3. **MAINTAIN** the context formatting with Harvard citations
4. **KEEP** max_tokens=600 and temperature=0.1 parameters

### GraphRAG
1. **DO NOT MODIFY** the citation format `[Data: <dataset> (ids)]`
2. **PRESERVE** the "5 record ids max" rule
3. **MAINTAIN** the response_type variable injection point
4. **KEEP** the markdown styling requirement

### Answer Format Integration Strategy
When adding answer format control:
- **Vector RAG:** Inject format template BETWEEN context and citation requirements
- **GraphRAG:** Inject format template INTO the `{response_type}` variable
- **Both:** Preserve all existing instructions around citations and evidence requirements

---

## Testing Baseline Outputs

Before making changes, test queries should be run and outputs saved:

### Test Queries
1. **Factual:** "What is machine learning?"
2. **Analytical:** "Why is deep learning effective for image recognition?"
3. **Summarization:** "Summarize the key concepts in neural networks"
4. **Comparison:** "Compare supervised and unsupervised learning"
5. **General:** "Tell me about artificial intelligence"

### Output Checklist
- [ ] Citations preserved with page numbers
- [ ] Query type correctly classified
- [ ] Answer quality maintained
- [ ] No hallucinations
- [ ] Markdown formatting (GraphRAG)
- [ ] Evidence-based claims only

---

## Change Control

**Any modifications to these prompts must:**
1. Create a new version of this backup document
2. Document what changed and why
3. Test against baseline outputs
4. Verify citation accuracy maintained
5. Get approval before deployment

**Version History:**
- v1.0 (2025-11-08) - Initial backup before answer format implementation

---

**End of Backup Document**
