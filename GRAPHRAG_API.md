# GraphRAG API Documentation

This document explains how Microsoft GraphRAG API works, what it expects, and how FileIntel should integrate with it.

## ⚠️ API Stability Warning
> **WARNING**: This API is under development and may undergo changes in future releases.
> Backwards compatibility is not guaranteed at this time.

## GraphRAG API Overview

GraphRAG provides three main APIs:
1. **Index API** (`build_index`) - Creates knowledge graph from documents
2. **Query API** (`global_search`, `local_search`) - Searches the knowledge graph
3. **Prompt Tuning API** (`generate_indexing_prompts`) - Optimizes extraction prompts

## 1. Index API - `build_index()`

### Purpose
Processes documents and builds a knowledge graph with entities, relationships, communities, and reports.

### Function Signature
```python
async def build_index(
    config: GraphRagConfig,
    method: IndexingMethod | str = IndexingMethod.Standard,
    is_update_run: bool = False,
    memory_profile: bool = False,
    callbacks: list[WorkflowCallbacks] | None = None,
    additional_context: dict[str, Any] | None = None,
    verbose: bool = False,
    input_documents: pd.DataFrame | None = None,
) -> list[PipelineRunResult]
```

### Key Parameters
- **`config`**: Full GraphRAG configuration (see Configuration section)
- **`input_documents`**: Optional DataFrame with documents to process
  - If provided, bypasses file loading and uses this data directly
  - Expected columns: `id`, `text`, `title` (optional), metadata columns
- **`method`**: Indexing approach (Standard, NLP+LLM, etc.)

### Output Structure
Creates parquet files in `{config.root_dir}/output/` directory:
- `entities.parquet` - Extracted entities with embeddings
- `relationships.parquet` - Entity relationships
- `communities.parquet` - Community detection results
- `community_reports.parquet` - LLM-generated community summaries
- `text_units.parquet` - Document chunks with embeddings
- `covariates.parquet` - Claims and facts (optional)

## 2. Query APIs

### Global Search - `global_search()`

**Purpose**: Searches across all communities for broad, comprehensive answers.

```python
async def global_search(
    config: GraphRagConfig,
    entities: pd.DataFrame,           # from entities.parquet
    communities: pd.DataFrame,        # from communities.parquet
    community_reports: pd.DataFrame,  # from community_reports.parquet
    community_level: int | None,      # hierarchy level to search
    dynamic_community_selection: bool, # adaptive level selection
    response_type: str,               # "text", "json", etc.
    query: str,                       # user question
    callbacks: list[QueryCallbacks] | None = None,
    verbose: bool = False,
) -> tuple[str | dict, str | dict]:   # (response, context_data)
```

### Local Search - `local_search()`

**Purpose**: Searches within specific entity neighborhoods for detailed answers.

```python
async def local_search(
    config: GraphRagConfig,
    entities: pd.DataFrame,           # from entities.parquet
    communities: pd.DataFrame,        # from communities.parquet
    community_reports: pd.DataFrame,  # from community_reports.parquet
    text_units: pd.DataFrame,         # from text_units.parquet
    relationships: pd.DataFrame,      # from relationships.parquet
    covariates: pd.DataFrame | None,  # from covariates.parquet (optional)
    community_level: int,             # hierarchy level
    response_type: str,               # "text", "json", etc.
    query: str,                       # user question
    callbacks: list[QueryCallbacks] | None = None,
    verbose: bool = False,
) -> tuple[str | dict, str | dict]:   # (response, context_data)
```

## 3. Configuration - GraphRagConfig

### Minimum Required Configuration
```python
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.config.models.language_model_config import LanguageModelConfig

config = GraphRagConfig(
    root_dir="/path/to/workspace",
    models={
        "default_chat_model": LanguageModelConfig(
            api_key="your-openai-key"
        ),
        "default_embedding_model": LanguageModelConfig(
            api_key="your-openai-key"
        )
    }
)
```

### Key Configuration Sections

#### Required Models Dictionary
GraphRAG **requires** exactly these two model keys:
- `"default_chat_model"` - For LLM operations (GPT-4, etc.)
- `"default_embedding_model"` - For embeddings (text-embedding-3-small, etc.)

#### Directory Structure
```
{root_dir}/
├── input/              # Source documents (if using file input)
├── output/             # Generated parquet files
│   ├── entities.parquet
│   ├── relationships.parquet
│   ├── communities.parquet
│   ├── community_reports.parquet
│   ├── text_units.parquet
│   └── covariates.parquet
└── settings.yaml       # Optional config file
```

## 4. Data Flow Integration

### FileIntel → GraphRAG Integration Pattern

```python
# 1. Convert FileIntel documents to DataFrame
def convert_fileintel_documents(chunks: List[DocumentChunk]) -> pd.DataFrame:
    return pd.DataFrame([{
        'id': chunk.id,
        'text': chunk.text,
        'title': chunk.metadata.get('title', ''),
        'source': chunk.metadata.get('filename', '')
    } for chunk in chunks])

# 2. Build index
documents_df = convert_fileintel_documents(document_chunks)
results = await build_index(
    config=graphrag_config,
    input_documents=documents_df
)

# 3. Load generated data for queries
entities = pd.read_parquet(f"{config.root_dir}/output/entities.parquet")
communities = pd.read_parquet(f"{config.root_dir}/output/communities.parquet")
community_reports = pd.read_parquet(f"{config.root_dir}/output/community_reports.parquet")

# 4. Execute queries
response, context = await global_search(
    config=config,
    entities=entities,
    communities=communities,
    community_reports=community_reports,
    community_level=2,
    dynamic_community_selection=True,
    response_type="text",
    query="What are the main themes?"
)
```

## 5. FileIntel Implementation Requirements

### Current Issues in FileIntel GraphRAG Integration:

#### ❌ **Wrong**: Current service calls
```python
# This doesn't match the actual API
result = await global_search(
    config=graphrag_config,
    community_level=2,
    response_type="text",
    query=query,
)
```

#### ✅ **Correct**: Required API calls
```python
# Must load parquet data first
entities = pd.read_parquet(f"{workspace}/output/entities.parquet")
communities = pd.read_parquet(f"{workspace}/output/communities.parquet")
community_reports = pd.read_parquet(f"{workspace}/output/community_reports.parquet")

# Then call with all required DataFrames
result, context = await global_search(
    config=graphrag_config,
    entities=entities,
    communities=communities,
    community_reports=community_reports,
    community_level=2,
    dynamic_community_selection=True,
    response_type="text",
    query=query,
)
```

### Required FileIntel Updates:

1. **Fix GraphRAGService.build_index()**:
   - Convert FileIntel DocumentChunks to pandas DataFrame
   - Pass DataFrame via `input_documents` parameter
   - Store workspace path for later data loading

2. **Fix GraphRAGService.global_search()**:
   - Load required parquet files from workspace
   - Pass all required DataFrames to GraphRAG API
   - Handle response format properly

3. **Fix GraphRAGService.local_search()**:
   - Load ALL required parquet files (entities, communities, reports, text_units, relationships, covariates)
   - Pass all DataFrames to API

4. **Update GraphRAGConfigAdapter**:
   - Ensure `models` dictionary has correct keys
   - Set up proper workspace directory structure
   - Configure input/output storage properly

## 6. Error Patterns to Avoid

### ❌ **Configuration Errors**:
- Missing `default_chat_model` or `default_embedding_model` keys
- Empty API keys in LanguageModelConfig
- Invalid `root_dir` that doesn't exist

### ❌ **Data Loading Errors**:
- Calling query APIs without loading parquet files first
- Missing required DataFrames (especially for local_search)
- Wrong DataFrame column names or structure

### ❌ **API Misuse**:
- Treating GraphRAG like a simple function call
- Not handling async/await properly
- Ignoring required parameters like `community_level`

## 7. Performance Considerations

- **Indexing**: Very expensive (multiple LLM calls per document chunk)
- **Query Loading**: Must load multiple large parquet files
- **Memory**: Keep DataFrames in memory for multiple queries
- **Caching**: Cache loaded parquet data between queries on same collection

## 8. Next Steps for FileIntel

1. **Immediate**: Fix GraphRAGService API calls to match actual GraphRAG signatures
2. **Data Pipeline**: Implement proper DataFrame conversion from DocumentChunks
3. **File Management**: Handle parquet file loading/caching efficiently
4. **Error Handling**: Add proper validation for missing index files
5. **Testing**: Create integration tests with actual GraphRAG API calls
