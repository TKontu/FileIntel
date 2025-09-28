# GraphRAG API Reference

This document provides a comprehensive overview of all GraphRAG API endpoints, interfaces, and services available in the FileIntel system. The GraphRAG implementation includes both the original Microsoft GraphRAG library and FileIntel-specific integrations.

## Core GraphRAG Library API (Microsoft GraphRAG)

### Index Management

#### `build_index()`
- **Module**: `graphrag.api.index`
- **Function**: `async def build_index(config, method=IndexingMethod.Standard, is_update_run=False, memory_profile=False, callbacks=None, additional_context=None, verbose=False, input_documents=None)`
- **Purpose**: Run the GraphRAG indexing pipeline to create knowledge graphs from documents
- **Parameters**:
  - `config` (GraphRagConfig): The GraphRAG configuration
  - `method` (IndexingMethod): Indexing method (Standard, etc.)
  - `is_update_run` (bool): Whether this is an update run
  - `memory_profile` (bool): Enable memory profiling
  - `callbacks` (list): Workflow callbacks for monitoring
  - `additional_context` (dict): Additional pipeline context
  - `verbose` (bool): Enable verbose logging
  - `input_documents` (pd.DataFrame): Override document loading with supplied dataframe
- **Returns**: `list[PipelineRunResult]` - List of pipeline run results

#### `initialize_project_at()`
- **Module**: `graphrag.cli.initialize`
- **Function**: `def initialize_project_at(path: Path, force: bool) -> None`
- **Purpose**: Initialize a new GraphRAG project with configuration templates
- **Parameters**:
  - `path` (Path): Directory path to initialize project
  - `force` (bool): Whether to overwrite existing project files
- **Creates**:
  - `settings.yaml` - Main configuration file
  - `.env` - Environment variables
  - `prompts/` directory with default prompt templates
- **Returns**: None (creates files on filesystem)

### Query Interfaces

#### Global Search

##### `global_search()`
- **Module**: `graphrag.api.query`
- **Function**: `async def global_search(config, entities, communities, community_reports, community_level, dynamic_community_selection, response_type, query, callbacks=None, verbose=False)`
- **Purpose**: Perform global search using community reports for high-level reasoning
- **Parameters**:
  - `config` (GraphRagConfig): Configuration from settings.yaml
  - `entities` (pd.DataFrame): Entities data from entities.parquet
  - `communities` (pd.DataFrame): Communities data from communities.parquet
  - `community_reports` (pd.DataFrame): Reports data from community_reports.parquet
  - `community_level` (int): Community hierarchy level to search
  - `dynamic_community_selection` (bool): Enable dynamic community selection
  - `response_type` (str): Type of response to return
  - `query` (str): User query string
  - `callbacks` (list): Query callbacks
  - `verbose` (bool): Enable verbose logging
- **Returns**: `tuple[str|dict|list, str|list[pd.DataFrame]|dict]` - Response and context data

##### `global_search_streaming()`
- **Module**: `graphrag.api.query`
- **Function**: `def global_search_streaming(...)`
- **Purpose**: Streaming version of global search returning AsyncGenerator
- **Parameters**: Same as `global_search()`
- **Returns**: `AsyncGenerator` - Streaming response chunks

#### Local Search

##### `local_search()`
- **Module**: `graphrag.api.query`
- **Function**: `async def local_search(config, entities, communities, community_reports, text_units, relationships, covariates, community_level, response_type, query, callbacks=None, verbose=False)`
- **Purpose**: Perform local search using entity relationships and text units
- **Parameters**:
  - Additional params beyond global search:
  - `text_units` (pd.DataFrame): Text units data from text_units.parquet
  - `relationships` (pd.DataFrame): Relationships data from relationships.parquet
  - `covariates` (pd.DataFrame): Covariates data from covariates.parquet
- **Returns**: Same as `global_search()`

##### `local_search_streaming()`
- **Module**: `graphrag.api.query`
- **Function**: `def local_search_streaming(...)`
- **Purpose**: Streaming version of local search
- **Parameters**: Same as `local_search()`
- **Returns**: `AsyncGenerator` - Streaming response chunks

#### DRIFT Search

##### `drift_search()`
- **Module**: `graphrag.api.query`
- **Function**: `async def drift_search(config, entities, communities, community_reports, text_units, relationships, community_level, response_type, query, callbacks=None, verbose=False)`
- **Purpose**: Perform DRIFT (Diverse Reasoning with Information from Text) search
- **Parameters**: Similar to local search without covariates
- **Returns**: Same format as other search methods

##### `drift_search_streaming()`
- **Module**: `graphrag.api.query`
- **Function**: `def drift_search_streaming(...)`
- **Purpose**: Streaming version of DRIFT search
- **Parameters**: Same as `drift_search()`
- **Returns**: `AsyncGenerator` - Streaming response chunks

#### Basic Search

##### `basic_search()`
- **Module**: `graphrag.api.query`
- **Function**: `async def basic_search(config, text_units, query, callbacks=None, verbose=False)`
- **Purpose**: Basic text similarity search without graph reasoning
- **Parameters**:
  - `config` (GraphRagConfig): Configuration
  - `text_units` (pd.DataFrame): Text units data
  - `query` (str): User query
  - `callbacks` (list): Query callbacks
  - `verbose` (bool): Enable verbose logging
- **Returns**: Same format as other search methods

##### `basic_search_streaming()`
- **Module**: `graphrag.api.query`
- **Function**: `def basic_search_streaming(...)`
- **Purpose**: Streaming version of basic search
- **Parameters**: Same as `basic_search()`
- **Returns**: `AsyncGenerator` - Streaming response chunks

#### Multi-Index Search

##### `multi_index_global_search()`
- **Module**: `graphrag.api.query`
- **Function**: `async def multi_index_global_search(config, entities_list, communities_list, community_reports_list, index_names, community_level, dynamic_community_selection, response_type, streaming, query, callbacks=None, verbose=False)`
- **Purpose**: Global search across multiple knowledge graph indexes
- **Parameters**:
  - `entities_list` (list[pd.DataFrame]): List of entities dataframes
  - `communities_list` (list[pd.DataFrame]): List of communities dataframes
  - `community_reports_list` (list[pd.DataFrame]): List of reports dataframes
  - `index_names` (list[str]): List of index names
  - `streaming` (bool): Whether to stream results (not yet implemented)
- **Returns**: Same format with index mapping

##### `multi_index_local_search()`
- **Module**: `graphrag.api.query`
- **Function**: `async def multi_index_local_search(...)`
- **Purpose**: Local search across multiple indexes
- **Parameters**: Extended version of local search with lists and index names
- **Returns**: Same format with index mapping

##### `multi_index_drift_search()`
- **Module**: `graphrag.api.query`
- **Function**: `async def multi_index_drift_search(...)`
- **Purpose**: DRIFT search across multiple indexes
- **Parameters**: Extended version of drift search with lists and index names
- **Returns**: Same format with index mapping

##### `multi_index_basic_search()`
- **Module**: `graphrag.api.query`
- **Function**: `async def multi_index_basic_search(config, text_units_list, index_names, streaming, query, callbacks=None, verbose=False)`
- **Purpose**: Basic search across multiple indexes
- **Parameters**:
  - `text_units_list` (list[pd.DataFrame]): List of text units dataframes
  - `index_names` (list[str]): List of index names
  - `streaming` (bool): Whether to stream results
- **Returns**: Same format as basic search

### Prompt Tuning

#### `generate_indexing_prompts()`
- **Module**: `graphrag.api.prompt_tune`
- **Function**: `async def generate_indexing_prompts(config, chunk_size, overlap, limit=15, selection_method=DocSelectionType.RANDOM, domain=None, language=None, max_tokens, discover_entity_types=True, min_examples_required=2, n_subset_max=300, k=15, verbose=False)`
- **Purpose**: Generate optimized prompts for entity extraction and summarization based on input documents
- **Parameters**:
  - `config` (GraphRagConfig): GraphRAG configuration
  - `chunk_size` (int): Chunk size for text processing
  - `overlap` (int): Overlap between chunks
  - `limit` (int): Number of chunks to process
  - `selection_method` (DocSelectionType): Document selection method (RANDOM, AUTO)
  - `domain` (str): Domain to map documents to
  - `language` (str): Language for prompts
  - `max_tokens` (int): Maximum tokens for entity extraction
  - `discover_entity_types` (bool): Whether to generate entity types
  - `min_examples_required` (int): Minimum examples for prompts
  - `n_subset_max` (int): Max text chunks to embed for auto selection
  - `k` (int): Number of documents to select for auto method
  - `verbose` (bool): Enable verbose logging
- **Returns**: `tuple[str, str, str]` - (entity_extraction_prompt, entity_summarization_prompt, community_summarization_prompt)

## FileIntel GraphRAG Integration

### REST API Endpoints

#### Index Management

##### `POST /api/v2/graphrag/index`
- **Purpose**: Create or rebuild GraphRAG index for a collection
- **Request Body**:
  ```json
  {
    "collection_id": "string",
    "force_rebuild": "boolean"
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "data": {
      "task_id": "string",
      "collection_id": "string",
      "status": "string",
      "message": "string"
    }
  }
  ```
- **Process**: Collection validation → Document/chunk retrieval → Background Celery task → Task response

##### `GET /api/v2/graphrag/{collection_id}/status`
- **Purpose**: Get GraphRAG index status for a collection
- **Path Parameters**: `collection_id` - Collection ID or name
- **Response**:
  ```json
  {
    "success": true,
    "data": {
      "status": "indexed|not_indexed|index_missing|error",
      "index_path": "string",
      "documents_count": "integer",
      "entities_count": "integer",
      "communities_count": "integer",
      "created_at": "datetime",
      "updated_at": "datetime"
    }
  }
  ```

##### `DELETE /api/v2/graphrag/{collection_id}/index`
- **Purpose**: Remove GraphRAG index for a collection
- **Path Parameters**: `collection_id` - Collection ID or name
- **Response**:
  ```json
  {
    "success": true,
    "data": {
      "status": "success",
      "message": "string",
      "workspace_removed": "string"
    }
  }
  ```

##### `GET /api/v2/graphrag/{collection_id}/entities`
- **Purpose**: List GraphRAG entities for a collection
- **Path Parameters**: `collection_id` - Collection ID or name
- **Query Parameters**: `limit` (optional) - Maximum entities to return
- **Response**:
  ```json
  {
    "success": true,
    "data": [
      {
        "name": "string",
        "type": "string",
        "importance_score": "float",
        "description": "string"
      }
    ]
  }
  ```

##### `GET /api/v2/graphrag/{collection_id}/communities`
- **Purpose**: List GraphRAG communities for a collection
- **Path Parameters**: `collection_id` - Collection ID or name
- **Query Parameters**: `limit` (optional) - Maximum communities to return
- **Response**:
  ```json
  {
    "success": true,
    "data": [
      {
        "title": "string",
        "rank": "integer",
        "summary": "string",
        "size": "integer"
      }
    ]
  }
  ```

##### `GET /api/v2/graphrag/status`
- **Purpose**: Get GraphRAG system status and capabilities
- **Response**:
  ```json
  {
    "success": true,
    "data": {
      "status": "operational",
      "collections": [],
      "capabilities": {
        "graph_search": "boolean",
        "adaptive_routing": "boolean"
      }
    }
  }
  ```

#### Query Integration

GraphRAG queries are handled through the main query API at `POST /api/v2/collections/{collection_id}/query` with `search_type` parameter:
- `"graph"` - Generic GraphRAG query
- `"global"` - Global search using community reports
- `"local"` - Local search using entity relationships

### FileIntel GraphRAG Service

#### Core Methods

##### `async def query(query: str, collection_id: str) -> Dict[str, Any]`
- **Purpose**: Standard query interface, defaults to global search
- **Parameters**:
  - `query` (str): User query string
  - `collection_id` (str): Collection identifier
- **Returns**:
  ```python
  {
    "answer": "string",
    "sources": "list",
    "confidence": "float",
    "metadata": "dict"
  }
  ```

##### `async def build_index(documents: List[DocumentChunk], collection_id: str) -> str`
- **Purpose**: Build GraphRAG index from document chunks
- **Parameters**:
  - `documents` (List[DocumentChunk]): Document chunks to index
  - `collection_id` (str): Collection identifier
- **Returns**: `str` - Workspace path where index was created

##### `async def global_search(query: str, collection_id: str) -> Any`
- **Purpose**: Perform global search using community summaries
- **Parameters**:
  - `query` (str): User query
  - `collection_id` (str): Collection identifier
- **Returns**: GraphRAG response object with answer and context

##### `async def local_search(query: str, collection_id: str, community: str = "") -> Any`
- **Purpose**: Perform local search using entity relationships
- **Parameters**:
  - `query` (str): User query
  - `collection_id` (str): Collection identifier
  - `community` (str): Optional community constraint
- **Returns**: GraphRAG response object with answer and context

##### `async def global_query(collection_id: str, query: str) -> Dict[str, Any]`
- **Purpose**: Wrapper for global_search matching orchestrator interface
- **Returns**: Formatted response dict

##### `async def local_query(collection_id: str, query: str) -> Dict[str, Any]`
- **Purpose**: Wrapper for local_search matching orchestrator interface
- **Returns**: Formatted response dict

##### `def is_collection_indexed(collection_id: str) -> bool`
- **Purpose**: Check if collection has GraphRAG index
- **Parameters**: `collection_id` (str)
- **Returns**: `bool` - True if indexed

##### `async def remove_index(collection_id: str) -> Dict[str, Any]`
- **Purpose**: Remove GraphRAG index and workspace
- **Parameters**: `collection_id` (str)
- **Returns**: Status dict with success/error info

##### `async def get_index_status(collection_id: str) -> Dict[str, Any]`
- **Purpose**: Get detailed index status with counts
- **Parameters**: `collection_id` (str)
- **Returns**: Status dict with entity/community counts

### Celery Tasks

#### `build_graphrag_index_task`
- **Task**: `@app.task(queue="memory_intensive", time_limit=3600)`
- **Function**: `def build_graphrag_index_task(self, collection_id: str, force_rebuild: bool = False)`
- **Purpose**: Background task to build GraphRAG index
- **Parameters**:
  - `collection_id` (str): Collection identifier
  - `force_rebuild` (bool): Whether to force rebuild existing index
- **Returns**: Task result dict with status and metrics

#### `query_graph_global`
- **Task**: `@app.task(queue="io_bound", rate_limit="5/m")`
- **Function**: `def query_graph_global(self, query: str, collection_id: str, **kwargs)`
- **Purpose**: Asynchronous global GraphRAG query
- **Parameters**:
  - `query` (str): User query
  - `collection_id` (str): Collection identifier
- **Returns**: Query result dict

#### `query_graph_local`
- **Task**: `@app.task(queue="io_bound", rate_limit="10/m")`
- **Function**: `def query_graph_local(self, query: str, collection_id: str, **kwargs)`
- **Purpose**: Asynchronous local GraphRAG query
- **Parameters**:
  - `query` (str): User query
  - `collection_id` (str): Collection identifier
- **Returns**: Query result dict

#### `adaptive_graphrag_query`
- **Task**: `@app.task(queue="io_bound")`
- **Function**: `def adaptive_graphrag_query(self, query: str, collection_id: str, **kwargs)`
- **Purpose**: Adaptive query choosing between global/local search
- **Parameters**:
  - `query` (str): User query
  - `collection_id` (str): Collection identifier
- **Returns**: Query result dict with routing explanation

#### `get_graphrag_index_status`
- **Task**: `@app.task(queue="document_processing")`
- **Function**: `def get_graphrag_index_status(self, collection_id: str)`
- **Purpose**: Get index status as background task
- **Parameters**: `collection_id` (str)
- **Returns**: Status dict

#### `remove_graphrag_index`
- **Task**: `@app.task(queue="document_processing")`
- **Function**: `def remove_graphrag_index(self, collection_id: str)`
- **Purpose**: Remove index as background task
- **Parameters**: `collection_id` (str)
- **Returns**: Removal result dict

### CLI Commands

#### Index Operations

##### `fileintel graphrag index <collection>`
- **Purpose**: Create GraphRAG index for a collection
- **Arguments**: `collection` - Collection name or ID
- **Options**:
  - `--wait, -w` - Wait for completion
  - `--force, -f` - Force rebuild existing index
- **API Call**: `POST /api/v2/graphrag/index`

##### `fileintel graphrag rebuild <collection>`
- **Purpose**: Rebuild GraphRAG index (removes existing first)
- **Arguments**: `collection` - Collection name or ID
- **Options**:
  - `--wait, -w` - Wait for completion
  - `--yes, -y` - Skip confirmation
- **API Calls**: `DELETE /api/v2/graphrag/{collection}/index` → `POST /api/v2/graphrag/index`

##### `fileintel graphrag status <collection>`
- **Purpose**: Get GraphRAG index status
- **Arguments**: `collection` - Collection name or ID
- **API Call**: `GET /api/v2/graphrag/{collection}/status`

##### `fileintel graphrag system-status`
- **Purpose**: Check GraphRAG system status
- **API Call**: `GET /api/v2/graphrag/status`

#### Query Operations

##### `fileintel graphrag query <collection> <question>`
- **Purpose**: Query collection using GraphRAG
- **Arguments**:
  - `collection` - Collection name or ID
  - `question` - Question to ask
- **API Call**: `POST /api/v2/collections/{collection}/query` with `search_type="graph"`

#### Data Operations

##### `fileintel graphrag entities <collection>`
- **Purpose**: List GraphRAG entities in collection
- **Arguments**: `collection` - Collection name or ID
- **Options**: `--limit, -l` - Max entities to show (default: 20)
- **API Call**: `GET /api/v2/graphrag/{collection}/entities`

##### `fileintel graphrag communities <collection>`
- **Purpose**: List GraphRAG communities in collection
- **Arguments**: `collection` - Collection name or ID
- **Options**: `--limit, -l` - Max communities to show (default: 10)
- **API Call**: `GET /api/v2/graphrag/{collection}/communities`

### Core Microsoft GraphRAG CLI Commands

#### `graphrag init`
- **Purpose**: Initialize GraphRAG project with default configuration
- **Options**:
  - `--root` - Project root directory
  - `--config` - Configuration file path
- **Creates**: `settings.yaml`, `.env`, and prompt templates

#### `graphrag index`
- **Purpose**: Build GraphRAG knowledge graph index from documents
- **Options**:
  - `--root` - Project root directory
  - `--config` - Configuration file path
  - `--verbose` - Enable verbose output
  - `--memprofile` - Enable memory profiling
  - `--dryrun` - Show what would be done without executing
  - `--update` - Update existing index

#### `graphrag update`
- **Purpose**: Update existing GraphRAG index with new data
- **Options**:
  - `--root` - Project root directory
  - `--config` - Configuration file path
  - `--verbose` - Enable verbose output
  - `--memprofile` - Enable memory profiling

#### `graphrag query`
- **Purpose**: Query the GraphRAG knowledge graph
- **Arguments**: Query text
- **Options**:
  - `--root` - Project root directory
  - `--config` - Configuration file path
  - `--data` - Data directory
  - `--method` - Query method (`local`, `global`, `drift`, `basic`)
  - `--community-level` - Community level for search
  - `--response-type` - Response format
  - `--streaming` - Enable streaming responses
  - `--verbose` - Enable verbose output

#### `graphrag prompt-tune`
- **Purpose**: Auto-tune prompts based on input data
- **Options**:
  - `--root` - Project root directory
  - `--config` - Configuration file path
  - `--domain` - Domain description for prompt tuning
  - `--chunk-size` - Text chunk size
  - `--language` - Content language
  - `--limit` - Number of chunks to use
  - `--max-tokens` - Maximum tokens for prompt generation
  - `--min-examples-required` - Minimum examples for entity types
  - `--discover-entity-types` - Auto-discover entity types
  - `--no-entity-types` - Skip entity type discovery
  - `--output` - Output directory for generated prompts

## Configuration

GraphRAG configuration is handled through the FileIntel settings system:

```yaml
rag:
  graph_rag:
    root_dir: "./graphrag_workspace"
    llm_model: "gpt-4"
    embedding_model: "text-embedding-ada-002"
    community_level: 2
    default_confidence: 0.8
```

Configuration is automatically adapted from FileIntel settings to GraphRAG format using `GraphRAGConfigAdapter`.

## Data Flow

1. **Index Creation**: Documents → Chunks → GraphRAG pipeline → Entities/Communities/Reports → Parquet files
2. **Query Processing**: User query → Search type routing → GraphRAG API → Formatted response
3. **Storage Integration**: PostgreSQL for metadata, file system for GraphRAG artifacts
4. **Caching**: DataFrame cache for performance, config cache for repeated operations

## Error Handling

All GraphRAG operations include comprehensive error handling:
- Collection validation
- Index existence checks
- Graceful fallbacks (adaptive → vector search)
- Detailed error messages with context
- Background task error reporting

## Performance Considerations

- **Memory-intensive tasks**: Index building uses dedicated queue with higher timeouts
- **Rate limiting**: Query tasks have rate limits to prevent API overload
- **Caching**: Config and DataFrame caching to avoid repeated processing
- **Async operations**: All I/O operations use asyncio.to_thread for non-blocking execution

## Limitations

- Multi-index streaming not yet implemented for global/local/drift search
- Task retry functionality requires stored task parameters (placeholder implementation)
- Entity/community data not included in standard query responses (architectural decision)
- GraphRAG workspace management requires sufficient disk space for large collections