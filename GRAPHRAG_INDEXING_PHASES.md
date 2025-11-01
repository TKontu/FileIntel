# GraphRAG Indexing Phases

This document describes the GraphRAG indexing workflow phases in execution order.

## Phase Order

### 1. load_input_documents
**Purpose**: Load and prepare document chunks for processing
**Input**: Document chunks from database
**Output**: Prepared text units
**Cache**: N/A (no cache)

---

### 2. create_base_text_units
**Purpose**: Create initial text units from documents
**Input**: Document chunks
**Output**: Base text units with IDs
**Cache**: `cache/create_base_text_units/`

---

### 3. create_final_documents
**Purpose**: Finalize document metadata and structure
**Input**: Base text units
**Output**: Final document records
**Cache**: `cache/create_final_documents/`

---

### 4. extract_graph
**Purpose**: Extract entities and relationships from text using LLM
**Input**: Text units
**Output**:
- `entities.parquet` - Extracted entities
- `relationships.parquet` - Relationships between entities
**Cache**: `cache/extract_graph/`
**Duration**: LONGEST PHASE (hours for large datasets)
**LLM intensive**: Yes - calls LLM for every text chunk

---

### 5. finalize_graph
**Purpose**: Clean, deduplicate, and finalize the knowledge graph
**Input**: Raw entities and relationships from extract_graph
**Output**: Cleaned and merged graph data
**Cache**: `cache/finalize_graph/`

---

### 6. extract_covariates
**Purpose**: Extract additional metadata and covariates
**Input**: Finalized graph
**Output**: Covariate data
**Cache**: `cache/extract_covariates/`

---

### 7. create_communities
**Purpose**: Detect communities using graph algorithms (Leiden)
**Input**: Finalized graph with entities and relationships
**Output**: Community assignments (multiple hierarchical levels 0-4)
**Cache**: `cache/create_communities/` or `cache/community_reporting/`
**Levels**: Creates hierarchical communities (level 0 = most granular, level 4 = highest abstraction)

---

### 8. create_final_text_units
**Purpose**: Finalize text units with community assignments
**Input**: Base text units + community data
**Output**: Text units with community links
**Cache**: `cache/create_final_text_units/`

---

### 9. create_community_reports
**Purpose**: Generate natural language summaries for each community using LLM
**Input**: Communities and their entities
**Output**: Community report summaries (one per community, per level)
**Cache**: `cache/community_reporting/` or `cache/summarize_descriptions/`
**Duration**: SECOND LONGEST PHASE
**LLM intensive**: Yes - generates summary for each community at each level

---

### 10. generate_text_embeddings
**Purpose**: Generate vector embeddings for semantic search
**Input**: Final text units and community reports
**Output**: Vector embeddings
**Cache**: `cache/generate_text_embeddings/`
**Model**: Uses embedding model (e.g., bge-large-en)

---

## Phase Dependencies

```
load_input_documents
    ↓
create_base_text_units
    ↓
create_final_documents
    ↓
extract_graph (entities + relationships)
    ↓
finalize_graph (clean & deduplicate)
    ↓
extract_covariates
    ↓
create_communities (hierarchical levels 0-4)
    ↓
create_final_text_units
    ↓
create_community_reports (summarize each community)
    ↓
generate_text_embeddings
    ↓
COMPLETE
```

## Most Time-Consuming Phases

1. **extract_graph** - Slowest (entity/relationship extraction via LLM)
2. **create_community_reports** - Second slowest (LLM summarization per community)
3. **generate_text_embeddings** - Fast (embedding model)

## Cache Folder Names

Different GraphRAG versions may use slightly different cache folder names:

| Phase | Possible Cache Names |
|-------|---------------------|
| extract_graph | `extract_graph/` |
| create_communities | `create_communities/` or `community_reporting/` |
| create_community_reports | `create_community_reports/` or `summarize_descriptions/` |

## Checkpoint Resume

The CheckpointManager detects completed workflows by checking:
- Existence of output `.parquet` files in cache folders
- Completeness of workflow outputs

**Important**: The system currently does NOT validate if partial results are complete (e.g., 21% vs 100%). It only checks if output files exist.

## Config Parameters

Key configuration parameters for throughput:

```yaml
graphrag:
  async:
    batch_size: 4                    # Concurrent chunks per batch
    max_concurrent_requests: 50      # Must match vLLM max_num_seqs
    batch_timeout: 900               # Timeout per batch (seconds)
```

## Status Tracking

Database field: `graphrag_indices.index_status`

Valid values:
- `building` - Indexing in progress
- `ready` - Indexing complete
- `failed` - Indexing failed
- `updating` - Re-indexing/updating

## Common Issues

1. **Phase marked complete at 21%**: extract_graph created output files but didn't finish → System thinks it's done
2. **Communities built from partial data**: If extract_graph is incomplete, communities will be incomplete
3. **Solution**: Delete community caches and let them rebuild after extract_graph completes
