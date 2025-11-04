# FileIntel System Schema

**Purpose**: Structured description of FileIntel's entities, operations, and interactions for schema generation and system modeling.

---

## Core Entities

### Collection
**Description**: Container for related documents with shared configuration and indexing state.

**Attributes**:
- `id` (UUID, PK): Unique identifier
- `name` (String, unique): Human-readable name
- `description` (String, optional): Purpose or contents description
- `status` (Enum): processing | ready | indexing | failed
- `document_count` (Integer): Number of documents in collection
- `chunk_count` (Integer): Total chunks across all documents
- `graphrag_indexed` (Boolean): Whether GraphRAG index exists
- `current_task_id` (String, optional): Currently running task
- `task_history` (JSONB): Historical task execution records
- `created_at` (DateTime): Creation timestamp
- `updated_at` (DateTime): Last modification timestamp
- `status_updated_at` (DateTime): Last status change timestamp

**Relationships**:
- Has many Documents (cascade delete)
- Has one GraphRAGIndex (optional)

**Constraints**:
- `name` must be unique
- `status` transitions: processing → ready → indexing → ready | failed

---

### Document
**Description**: Single file (PDF, EPUB, MOBI) processed into chunks.

**Attributes**:
- `id` (UUID, PK): Unique identifier
- `collection_id` (UUID, FK): Parent collection
- `filename` (String): Original filename
- `file_path` (String): Storage location
- `file_size` (Integer): Size in bytes
- `file_type` (String): pdf | epub | mobi
- `content_fingerprint` (UUID): UUID v5 hash of content (deduplication)
- `status` (Enum): pending | processing | completed | failed
- `metadata` (JSONB): Extracted metadata (title, author, year, etc.)
- `chunk_count` (Integer): Number of chunks generated
- `processing_metadata` (JSONB): Processor used, extraction settings
- `created_at` (DateTime): Upload timestamp
- `processed_at` (DateTime, optional): Completion timestamp

**Relationships**:
- Belongs to Collection
- Has many Chunks (cascade delete)
- Has many DocumentStructure entries (cascade delete)

**Constraints**:
- `content_fingerprint` can identify duplicates within collection
- `status` must be updated atomically with chunk creation

---

### Chunk
**Description**: Text segment with embedding for semantic search.

**Attributes**:
- `id` (UUID, PK): Unique identifier
- `document_id` (UUID, FK): Source document
- `collection_id` (UUID, FK): Parent collection (denormalized)
- `content` (Text): Actual text content
- `embedding` (Vector): pgvector embedding (dimension: 1024 for bge-large-en)
- `chunk_index` (Integer): Position in document (0-based)
- `metadata` (JSONB): Additional context
  - `page_number` (Integer, optional): Source page
  - `content_type` (String): prose | bullet_list | citation_heavy | structured_sections
  - `element_type` (String, optional): text | title | table | image_caption
  - `token_count` (Integer): Approximate token count
- `created_at` (DateTime): Creation timestamp

**Relationships**:
- Belongs to Document
- Belongs to Collection

**Constraints**:
- `chunk_index` unique per document
- `embedding` dimension must match model (1024 for bge-large-en)
- Content length ≤ 450 tokens (with 10-20% overage allowance)

**Indexes**:
- Vector similarity index on `embedding` (IVFFlat or HNSW)
- Index on `collection_id` for filtering
- Index on `document_id, chunk_index` for ordering

---

### DocumentStructure
**Description**: Extracted structural elements (TOC, LOF, LOT) stored separately from chunks.

**Attributes**:
- `id` (UUID, PK): Unique identifier
- `document_id` (UUID, FK): Source document
- `structure_type` (Enum): toc | lof | lot | headers | filtered_content
- `data` (JSONB): Structured entries
  - For TOC: `[{section: "1.2", title: "Introduction", page: 5}, ...]`
  - For LOF/LOT: `[{number: "Figure 1", caption: "...", page: 10}, ...]`
- `created_at` (DateTime): Extraction timestamp

**Relationships**:
- Belongs to Document

**Constraints**:
- One entry per `(document_id, structure_type)` combination
- `data` schema depends on `structure_type`

---

### GraphRAGIndex
**Description**: GraphRAG index metadata for a collection.

**Attributes**:
- `id` (UUID, PK): Unique identifier
- `collection_id` (UUID, FK, unique): Parent collection
- `index_path` (String): Filesystem path to index directory
- `status` (Enum): pending | indexing | completed | failed
- `entity_count` (Integer): Number of entities extracted
- `community_count` (Integer): Number of communities detected
- `relationship_count` (Integer): Number of relationships
- `indexing_config` (JSONB): Configuration snapshot
  - `llm_model` (String): Model used for extraction
  - `embedding_model` (String): Model used for embeddings
  - `max_cluster_size` (Integer): Leiden max cluster size
  - `leiden_resolution` (Float): Resolution parameter
  - `pyramid_hierarchy_enabled` (Boolean): Whether pyramid used
- `checkpoint_state` (JSONB, optional): Current workflow progress
  - `last_completed_workflow` (String): Workflow name
  - `completed_at` (DateTime): Timestamp
- `completeness_report` (JSONB, optional): Validation results
  - `entity_description_completeness` (Float): 0.0-1.0
  - `community_report_completeness` (JSONB): Per-level stats
- `created_at` (DateTime): Index creation start
- `completed_at` (DateTime, optional): Index completion
- `error_message` (String, optional): Failure reason

**Relationships**:
- Belongs to Collection (one-to-one)
- Has many GraphRAGEntity entries
- Has many GraphRAGCommunity entries
- Has many GraphRAGRelationship entries

**Constraints**:
- `collection_id` unique (one index per collection)
- `status` cannot regress from completed to indexing

---

### GraphRAGEntity
**Description**: Named entity extracted from document chunks.

**Attributes**:
- `id` (UUID, PK): Unique identifier
- `index_id` (UUID, FK): Parent GraphRAG index
- `name` (String): Entity name
- `type` (String): Entity type (PERSON, ORG, CONCEPT, etc.)
- `description` (Text): LLM-generated entity description
- `embedding` (Vector, optional): Entity embedding
- `text_unit_ids` (JSONB): Source chunk IDs
- `created_at` (DateTime): Extraction timestamp

**Relationships**:
- Belongs to GraphRAGIndex
- Participates in GraphRAGRelationship entries (source or target)

**Constraints**:
- `(index_id, name)` should be unique (entities deduplicated by name)

---

### GraphRAGCommunity
**Description**: Hierarchical community in the entity graph.

**Attributes**:
- `id` (UUID, PK): Unique identifier
- `index_id` (UUID, FK): Parent GraphRAG index
- `community_id` (Integer): GraphRAG-assigned community ID
- `level` (Integer): Hierarchy level (0 = ROOT, N = BASE)
- `parent_id` (Integer, optional): Parent community ID (-1 for root)
- `title` (String): Community title
- `summary` (Text): LLM-generated community summary
- `full_content` (Text): Detailed community report
- `rank` (Float, optional): Importance ranking
- `entity_ids` (JSONB): Member entity IDs
- `relationship_ids` (JSONB): Member relationship IDs
- `created_at` (DateTime): Creation timestamp

**Relationships**:
- Belongs to GraphRAGIndex
- Has parent Community (self-referential, optional)

**Constraints**:
- `(index_id, community_id)` unique
- `level` 0 = coarsest (ROOT), higher levels = finer-grained
- `parent_id` must reference community at level-1 (or -1 for root)

**Indexes**:
- Index on `(index_id, level)` for level-based queries

---

### GraphRAGRelationship
**Description**: Directed relationship between two entities.

**Attributes**:
- `id` (UUID, PK): Unique identifier
- `index_id` (UUID, FK): Parent GraphRAG index
- `source_id` (UUID, FK): Source entity
- `target_id` (UUID, FK): Target entity
- `description` (Text): Relationship description
- `weight` (Float, optional): Relationship strength
- `text_unit_ids` (JSONB): Source chunk IDs
- `created_at` (DateTime): Extraction timestamp

**Relationships**:
- Belongs to GraphRAGIndex
- References GraphRAGEntity (source)
- References GraphRAGEntity (target)

**Constraints**:
- `source_id` and `target_id` must exist in same index

---

## Core Operations

### Document Upload & Processing

**Operation**: `upload_document`

**Input**:
- `collection_id` (UUID): Target collection
- `file` (Binary): File content
- `filename` (String): Original filename
- `auto_process` (Boolean, default: true): Process immediately

**Process**:
1. Validate file type (pdf | epub | mobi)
2. Check file size < max_file_size_mb (100MB default)
3. Generate content_fingerprint (UUID v5 from content)
4. Check for duplicates in collection
5. Store file to uploads directory
6. Create Document record (status: pending)
7. If auto_process: Submit process_document task

**Output**:
- `document_id` (UUID): Created document
- `task_id` (String): Celery task ID

**Side Effects**:
- Collection.status → processing
- Collection.document_count incremented

---

**Operation**: `process_document`

**Input**:
- `document_id` (UUID): Document to process

**Process**:
1. Update Document.status → processing
2. Extract content using processor:
   - Primary: MinerU (OCR + layout detection)
   - Fallback: Traditional PDF extractor
3. Extract metadata (title, author, year, etc.)
4. Filter TOC/LOF/LOT elements:
   - Store in DocumentStructure table
   - Remove from content stream
5. Apply type-aware chunking:
   - Analyze text statistics (line length, quote density, bullet patterns)
   - Classify content type: bullet_list | citation_heavy | structured_sections | prose
   - Apply specialized chunking strategy
6. Generate embeddings (batched, 25 chunks/batch):
   - Prepare text (clean OCR artifacts, truncate to 450 tokens)
   - Call embedding model
   - Store as pgvector
7. Create Chunk records
8. Update Document (status: completed, chunk_count, metadata)

**Output**:
- `chunk_ids` (List[UUID]): Created chunks
- `metadata` (Dict): Extracted metadata

**Side Effects**:
- Collection.chunk_count updated
- Collection.status → ready (if all docs processed)

**Error Handling**:
- Document.status → failed
- Collection.status → failed (if critical)
- Error message stored in processing_metadata

---

### Vector RAG Query

**Operation**: `query_collection`

**Input**:
- `collection_id` (UUID): Target collection
- `query` (String): User question
- `top_k` (Integer, default: 5): Number of results
- `strategy` (Enum, optional): vector | graph | hybrid (auto-detected if null)
- `reranking_enabled` (Boolean, default: true): Use semantic reranking

**Process**:
1. Classify query strategy (if not provided):
   - Method: hybrid (LLM with keyword fallback)
   - LLM analyzes query semantics
   - Fallback to keyword patterns on timeout (5s)
   - Cache result (TTL: 3600s)
2. Execute retrieval based on strategy:
   - **Vector**: Semantic similarity search
     - Generate query embedding
     - pgvector cosine similarity search
     - Retrieve top_k (or initial_retrieval_k if reranking)
   - **Graph**: GraphRAG search (requires index)
     - Load GraphRAG index
     - Execute local/global search
     - Return community reports + entities
   - **Hybrid**: Combine both
     - Execute both searches
     - Merge and deduplicate results
3. Optional reranking:
   - If enabled: call vLLM reranker API
   - Over-retrieve (20 chunks) → rerank → return top 5
   - Latency: +50-200ms
4. Generate answer:
   - Format context from chunks
   - Call LLM with prompt template
   - Extract citations and sources
5. Return structured response

**Output**:
- `answer` (String): Generated answer
- `citations` (List): Source references with page numbers
- `chunks` (List): Retrieved chunks with scores
- `strategy_used` (String): Actual strategy used
- `metadata` (Dict): Query stats (latency, token count, etc.)

**Side Effects**:
- Query classification cached in Redis
- LLM response may be cached (TTL: 3600s)

---

### GraphRAG Indexing

**Operation**: `index_collection_graphrag`

**Input**:
- `collection_id` (UUID): Target collection
- `resume_from_checkpoint` (Boolean, default: true): Resume if possible

**Process**:
1. Validate collection ready (status: ready, chunk_count > 0)
2. Create or load GraphRAGIndex record
3. Check for existing checkpoint:
   - If exists and valid: resume from last completed workflow
   - If not: start from beginning
4. Execute GraphRAG workflows (checkpointed):
   - **extract_graph**: Extract entities and relationships
     - Batch LLM requests (4-50 concurrent)
     - Retry failed items with exponential backoff (up to 20 attempts)
     - Store entities and relationships
   - **create_graph**: Build NetworkX graph
   - **cluster_graph**: Apply pyramid hierarchy clustering
     - Adaptive scaling: ~20 entities per base community
     - Build pyramid: Level 0 = ROOT, Level N = BASE
     - Store community hierarchy
   - **summarize_descriptions**: Generate entity descriptions
     - Gap prevention: retry failed items
     - Validate completeness (99% threshold)
   - **create_communities**: Formalize community structure
   - **create_community_reports**: Generate community summaries
     - Gap prevention: retry failed items
     - Validate completeness per level
   - **embed_entities**: Generate entity embeddings
   - **embed_graph**: Generate graph embeddings
5. Validate completeness:
   - Check entity description completeness
   - Check community report completeness per level
   - Log warnings if < 99%
6. Update GraphRAGIndex (status: completed, counts, completeness_report)
7. Update Collection (graphrag_indexed: true)

**Output**:
- `index_id` (UUID): GraphRAG index
- `stats` (Dict): Entity/community/relationship counts
- `completeness_report` (Dict): Validation results

**Side Effects**:
- Parquet files written to index_path
- Collection.status → indexing → ready
- Checkpoint state updated after each workflow

**Error Handling**:
- Checkpoint state preserved
- GraphRAGIndex.status → failed
- Resume possible from last successful workflow

**Configuration**:
- `enable_checkpoint_resume`: Enable auto-resume
- `validate_completeness`: Post-indexing validation
- `gap_prevention.enabled`: In-phase retry
- `async_processing.batch_size`: Concurrent LLM requests

---

### GraphRAG Query

**Operation**: `query_graphrag`

**Input**:
- `collection_id` (UUID): Target collection
- `query` (String): User question
- `mode` (Enum): local | global
- `community_level` (Integer, optional): Target level for local search

**Process**:
1. Validate GraphRAG index exists and is completed
2. Load GraphRAG index from filesystem
3. Execute search based on mode:
   - **Local**: Community-focused search
     - Find relevant communities at specified level
     - Retrieve community reports and entities
     - Generate answer from local context
   - **Global**: Corpus-wide search
     - Traverse hierarchy from Level 0 (ROOT)
     - Aggregate community reports
     - Generate answer from global context
4. Format response with citations
5. Return structured response

**Output**:
- `answer` (String): Generated answer
- `communities` (List): Retrieved communities
- `entities` (List): Relevant entities
- `relationships` (List): Relevant relationships
- `metadata` (Dict): Search stats

**Side Effects**:
- LLM response may be cached

---

### Batch Processing

**Operation**: `batch_upload_documents`

**Input**:
- `collection_id` (UUID): Target collection
- `files` (List[Binary]): Multiple files
- `auto_process` (Boolean, default: true): Process after upload

**Process**:
1. Validate batch size ≤ max_upload_batch_size (50 default)
2. Validate each file size ≤ max_file_size_mb
3. For each file: upload_document (sequential or parallel)
4. If auto_process: Submit batch_process_documents workflow

**Output**:
- `document_ids` (List[UUID]): Created documents
- `task_ids` (List[String]): Celery task IDs

---

**Operation**: `batch_process_documents`

**Input**:
- `document_ids` (List[UUID]): Documents to process

**Process**:
1. Create Celery group of process_document tasks
2. Execute in parallel (respecting worker concurrency)
3. Aggregate results via chord callback

**Output**:
- `results` (List[Dict]): Per-document results
- `success_count` (Integer): Successful processing count
- `failed_count` (Integer): Failed processing count

---

## System Interactions

### Document Processing Pipeline

```
Client → API → Celery Queue → Worker
  ↓        ↓          ↓            ↓
Upload  Validate   Route      Process:
File    Request    Task       1. Extract (MinerU)
                              2. Filter TOC/LOF
                              3. Classify content
                              4. Chunk (5 strategies)
                              5. Batch embed (25/batch)
                              6. Store chunks
  ↓                              ↓
PostgreSQL ← ← ← ← ← ← ← ← ← ← ←
(Document, Chunks, Structure)
```

### Query Processing Flow

```
Client → API → Query Classifier (LLM/Keyword/Hybrid)
                      ↓
              Strategy Decision: Vector | Graph | Hybrid
                      ↓
         ┌────────────┼────────────┐
         ↓            ↓            ↓
      Vector       Graph       Hybrid
      Search      Search      (Both)
         ↓            ↓            ↓
    pgvector    GraphRAG     Merge Results
    Similarity   Index           ↓
         ↓            ↓       Reranker (optional)
         └────────────┴───────────┘
                      ↓
                 LLM Answer Generation
                      ↓
              Client ← API
         (Answer + Citations)
```

### GraphRAG Indexing Flow

```
Client → API → Celery Queue → Worker
                                  ↓
                         Checkpoint Manager
                         (Check resume point)
                                  ↓
                    ┌─────────────┴──────────────┐
                    ↓                            ↓
               Fresh Start                  Resume from
               (Workflow 0)              Last Checkpoint
                    ↓                            ↓
                    └─────────────┬──────────────┘
                                  ↓
                         Execute Workflows:
                         1. Extract entities/rels (async batch)
                         2. Build graph
                         3. Cluster (pyramid hierarchy)
                         4. Summarize (gap prevention)
                         5. Create communities
                         6. Generate reports (gap prevention)
                         7. Embed entities/graph
                                  ↓
                         After each workflow:
                         - Write parquet checkpoint
                         - Update checkpoint_state
                                  ↓
                         Completeness Validation
                         (Entity desc, Community reports)
                                  ↓
                         Store results in PostgreSQL
                         (Entities, Communities, Relationships)
                                  ↓
                         Update GraphRAGIndex.status → completed
```

### Caching Strategy

```
Query → Classification Cache (Redis)
  ↓            ↓
  Miss       Hit (70%+)
  ↓            ↓
  LLM      Return cached
Classification  strategy
  ↓
Store in cache (TTL: 3600s)
  ↓
Continue query processing
  ↓
LLM Response Cache (Redis)
  ↓            ↓
  Miss       Hit
  ↓            ↓
  Generate   Return cached
  Answer     response
  ↓
Store in cache (TTL: 3600s)
```

---

## State Machines

### Document Status Transitions

```
pending → processing → completed
                    ↓
                  failed
```

**States**:
- `pending`: Uploaded, awaiting processing
- `processing`: Currently being extracted/chunked/embedded
- `completed`: Successfully processed, chunks created
- `failed`: Processing error, see processing_metadata.error

**Triggers**:
- `pending → processing`: process_document task started
- `processing → completed`: All chunks created successfully
- `processing → failed`: Extraction/chunking/embedding error

---

### Collection Status Transitions

```
processing → ready → indexing → ready
          ↓       ↓           ↓
        failed  failed      failed
```

**States**:
- `processing`: Documents being uploaded/processed
- `ready`: All documents processed, ready for queries
- `indexing`: GraphRAG index being built
- `failed`: Critical error in processing or indexing

**Triggers**:
- `processing → ready`: All documents.status = completed
- `ready → indexing`: index_collection_graphrag started
- `indexing → ready`: GraphRAG indexing completed
- `* → failed`: Unrecoverable error

---

### GraphRAGIndex Status Transitions

```
pending → indexing → completed
                  ↓
                failed
```

**States**:
- `pending`: Index record created, not started
- `indexing`: Workflows executing
- `completed`: All workflows successful, index ready
- `failed`: Workflow error, resume possible from checkpoint

**Triggers**:
- `pending → indexing`: First workflow started
- `indexing → completed`: All workflows successful + validation passed
- `indexing → failed`: Workflow error (checkpoint preserved)

---

## Data Consistency Rules

### Referential Integrity

1. **Collection → Documents**: Cascade delete
   - Deleting collection deletes all documents
   - Deleting collection deletes all chunks
   - Deleting collection deletes GraphRAG index

2. **Document → Chunks**: Cascade delete
   - Deleting document deletes all chunks
   - Deleting document deletes structure entries

3. **GraphRAGIndex → Entities/Communities/Relationships**: Cascade delete
   - Deleting index deletes all graph data

### Atomic Operations

1. **Document Processing**:
   - Chunks created in transaction
   - Document.status and chunk_count updated atomically
   - Rollback on embedding failure

2. **Collection Status**:
   - Status updated only after all documents processed
   - Document_count and chunk_count updated atomically

3. **GraphRAG Workflow**:
   - Checkpoint written after each workflow completes
   - Index status updated only after all workflows complete

### Idempotency

1. **Document Upload**:
   - content_fingerprint prevents duplicates
   - Same file uploaded twice: return existing document_id

2. **Chunk Embedding**:
   - Batch embedding failures: retry individual chunks
   - Same chunk embedded twice: update existing embedding

3. **GraphRAG Indexing**:
   - Resume from checkpoint: skip completed workflows
   - Re-running completed index: detect and skip

---

## Configuration Schema

### LLM Configuration

```yaml
llm:
  provider: openai | anthropic
  model: string
  temperature: float (0.0-1.0)
  context_length: integer
  rate_limit: integer (requests/minute)
  task_timeout_seconds: integer (null = no limit)
  http_timeout_seconds: integer
  max_retries: integer
  retry_backoff_min: integer (seconds)
  retry_backoff_max: integer (seconds)
  openai:
    base_url: string (URL)
    embedding_base_url: string (URL, optional)
    api_key: string
  anthropic:
    api_key: string
```

### RAG Configuration

```yaml
rag:
  strategy: separate | merge
  embedding_provider: openai
  embedding_model: string
  embedding_max_tokens: integer
  enable_two_tier_chunking: boolean

  chunking:
    chunk_size: integer (characters)
    chunk_overlap: integer (characters)
    target_sentences: integer
    overlap_sentences: integer

  embedding_processing:
    batch_size: integer (1-100)
    fallback_to_single: boolean
    retry_failed_individually: boolean

  classification_method: llm | keyword | hybrid
  classification_model: string
  classification_temperature: float
  classification_timeout_seconds: integer
  classification_cache_enabled: boolean
  classification_cache_ttl: integer (seconds)

  reranking:
    enabled: boolean
    base_url: string (URL)
    model_name: string
    initial_retrieval_k: integer
    final_top_k: integer
    rerank_vector_results: boolean
    rerank_graph_results: boolean
    rerank_hybrid_results: boolean
```

### GraphRAG Configuration

```yaml
graphrag:
  llm_model: string
  embedding_model: string
  community_levels: integer (query parameter)
  max_cluster_size: integer (50-150)
  leiden_resolution: float (0.5-2.0)
  max_tokens: integer
  index_base_path: string (path)

  async_processing:
    enabled: boolean
    batch_size: integer (concurrent requests)
    max_concurrent_requests: integer
    batch_timeout: integer (seconds)
    fallback_to_sequential: boolean

  enable_checkpoint_resume: boolean
  validate_checkpoints: boolean

  gap_prevention:
    enabled: boolean
    max_retries_per_item: integer
    retry_backoff_base: float
    retry_backoff_max: float
    retry_jitter: boolean
    gap_fill_concurrency: integer

  validate_completeness: boolean
  completeness_threshold: float (0.0-1.0)
```

### Document Processing Configuration

```yaml
document_processing:
  chunk_size: integer (characters)
  overlap: integer (characters)
  max_file_size: string (e.g., "100MB")
  supported_formats: [pdf, epub, mobi]
  primary_pdf_processor: mineru | traditional
  use_type_aware_chunking: boolean

  mineru:
    api_type: selfhosted | commercial
    base_url: string (URL)
    timeout: integer (seconds, null = no timeout)
    model_version: pipeline | vlm
    use_element_level_types: boolean
    enable_element_filtering: boolean
    save_outputs: boolean
    output_directory: string (path)
```

### Celery Configuration

```yaml
celery:
  task_soft_time_limit: integer (seconds, null = no limit)
  task_time_limit: integer (seconds, null = no limit)
  worker_pool_restarts: boolean
  worker_pool_restart_timeout: integer (seconds)
  worker_max_tasks_per_child: integer (null = never restart)
```

### Storage Configuration

```yaml
storage:
  type: postgres
  connection_string: string (connection URL)
  pool_size: integer (connections per worker)
  max_overflow: integer (additional burst capacity)
  pool_timeout: integer (seconds)
```

---

## API Schema (v2)

### REST Endpoints

**Collections**:
- `POST /api/v2/collections` - Create collection
- `GET /api/v2/collections` - List collections
- `GET /api/v2/collections/{id}` - Get collection details
- `DELETE /api/v2/collections/{id}` - Delete collection

**Documents**:
- `POST /api/v2/documents` - Upload document
- `GET /api/v2/documents?collection_id={id}` - List documents
- `GET /api/v2/documents/{id}` - Get document details
- `DELETE /api/v2/documents/{id}` - Delete document

**Query**:
- `POST /api/v2/query` - Query collection
- `POST /api/v2/query/batch` - Batch query

**GraphRAG**:
- `POST /api/v2/graphrag/index` - Build GraphRAG index
- `POST /api/v2/graphrag/query` - Query with GraphRAG
- `GET /api/v2/graphrag/status/{collection_id}` - Index status

**Tasks**:
- `GET /api/v2/tasks/{task_id}` - Get task status
- `GET /api/v2/tasks/metrics` - System metrics

### Request/Response Schemas

**Create Collection**:
```json
// Request
{
  "name": "string (required, unique)",
  "description": "string (optional)"
}

// Response
{
  "status": "success",
  "data": {
    "id": "uuid",
    "name": "string",
    "description": "string",
    "status": "ready",
    "created_at": "datetime"
  }
}
```

**Upload Document**:
```json
// Request (multipart/form-data)
{
  "file": "binary (required)",
  "collection_id": "uuid (required)",
  "auto_process": "boolean (optional, default: true)"
}

// Response
{
  "status": "success",
  "data": {
    "document_id": "uuid",
    "task_id": "string (if auto_process)",
    "filename": "string",
    "file_size": "integer",
    "status": "pending | processing"
  }
}
```

**Query Collection**:
```json
// Request
{
  "collection_id": "uuid (required)",
  "query": "string (required)",
  "top_k": "integer (optional, default: 5)",
  "strategy": "vector | graph | hybrid (optional, auto-detect)",
  "reranking_enabled": "boolean (optional, default: true)"
}

// Response
{
  "status": "success",
  "data": {
    "answer": "string",
    "citations": [
      {
        "text": "string",
        "source": "string (filename)",
        "page": "integer (optional)"
      }
    ],
    "chunks": [
      {
        "content": "string",
        "score": "float",
        "document_id": "uuid",
        "page_number": "integer (optional)"
      }
    ],
    "strategy_used": "vector | graph | hybrid",
    "metadata": {
      "latency_ms": "integer",
      "token_count": "integer",
      "reranking_applied": "boolean"
    }
  }
}
```

**Index GraphRAG**:
```json
// Request
{
  "collection_id": "uuid (required)",
  "resume_from_checkpoint": "boolean (optional, default: true)"
}

// Response
{
  "status": "success",
  "data": {
    "task_id": "string",
    "index_id": "uuid",
    "status": "indexing"
  }
}
```

**GraphRAG Status**:
```json
// Response
{
  "status": "success",
  "data": {
    "index_id": "uuid",
    "status": "completed | indexing | failed",
    "entity_count": "integer",
    "community_count": "integer",
    "relationship_count": "integer",
    "checkpoint_state": {
      "last_completed_workflow": "string",
      "completed_at": "datetime"
    },
    "completeness_report": {
      "entity_description_completeness": "float (0.0-1.0)",
      "community_report_completeness": {
        "level_0": "float",
        "level_1": "float",
        ...
      }
    },
    "created_at": "datetime",
    "completed_at": "datetime (optional)"
  }
}
```

---

## Event Schema

### Task Events (Celery)

**Document Processing Events**:
- `document.processing.started` - Processing initiated
- `document.processing.completed` - Processing successful
- `document.processing.failed` - Processing error
- `document.chunks.created` - Chunks created (count)
- `document.embeddings.generated` - Embeddings generated (batch count)

**GraphRAG Events**:
- `graphrag.indexing.started` - Indexing initiated
- `graphrag.workflow.started` - Workflow execution started
- `graphrag.workflow.completed` - Workflow execution completed
- `graphrag.workflow.failed` - Workflow execution failed
- `graphrag.checkpoint.saved` - Checkpoint written
- `graphrag.validation.completed` - Completeness validation done
- `graphrag.indexing.completed` - Indexing finished

**Query Events**:
- `query.classification.started` - Query classification started
- `query.classification.completed` - Strategy determined
- `query.retrieval.started` - Retrieval started
- `query.retrieval.completed` - Chunks retrieved
- `query.reranking.started` - Reranking started
- `query.reranking.completed` - Reranking finished
- `query.generation.started` - Answer generation started
- `query.generation.completed` - Answer generated

### Event Payload Schema

```json
{
  "event_type": "string (e.g., document.processing.completed)",
  "timestamp": "datetime (ISO 8601)",
  "entity_type": "document | collection | graphrag_index | query",
  "entity_id": "uuid",
  "data": {
    // Event-specific data
  },
  "metadata": {
    "task_id": "string (Celery task ID)",
    "worker_name": "string",
    "duration_ms": "integer (optional)"
  }
}
```

---

## Performance Characteristics

### Latency Targets

**Document Processing**:
- Upload: < 1s (per document)
- Extraction (MinerU pipeline): 2-10s per document
- Extraction (MinerU VLM): 30-60s first request, 10-20s subsequent
- Chunking: < 1s per document (100-500 chunks)
- Embedding (batched): ~2s per batch (25 chunks)
- Total: 10-60s per document (depends on processor)

**Query Processing**:
- Classification (cached): < 5ms
- Classification (LLM): 100-300ms
- Vector retrieval: 50-200ms
- GraphRAG retrieval: 500-2000ms
- Reranking: +50-200ms
- Answer generation: 1-5s
- Total (vector): 2-7s
- Total (graph): 3-10s

**GraphRAG Indexing**:
- Small collection (100 docs, 5K chunks): 10-30 min
- Medium collection (1000 docs, 50K chunks): 4-8 hours
- Large collection (2000 docs, 150K chunks): 48-96 hours

### Throughput

**Document Processing**:
- Sequential: 6-60 docs/hour (depends on processor)
- Parallel (4 workers): 24-240 docs/hour

**Embedding Generation**:
- Without batching: ~1000 chunks/hour
- With batching (25/batch): 10,000-25,000 chunks/hour

**Query Processing**:
- Vector queries: 30-60 queries/minute (per worker)
- Graph queries: 6-12 queries/minute (per worker)

### Resource Usage

**Memory (per worker)**:
- Base: 1-2 GB
- With active processing: 2-4 GB
- GraphRAG indexing: 4-8 GB

**Database**:
- Document: ~1 KB
- Chunk: ~1 KB + embedding (4 KB for 1024-dim float32)
- Total per document (100 chunks): ~500 KB
- Large collection (150K chunks): ~750 MB

**Disk**:
- Uploads: File size + 10% (metadata)
- GraphRAG index: ~100 MB per 10K chunks (parquet files)

---

## Error Handling Schema

### Error Categories

1. **Validation Errors** (400):
   - Invalid input format
   - Missing required fields
   - File type not supported
   - File size exceeds limit

2. **Not Found Errors** (404):
   - Collection not found
   - Document not found
   - GraphRAG index not found

3. **Conflict Errors** (409):
   - Duplicate collection name
   - Collection not ready for indexing

4. **Processing Errors** (500):
   - Document extraction failed
   - Embedding generation failed
   - GraphRAG workflow failed

### Error Response Schema

```json
{
  "status": "error",
  "error": {
    "code": "string (e.g., DOCUMENT_PROCESSING_FAILED)",
    "message": "string (human-readable)",
    "details": {
      // Error-specific details
    }
  },
  "metadata": {
    "timestamp": "datetime",
    "request_id": "uuid"
  }
}
```

### Retry Policies

**Document Processing**:
- Max retries: 3
- Backoff: exponential (2s, 4s, 8s)
- Retry on: extraction timeout, embedding API 503

**GraphRAG Indexing**:
- Max retries per item: 20 (gap prevention)
- Backoff: exponential (2s, 4s, 8s, ..., 120s max)
- Retry on: LLM API 503, timeout

**Query Processing**:
- Max retries: 5
- Backoff: exponential (2s, 4s, 8s, 16s, 32s)
- Retry on: LLM API 503, embedding API timeout

---

This schema document provides a comprehensive, structured description of FileIntel's entities, operations, state machines, and interactions, suitable for generating database schemas, API specifications, or system diagrams.
