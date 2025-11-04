# FileIntel Architecture

High-level overview of FileIntel's distributed architecture, components, and key workflows.

## System Components

### Services (Docker Compose)

- **api** - FastAPI application serving v2 task-based endpoints
- **celery-worker** - Distributed task processors (scalable horizontally)
- **postgres** - PostgreSQL with pgvector extension for embeddings and relational data
- **redis** - Celery message broker, result backend, and LLM response cache
- **flower** - Celery monitoring dashboard (http://localhost:5555)
- **backup** - Automated daily PostgreSQL backups (production)

### External Integrations

- **LLM Providers** - OpenAI, Anthropic, or local vLLM/Ollama
- **MinerU** - Advanced PDF extraction with OCR and layout detection
- **GraphRAG** - Microsoft's graph-based RAG (embedded library)

## Source Structure

```
src/fileintel/
├── api/                      # FastAPI application (v2 task-based endpoints)
├── tasks/                    # Celery task definitions
│   ├── document_tasks.py     # Document processing
│   ├── graphrag_tasks.py     # GraphRAG indexing/querying
│   ├── llm_tasks.py          # LLM/embedding generation (batched)
│   └── workflow_tasks.py     # Multi-step orchestration
├── cli/                      # Typer-based command-line interface
├── storage/                  # PostgreSQL storage and SQLAlchemy models
├── document_processing/      # Document parsing, chunking, metadata extraction
│   ├── type_aware_chunking.py    # Semantic content-type detection
│   ├── element_filter.py         # TOC/LOF filtering
│   └── processors/               # PDF/EPUB/MOBI extractors
├── rag/
│   ├── vector_rag/          # Embedding-based semantic search
│   └── graph_rag/           # GraphRAG integration layer
│       ├── execution/       # Phase execution with retry logic
│       └── validators/      # Completeness validation
├── llm_integration/         # LLM clients, rate limiting, batching
├── citation/                # Citation extraction and formatting
├── prompt_management/       # Prompt template management
├── core/                    # Configuration, logging, exceptions
└── celery_config.py         # Celery app configuration

src/graphrag/                # Microsoft GraphRAG (customized)
├── index/operations/
│   └── cluster_graph.py     # Custom pyramid hierarchy algorithm
└── index/run/
    └── checkpoint_manager.py # Workflow resume system
```

## Key Features

### Hybrid RAG System

**Query Classification:**
- LLM-based semantic classification (90-95% accuracy)
- Keyword-based fallback (fast, deterministic)
- Hybrid mode with automatic fallback

**RAG Strategies:**
- **Vector RAG** - Semantic similarity search via pgvector
- **GraphRAG** - Relationship and entity-based queries
- **Hybrid** - Combines both approaches

**Result Reranking:**
- Optional vLLM-based semantic reranking
- Over-retrieve (20 chunks) → rerank → return top K
- 50-200ms additional latency for improved relevance

### Document Processing

**Multi-Format Support:**
- PDF, EPUB, MOBI via MinerU or traditional extractors
- OCR and layout detection for scanned documents
- Metadata extraction (titles, authors, citations)

**Type-Aware Chunking:**
- Statistical heuristics classify content: bullet lists, citation-heavy prose, structured sections, or prose
- Specialized chunking strategies per content type (5 strategies)
- Element filtering removes TOC/LOF/LOT before chunking (stored separately)
- Respects ~450 token limit with adaptive overage allowance (10-20%)

**Citation Management:**
- Automatic citation extraction from documents
- Source tracking with page numbers
- Multiple citation formats (Harvard, APA, etc.)

### Task Orchestration

**Celery Patterns:**
- **Groups** - Parallel task execution
- **Chains** - Sequential pipelines
- **Chords** - Parallel execution + callback

**Performance Optimizations:**
- Embedding batch processing: 25 chunks/task (10-25x throughput improvement)
- Async GraphRAG processing: configurable concurrency (4-50 concurrent LLM requests)
- Connection pooling: 10 base + 20 overflow per worker

**Workflow Examples:**
- Upload → Process → Chunk → Embed (batched) → Index
- Upload → Process → Auto-index GraphRAG (if enabled)
- Batch document processing with progress tracking

## Core Workflows

### 1. Document Ingestion

```
User Upload → API creates Document record
           → Celery task submitted to queue
           → Worker processes document
              - Extract text/metadata (MinerU)
              - Type-aware chunking
              - Generate embeddings
              - Extract citations
           → Store chunks in PostgreSQL
           → Auto-index GraphRAG (optional)
```

### 2. Query Processing

```
User Query → API submits query task
          → Query classifier determines strategy
             - Vector: Find similar chunks via pgvector
             - Graph: Use GraphRAG global/local search
             - Hybrid: Combine both results
          → Optional reranking (vLLM)
          → Generate answer with LLM
          → Return with citations and sources
```

### 3. GraphRAG Indexing

```
Collection ready → User triggers indexing
                → Celery task extracts entities/relationships
                   - Checkpoint after each workflow step
                   - Auto-resume from last checkpoint on failure
                → Build community graph (pyramid hierarchy)
                   - Adaptive scaling: ~20 entities per base community
                   - Level 0 = ROOT (coarse), Level N = BASE (fine-grained)
                → Generate community reports with gap prevention
                   - Retry failed items with exponential backoff
                   - Validate completeness (99% threshold)
                → Store in GraphRAG index directory
```

## Data Flow

```
┌─────────────┐
│   CLI/API   │
└──────┬──────┘
       │
   ┌───┴───┐
   │ Redis │ (Message Broker + Cache)
   └───┬───┘
       │
┌──────┴──────────┐
│ Celery Workers  │
│  - Process Docs │
│  - Batch Embed  │  <-- 25 chunks/batch
│  - Query LLMs   │
│  - Index Graph  │  <-- Checkpoint/Resume
└──────┬──────────┘
       │
┌──────┴──────────┐
│   PostgreSQL    │
│   + pgvector    │
│                 │
│  - Documents    │
│  - Chunks       │
│  - Embeddings   │
│  - Structure    │  <-- TOC/LOF/LOT
│  - GraphRAG     │  <-- Entities/Communities
└─────────────────┘
```

## Scaling Considerations

**Horizontal Scaling:**
- Multiple Celery workers (different queues: default, llm, graphrag)
- Multiple API instances behind load balancer
- PostgreSQL connection pooling (10 base + 20 overflow per worker)

**Performance Tuning:**
- Embedding batch processing: 25 chunks/task (10-25x improvement)
- LLM response caching in Redis (TTL: 3600s)
- Query classification caching (70%+ hit rate)
- GraphRAG async processing: 4-50 concurrent LLM requests
- Worker optimization: gevent, no max-tasks-per-child (prevents fork bomb)

**Large-Scale Processing (150K+ chunks):**
- Checkpoint/resume system: auto-resume from last successful workflow step
- Gap prevention: retry failed items with exponential backoff (20 attempts)
- Completeness validation: ensures 99%+ successful entity/community generation
- Pyramid hierarchy: adaptive scaling for graphs of any size (560 to 150K+ entities)

**Resource Requirements:**
- Minimum: 4 CPU, 16GB RAM, 100GB disk
- Recommended: 8+ CPU, 32GB+ RAM, 500GB SSD
- Large-scale (150K chunks): 16+ CPU, 64GB+ RAM, 1TB SSD
- GPU optional (for local LLM inference)

## Configuration

All configuration in `config/default.yaml`:

**Key sections:**
- `llm` - LLM provider, models, API keys, HTTP timeouts (900s for high queue depths)
- `rag` - Strategy, chunking, classification (hybrid), reranking, embedding batching
- `graphrag` - Pyramid hierarchy, checkpoint/resume, gap prevention, async processing
- `document_processing` - PDF processor, type-aware chunking, element filtering
- `celery` - Worker concurrency, timeouts, pool restart settings
- `storage` - Connection pooling (base + overflow), timeout settings
- `logging` - Component-level log configuration (46 component overrides)

Environment variables override YAML via `${VAR:-default}` syntax.

**Critical Settings for Large-Scale:**
```yaml
graphrag:
  enable_checkpoint_resume: true        # Auto-resume on failure
  validate_completeness: true           # Ensure 99%+ completeness
  gap_prevention.enabled: false         # In-phase retry (optional)
  async_processing.batch_size: 4        # Concurrent LLM requests

rag:
  embedding_processing.batch_size: 25   # Chunks per embedding batch

celery:
  worker_max_tasks_per_child: null      # Prevent fork bomb
  task_soft_time_limit: null            # No timeout for large batches
```

## Monitoring

- **Flower Dashboard** - http://localhost:5555 (Celery task monitoring)
- **API Health** - http://localhost:8000/health
- **Logs** - Component-level log configuration (INFO for progress, suppress verbose libraries)
- **Metrics** - Task success/failure rates via Flower
- **Completeness Validation** - Post-indexing reports for entity/community generation
- **Checkpoint Status** - Track workflow progress and resume points

## Storage Schema

**Key Tables:**
- `documents` - Document metadata, fingerprinting (UUID v5 from content)
- `chunks` - Text chunks with pgvector embeddings
- `document_structure` - Filtered TOC/LOF/LOT (separate from chunks)
- `graphrag_entities` - Entity storage with embeddings
- `graphrag_communities` - Community hierarchy (pyramid structure)
- `graphrag_relationships` - Relationship storage

**Storage Optimizations:**
- Content deduplication via fingerprinting
- Cascade deletions for collection cleanup
- pgvector indexing for semantic search
- JSONB for structured metadata and task history

## Advanced Features

### GraphRAG Pyramid Hierarchy

Custom clustering algorithm replacing standard hierarchical Leiden:
- **Adaptive scaling**: Automatically adjusts for graph size (560 to 150K+ entities)
- **Target base size**: ~20 entities per base community
- **Query-optimized structure**: Level 0 = ROOT (coarse), Level N = BASE (fine-grained)
- **Configurable consolidation**: Moderate pyramid steepness (6-8 levels)

**Environment Variables:**
- `GRAPHRAG_USE_PYRAMID_HIERARCHY=true` (default)
- `GRAPHRAG_USE_ADAPTIVE_SCALING=true` (default)
- `GRAPHRAG_ADAPTIVE_TARGET_BASE_SIZE=20` (entities per base community)

### Checkpoint & Resume System

Automatic workflow recovery for long-running GraphRAG indexing:
- Detects completed workflow steps via parquet file analysis
- Resumes from last successful checkpoint
- Validates data consistency before resume
- Critical for 96-hour indexing jobs (saves ~95 hours on mid-run failures)

### Gap Prevention & Completeness Validation

Ensures data integrity during GraphRAG indexing:
- **In-phase retry**: Exponential backoff for failed items (up to 20 attempts)
- **Gap filling**: Reduced concurrency to avoid 503 errors
- **Completeness validation**: Post-indexing report ensures 99%+ success rate
- **Configuration**: `graphrag.gap_prevention.enabled` (optional, default: false)

### Type-Aware Chunking

Intelligent content-type detection and specialized chunking:
- **Statistical heuristics**: Analyze line length, quote density, bullet patterns
- **Content types**: Bullet lists, citation-heavy prose, structured sections, prose
- **Specialized strategies**: 5 chunking algorithms per content type
- **Element filtering**: Remove TOC/LOF/LOT before chunking (stored separately)

## Testing Architecture

Test organization with pytest markers:
- `@pytest.mark.unit` - Unit tests for individual components
- `@pytest.mark.integration` - Integration tests (requires database)
- `@pytest.mark.celery` - Celery task tests
- `@pytest.mark.graphrag` - GraphRAG workflow tests
- `@pytest.mark.v2_api` - V2 API endpoint tests

**Execution:**
```bash
python run_tests.py celery      # Celery task tests
python run_tests.py api_v2      # V2 API tests
python run_tests.py graphrag    # GraphRAG tests
```

## Security

- Non-root container execution (UID 1000)
- Internal Docker network (no external DB/Redis ports)
- Environment-based secrets (or Docker secrets in production)
- Configurable resource limits per service

---

For deployment details, see [deployment.md](deployment.md).
For API reference, see [API_REFERENCE.md](API_REFERENCE.md).
For architectural decisions, see [docs/adr/](docs/adr/).
