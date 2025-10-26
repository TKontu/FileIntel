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
│   ├── llm_tasks.py          # LLM/embedding generation
│   └── workflow_tasks.py     # Multi-step orchestration
├── cli/                      # Typer-based command-line interface
├── storage/                  # PostgreSQL storage and SQLAlchemy models
├── document_processing/      # Document parsing, chunking, metadata extraction
├── rag/
│   ├── vector_rag/          # Embedding-based semantic search
│   └── graph_rag/           # GraphRAG integration layer
├── llm_integration/         # LLM clients, rate limiting, circuit breakers
├── citation/                # Citation extraction and formatting
├── prompt_management/       # Prompt template management
├── core/                    # Configuration, logging, exceptions
└── celery_config.py         # Celery app configuration
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
- Content-aware semantic chunking
- Respects document structure (paragraphs, sections, tables)
- Two-tier mode: graph chunks contain multiple vector chunks

**Citation Management:**
- Automatic citation extraction from documents
- Source tracking with page numbers
- Multiple citation formats (Harvard, APA, etc.)

### Task Orchestration

**Celery Patterns:**
- **Groups** - Parallel task execution
- **Chains** - Sequential pipelines
- **Chords** - Parallel execution + callback

**Workflow Examples:**
- Upload → Process → Chunk → Embed → Index
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
                → Build community graph
                → Generate community reports
                → Store in GraphRAG index directory
```

## Data Flow

```
┌─────────────┐
│   CLI/API   │
└──────┬──────┘
       │
   ┌───┴───┐
   │ Redis │ (Message Broker)
   └───┬───┘
       │
┌──────┴──────────┐
│ Celery Workers  │
│  - Process Docs │
│  - Generate     │
│    Embeddings   │
│  - Query LLMs   │
│  - Index Graph  │
└──────┬──────────┘
       │
┌──────┴──────────┐
│   PostgreSQL    │
│   + pgvector    │
│                 │
│  - Documents    │
│  - Chunks       │
│  - Embeddings   │
│  - Metadata     │
└─────────────────┘
```

## Scaling Considerations

**Horizontal Scaling:**
- Multiple Celery workers (different queues: default, llm, graphrag)
- Multiple API instances behind load balancer
- PostgreSQL connection pooling

**Performance Tuning:**
- Async batch processing for embeddings
- LLM response caching in Redis
- Query classification caching (70%+ hit rate)
- Configurable worker concurrency and timeouts

**Resource Requirements:**
- Minimum: 4 CPU, 16GB RAM, 100GB disk
- Recommended: 8+ CPU, 32GB+ RAM, 500GB SSD
- GPU optional (for local LLM inference)

## Configuration

All configuration in `config/default.yaml`:

**Key sections:**
- `llm` - LLM provider, models, API keys
- `rag` - Strategy, chunking, classification, reranking
- `graphrag` - Community levels, auto-indexing
- `document_processing` - PDF processor, type-aware chunking
- `celery` - Worker concurrency, timeouts
- `logging` - Component-level log configuration

Environment variables override YAML via `${VAR:-default}` syntax.

## Monitoring

- **Flower Dashboard** - http://localhost:5555 (Celery task monitoring)
- **API Health** - http://localhost:8000/health
- **Logs** - Configurable per-component logging (INFO for progress, DEBUG for details)
- **Metrics** - Task success/failure rates via Flower

## Security

- Non-root container execution (UID 1000)
- Internal Docker network (no external DB/Redis ports)
- Environment-based secrets (or Docker secrets in production)
- Configurable resource limits per service

---

For deployment details, see [deployment.md](deployment.md).
For API reference, see [API_REFERENCE.md](API_REFERENCE.md).
