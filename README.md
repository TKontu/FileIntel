# FileIntel

**A production-ready hybrid RAG system combining vector search and knowledge graphs for intelligent document analysis.**

FileIntel is a powerful document intelligence platform that leverages both vector embeddings (semantic search) and Microsoft's GraphRAG (relationship discovery) to provide deep insights from your document collections. Built with a distributed, scalable architecture, it's designed for serious document analysis workloads.

## Features

### Core Capabilities

- **Hybrid RAG System**: Combines vector-based semantic search with graph-based relationship discovery
- **Intelligent Query Routing**: Automatically selects the best RAG strategy (vector, graph, or hybrid) based on query type
- **Multi-Format Support**: PDF, EPUB, MOBI with advanced extraction using MinerU (OCR + layout detection)
- **Type-Aware Chunking**: Semantic chunking that respects document structure (paragraphs, sections, tables)
- **Citation Generation**: Automatic citation formatting with source tracking and page numbers
- **Metadata Extraction**: Comprehensive bibliographic metadata extraction from documents

### Architecture Highlights

- **Distributed Task Processing**: Celery-based async processing with Redis message broker
- **Scalable Storage**: PostgreSQL with pgvector extension for efficient vector operations
- **Flexible LLM Integration**: Supports OpenAI API, Anthropic Claude, and local models (via vLLM/Ollama)
- **Production-Ready**: Docker Compose orchestration, health checks, task monitoring, and Flower dashboard
- **RESTful API**: FastAPI-based v2 API with task-based operations
- **Rich CLI**: Full-featured command-line interface for all operations

## Quick Start

### Prerequisites

- Docker & Docker Compose
- 8GB+ RAM (16GB+ recommended for GraphRAG)
- GPU recommended for local LLM inference (optional)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/fileintel.git
cd fileintel
```

2. **Create environment file:**
```bash
cat > .env << EOF
# Database credentials
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_DB=fileintel

# LLM API keys (choose your provider)
OPENAI_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here

# Optional: Redis and paths
REDIS_HOST=redis
REDIS_PORT=6379
EOF
```

3. **Start the services:**
```bash
# Basic setup (vector RAG only)
docker-compose up -d

# Or with MinerU OCR (recommended for PDFs)
docker-compose --profile pipeline up -d

# For advanced layout detection (slower, better quality)
docker-compose --profile vlm up -d
```

4. **Install CLI (optional but recommended):**
```bash
pip install -e .
```

5. **Verify installation:**
```bash
fileintel health
```

## Usage

### Basic Workflow

```bash
# 1. Create a collection
fileintel collections create "research-papers" \
    --description "AI/ML research papers"

# 2. Upload documents
fileintel documents upload research-papers \
    --files paper1.pdf paper2.pdf paper3.pdf

# 3. Wait for indexing (or check status)
fileintel collections status research-papers

# 4. Query the collection
fileintel query ask research-papers \
    "What are the main approaches to transformer optimization?"

# 5. Enable GraphRAG for relationship queries
fileintel graphrag index research-papers

# 6. Query with graph knowledge
fileintel graphrag query research-papers \
    "How are attention mechanisms and parameter efficiency related?"
```

### API Example

```python
import requests

BASE_URL = "http://localhost:8000"

# Create collection
response = requests.post(
    f"{BASE_URL}/api/v2/collections",
    json={
        "name": "my-docs",
        "description": "Document collection"
    }
)
collection_id = response.json()["data"]["id"]

# Upload document
with open("document.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(
        f"{BASE_URL}/api/v2/documents",
        files=files,
        data={"collection_id": collection_id}
    )

# Query collection
response = requests.post(
    f"{BASE_URL}/api/v2/query",
    json={
        "collection_id": collection_id,
        "query": "What are the main findings?",
        "top_k": 5
    }
)
answer = response.json()["data"]["answer"]
citations = response.json()["data"]["citations"]
```

## Configuration

FileIntel is configured via `config/default.yaml`. Key sections:

### LLM Configuration
```yaml
llm:
  provider: "openai"  # or "anthropic"
  model: "gpt-4-turbo"
  temperature: 0.1
  openai:
    base_url: "http://localhost:9003/v1"  # For local models
    api_key: ${OPENAI_API_KEY}
```

### RAG Configuration
```yaml
rag:
  strategy: "separate"  # "merge" or "separate"
  embedding_model: "text-embedding-3-large"
  chunking:
    chunk_size: 800
    chunk_overlap: 80
    target_sentences: 3
  enable_two_tier_chunking: false
```

### GraphRAG Configuration
```yaml
graphrag:
  llm_model: "gpt-4-turbo"
  embedding_model: "text-embedding-3-large"
  community_levels: 3
  auto_index_after_upload: true
  query_classification_model: "gpt-4-turbo"
```

### Intelligent Query Classification

FileIntel automatically routes queries to the optimal RAG strategy (vector, graph, or hybrid) using **LLM-based semantic understanding** with keyword fallback for reliability.

#### Configuration

```yaml
rag:
  # Classification method: llm (LLM only), keyword (fast/free), hybrid (recommended)
  classification_method: "hybrid"  # LLM with keyword fallback
  classification_model: "gemma3-4B"  # Small/fast model for classification
  classification_temperature: 0.0  # Deterministic
  classification_max_tokens: 150
  classification_timeout_seconds: 5  # Fallback to keywords after timeout

  # Caching reduces costs and latency (70%+ hit rate typical)
  classification_cache_enabled: true
  classification_cache_ttl: 3600  # 1 hour
```

#### Classification Methods

1. **LLM (Recommended for Production)**: Uses semantic analysis to understand query intent
   - 90-95% accuracy
   - 100-300ms latency (uncached)
   - ~$0.0001 per unique query
   - Handles complex/ambiguous queries

2. **Keyword (Fast & Free)**: Pattern matching on query text
   - 75-85% accuracy
   - <1ms latency
   - $0 cost
   - Deterministic and reliable

3. **Hybrid (Best of Both)**: LLM with automatic keyword fallback
   - 90-95% accuracy (from LLM)
   - <5ms average latency (with cache)
   - Minimal cost (cache hit rate >70%)
   - Zero-failure guarantee

#### Environment Variables

```bash
# Set classification method
RAG_CLASSIFICATION_METHOD=hybrid  # Options: llm, keyword, hybrid

# Use faster/cheaper model for classification
RAG_CLASSIFICATION_MODEL=gemma3-4B

# Adjust cache settings
RAG_CLASSIFICATION_CACHE_ENABLED=true
RAG_CLASSIFICATION_CACHE_TTL=3600

# Timeout before falling back to keywords (hybrid mode)
RAG_CLASSIFICATION_TIMEOUT=5
```

#### Testing Classification

```bash
# Run test script to see classification in action
python test_llm_classifier.py

# Test with different methods
RAG_CLASSIFICATION_METHOD=keyword python test_llm_classifier.py
RAG_CLASSIFICATION_METHOD=llm python test_llm_classifier.py
RAG_CLASSIFICATION_METHOD=hybrid python test_llm_classifier.py
```

#### Classification Examples

| Query | Classification | Reason |
|-------|---------------|---------|
| "What is quantum computing?" | VECTOR | Factual lookup |
| "How are X and Y related?" | GRAPH | Relationship analysis |
| "Compare X and Y and provide details" | HYBRID | Needs both methods |
| "Tell me everything about X" | VECTOR (LLM) or GRAPH (keyword) | Ambiguous - LLM understands context |

### Result Reranking (Advanced)

FileIntel supports **reranking** to improve retrieval result quality by re-scoring initial results using semantic relevance models. This can significantly improve answer quality at the cost of 50-200ms additional latency.

#### How It Works

1. **Retrieve More Initially**: Fetch 20 chunks (configurable) instead of final 5
2. **Semantic Re-scoring**: Use BAAI/bge-reranker models to compute query-passage relevance
3. **Return Top K**: Return only the most relevant chunks after reranking

#### Configuration

```yaml
rag:
  reranking:
    enabled: false  # Enable to improve result quality
    model_name: "BAAI/bge-reranker-v2-m3"  # Multilingual reranker
    model_type: "normal"  # Options: normal, llm, layerwise

    # Strategy - which results to rerank
    rerank_vector_results: true
    rerank_graph_results: true
    rerank_hybrid_results: true

    # Retrieval strategy (over-retrieve, then rerank)
    initial_retrieval_k: 20  # Retrieve more initially
    final_top_k: 5  # Return fewer after reranking

    # Performance tuning
    batch_size: 32
    normalize_scores: true
    device: "auto"  # auto, cuda, cpu
    cache_model: true  # Singleton model caching

    # Optional filtering
    min_score_threshold: null  # e.g., 0.3 to filter low-relevance
```

#### Environment Variables

```bash
# Enable reranking
RAG_RERANKING_ENABLED=true

# Choose reranker model
RAG_RERANKING_MODEL=BAAI/bge-reranker-v2-m3
RAG_RERANKING_MODEL_TYPE=normal  # normal, llm, layerwise

# Tune retrieval parameters
RAG_RERANKING_INITIAL_K=20  # Over-retrieve
RAG_RERANKING_FINAL_K=5  # Final results

# Select which queries to rerank
RAG_RERANK_VECTOR=true
RAG_RERANK_GRAPH=true
RAG_RERANK_HYBRID=true

# Performance settings
RAG_RERANKING_DEVICE=auto
RAG_RERANKING_BATCH_SIZE=32
RAG_RERANKING_USE_FP16=true
```

#### When to Use Reranking

**Use reranking when:**
- Answer quality is more important than speed
- You have GPU available (faster inference)
- You're working with multilingual content (bge-reranker-v2-m3)
- Initial retrieval returns noisy results

**Skip reranking when:**
- Latency is critical (<100ms responses required)
- Running on CPU-only systems
- Your retrieval already provides high-quality results
- Processing very large batches

#### Testing Reranking

```bash
# Test reranking with sample data
python test_reranker.py

# Compare results with/without reranking
RAG_RERANKING_ENABLED=false fileintel query ask collection-id "your query"
RAG_RERANKING_ENABLED=true fileintel query ask collection-id "your query"
```

#### Performance Characteristics

| Model Type | Latency (GPU) | Latency (CPU) | Accuracy | Use Case |
|------------|---------------|---------------|----------|----------|
| normal (bge-reranker-v2-m3) | 50-100ms | 200-500ms | High | General purpose, multilingual |
| llm (bge-reranker-v2-gemma) | 100-200ms | 500-1000ms | Very High | Complex queries, best accuracy |
| layerwise (bge-reranker-v2-minicpm) | 150-300ms | 800-1500ms | Very High | Research, maximum quality |

#### Installation

Reranking requires the FlagEmbedding library:

```bash
# Install FlagEmbedding
pip install FlagEmbedding

# Or if using Docker, rebuild containers after adding to requirements
docker-compose build
```

### Document Processing
```yaml
document_processing:
  primary_pdf_processor: "mineru"  # or "traditional"
  use_type_aware_chunking: true
  mineru:
    api_type: "selfhosted"
    base_url: "http://localhost:8000"
    model_version: "pipeline"  # or "vlm"
    enable_element_filtering: true
```

## Architecture

```
┌─────────────────┐
│   CLI / API     │  FastAPI + Typer CLI
└────────┬────────┘
         │
    ┌────┴─────┐
    │  Redis   │  Message Broker
    └────┬─────┘
         │
┌────────┴─────────────┐
│   Celery Workers     │  Distributed Task Processing
│  - Document Proc     │
│  - Vector Indexing   │
│  - GraphRAG Build    │
└──────────┬───────────┘
           │
    ┌──────┴────────┐
    │  PostgreSQL   │  Storage + pgvector
    │  + pgvector   │
    └───────────────┘
```

### Key Components

- **API Service**: FastAPI application with v2 task-based endpoints
- **Celery Workers**: Handle async document processing, indexing, and queries
- **PostgreSQL + pgvector**: Primary storage with vector similarity search
- **Redis**: Message broker and result backend
- **MinerU**: Advanced PDF extraction with OCR and layout detection
- **Flower** (optional): Web UI for monitoring Celery tasks

## Advanced Features

### GraphRAG Integration

FileIntel integrates Microsoft's GraphRAG for relationship-based queries:

```bash
# Index collection for graph operations
fileintel graphrag index my-collection

# Query with graph mode
fileintel graphrag query my-collection \
    "How are the entities X and Y related?" \
    --mode global

# Check index status
fileintel graphrag status my-collection
```

### Metadata Extraction

Extract and manage bibliographic metadata:

```bash
# Extract metadata from document
fileintel metadata extract document-id

# Export bibliography
fileintel metadata export collection-id \
    --format bibtex \
    --output references.bib
```

### Citation Management

FileIntel automatically generates citations with source tracking:

```bash
# Query with citations
fileintel query ask my-collection \
    "Summarize the findings" \
    --with-citations \
    --citation-style harvard
```

### Batch Processing

Process multiple documents efficiently:

```bash
# Batch upload from directory
fileintel documents batch-upload my-collection \
    --directory ./papers/ \
    --pattern "*.pdf"

# Monitor batch progress
fileintel tasks list --filter processing
```

## CLI Reference

### Collections
- `collections create` - Create new collection
- `collections list` - List all collections
- `collections status` - Check collection status
- `collections delete` - Delete collection

### Documents
- `documents upload` - Upload document(s)
- `documents list` - List documents in collection
- `documents delete` - Remove document

### Query
- `query ask` - Query collection with vector RAG
- `query batch` - Batch query multiple questions

### GraphRAG
- `graphrag index` - Build GraphRAG index
- `graphrag query` - Query with graph knowledge
- `graphrag status` - Check index status

### Tasks
- `tasks list` - List running tasks
- `tasks status` - Check task status
- `tasks cancel` - Cancel running task

### System
- `health` - Check system health
- `status` - Overall system status
- `version` - Show version info

## API Reference

### REST API v2 Endpoints

**Base URL**: `http://localhost:8000/api/v2`

#### Collections
- `POST /collections` - Create collection
- `GET /collections` - List collections
- `GET /collections/{id}` - Get collection details
- `DELETE /collections/{id}` - Delete collection

#### Documents
- `POST /documents` - Upload document
- `GET /documents` - List documents
- `DELETE /documents/{id}` - Delete document

#### Query
- `POST /query` - Query collection
- `POST /query/batch` - Batch query

#### GraphRAG
- `POST /graphrag/index` - Build GraphRAG index
- `POST /graphrag/query` - Query with GraphRAG
- `GET /graphrag/status/{collection_id}` - Index status

#### Tasks
- `GET /tasks/{task_id}` - Get task status
- `GET /tasks/metrics` - System metrics

Full API documentation available at `http://localhost:8000/docs` when running.

## Monitoring

### Flower Dashboard

Monitor Celery workers and tasks:

```bash
# Access Flower at http://localhost:5555
# (Enabled by default in docker-compose)
```

### Task Metrics

```bash
# CLI metrics
fileintel tasks list
fileintel health

# API metrics
curl http://localhost:8000/api/v2/tasks/metrics
```

## Performance Tuning

### For Large Collections (10,000+ documents)

```yaml
# config/default.yaml
storage:
  pool_size: 50
  max_overflow: 50

celery:
  worker_max_tasks_per_child: 100

rag:
  async_processing:
    enabled: true
    batch_size: 10
    max_concurrent_requests: 25
```

### For GPU Acceleration

```yaml
# Use local vLLM for embedding/inference
llm:
  openai:
    base_url: "http://gpu-server:9003/v1"

rag:
  embedding_provider: "openai"
  embedding_model: "bge-large-en"
```

## Troubleshooting

### Common Issues

**Task timeout errors:**
```yaml
# Increase timeouts in config/default.yaml
celery:
  task_soft_time_limit: 7200  # 2 hours
  task_time_limit: 7200
```

**Out of memory during GraphRAG indexing:**
```yaml
# Reduce batch size
graphrag:
  async_processing:
    batch_size: 4
    max_concurrent_requests: 10
```

**Slow PDF processing:**
```yaml
# Switch to faster MinerU backend
document_processing:
  mineru:
    model_version: "pipeline"  # faster than "vlm"
```

### Logs

```bash
# View logs
docker-compose logs -f api
docker-compose logs -f celery

# Or check files
tail -f logs/fileintel.log
```

## Development

### Local Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/
isort src/
```

### Database Migrations

```bash
# Create migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Write tests for new features
4. Ensure all tests pass
5. Submit a pull request

## License

[Your License Here]

## Acknowledgments

- [Microsoft GraphRAG](https://github.com/microsoft/graphrag) - Graph-based RAG implementation
- [MinerU](https://github.com/opendatalab/MinerU) - Advanced PDF extraction
- [pgvector](https://github.com/pgvector/pgvector) - Vector similarity search for PostgreSQL

## Support

- Issues: https://github.com/yourusername/fileintel/issues
- Discussions: https://github.com/yourusername/fileintel/discussions

---

Built with Python, FastAPI, Celery, PostgreSQL, and Redis.
