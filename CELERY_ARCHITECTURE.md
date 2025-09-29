# Celery Task Queue Architecture

This document provides detailed information about FileIntel's Celery-based distributed task processing architecture, including worker configuration, monitoring, and operational considerations.

## Architecture Overview

FileIntel uses Celery for distributed task processing, replacing the previous job management system. This provides better scalability, fault tolerance, and monitoring capabilities.

### Core Components

1. **Celery Application** (`src/fileintel/celery_config.py`)
   - Centralized configuration and task discovery
   - Queue routing and task serialization settings
   - Worker monitoring and health checks

2. **Task Modules** (`src/fileintel/tasks/`)
   - `document_tasks.py`: Document processing and indexing
   - `graphrag_tasks.py`: GraphRAG index building and search operations
   - `llm_tasks.py`: Language model integration tasks
   - `workflow_tasks.py`: Complex multi-step workflow orchestration

3. **Message Broker** (Redis)
   - Task queue management
   - Result backend for task status and results
   - Message routing and persistence

## Task Categories and Queues

Tasks are organized into specialized queues for optimal resource allocation:

### Document Processing Queue (`document_processing`)
- **Tasks**: `process_document`, `process_collection`, `extract_metadata`, `complete_collection_analysis`
- **Characteristics**: I/O intensive, medium duration (30s-5min)
- **Worker Requirements**: Standard CPU, sufficient memory for document processing

### Embedding Processing Queue (`embedding_processing`)
- **Tasks**: `generate_text_embedding`, `generate_and_store_chunk_embedding`, `generate_collection_embeddings_simple`
- **Characteristics**: High throughput, network intensive, parallel batch processing
- **Worker Requirements**: High concurrency (8+ workers), moderate memory, network optimized

### LLM Processing Queue (`llm_processing`)
- **Tasks**: `summarize_content`, text generation tasks
- **Characteristics**: Complex reasoning, variable duration, high context memory
- **Worker Requirements**: Lower concurrency (2-4 workers), high memory (2GB+ per worker)

### RAG Processing Queue (`rag_processing`)
- **Tasks**: Vector queries, `get_graphrag_index_status`, `remove_graphrag_index`
- **Characteristics**: Quick execution, lightweight operations
- **Worker Requirements**: Standard configuration, moderate concurrency

### GraphRAG Indexing Queue (`graphrag_indexing`)
- **Tasks**: `build_graph_index`, `build_graphrag_index_task`, `update_collection_index`
- **Characteristics**: Extremely memory intensive, long duration (30-60min)
- **Worker Requirements**: Dedicated workers, very high memory (4GB+), low concurrency (1-2 workers)

### GraphRAG Queries Queue (`graphrag_queries`)
- **Tasks**: `query_graph_global`, `query_graph_local`, `adaptive_graphrag_query`
- **Characteristics**: Memory intensive, variable duration, graph traversal
- **Worker Requirements**: Moderate memory (1GB), balanced concurrency (3-4 workers)

## Worker Configuration

### Basic Worker Setup

```bash
# Start a general-purpose worker
celery -A fileintel.celery_config worker --loglevel=info

# Start workers for specific queues
celery -A fileintel.celery_config worker --queues=document_processing --loglevel=info
celery -A fileintel.celery_config worker --queues=embedding_processing --concurrency=8 --loglevel=info
celery -A fileintel.celery_config worker --queues=graphrag_indexing --concurrency=1 --loglevel=info
```

### Production Worker Configuration

#### Document Processing Workers
```bash
celery -A fileintel.celery_config worker \
  --queues=document_processing \
  --concurrency=4 \
  --max-tasks-per-child=10 \
  --time-limit=600 \
  --soft-time-limit=480 \
  --loglevel=info
```

#### Embedding Processing Workers
```bash
celery -A fileintel.celery_config worker \
  --queues=embedding_processing \
  --concurrency=8 \
  --max-memory-per-child=512000 \
  --time-limit=300 \
  --soft-time-limit=240 \
  --loglevel=info
```

#### LLM Processing Workers
```bash
celery -A fileintel.celery_config worker \
  --queues=llm_processing \
  --concurrency=2 \
  --max-memory-per-child=2000000 \
  --time-limit=1800 \
  --soft-time-limit=1500 \
  --loglevel=info
```

#### GraphRAG Indexing Workers
```bash
celery -A fileintel.celery_config worker \
  --queues=graphrag_indexing \
  --concurrency=1 \
  --max-memory-per-child=4000000 \
  --time-limit=3600 \
  --soft-time-limit=3300 \
  --loglevel=info
```

#### GraphRAG Query Workers
```bash
celery -A fileintel.celery_config worker \
  --queues=graphrag_queries \
  --concurrency=3 \
  --max-memory-per-child=1000000 \
  --time-limit=600 \
  --soft-time-limit=480 \
  --loglevel=info
```

#### RAG Processing Workers
```bash
celery -A fileintel.celery_config worker \
  --queues=rag_processing \
  --concurrency=4 \
  --max-memory-per-child=500000 \
  --time-limit=300 \
  --soft-time-limit=240 \
  --loglevel=info
```

## Monitoring and Management

### Celery Flower (Web UI)
```bash
# Start Flower monitoring
celery -A fileintel.celery_config flower --port=5555
```

Access web interface at `http://localhost:5555` for:
- Real-time task monitoring
- Worker status and statistics
- Task history and results
- Queue length monitoring

### CLI Monitoring Commands

```bash
# Check worker status
celery -A fileintel.celery_config status

# Monitor active tasks
celery -A fileintel.celery_config inspect active

# Check queue lengths
celery -A fileintel.celery_config inspect reserved

# Worker statistics
celery -A fileintel.celery_config inspect stats
```

### API v2 Monitoring Endpoints

The v2 API provides programmatic access to task monitoring:

- `GET /api/v2/tasks/metrics` - Worker and queue statistics
- `GET /api/v2/tasks/active` - Currently running tasks
- `GET /api/v2/tasks/{task_id}` - Individual task status
- `POST /api/v2/tasks/{task_id}/cancel` - Cancel running tasks

## Task Orchestration Patterns

### Groups (Parallel Execution)
```python
from celery import group
from tasks.document_tasks import process_document

# Process multiple documents in parallel
job = group(
    process_document.s(file_path, doc_id, collection_id)
    for file_path, doc_id in document_list
)
result = job.apply_async()
```

### Chains (Sequential Execution)
```python
from celery import chain
from tasks.document_tasks import process_document
from tasks.llm_tasks import generate_batch_embeddings

# Sequential processing pipeline
pipeline = chain(
    process_document.s(file_path, doc_id, collection_id),
    generate_batch_embeddings.s(),
)
result = pipeline.apply_async()
```

### Chords (Group + Callback)
```python
from celery import chord
from tasks.workflow_tasks import update_collection_index

# Process multiple documents, then update index
callback = update_collection_index.s(collection_id)
job = chord(group_tasks)(callback)
result = job.apply_async()
```

## Operational Considerations

### Scaling Workers

1. **Horizontal Scaling**: Add more worker processes
   ```bash
   # Scale by adding workers on different machines
   celery -A fileintel.celery_config worker --hostname=worker1@%h
   celery -A fileintel.celery_config worker --hostname=worker2@%h
   ```

2. **Vertical Scaling**: Increase concurrency per worker
   ```bash
   # Increase concurrent tasks per worker
   celery -A fileintel.celery_config worker --concurrency=8
   ```

### Memory Management

- Use `--max-memory-per-child` to prevent memory leaks
- Monitor memory usage with `--max-tasks-per-child`
- Set appropriate task time limits to prevent runaway processes

### Error Handling and Retries

Tasks implement automatic retry logic:
```python
@app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3})
def example_task(self, ...):
    # Task implementation
    pass
```

### Health Checks

Implement health checks for workers:
```bash
# Check if workers are responding
celery -A fileintel.celery_config inspect ping
```

## Configuration Files

### Environment Variables
```bash
# Celery Configuration
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
CELERY_TASK_SERIALIZER=json
CELERY_RESULT_SERIALIZER=json
CELERY_ACCEPT_CONTENT=['json']
CELERY_TIMEZONE=UTC
CELERY_ENABLE_UTC=True

# Worker Configuration
CELERY_WORKER_PREFETCH_MULTIPLIER=1
CELERY_TASK_ACKS_LATE=True
CELERY_WORKER_DISABLE_RATE_LIMITS=False
```

### Docker Compose Configuration
```yaml
services:
  celery:
    build: .
    command: celery -A fileintel.celery_config worker --loglevel=info
    volumes:
      - ./src:/app/src
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
    depends_on:
      - redis
      - postgres
    deploy:
      replicas: 3

  celery-flower:
    build: .
    command: celery -A fileintel.celery_config flower --port=5555
    ports:
      - "5555:5555"
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
    depends_on:
      - redis
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Reduce worker concurrency
   - Implement task chunking for large datasets
   - Use `--max-memory-per-child`

2. **Slow Task Processing**
   - Check queue lengths and worker availability
   - Optimize task implementation
   - Scale workers horizontally

3. **Task Failures**
   - Check worker logs for exceptions
   - Verify database and Redis connectivity
   - Review task timeout settings

### Logging Configuration

```python
# In celery_config.py
CELERY_WORKER_LOG_FORMAT = '[%(asctime)s: %(levelname)s/%(processName)s] %(message)s'
CELERY_WORKER_TASK_LOG_FORMAT = '[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s'
```

## Migration from Job Management System

The Celery architecture replaces the previous job management system with the following improvements:

1. **Better Fault Tolerance**: Automatic task retry and recovery
2. **Horizontal Scalability**: Easy worker scaling across machines
3. **Advanced Patterns**: Groups, chains, and chords for complex workflows
4. **Rich Monitoring**: Built-in monitoring and management tools
5. **Industry Standard**: Well-established patterns and best practices

### API Compatibility

- **v1 API**: Legacy endpoints maintained for backward compatibility
- **v2 API**: New task-based endpoints with direct Celery integration
- **CLI**: Updated to use v2 API endpoints and task monitoring

This architecture provides a robust foundation for distributed document processing and analysis at scale.
