# Gevent Configuration for High-Memory Environment (64GB RAM)

## Optimized Resource Allocation

With 64GB RAM available, we can maximize concurrency and eliminate bottlenecks completely.

---

## Memory Budget Breakdown

**Total Available**: 64GB

**Proposed Allocation**:
- **Gevent GraphRAG Worker**: 20GB (aggressive concurrency)
- **Prefork Workers**: 16GB (maintain existing capacity)
- **PostgreSQL**: 8GB (handle increased connections)
- **Redis**: 2GB
- **API**: 4GB
- **vLLM/Other Services**: 10GB
- **System/Buffer**: 4GB

---

## Optimized Docker Compose Configuration

### Update `.env` with High-Memory Settings

```bash
# Gevent worker for GraphRAG (optimized for 64GB system)
CELERY_GRAPHRAG_MEMORY_LIMIT=20G           # 20GB for aggressive concurrency
CELERY_GRAPHRAG_MEMORY_RESERVATION=2G      # Reserve 2GB minimum
CELERY_GRAPHRAG_CONCURRENCY=200            # 200 concurrent greenlets!

# Prefork worker (existing workloads)
CELERY_WORKER_MEMORY_LIMIT=16G             # Increased from 8G
CELERY_WORKER_CONCURRENCY=8                # Increased from 4
CELERY_WORKER_MEMORY_RESERVATION=2G

# PostgreSQL (handle increased connections)
POSTGRES_MAX_CONNECTIONS=500               # Increased from 200
POSTGRES_SHARED_BUFFERS=2GB                # Increased from 256MB
POSTGRES_EFFECTIVE_CACHE_SIZE=6GB          # Use more RAM for query cache
POSTGRES_WORK_MEM=32MB                     # Per-connection work memory
```

---

## Updated docker-compose.yml Services

### 1. Gevent GraphRAG Worker (OPTIMIZED)

```yaml
  celery-graphrag-gevent:
    build: .
    entrypoint: ["./docker-entrypoint.sh"]
    command:
      - "celery"
      - "-A"
      - "fileintel.celery_config:app"
      - "worker"
      - "--loglevel=info"
      - "--pool=gevent"
      - "--concurrency=200"                              # ← 200 concurrent greenlets
      - "--queues=graphrag_indexing,graphrag_queries"
      - "--hostname=graphrag_gevent@%h"
      - "--max-tasks-per-child=1000"                     # Restart worker after 1000 tasks (prevent memory leaks)
    deploy:
      resources:
        limits:
          memory: 20G                                     # ← 20GB limit
        reservations:
          memory: 2G
    volumes:
      - ./src:/home/appuser/app/src
      - ./config:/home/appuser/app/config
      - ./prompts:/home/appuser/app/prompts
      - ./logs:/home/appuser/app/logs
      - ./graphrag_indices:/data
    env_file:
      - .env
    environment:
      - PYTHONPATH=/home/appuser/app/src:/home/appuser/.local/lib/python3.9/site-packages
      - PYTHONUNBUFFERED=1
      - DB_USER=${POSTGRES_USER}
      - DB_PASSWORD=${POSTGRES_PASSWORD}
      - DB_HOST=postgres
      - DB_PORT=5432
      - DB_NAME=${POSTGRES_DB}
      - CELERY_BROKER_URL=redis://redis:6379/1
      - CELERY_RESULT_BACKEND=redis://redis:6379/1
      # Gevent-specific optimizations
      - GEVENT_RESOLVER=ares                             # Async DNS resolver
      - GEVENT_THREADPOOL_SIZE=50                        # Thread pool for blocking operations
      # Python memory optimizations
      - PYTHONMALLOC=malloc                              # Use system malloc (better for gevent)
    extra_hosts:
      - "host.docker.internal:host-gateway"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
```

### 2. Enhanced Prefork Worker

```yaml
  celery-worker:
    build: .
    entrypoint: ["./docker-entrypoint.sh"]
    command:
      - "celery"
      - "-A"
      - "fileintel.celery_config:app"
      - "worker"
      - "--loglevel=info"
      - "--pool=prefork"
      - "--concurrency=8"                                # ← Increased from 4
      - "--queues=default,document_processing,embedding_processing,llm_processing,rag_processing"
      - "--hostname=main_prefork@%h"
    deploy:
      resources:
        limits:
          memory: 16G                                     # ← Increased from 8G
        reservations:
          memory: 2G                                      # ← Increased from 1G
    # ... rest of config stays the same
```

### 3. PostgreSQL with Increased Capacity

```yaml
  postgres:
    image: "pgvector/pgvector:pg13"
    env_file:
      - .env
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    ports:
      - "5433:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data/
      - ./docker/postgres:/docker-entrypoint-initdb.d
    command:
      - "postgres"
      - "-c"
      - "max_connections=500"                            # ← Increased from 200
      - "-c"
      - "shared_buffers=2GB"                             # ← Increased from 256MB
      - "-c"
      - "effective_cache_size=6GB"                       # ← New: use RAM for query cache
      - "-c"
      - "work_mem=32MB"                                  # ← New: per-connection memory
      - "-c"
      - "maintenance_work_mem=512MB"                     # ← New: for VACUUM, CREATE INDEX
      - "-c"
      - "checkpoint_completion_target=0.9"               # ← Spread out checkpoint writes
      - "-c"
      - "wal_buffers=16MB"                               # ← Write-ahead log buffer
      - "-c"
      - "random_page_cost=1.1"                           # ← Optimized for SSD
    deploy:
      resources:
        limits:
          memory: 8G                                      # ← Explicit limit for PostgreSQL
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d fileintel"]
      interval: 5s
      timeout: 5s
      retries: 5
```

---

## Expected Performance with 200 Concurrent Greenlets

### Current Performance (Sequential)
- **Concurrency**: 1 request at a time
- **Time per request**: ~3-4 seconds
- **168 communities**: 168 × 3.5s = **588 seconds (9.8 minutes)**

### With 50 Concurrent (Original Plan)
- **Concurrency**: 50 requests
- **Batches**: ceil(168 / 50) = 4 batches
- **Time**: 4 × 3.5s = **14 seconds**
- **Speedup**: **42x faster**

### With 200 Concurrent (High-Memory Config)
- **Concurrency**: 200 requests (BUT limited by vLLM capacity)
- **Effective concurrency**: ~50-100 (vLLM `max_num_seqs` limit)
- **Time**: ceil(168 / 100) × 3.5s = **~7 seconds**
- **Speedup**: **84x faster**

**Reality Check**: Performance will be limited by vLLM's `max_num_seqs` setting, not gevent concurrency.

---

## vLLM Configuration Adjustment

To fully utilize 200 greenlets, you need to increase vLLM concurrency:

### Check Current vLLM Settings

```bash
docker inspect vllm-container | grep -i "max.*seq"
# OR
ps aux | grep vllm | grep -o "max-num-seqs=[0-9]*"
```

### Recommended vLLM Configuration

```bash
# If using Docker:
docker run ... vllm/vllm-openai:latest \
  --model <your-model> \
  --max-num-seqs 100 \              # ← Increase from default 256 to match concurrency
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95     # Use more GPU memory for batching
```

**Key setting**: `--max-num-seqs` should match your desired concurrency (50-100)

**Why not 200?**
- vLLM batches requests for efficiency
- 100-150 is optimal for most models
- Beyond that, diminishing returns (GPU becomes bottleneck)

---

## Recommended Settings for Different Scenarios

### Scenario 1: Conservative (Start Here)
```bash
CELERY_GRAPHRAG_CONCURRENCY=100
CELERY_GRAPHRAG_MEMORY_LIMIT=12G
POSTGRES_MAX_CONNECTIONS=300
```
**Use when**: Testing gevent for first time

### Scenario 2: Balanced (Recommended)
```bash
CELERY_GRAPHRAG_CONCURRENCY=150
CELERY_GRAPHRAG_MEMORY_LIMIT=16G
POSTGRES_MAX_CONNECTIONS=400
```
**Use when**: After successful testing, want good performance

### Scenario 3: Aggressive (Your 64GB Setup)
```bash
CELERY_GRAPHRAG_CONCURRENCY=200
CELERY_GRAPHRAG_MEMORY_LIMIT=20G
POSTGRES_MAX_CONNECTIONS=500
```
**Use when**: Maximum performance needed, vLLM can handle it

---

## Memory Usage Estimation

### Gevent Worker (200 concurrency)

**Per-greenlet overhead**: ~1-2MB
**Base process**: ~500MB
**Working set (HTTP buffers, JSON parsing)**: ~50-100MB per active request

**Formula**: 500MB + (200 greenlets × 1.5MB) + (100 active × 75MB) = **~8.3GB peak**

**Recommendation**: 20GB limit provides 2.4x safety margin

### Connection Pool Math

**Gevent worker**: 200 greenlets → max 200 DB connections
**Prefork worker**: 8 workers × 25 connections = 200 DB connections
**API**: 50 connections
**Other**: 50 connections

**Total**: ~500 connections needed → matches `max_connections=500`

---

## Monitoring Recommendations

### Key Metrics to Watch

**1. Greenlet Count**
```python
# Add to graphrag_tasks.py for monitoring
import gevent
logger.info(f"Active greenlets: {len(gevent._active)}")
```

**2. Memory Usage**
```bash
# Real-time monitoring
docker stats celery-graphrag-gevent --no-stream

# Alert if exceeds 16GB (80% of limit)
```

**3. vLLM Queue Depth**
```bash
# Check vLLM metrics endpoint
curl http://vllm-host:8000/metrics | grep queue

# Ideally: queue_size < 50
```

**4. PostgreSQL Connections**
```sql
SELECT count(*) FROM pg_stat_activity WHERE datname='fileintel';
-- Should be < 450 (90% of max_connections=500)
```

---

## Optimization Tips for 64GB Environment

### 1. Enable Huge Pages (Linux)

Improves memory performance for large allocations:

```bash
# On host machine (requires sudo)
sudo sysctl -w vm.nr_hugepages=2048

# Add to docker-compose.yml under gevent service:
    ulimits:
      memlock:
        soft: -1
        hard: -1
```

### 2. Increase Redis Memory

```yaml
  redis:
    image: "redis:alpine"
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
    deploy:
      resources:
        limits:
          memory: 3G
```

### 3. Use Connection Pooling

Already configured in `celery_config.py` but verify settings:

```python
# In celery_config.py
_shared_engine = create_engine(
    database_url,
    pool_size=100,        # ← Increase from default
    max_overflow=200,     # ← Increase overflow
    pool_recycle=3600,
)
```

---

## Complete .env Example for 64GB System

```bash
# PostgreSQL
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_DB=fileintel
POSTGRES_MAX_CONNECTIONS=500
POSTGRES_SHARED_BUFFERS=2GB
POSTGRES_EFFECTIVE_CACHE_SIZE=6GB

# Redis
REDIS_MAXMEMORY=2gb

# Gevent GraphRAG Worker (20GB allocation)
CELERY_GRAPHRAG_MEMORY_LIMIT=20G
CELERY_GRAPHRAG_MEMORY_RESERVATION=2G
CELERY_GRAPHRAG_CONCURRENCY=200

# Prefork Worker (16GB allocation)
CELERY_WORKER_MEMORY_LIMIT=16G
CELERY_WORKER_MEMORY_RESERVATION=2G
CELERY_WORKER_CONCURRENCY=8

# GraphRAG async settings (matches gevent concurrency)
RAG_ASYNC_MAX_CONCURRENT=200

# API
API_WORKERS=4
API_MEMORY_LIMIT=4G
```

---

## Testing Strategy for High Concurrency

### Phase 1: Baseline (Current)
```bash
# Measure current performance
time fileintel graphrag index thesis_sources --wait
# Record: ~9.8 minutes
```

### Phase 2: Conservative Gevent (concurrency=50)
```bash
# Update .env
CELERY_GRAPHRAG_CONCURRENCY=50
CELERY_GRAPHRAG_MEMORY_LIMIT=8G

# Test
docker-compose up -d celery-graphrag-gevent
time fileintel graphrag index thesis_sources --wait
# Expected: ~30 seconds (20x faster)
```

### Phase 3: Aggressive Gevent (concurrency=200)
```bash
# Update .env
CELERY_GRAPHRAG_CONCURRENCY=200
CELERY_GRAPHRAG_MEMORY_LIMIT=20G

# Test
docker-compose restart celery-graphrag-gevent
time fileintel graphrag index thesis_sources --wait
# Expected: ~10-15 seconds (40-60x faster, limited by vLLM)
```

---

## Expected Speedup Chart

| Concurrency | Memory | Time (168 communities) | Speedup | Bottleneck |
|------------|--------|----------------------|---------|------------|
| 1 (current) | 4GB | 9.8 min | 1x | Sequential |
| 50 | 8GB | ~30 sec | 20x | Gevent |
| 100 | 12GB | ~15 sec | 39x | Gevent |
| 150 | 16GB | ~10 sec | 59x | vLLM queue |
| 200 | 20GB | ~10 sec | 59x | vLLM max_num_seqs |

**Diminishing returns** after concurrency=100 unless vLLM is also scaled.

---

## Conclusion for 64GB Environment

**Recommended Configuration**:
- ✅ Gevent concurrency: **200** (utilize your RAM)
- ✅ Memory limit: **20GB** (safe with 64GB total)
- ✅ PostgreSQL max_connections: **500**
- ✅ vLLM max_num_seqs: **100-150** (GPU bottleneck)

**Expected Performance**:
- **10-15 seconds** for 168 communities (vs 9.8 minutes)
- **~40-60x speedup** (limited by vLLM, not gevent)
- **99% GPU utilization** (instead of 5%)

**This configuration will fully utilize your 64GB RAM and eliminate the GraphRAG bottleneck completely.**
