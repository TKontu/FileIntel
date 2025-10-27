# Fixing Slow Embedding Generation

## Problem

With default settings, the Celery worker runs in "solo" mode with concurrency=1:
- Only **1 embedding at a time** (completely sequential)
- Embeddings that could complete in 2 minutes take 20+ minutes
- No parallelism for I/O-bound API calls

## Quick Fix

Add these lines to your `.env` file:

```bash
# Use gevent for async I/O (perfect for embeddings)
CELERY_WORKER_POOL=gevent

# Run 16 concurrent embedding tasks
CELERY_WORKER_CONCURRENCY=16

# Keep memory limit appropriate
CELERY_WORKER_MEMORY_LIMIT=4G
```

Then restart the worker:
```bash
docker compose restart celery-worker
```

## Why This Works

### Pool Types Comparison

| Pool Type | Memory | Speed (Embeddings) | Speed (GraphRAG) | Best For |
|-----------|--------|-------------------|------------------|----------|
| **solo** (current) | Lowest | ⭐ (1x) | ⭐ (1x) | Very limited RAM |
| **gevent** (recommended) | Low | ⭐⭐⭐⭐⭐ (10-20x) | ⭐⭐⭐ (5-8x) | I/O-bound tasks |
| **threads** | Medium | ⭐⭐⭐⭐ (8-12x) | ⭐⭐ (3-5x) | Mixed workloads |
| **prefork** | High | ⭐⭐⭐⭐ (8-12x) | ⭐⭐⭐⭐⭐ (10-15x) | CPU-bound tasks |

### Gevent Advantages

1. **Low memory overhead** - runs in single process with green threads
2. **Perfect for I/O** - while one task waits for API, others run
3. **High concurrency** - can handle 50+ concurrent tasks
4. **No forking** - avoids memory duplication

## Concurrency Recommendations

### Conservative (4GB RAM)
```bash
CELERY_WORKER_POOL=gevent
CELERY_WORKER_CONCURRENCY=8
CELERY_WORKER_MEMORY_LIMIT=4G
```

### Balanced (8GB RAM)
```bash
CELERY_WORKER_POOL=gevent
CELERY_WORKER_CONCURRENCY=16
CELERY_WORKER_MEMORY_LIMIT=6G
```

### Aggressive (16GB+ RAM)
```bash
CELERY_WORKER_POOL=gevent
CELERY_WORKER_CONCURRENCY=32
CELERY_WORKER_MEMORY_LIMIT=8G
```

## Alternative: Prefork for Mixed Workloads

If you process a lot of GraphRAG (CPU-intensive), consider prefork:

```bash
# Prefork uses more memory but better for CPU tasks
CELERY_WORKER_POOL=prefork
CELERY_WORKER_CONCURRENCY=4  # Lower concurrency due to memory
CELERY_WORKER_MEMORY_LIMIT=8G
```

## Expected Performance Improvement

With gevent + concurrency=16:

| Task | Before (solo) | After (gevent) | Speedup |
|------|--------------|----------------|---------|
| 100 embeddings | 10-15 min | 1-2 min | **10x** |
| 1000 embeddings | 100-150 min | 10-15 min | **10x** |
| Document processing | Same | Same | 1x |
| GraphRAG building | Slow | 5-8x faster | **5-8x** |

## Monitoring Performance

After making changes, monitor:

```bash
# Watch task throughput
docker compose logs celery-worker -f | grep "succeeded in"

# Watch concurrent tasks
watch -n 2 'docker compose logs celery-worker --tail=50 | grep "received\|succeeded" | tail -20'

# Check memory usage
./scripts/monitor_worker_memory.sh
```

## Troubleshooting

### If you see "gevent not installed"

Install gevent in the container:
```bash
docker compose exec celery-worker pip install gevent
```

Or add to your `pyproject.toml`:
```toml
[tool.poetry.dependencies]
gevent = "^24.2.1"
```

### If memory usage is still high

1. Reduce concurrency:
   ```bash
   CELERY_WORKER_CONCURRENCY=8
   ```

2. Or use threads instead:
   ```bash
   CELERY_WORKER_POOL=threads
   CELERY_WORKER_CONCURRENCY=8
   ```

### If embeddings are still slow

Check your embedding server:
```bash
# Make sure it's responding quickly
curl -X POST http://192.168.0.136:9003/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "test", "model": "bge-large-en"}'
```

## API Rate Limiting

If your embedding API has rate limits:

1. **Respect the limits** - set concurrency accordingly
2. **Example**: If API allows 60 requests/min, use concurrency=10-15
3. **Monitor retry logs** - look for rate limit errors

The retry decorator will automatically back off:
- First retry: wait 4 seconds
- Second retry: wait ~6 seconds
- Third retry: wait ~10 seconds

## Recommended Configuration

For most users with embedding-heavy workloads:

```bash
# .env configuration
CELERY_WORKER_POOL=gevent
CELERY_WORKER_CONCURRENCY=16
CELERY_WORKER_MEMORY_LIMIT=6G
CELERY_WORKER_MAX_TASKS_PER_CHILD=100
```

This provides excellent throughput while keeping memory reasonable.
