# Celery Worker Configuration Guide

## Overview

The Celery worker pool type is now configurable to support different deployment environments with varying memory constraints.

## Configuration Options

### Environment Variables

Add these to your `.env` file:

```bash
# Worker Pool Type
CELERY_WORKER_POOL=solo  # or prefork, threads, gevent

# Worker Concurrency (only applies to prefork/threads/gevent)
CELERY_WORKER_CONCURRENCY=1

# Worker Limits
CELERY_WORKER_MAX_TASKS_PER_CHILD=50
CELERY_WORKER_MEMORY_LIMIT=4G
CELERY_WORKER_MEMORY_RESERVATION=2G
CELERY_MEMORY_OVERHEAD_GB=1.0
```

## Pool Types Explained

### 1. **solo** (Default - Low Memory)
- **Best for:** Development, limited RAM environments (8-16GB total)
- **Pros:** Minimal memory footprint, no forking overhead
- **Cons:** Single-threaded, no parallel task execution
- **Memory:** ~2-4GB per container
- **Use when:** Total system RAM < 16GB

```bash
CELERY_WORKER_POOL=solo
# Concurrency setting is ignored
```

### 2. **prefork** (Production - High Memory)
- **Best for:** Production servers with 32GB+ RAM
- **Pros:** True parallelism, scales with CPU cores
- **Cons:** High memory usage (each worker duplicates models)
- **Memory:** ~4-8GB per worker × concurrency
- **Use when:** Total system RAM ≥ 32GB

```bash
CELERY_WORKER_POOL=prefork
CELERY_WORKER_CONCURRENCY=4  # Number of worker processes
CELERY_WORKER_MEMORY_LIMIT=16G
CELERY_MEMORY_OVERHEAD_GB=2.0
```

### 3. **threads** (Medium Memory)
- **Best for:** I/O-bound tasks, medium RAM (16-32GB)
- **Pros:** Lower memory than prefork, Python GIL aware
- **Cons:** Not true parallelism for CPU-bound tasks
- **Memory:** ~4-6GB total
- **Use when:** I/O-heavy workloads

```bash
CELERY_WORKER_POOL=threads
CELERY_WORKER_CONCURRENCY=4  # Number of threads
```

### 4. **gevent** (Async - Low Memory)
- **Best for:** Async I/O, high concurrency, low memory
- **Pros:** Very efficient for I/O operations
- **Cons:** Requires greenlet-compatible code
- **Memory:** ~3-5GB total
- **Use when:** Async-compatible codebase

```bash
CELERY_WORKER_POOL=gevent
CELERY_WORKER_CONCURRENCY=100  # Number of greenlets
```

## Recommended Configurations

### Local Development (16GB RAM)
```bash
CELERY_WORKER_POOL=solo
CELERY_WORKER_MEMORY_LIMIT=4G
```

### Small Production Server (32GB RAM)
```bash
CELERY_WORKER_POOL=prefork
CELERY_WORKER_CONCURRENCY=2
CELERY_WORKER_MEMORY_LIMIT=16G
CELERY_WORKER_MEMORY_RESERVATION=8G
CELERY_MEMORY_OVERHEAD_GB=2.0
```

### Large Production Server (64GB+ RAM)
```bash
CELERY_WORKER_POOL=prefork
CELERY_WORKER_CONCURRENCY=4
CELERY_WORKER_MEMORY_LIMIT=32G
CELERY_WORKER_MEMORY_RESERVATION=16G
CELERY_MEMORY_OVERHEAD_GB=4.0
```

## Troubleshooting

### Workers being killed with SIGKILL

**Symptoms:**
```
ERROR/MainProcess] Process 'ForkPoolWorker-X' pid:XXX exited with 'signal 9 (SIGKILL)'
```

**Cause:** System OOM killer terminating workers due to memory exhaustion

**Solutions:**
1. Switch to `solo` pool:
   ```bash
   CELERY_WORKER_POOL=solo
   ```

2. Reduce concurrency:
   ```bash
   CELERY_WORKER_CONCURRENCY=1
   ```

3. Increase container memory:
   ```bash
   CELERY_WORKER_MEMORY_LIMIT=8G
   ```

### Workers hanging during startup

**Symptoms:**
```
Timed out waiting for UP message from ForkPoolWorker
```

**Solutions:**
1. Use `solo` pool for low-memory environments
2. Check if embedding models are loading in every fork
3. Increase startup timeout (not recommended)

## Memory Calculation Formula

For **prefork** pool:
```
Total Memory Needed = (Worker Memory × Concurrency) + Overhead + Main Process

Example:
- Worker Memory: 4GB
- Concurrency: 3
- Overhead: 2GB
- Main Process: 2GB
Total = (4GB × 3) + 2GB + 2GB = 16GB
```

For **solo** pool:
```
Total Memory Needed = Worker Memory + Overhead

Example:
- Worker Memory: 3GB
- Overhead: 1GB
Total = 3GB + 1GB = 4GB
```

## Migration Guide

### From prefork to solo (reduce memory)
1. Update `.env`:
   ```bash
   CELERY_WORKER_POOL=solo
   CELERY_WORKER_MEMORY_LIMIT=4G
   ```

2. Restart worker:
   ```bash
   docker compose restart celery-worker
   ```

### From solo to prefork (scale up)
1. Ensure sufficient memory (32GB+ recommended)

2. Update `.env`:
   ```bash
   CELERY_WORKER_POOL=prefork
   CELERY_WORKER_CONCURRENCY=4
   CELERY_WORKER_MEMORY_LIMIT=16G
   CELERY_MEMORY_OVERHEAD_GB=2.0
   ```

3. Restart worker:
   ```bash
   docker compose restart celery-worker
   ```

## Monitoring

Check worker memory usage:
```bash
docker stats fileintel-celery-worker-1
```

Watch worker logs:
```bash
docker logs -f fileintel-celery-worker-1
```

Check for OOM kills:
```bash
dmesg | grep -i "killed process"
```
