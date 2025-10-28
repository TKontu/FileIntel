# Fork Memory Optimization

## Problem: Workers Killed During Fork

When using `prefork` pool with multiple workers, you may see:
```
Process 'ForkPoolWorker-29' pid:158 exited with 'signal 9 (SIGKILL)'
```

This happens when:
1. Worker reaches `max-tasks-per-child` limit and needs to restart
2. Celery calls `fork()` to create new worker process
3. **Fork temporarily doubles memory** (copy-on-write)
4. System runs out of memory → **OOM killer terminates process**

## Solutions Implemented

### 1. Worker Process Hooks (in celery_config.py)

Added signal handlers that run after fork:

```python
@worker_process_init.connect
def init_worker_process(sender=None, **kwargs):
    """Reset global state in child process after fork"""
    global _shared_engine, _shared_session_factory, _storage_lock

    # Reset shared storage - force recreation in this process
    _shared_engine = None
    _shared_session_factory = None
    _storage_lock = None

    # Force garbage collection
    gc.collect()
```

**What this does:**
- Runs in the **child process** after fork
- Clears inherited global state
- Forces garbage collection to free inherited objects
- Ensures each worker creates its own fresh database connections

### 2. Use `exec()` Instead of `fork()` (docker-compose.yml)

```yaml
environment:
  - CELERYD_FORCE_EXECV=true
```

**What this does:**
- Instead of `fork()` (duplicate process), uses `exec()` (create fresh process)
- **No memory duplication** during worker creation
- Slightly higher startup cost, but avoids OOM killer
- **Recommended for production** when memory is tight

### 3. Memory Allocator Tuning (docker-compose.yml)

```yaml
environment:
  - MALLOC_MMAP_THRESHOLD_=65536
  - MALLOC_TRIM_THRESHOLD_=131072
  - MALLOC_MMAP_MAX_=65536
```

**What this does:**
- Tunes glibc memory allocator for better copy-on-write behavior
- Uses mmap for larger allocations (more COW-friendly)
- Reduces heap fragmentation
- Helps reduce memory spike during fork

## Configuration Options

### Recommended Setup (Your Machine - 14GB RAM)

```bash
# .env
CELERY_WORKER_POOL=prefork
CELERY_WORKER_CONCURRENCY=4
CELERY_WORKER_MEMORY_LIMIT=8G
CELERY_WORKER_MAX_TASKS_PER_CHILD=100
CELERYD_FORCE_EXECV=true  # Key: use exec instead of fork
```

### Alternative: Threads (No Forking)

If you still have issues:

```bash
# .env
CELERY_WORKER_POOL=threads
CELERY_WORKER_CONCURRENCY=8
CELERY_WORKER_MEMORY_LIMIT=6G
# No CELERYD_FORCE_EXECV needed (threads don't fork)
```

## How It Works

### Before (OOM During Fork)

```
Parent Process: 800MB
    |
    | fork() → temporarily 1600MB (duplication)
    |
    v
Child Process: OOM KILL! (not enough memory)
```

### After (exec Creates Fresh Process)

```
Parent Process: 800MB
    |
    | exec() → creates fresh process
    |
    v
Child Process: 200MB (fresh start, no duplication)
```

## Testing

Restart your worker to apply changes:

```bash
docker compose down
docker compose up --build api celery-worker
```

Watch for worker initialization:

```bash
docker compose logs celery-worker -f | grep "Worker process"
```

You should see:
```
Worker process initializing (PID: 123)
Worker process initialized with clean state (PID: 123)
```

## Monitoring

Check if workers are still being killed:

```bash
# Look for SIGKILL
docker compose logs celery-worker | grep "signal 9"

# Monitor memory during operation
./scripts/monitor_worker_memory.sh
```

## Performance Impact

| Setting | Memory Spike | Startup Time | Memory Efficiency |
|---------|-------------|--------------|-------------------|
| `fork()` (old) | **High** (2x parent) | Fast | Poor |
| `exec()` (new) | **Low** (fresh) | Slightly slower | Excellent |
| `threads` | **None** (no fork) | Fast | Good |

## Troubleshooting

### Still seeing workers killed?

1. **Check system memory:**
   ```bash
   free -h
   ```
   If swap is heavily used (>50%), reduce `CELERY_WORKER_CONCURRENCY`

2. **Try threads instead:**
   ```bash
   CELERY_WORKER_POOL=threads
   CELERY_WORKER_CONCURRENCY=8
   ```

3. **Increase max-tasks-per-child** (fork less often):
   ```bash
   CELERY_WORKER_MAX_TASKS_PER_CHILD=200
   ```

4. **Free up system memory:**
   Check what's using memory:
   ```bash
   docker stats
   ps aux --sort=-%mem | head -20
   ```

### Workers slow to start?

This is normal with `CELERYD_FORCE_EXECV=true`:
- `exec()` creates fresh process (slower)
- But prevents OOM killer
- **Trade-off: 1-2s startup time vs. no crashes**

## Summary

The optimizations:
- ✅ Clear inherited state after fork
- ✅ Use `exec()` to avoid memory duplication
- ✅ Tune memory allocator for COW
- ✅ Add worker cleanup hooks

Expected result:
- **No more SIGKILL** during worker creation
- **Stable parallel processing** with 4 workers
- **Fast embeddings** with proper concurrency
