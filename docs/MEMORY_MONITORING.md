# Memory Monitoring Guide

This guide explains how to monitor memory consumption of Celery workers and the FileIntel system.

## Overview

FileIntel now includes comprehensive memory monitoring capabilities:

1. **Automatic Task-Level Monitoring**: Every Celery task automatically tracks memory usage
2. **Garbage Collection Management**: Automatic cleanup after tasks complete
3. **Real-Time Monitoring Scripts**: Tools to monitor workers in real-time
4. **Memory Reports**: Generate comprehensive memory usage reports

## Automatic Memory Monitoring

### Task-Level Monitoring

All Celery tasks now automatically log memory usage at key points:

- **START**: When task begins execution
- **Checkpoints**: At strategic points during execution (if configured)
- **FINISH**: When task completes (includes delta and peak memory)
- **FAILURE**: When task fails (helps identify memory-related failures)

Example log output:
```
Memory [fileintel.tasks.document_tasks.process_document] START: RSS=245.3MB (peak=245.3MB), Process%=6.1%, System=45.2%, GC objects=127,453
Memory [fileintel.tasks.document_tasks.process_document] checkpoint:after_chunking: RSS=512.7MB (peak=512.7MB), Process%=12.8%, System=47.1%, GC objects=234,567
Memory [fileintel.tasks.document_tasks.process_document] FINISH (delta=+125.4MB, peak=512.7MB): RSS=370.7MB (peak=512.7MB), Process%=9.3%, System=46.8%, GC objects=145,234
```

### Memory Metrics Explained

- **RSS (Resident Set Size)**: Actual RAM used by the process (most important)
- **Peak**: Maximum RSS observed during task execution
- **Delta**: Change in memory from start to finish
- **Process%**: Percentage of system memory used by this process
- **System%**: Total system memory utilization
- **GC objects**: Number of Python objects tracked by garbage collector

### Adding Memory Checkpoints

To add custom checkpoints in your tasks:

```python
@app.task(base=BaseFileIntelTask, bind=True)
def my_task(self, data):
    # Memory is automatically logged at START

    # Do some processing
    process_data(data)

    # Log memory at checkpoint
    self.memory_checkpoint("after_processing")

    # More processing
    heavy_computation()

    # Another checkpoint
    self.memory_checkpoint("after_computation")

    # Memory is automatically logged at FINISH
    return result
```

## Real-Time Monitoring

### Monitor Worker Memory (Live Updates)

Use the monitoring script to watch worker memory in real-time:

```bash
# Monitor with default 5-second interval
./scripts/monitor_worker_memory.sh

# Monitor with 10-second interval
./scripts/monitor_worker_memory.sh 10
```

Output shows:
- Docker container memory usage and percentage
- Process-level details inside the container
- CPU usage

### Generate Memory Report

Generate a comprehensive snapshot of system memory:

```bash
# Text format (human-readable)
python scripts/memory_report.py

# JSON format (for parsing/logging)
python scripts/memory_report.py --format json
```

The report includes:
- Docker container memory for all services
- PostgreSQL database size and largest tables
- Redis memory usage
- System memory statistics

## Docker-Based Monitoring

### View Container Stats

```bash
# Real-time stats for all containers
docker stats

# Stats for just the Celery worker
docker stats fileintel-celery-worker-1

# One-time snapshot
docker stats --no-stream
```

### Check Worker Container Memory

```bash
# View memory configuration
docker inspect fileintel-celery-worker-1 | grep -A 10 Memory

# Check actual memory usage
docker exec fileintel-celery-worker-1 ps aux

# View Python process memory specifically
docker exec fileintel-celery-worker-1 ps aux | grep celery
```

## Memory Limits and Configuration

### Current Configuration

The Celery worker memory limits are configured in `docker-compose.yml`:

```yaml
celery-worker:
  deploy:
    resources:
      limits:
        memory: ${CELERY_WORKER_MEMORY_LIMIT:-4G}  # Hard limit
      reservations:
        memory: ${CELERY_WORKER_MEMORY_RESERVATION:-1G}  # Soft guarantee
```

### Adjusting Memory Limits

Set in `.env` file:

```bash
# Maximum memory (hard limit - container killed if exceeded)
CELERY_WORKER_MEMORY_LIMIT=4G

# Minimum memory (soft reservation for scheduling)
CELERY_WORKER_MEMORY_RESERVATION=1G
```

## Garbage Collection

### Automatic GC

The memory monitor automatically performs garbage collection after each task completes:

- Collects all 3 Python GC generations (gen0, gen1, gen2)
- Logs amount of memory freed (if significant)
- Helps prevent memory leaks from accumulating

### Manual GC

To force garbage collection in a task:

```python
from fileintel.utils.memory_monitor import MemoryMonitor

monitor = MemoryMonitor(task_name="my_task")
gc_stats = monitor.force_garbage_collection()
logger.info(f"Freed {gc_stats['freed_mb']}MB")
```

## Troubleshooting High Memory Usage

### Warning Signs

The system will log warnings when:

- Task uses >1000MB peak memory
- Process uses >80% of available memory
- Memory threshold exceeded (configurable)

### Investigation Steps

1. **Check which tasks are using memory:**
   ```bash
   docker compose logs celery-worker | grep "Memory.*FINISH" | tail -20
   ```

2. **Look for high delta values** (memory not being freed):
   ```bash
   docker compose logs celery-worker | grep "delta=+" | grep -v "delta=+[0-9]\." | tail -20
   ```

3. **Monitor peak memory over time:**
   ```bash
   docker compose logs celery-worker | grep "peak=" | tail -50
   ```

4. **Check for memory leaks** (gradual RSS increase):
   ```bash
   ./scripts/monitor_worker_memory.sh 30  # 30-second intervals
   ```

### Common Issues and Solutions

#### Issue: Memory grows continuously

**Solution**: Check for circular references or cached data not being cleared

```python
# Add explicit cleanup
def cleanup_large_objects(self):
    self.memory_checkpoint("before_cleanup")

    # Clear large data structures
    self.large_cache.clear()
    del self.temporary_data

    # Force GC
    import gc
    gc.collect()

    self.memory_checkpoint("after_cleanup")
```

#### Issue: Tasks killed with OOM (Out of Memory)

**Symptoms**: Container logs show "Killed" or exit code 137

**Solutions**:
1. Increase memory limit in `.env`
2. Reduce concurrency: `CELERY_WORKER_CONCURRENCY=1`
3. Enable memory optimization in task code
4. Process fewer documents per batch

#### Issue: High baseline memory usage

**Solution**: Check for large loaded models or persistent caches

```bash
# View baseline memory before any tasks run
docker compose restart celery-worker
sleep 10
docker stats --no-stream fileintel-celery-worker-1
```

## Monitoring Best Practices

### Regular Monitoring

Set up periodic monitoring:

```bash
# Add to cron for hourly memory reports
0 * * * * cd /path/to/fileintel && python scripts/memory_report.py >> logs/memory_hourly.log 2>&1
```

### Alert Thresholds

Configure alerts when memory exceeds thresholds:

```python
from fileintel.utils.memory_monitor import check_memory_threshold

if not check_memory_threshold(threshold_mb=3000):
    # Send alert (email, Slack, etc.)
    logger.critical("Memory threshold exceeded!")
```

### Production Monitoring

For production environments, consider:

1. **External Monitoring**: Prometheus + Grafana
2. **Container Orchestration**: Kubernetes memory limits and monitoring
3. **Log Aggregation**: ELK stack or similar to track memory patterns
4. **Automated Restarts**: Use Celery's `--max-tasks-per-child` to prevent leaks

```yaml
# In docker-compose.yml
command: ["celery", "-A", "fileintel.celery_config:app", "worker",
          "--loglevel=info",
          "--max-tasks-per-child=50"]  # Restart worker after 50 tasks
```

## Advanced: Memory Profiling

For deep investigation of memory issues:

### Using memory_profiler

```bash
# Install memory_profiler
pip install memory-profiler

# Profile a specific function
python -m memory_profiler src/fileintel/tasks/document_tasks.py
```

### Using pympler

```python
from pympler import asizeof, tracker

# Track memory allocation
tr = tracker.SummaryTracker()

# Your code here

# Print summary
tr.print_diff()
```

### Docker Memory Profiling

```bash
# Enable memory profiling in Docker
docker run --memory-swap-limit=4g --memory=4g --oom-kill-disable=false ...
```

## Reference: Memory Monitoring API

### MemoryMonitor Class

```python
from fileintel.utils.memory_monitor import MemoryMonitor

# Create monitor
monitor = MemoryMonitor(task_name="my_task", enable_gc=True)

# Start monitoring
monitor.start_monitoring()

# Log at checkpoint
monitor.checkpoint("processing_phase")

# Get detailed info
mem_info = monitor.get_memory_info()

# Finish and get summary
summary = monitor.finish_monitoring()
```

### Utility Functions

```python
from fileintel.utils.memory_monitor import (
    log_worker_memory_summary,  # Log overall worker memory
    check_memory_threshold,     # Check if memory exceeds limit
    get_system_memory_summary,  # Get system-wide memory info
)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CELERY_WORKER_MEMORY_LIMIT` | 4G | Hard memory limit for worker container |
| `CELERY_WORKER_MEMORY_RESERVATION` | 1G | Soft memory reservation |
| `CELERY_WORKER_CONCURRENCY` | 1 | Number of concurrent workers (affects total memory) |
| `CELERY_WORKER_MAX_TASKS_PER_CHILD` | 50 | Restart worker after N tasks (prevents leaks) |

## Monitoring Checklist

- [ ] Monitor baseline memory after worker starts
- [ ] Check memory after batch processing
- [ ] Review logs for high peak memory tasks
- [ ] Monitor memory delta to detect leaks
- [ ] Set up periodic memory reports
- [ ] Configure appropriate memory limits
- [ ] Enable garbage collection logging
- [ ] Test with realistic workloads
- [ ] Document memory requirements per document type
- [ ] Set up alerts for abnormal memory usage
