# Gevent Implementation Plan for GraphRAG Performance Optimization

## Executive Summary

This document provides a complete implementation plan for adding gevent support to enable 20x faster GraphRAG indexing by allowing true concurrent HTTP requests.

**Current State**: Sequential processing (1 request at a time) in prefork workers
**Target State**: Concurrent processing (50+ requests) using gevent workers for GraphRAG only
**Expected Performance**: 9.8 minutes → ~30 seconds for community reports generation

---

## Analysis of Current Setup

### Docker Compose Configuration (`docker-compose.yml`)

**Line 86**: Current Celery worker command
```yaml
command: ["celery", "-A", "fileintel.celery_config:app", "worker", "--loglevel=info", "--pool=${CELERY_WORKER_POOL:-prefork}", "--concurrency=${CELERY_WORKER_CONCURRENCY:-4}"]
```

**Key findings**:
- ✅ Already has `CELERY_WORKER_POOL` environment variable support
- ✅ Pool type is configurable (defaults to `prefork`)
- ✅ Single worker service handles all queues
- ⚠️ Need to add separate gevent worker service

### Dependencies (`pyproject.toml`)

**Current**: No gevent dependency (lines 1-80 checked)
**Required**: Add `gevent` to dependencies

### Dockerfile

**Current**: Standard Python 3.12 slim image
**Required**: No changes needed (gevent installs via pip)

---

## Required Changes

### 1. Add Gevent Dependency

**File**: `pyproject.toml`

**Change** (add after line 76):
```toml
[tool.poetry.dependencies]
python = ">=3.10,<3.13"
# ... existing dependencies ...
transformers = "^4.30.0"
gevent = ">=24.2.1"  # ← ADD THIS LINE
```

**Why this version**:
- gevent 24.2.1 is the latest stable (as of 2024)
- Full Python 3.12 support
- Includes greenlet 3.0+ with improved performance

---

### 2. Add Dedicated Gevent Worker Service

**File**: `docker-compose.yml`

**Add new service** (after line 140, before `flower:` service):

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
      - "--concurrency=100"
      - "--queues=graphrag_indexing,graphrag_queries"
      - "--hostname=graphrag_gevent@%h"
    deploy:
      resources:
        limits:
          memory: ${CELERY_GRAPHRAG_MEMORY_LIMIT:-4G}
        reservations:
          memory: ${CELERY_GRAPHRAG_MEMORY_RESERVATION:-512M}
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
      # Gevent-specific settings
      - GEVENT_RESOLVER=ares  # Use async DNS resolver
    extra_hosts:
      - "host.docker.internal:host-gateway"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
```

---

### 3. Update Existing Celery Worker Service

**File**: `docker-compose.yml`

**Modify line 86** to exclude GraphRAG queues:

**Before**:
```yaml
command: ["celery", "-A", "fileintel.celery_config:app", "worker", "--loglevel=info", "--pool=${CELERY_WORKER_POOL:-prefork}", "--concurrency=${CELERY_WORKER_CONCURRENCY:-4}"]
```

**After**:
```yaml
command:
  - "celery"
  - "-A"
  - "fileintel.celery_config:app"
  - "worker"
  - "--loglevel=info"
  - "--pool=${CELERY_WORKER_POOL:-prefork}"
  - "--concurrency=${CELERY_WORKER_CONCURRENCY:-4}"
  - "--queues=default,document_processing,embedding_processing,llm_processing,rag_processing"
  - "--hostname=main_prefork@%h"
```

**Key changes**:
- ✅ Added explicit `--queues` to exclude `graphrag_indexing` and `graphrag_queries`
- ✅ Added `--hostname` to differentiate workers in Flower monitoring
- ✅ Keeps all existing queues except GraphRAG

---

### 4. Add Environment Variables

**File**: `.env`

**Add these lines**:
```bash
# Gevent worker configuration (for GraphRAG only)
CELERY_GRAPHRAG_MEMORY_LIMIT=4G
CELERY_GRAPHRAG_MEMORY_RESERVATION=512M
```

**Why these values**:
- **4G limit**: Gevent uses less memory than prefork (single process)
- **512M reservation**: GraphRAG mostly I/O-bound, minimal baseline memory

---

## Dependency Installation Steps

### Step 1: Add gevent to pyproject.toml

```bash
# Edit pyproject.toml manually or use:
poetry add gevent@^24.2.1
```

### Step 2: Update lock file and rebuild Docker image

```bash
# Generate new requirements.txt from poetry
poetry lock
poetry export -f requirements.txt --output requirements.txt --without-hashes

# Rebuild Docker images
docker-compose build
```

### Step 3: Verify gevent installation

```bash
# Start a test container
docker-compose run --rm api python -c "import gevent; print(f'gevent {gevent.__version__} installed')"
```

**Expected output**: `gevent 24.2.1 installed`

---

## Deployment Steps

### Phase 1: Test with Both Workers Running

```bash
# 1. Stop existing services
docker-compose down

# 2. Start all services (including new gevent worker)
docker-compose up -d

# 3. Verify both workers are running
docker-compose ps

# Expected output:
# celery-worker           running (prefork, all queues except GraphRAG)
# celery-graphrag-gevent  running (gevent, GraphRAG queues only)
```

### Phase 2: Monitor Worker Status

```bash
# Check Flower UI
open http://localhost:5555

# Check worker status via Celery
docker-compose exec celery-worker celery -A fileintel.celery_config:app inspect active_queues

# Expected:
# - main_prefork@...: ["default", "document_processing", "embedding_processing", "llm_processing", "rag_processing"]
# - graphrag_gevent@...: ["graphrag_indexing", "graphrag_queries"]
```

### Phase 3: Test GraphRAG Indexing

```bash
# Trigger GraphRAG indexing via CLI
docker-compose exec api fileintel graphrag index thesis_sources --wait

# OR via API
curl -X POST http://localhost:8001/api/v2/graphrag/index \
  -H "Content-Type: application/json" \
  -d '{"collection_id": "thesis_sources", "force_rebuild": false}'
```

### Phase 4: Verify Performance Improvement

**Monitor vLLM logs**:
```bash
# Should see 20-50 concurrent requests instead of 1
docker logs <vllm-container> -f | grep "Running:"
```

**Expected before**:
```
Running: 1 reqs, Waiting: 0 reqs
```

**Expected after**:
```
Running: 28 reqs, Waiting: 15 reqs
```

**Monitor GraphRAG logs**:
```bash
docker-compose logs -f celery-graphrag-gevent | grep "Community reports configuration"
```

**Expected output**:
```
Community reports configuration: async_mode=AsyncType.AsyncIO, concurrent_requests=50
```

---

## Potential Issues and Solutions

### Issue 1: Import Error - gevent not found

**Symptom**:
```
ImportError: No module named 'gevent'
```

**Cause**: Docker image built before adding gevent dependency

**Solution**:
```bash
# Force rebuild Docker image
docker-compose build --no-cache api celery-worker celery-graphrag-gevent

# Restart services
docker-compose up -d
```

---

### Issue 2: Both Workers Claiming Same Queue

**Symptom**: GraphRAG tasks processed by both workers

**Diagnosis**:
```bash
docker-compose exec celery-worker celery -A fileintel.celery_config:app inspect active_queues
```

**Solution**: Verify `--queues` parameter in docker-compose.yml commands

---

### Issue 3: Gevent Worker Crashes with "greenlet" Error

**Symptom**:
```
greenlet.error: cannot switch to a different thread
```

**Cause**: PostgreSQL session created in main thread, used in greenlet

**Solution**: Already handled by `get_shared_storage()` in `celery_config.py` (creates new session per task)

**Verify**:
```python
# In graphrag_tasks.py:649 (build_graphrag_index_task)
storage = get_shared_storage()
try:
    # Use storage
finally:
    storage.close()  # ← Always closes session
```

---

### Issue 4: Gevent Worker High Memory Usage

**Symptom**: Container OOM (Out of Memory) kills

**Diagnosis**:
```bash
docker stats celery-graphrag-gevent
```

**Solution**:
```bash
# Reduce concurrency
# In docker-compose.yml, change:
--concurrency=100
# To:
--concurrency=50

# OR increase memory limit in .env:
CELERY_GRAPHRAG_MEMORY_LIMIT=6G
```

---

### Issue 5: PostgreSQL "Too Many Connections"

**Symptom**:
```
psycopg2.OperationalError: FATAL: sorry, too many clients already
```

**Cause**: Gevent creates more concurrent connections than prefork

**Solution**:
```yaml
# In docker-compose.yml, postgres service (line 15):
command: ["postgres", "-c", "max_connections=300", "-c", "shared_buffers=256MB"]
# Changed from 200 to 300
```

**OR** reduce gevent concurrency:
```yaml
--concurrency=50  # Instead of 100
```

---

## Rollback Plan

### If Gevent Causes Issues

**Option A: Disable gevent worker only** (keeps other tasks working)

```bash
# Stop gevent worker
docker-compose stop celery-graphrag-gevent

# GraphRAG tasks will now queue up
# Start prefork worker with GraphRAG queues temporarily:
docker-compose exec celery-worker celery -A fileintel.celery_config:app worker \
  --pool=prefork \
  --concurrency=4 \
  --queues=graphrag_indexing,graphrag_queries \
  --hostname=graphrag_fallback@%h
```

**Option B: Remove gevent completely** (full rollback)

```bash
# 1. Stop all services
docker-compose down

# 2. Edit docker-compose.yml - remove celery-graphrag-gevent service

# 3. Revert celery-worker command to original (remove --queues parameter)

# 4. Restart services
docker-compose up -d
```

**Option C: Git revert** (if using version control)

```bash
git revert <commit-hash>
docker-compose build
docker-compose up -d
```

---

## Testing Checklist

Before deploying to production:

- [ ] gevent installed successfully (`import gevent` works)
- [ ] Both workers start without errors
- [ ] Flower shows 2 workers with correct queues
- [ ] GraphRAG task goes to gevent worker (check Flower)
- [ ] Other tasks (document processing, LLM) go to prefork worker
- [ ] vLLM shows 20-50 concurrent requests during GraphRAG indexing
- [ ] GraphRAG indexing completes successfully
- [ ] Community reports generation ~20x faster
- [ ] No PostgreSQL connection errors
- [ ] Memory usage within limits
- [ ] All existing tests pass

---

## Monitoring After Deployment

### Metrics to Watch

**1. Worker Health**
```bash
# Check worker status
docker-compose ps
celery -A fileintel.celery_config:app inspect ping

# Expected:
# graphrag_gevent@...: OK
# main_prefork@...: OK
```

**2. Queue Distribution**
```bash
# Via Flower: http://localhost:5555/workers
# Check that:
# - graphrag_indexing tasks → graphrag_gevent worker
# - Other tasks → main_prefork worker
```

**3. Performance**
```bash
# GraphRAG indexing time (should be ~20x faster)
# Before: 9.8 minutes for 168 communities
# After: ~30 seconds for 168 communities
```

**4. Resource Usage**
```bash
# Memory
docker stats --no-stream celery-graphrag-gevent celery-worker

# Expected:
# celery-graphrag-gevent: ~1-2GB (single process)
# celery-worker: ~2-4GB (4 prefork workers)
```

**5. Error Rate**
```bash
# Check logs for errors
docker-compose logs -f celery-graphrag-gevent | grep -i error

# Should be minimal/none
```

---

## Configuration Summary

### Files to Modify

1. **`pyproject.toml`** - Add gevent dependency
2. **`docker-compose.yml`** - Add gevent worker service, modify existing worker
3. **`.env`** - Add memory limits for gevent worker

### No Changes Required

- ✅ `Dockerfile` - gevent installs via pip, no system deps needed
- ✅ `celery_config.py` - already compatible (no pool-specific code)
- ✅ `graphrag_tasks.py` - already uses asyncio correctly
- ✅ Application code - no changes needed

---

## Success Criteria

**Deployment is successful if**:

1. ✅ Both workers (prefork + gevent) running without errors
2. ✅ GraphRAG tasks route to gevent worker
3. ✅ Other tasks route to prefork worker
4. ✅ vLLM shows 20-50 concurrent requests during GraphRAG
5. ✅ Community report generation ~20x faster
6. ✅ No increase in error rates
7. ✅ Memory usage within limits
8. ✅ All existing functionality works

**If ANY criterion fails**: Execute rollback plan

---

## Timeline Estimate

- **Dependency update**: 10 minutes
- **Docker Compose changes**: 15 minutes
- **Build and deploy**: 10 minutes
- **Testing**: 30 minutes
- **Monitoring**: Ongoing (first 24 hours critical)

**Total**: ~1 hour for initial deployment + 24 hour monitoring period

---

## Next Steps

1. Review this plan with team
2. Test in development/staging environment first
3. Schedule deployment window (low-traffic period recommended)
4. Have rollback plan ready
5. Monitor closely for first 24 hours

---

## Questions to Answer Before Deployment

- [ ] Do we have a staging environment to test this first?
- [ ] What is the rollback SLA if issues occur?
- [ ] Who will monitor the deployment?
- [ ] What is the acceptable error rate threshold?
- [ ] Should we start with concurrency=50 instead of 100?

---

## Conclusion

This implementation is **low-risk** with **high reward**:

- ✅ Minimal code changes (mostly config)
- ✅ Easy rollback (stop gevent worker)
- ✅ Isolated to GraphRAG only (other tasks unaffected)
- ✅ Expected 20x performance improvement
- ✅ No breaking changes

**Recommendation**: Proceed with implementation in staging environment first, then production after 24-48 hours of successful testing.
