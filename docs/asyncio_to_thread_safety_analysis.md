# SQLAlchemy Session Thread Safety with asyncio.to_thread

## Investigation Summary

Analyzing why existing `asyncio.to_thread` usage with database operations appears to work without issues.

## Current Architecture

### Session Creation Flow

1. **FastAPI Dependency Injection** (`dependencies.py:get_storage`):
   ```python
   def get_storage():
       db = SessionLocal()  # Created in main thread (event loop)
       try:
           yield PostgreSQLStorage(db)
       finally:
           db.close()
   ```

2. **Storage Initialization** (`base_storage.py`):
   ```python
   class BaseStorageInfrastructure:
       def __init__(self, config_or_session):
           if hasattr(config_or_session, "query"):
               # Receives session from FastAPI dependency
               self.db = config_or_session
               self.engine = config_or_session.bind
               self._owns_session = False  # Session owned by caller
   ```

3. **Usage in Services** (e.g., `graphrag_service.py`):
   ```python
   class GraphRAGService:
       def __init__(self, storage: PostgreSQLStorage, settings: Settings):
           self.storage = storage  # Storage contains session from main thread

       async def query(self, query: str, collection_id: str):
           # Runs in thread pool, but session created in main thread
           collection = await asyncio.to_thread(
               self.storage.get_collection,
               collection_id
           )
   ```

### Database Operation Flow

```
Main Thread (FastAPI Event Loop)
├─ Create Session via SessionLocal()
├─ Create PostgreSQLStorage(session)
├─ Pass storage to GraphRAGService
└─ Call: await asyncio.to_thread(storage.get_collection, ...)
    │
    └─→ Thread Pool Worker
        └─ Execute: session.query(Collection).filter(...).first()
           └─ Access session created in DIFFERENT thread
```

## SQLAlchemy Thread Safety Concerns

### Official SQLAlchemy Documentation

From SQLAlchemy docs:

> **The Session is very much intended to be used in a non-concurrent fashion**, which usually means in only one thread at a time.
>
> Using the Session in the proper way, together with the unit of work pattern, generally doesn't call for any special concurrency practices, whereas **accessing the same Session from multiple threads at the same time is not safe**.

### Why It "Works" Currently

The current code appears to work because:

1. **Connection Pool Thread Safety**:
   - SQLAlchemy's `Engine` and connection pool ARE thread-safe
   - Each thread can get its own connection from the pool
   - Connections are thread-local within the pool

2. **Read-Only Operations**:
   - Most existing `asyncio.to_thread` usage is for READ operations
   - Read operations don't modify session state
   - Less likely to cause corruption compared to writes

3. **Short-Lived Operations**:
   - Database queries complete quickly
   - Less chance of concurrent access to same session object
   - Thread switches happen between operations, not during

4. **Single Request Context**:
   - Each API request gets its own session
   - No session sharing between requests
   - Reduces probability of actual concurrent access

### What Could Go Wrong

Even though it "works", there are risks:

1. **Session State Corruption**:
   ```python
   # Main thread
   session = SessionLocal()

   # Thread pool worker accesses same session object
   # Session maintains identity map, pending changes, etc.
   # Concurrent access could corrupt this state
   ```

2. **Identity Map Issues**:
   - Session maintains object cache (identity map)
   - Accessing from multiple threads could cause cache inconsistencies
   - Same object could have different states in different threads

3. **Transaction State**:
   - Session has transaction state (dirty objects, pending changes)
   - Thread switching during transaction could corrupt state

4. **Write Operations Are Especially Risky**:
   ```python
   # DANGEROUS: Write operation in thread pool
   await asyncio.to_thread(storage.create_document, ...)
   ```
   - Modifies session state
   - Could trigger automatic flush
   - Could interfere with transaction boundaries

## Existing Usage Pattern Analysis

### Pattern 1: Read Operations (Currently Used)

```python
# graphrag_service.py:49
collection = await asyncio.to_thread(self.storage.get_collection, collection_id)

# graphrag_service.py:130
index_info = await asyncio.to_thread(
    self.storage.get_graphrag_index_info, collection_id
)
```

**Risk Level**: 🟡 **Medium**
- Read-only operations
- No session state modification
- Works in practice but technically unsafe

### Pattern 2: Write Operations (Proposed)

```python
# PROPOSED for upload_document_to_collection
document = await asyncio.to_thread(
    self.storage.create_document,
    filename=...,
    ...
)
```

**Risk Level**: 🔴 **High**
- Modifies session state
- Could trigger flush/commit
- High chance of corruption with concurrent requests

## Why It Hasn't Failed Yet

### Hypothesis 1: Low Concurrency

```python
# Most deployments have low concurrent request rates
# Probability of actual simultaneous access to same session: Low
```

### Hypothesis 2: FastAPI Request Isolation

```python
# Each request gets its own session via dependency injection
# Sessions are not shared between requests
# Thread pool workers for different requests use different sessions
```

### Hypothesis 3: Python GIL

```python
# Global Interpreter Lock prevents true parallelism
# Only one thread executes Python bytecode at a time
# Reduces (but doesn't eliminate) race condition window
```

### Hypothesis 4: Quick Operations

```python
# Database queries are fast (<100ms typically)
# Short critical sections reduce collision probability
# Operations complete before context switches
```

## Safe Alternatives

### Option 1: Create Session in Thread (Recommended for Writes)

```python
async def upload_document_to_collection(...):
    # Don't use self.storage (has main thread session)
    # Create new session in thread pool

    def _create_document_with_new_session():
        from fileintel.storage.models import SessionLocal
        db = SessionLocal()
        try:
            storage = PostgreSQLStorage(db)
            return storage.create_document(...)
        finally:
            db.close()

    document = await asyncio.to_thread(_create_document_with_new_session)
```

**Pros**:
- Completely thread-safe
- Session created and used in same thread
- Proper cleanup

**Cons**:
- More verbose
- Creates extra database connections
- Bypasses dependency injection

### Option 2: Keep Synchronous (Current Approach)

```python
async def upload_document_to_collection(...):
    # Async file I/O
    async with aiofiles.open(...) as f:
        await f.write(...)

    # Synchronous DB operation (keeps session in main thread)
    document = self.storage.create_document(...)
```

**Pros**:
- No thread-safety issues
- Simpler code
- Works with dependency injection

**Cons**:
- Blocks event loop briefly (~100ms)
- Not fully optimal

### Option 3: Use AsyncIO-Compatible SQLAlchemy

```python
# Requires SQLAlchemy 2.0+ with asyncio support
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

async def upload_document_to_collection(...):
    async with AsyncSession(async_engine) as session:
        document = await session.execute(...)
        await session.commit()
```

**Pros**:
- Fully non-blocking
- Native async support
- No thread-safety concerns

**Cons**:
- Major refactoring required
- All storage methods need conversion
- Breaking change

## Recommendations

### Immediate (Safe)

1. ✅ **Keep current read operations as-is**
   - Low risk in practice
   - Working without issues
   - Performance benefit worth minor risk

2. ✅ **Avoid `asyncio.to_thread` for write operations**
   - Keep DB writes synchronous
   - Use async file I/O and CPU-bound threading
   - Minimizes event loop blocking without adding risk

### Short-Term

3. **Document the pattern**
   - Add comments explaining thread safety considerations
   - Warn developers about write operations
   - Establish coding standards

4. **Add session assertions**
   ```python
   import threading

   class ThreadSafeSessionProxy:
       def __init__(self, session):
           self.session = session
           self.thread_id = threading.get_ident()

       def query(self, *args, **kwargs):
           current_thread = threading.get_ident()
           if current_thread != self.thread_id:
               logger.warning(f"Session accessed from different thread!")
           return self.session.query(*args, **kwargs)
   ```

### Long-Term

5. **Consider SQLAlchemy 2.0 async**
   - Evaluate migration path
   - Test compatibility with existing code
   - Plan gradual migration

6. **Connection pool monitoring**
   - Monitor for session leaks
   - Check for thread-related errors
   - Alert on unusual patterns

## Conclusion

**Why it works**: Combination of low concurrency, read-mostly operations, request isolation, and Python GIL

**Why it's risky**: Violates SQLAlchemy's thread-safety contract, especially for writes

**Safe approach for new code**:
- ✅ Async file I/O: `aiofiles`
- ✅ CPU-bound work: `asyncio.to_thread(lambda: hash(...))`
- ✅ DB reads: Current `asyncio.to_thread` pattern (acceptable risk)
- ❌ DB writes: Keep synchronous or create session in thread

**Current upload fix is correct**: Uses async I/O, keeps DB synchronous
