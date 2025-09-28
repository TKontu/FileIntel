# Storage Layer Architecture

This document explains the storage layer architecture decisions in FileIntel, including why certain abstractions exist, which implementations to use, and the design rationale.

## Current Architecture

### Storage Implementations

FileIntel currently has the following storage implementations:

1. **PostgreSQLStorage** (`postgresql_storage.py`)
   - **Primary storage implementation**
   - Handles all database operations for documents, collections, chunks, and metadata
   - Implements both `DocumentStorageInterface` and `GraphRAGStorageInterface`
   - Uses SQLAlchemy ORM for database operations
   - Supports vector similarity search via pgvector extension

2. **SimpleCache** (`simple_cache.py`)
   - **Auxiliary caching layer**
   - Provides Redis-based caching for LLM responses and embeddings
   - Replaces the over-engineered cache_manager.py with focused functionality
   - Used primarily for performance optimization

### Interface Design

The storage layer uses focused interfaces:

1. **DocumentStorageInterface** (`base.py`)
   - **Purpose**: Defines core document and collection operations
   - **Justification**: Separates document operations from GraphRAG-specific operations
   - **Methods**: Collection CRUD, document CRUD, chunk retrieval

2. **GraphRAGStorageInterface** (`base.py`)
   - **Purpose**: Defines GraphRAG-specific storage operations
   - **Justification**: GraphRAG has unique storage requirements (entities, communities, relationships)
   - **Methods**: GraphRAG index metadata, entities, communities, relationships

## Architectural Decisions

### Decision 1: Keep Focused Interfaces

**Decision**: Maintain separate `DocumentStorageInterface` and `GraphRAGStorageInterface` instead of a single monolithic storage interface.

**Rationale**:
- **Separation of Concerns**: Document operations and GraphRAG operations serve different purposes
- **Interface Segregation Principle**: Components only depend on methods they actually use
- **Modularity**: GraphRAG functionality can be disabled without affecting core document operations
- **Clarity**: Clear distinction between core storage and specialized graph operations

**Usage Patterns**:
```python
# Core document operations
storage: DocumentStorageInterface = get_storage()
collection = storage.create_collection("My Collection")
document = storage.create_document(filename, hash, size, mime_type, collection_id)

# GraphRAG operations
graphrag_storage: GraphRAGStorageInterface = get_storage()  # Same instance, different interface
graphrag_storage.save_graphrag_index_info(collection_id, index_path)
```

### Decision 2: Single Implementation Strategy

**Decision**: Use PostgreSQLStorage as the single primary implementation for both interfaces.

**Rationale**:
- **Simplicity**: No need for complex abstraction when only one implementation exists
- **PostgreSQL Capabilities**: Handles both relational data and vector operations efficiently
- **Transactional Consistency**: Single database ensures data consistency across operations
- **Performance**: Direct access eliminates abstraction overhead

**Anti-Pattern Avoided**: Creating multiple storage implementations "for flexibility" when no alternative implementations are needed.

### Decision 3: Simplified Caching

**Decision**: Replace complex cache management with SimpleCache focused on essential operations.

**Rationale**:
- **Code Bloat Elimination**: Removed unnecessary abstractions and complexity
- **Single Responsibility**: Cache only handles caching, not business logic
- **Performance Focus**: Targets specific high-value caching scenarios (LLM responses, embeddings)
- **Maintainability**: Simple implementation easier to debug and maintain

**Before (cache_manager.py)**:
- Complex cache hierarchy and abstractions
- Multiple cache types and strategies
- Over-engineered for actual use cases

**After (simple_cache.py)**:
- Direct Redis operations with error handling
- Focused on actual caching needs
- Clear TTL management for different data types

## Usage Guidelines

### When to Use Each Component

1. **PostgreSQLStorage**
   - ✅ **Use for**: All persistent data operations
   - ✅ **Use for**: Document and collection management
   - ✅ **Use for**: GraphRAG metadata and entities
   - ✅ **Use for**: Vector similarity searches
   - ✅ **Use for**: Transactional operations

2. **SimpleCache**
   - ✅ **Use for**: Caching LLM responses to reduce API costs
   - ✅ **Use for**: Caching computed embeddings for reuse
   - ✅ **Use for**: Temporary data that can be regenerated
   - ❌ **Don't use for**: Persistent data storage
   - ❌ **Don't use for**: Critical data that cannot be lost

### Code Patterns

#### Correct Dependency Injection
```python
# In API endpoints
def some_endpoint(storage: PostgreSQLStorage = Depends(get_storage)):
    # Direct use of concrete type - no unnecessary abstraction
    collection = storage.create_collection(name)
    return collection

# In services that need caching
class SomeService:
    def __init__(self, storage: PostgreSQLStorage):
        self.storage = storage
        self.cache = get_cache()  # Simple, direct access
```

#### Avoid Over-Abstraction
```python
# ❌ Don't create unnecessary storage service layers
class StorageService:
    def __init__(self, storage: DocumentStorageInterface):
        self.storage = storage

    def create_collection(self, name: str):
        return self.storage.create_collection(name)  # Just delegates

# ✅ Use storage directly
def create_collection(name: str, storage: PostgreSQLStorage = Depends(get_storage)):
    return storage.create_collection(name)
```

## Implementation Details

### PostgreSQL Storage Features

1. **Vector Operations**
   - Uses pgvector extension for similarity search
   - Efficient storage and querying of document embeddings
   - Supports various distance metrics

2. **GraphRAG Integration**
   - Stores GraphRAG index metadata
   - Manages entities, communities, and relationships
   - Provides efficient retrieval for graph operations

3. **Transaction Management**
   - Proper session management in dependencies.py
   - Ensures data consistency across operations
   - Handles connection pooling and cleanup

### Caching Strategy

1. **LLM Response Caching**
   - TTL: 30 minutes for query results
   - Key pattern: `llm_response:{hash_of_input}`
   - Reduces API costs and improves response times

2. **Embedding Caching**
   - TTL: 24 hours for computed embeddings
   - Key pattern: `embedding:{content_hash}`
   - Avoids recomputing expensive embeddings

3. **Cache Invalidation**
   - Pattern-based clearing for related data
   - Automatic expiration via TTL
   - Manual invalidation when data changes

## Migration History

### From Job Management to Celery

The storage layer was simplified during the Celery migration:

**Removed Components**:
- `cache_manager.py`: Over-engineered caching abstractions
- `redis_storage.py`: Unnecessary Redis storage layer
- Complex storage service wrappers

**Simplified Components**:
- Direct PostgreSQL storage usage
- Focused caching with SimpleCache
- Eliminated unnecessary abstraction layers

### Interface Consolidation

**Previous State**: Single monolithic storage interface with many methods

**Current State**: Focused interfaces with clear separation of concerns

**Benefits**:
- Clearer dependencies
- Better testability
- Easier to understand and maintain
- Follows Interface Segregation Principle

## Best Practices

### Do's
1. **Use concrete types** (PostgreSQLStorage) in dependency injection
2. **Cache expensive operations** (LLM calls, embedding generation)
3. **Handle cache failures gracefully** (fallback to computation)
4. **Use appropriate TTLs** for different data types
5. **Leverage PostgreSQL features** (vector operations, transactions)

### Don'ts
1. **Don't create storage service wrappers** unless they add real value
2. **Don't abstract for theoretical flexibility** - YAGNI principle
3. **Don't cache persistent data** that should be in the database
4. **Don't create multiple storage implementations** without clear requirements
5. **Don't mix business logic** with storage operations

## Performance Considerations

### Database Performance
- Use connection pooling (handled by SQLAlchemy)
- Optimize vector similarity queries with appropriate indexes
- Batch operations when processing multiple documents
- Use transactions appropriately

### Cache Performance
- Monitor cache hit rates
- Adjust TTLs based on usage patterns
- Use pattern-based invalidation sparingly
- Consider memory usage of cached data

## Testing Strategy

### Unit Testing
- Mock storage dependencies with specific interfaces
- Test cache behavior with Redis test instances
- Use in-memory databases for isolated tests

### Integration Testing
- Test actual PostgreSQL operations
- Verify vector similarity search functionality
- Test cache integration with real Redis

This architecture provides a clean, maintainable storage layer that avoids over-engineering while supporting the actual needs of the FileIntel application.
