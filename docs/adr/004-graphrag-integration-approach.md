# ADR-004: GraphRAG Integration as Black-Box Dependency

## Status
Accepted

## Context

FileIntel needed to integrate Microsoft's GraphRAG library for advanced graph-based retrieval-augmented generation capabilities. The challenge was determining the appropriate integration approach that would:

1. **Leverage GraphRAG Capabilities**: Utilize the sophisticated graph-based knowledge representation and querying
2. **Maintain Architectural Boundaries**: Keep GraphRAG as an external dependency without tight coupling
3. **Avoid Code Bloat**: Prevent accumulation of unused GraphRAG wrapper code
4. **Enable Performance**: Provide efficient graph operations while maintaining system responsiveness
5. **Support Scalability**: Allow GraphRAG operations to scale with the distributed task architecture

Initial integration attempts resulted in:
- Complex custom wrapper classes that duplicated GraphRAG functionality
- Unused abstraction layers that provided no additional value
- Memory-intensive operations that didn't integrate well with the task system
- Dead code accumulation (834 lines removed during audit)

## Decision

We decided to treat GraphRAG as a black-box external dependency with minimal integration wrappers, focusing on essential operations only.

### Integration Principles:

1. **Black-Box Approach**: Treat GraphRAG as an external library, not as part of FileIntel's core architecture
2. **Minimal Wrappers**: Create only essential adapter classes that provide genuine value
3. **Essential Operations Only**: Support only operations actually used by FileIntel
4. **Clean Boundaries**: Clear separation between FileIntel and GraphRAG code
5. **Performance Integration**: Integrate with Celery task system for distributed processing

### Key Components:

1. **GraphRAGService** (`graphrag_service.py`):
   - **Purpose**: Primary interface for GraphRAG operations
   - **Scope**: Index building, global search, local search, status checking
   - **Justification**: Provides FileIntel-specific configuration and error handling

2. **GraphRAGConfigAdapter** (`config_adapter.py`):
   - **Purpose**: Convert FileIntel settings to GraphRAG configuration format
   - **Scope**: Model configuration, API settings, workspace management
   - **Justification**: Bridges FileIntel's configuration system with GraphRAG requirements

3. **GraphRAGDataAdapter** (`data_adapter.py`):
   - **Purpose**: Convert between FileIntel and GraphRAG data formats
   - **Scope**: Document format conversion, response formatting
   - **Justification**: Handles data transformation without business logic

4. **DataFrameCache and ParquetLoader**:
   - **Purpose**: Performance optimization for large GraphRAG datasets
   - **Scope**: Caching and loading of GraphRAG parquet files
   - **Justification**: Essential for performance with large graph datasets

### Eliminated Components:

1. **DirectOpenAIBypass**: 67 lines of unused API bypass code
2. **DirectGraphRAGEmbeddingProvider**: 101 lines of unused embedding provider
3. **GraphRAGBatchProcessor**: 274 lines of complex async batching never used
4. **VRAMMonitor**: 392 lines of GPU monitoring complexity never used

## Consequences

### Positive:
1. **Clean Architecture**: Clear separation between FileIntel and GraphRAG concerns
2. **Reduced Complexity**: Eliminated 834 lines of dead code and unnecessary abstractions
3. **Better Maintainability**: Only maintain code that serves actual business needs
4. **Performance Focus**: Optimization efforts focused on actually used components
5. **Easier Updates**: GraphRAG updates don't require extensive wrapper modifications
6. **Clear Dependencies**: Explicit dependency on GraphRAG library versions

### Negative:
1. **Limited Customization**: Less ability to customize GraphRAG behavior
2. **GraphRAG Dependency**: Tightly coupled to GraphRAG library evolution
3. **Error Handling**: Limited control over GraphRAG internal error conditions

### Neutral:
1. **Configuration Mapping**: Need to maintain mapping between FileIntel and GraphRAG configs
2. **Data Format Handling**: Ongoing need to handle data format conversions
3. **Performance Tuning**: GraphRAG performance depends on library implementation

## Technical Implementation

### Service Layer Pattern:
```python
class GraphRAGService:
    """Essential GraphRAG operations only"""

    def build_index(self, documents: List[DocumentChunk], collection_id: str):
        # FileIntel-specific setup and error handling
        # Direct GraphRAG library calls

    def global_search(self, query: str, collection_id: str):
        # FileIntel data preparation
        # GraphRAG library search
        # FileIntel response formatting
```

### Configuration Adapter:
```python
class GraphRAGConfigAdapter:
    """Convert FileIntel settings to GraphRAG format"""

    def adapt_config(self, settings: Settings, collection_id: str) -> GraphRagConfig:
        # Extract hardcoded values to module constants
        # Map FileIntel config to GraphRAG config structure
        # Handle workspace and API configuration
```

### Celery Integration:
```python
@app.task(base=BaseFileIntelTask)
def build_graph_index(documents: List[dict], collection_id: str):
    """Celery task for GraphRAG index building"""
    service = GraphRAGService(storage, settings)
    return service.build_index(documents, collection_id)
```

## Performance Considerations

### Caching Strategy:
- **DataFrameCache**: Cache loaded GraphRAG parquet files to avoid repeated I/O
- **Configuration Cache**: Cache GraphRAG configurations per collection
- **Memory Management**: Handle large datasets efficiently in distributed environment

### Task Distribution:
- **Index Building**: Long-running task in `memory_intensive` queue
- **Search Operations**: Moderate duration tasks in `graphrag` queue
- **Result Caching**: Cache search results for frequently asked questions

## Code Quality Standards Applied

### SOLID Principles:
1. **Single Responsibility**: Each adapter has one clear purpose
2. **Open/Closed**: Can extend GraphRAG integration without modifying existing code
3. **Interface Segregation**: Separate interfaces for different GraphRAG concerns
4. **Dependency Inversion**: Depend on GraphRAG abstractions, not implementations

### Code Bloat Elimination:
1. **Dead Code Removal**: Eliminated 834 lines of unused functionality
2. **Magic Number Extraction**: Hardcoded values moved to module constants
3. **Unused Import Cleanup**: Removed imports for deleted functionality
4. **Wrapper Elimination**: Removed unnecessary abstraction layers

### Architecture Improvements:
1. **Logging Separation**: Moved logging imports to module level
2. **Error Handling**: Simplified error handling patterns for different contexts
3. **Configuration Management**: Centralized GraphRAG configuration logic
4. **Progress Tracking**: Integrated with Celery progress reporting

## Guidelines for Future Development

### Do's:
1. **Question New Wrappers**: Ensure any new GraphRAG wrapper provides genuine value
2. **Use Direct API**: Call GraphRAG library directly when wrappers don't add value
3. **Focus on Performance**: Optimize for actual usage patterns
4. **Maintain Boundaries**: Keep FileIntel and GraphRAG concerns separate

### Don'ts:
1. **Don't Create Unnecessary Abstractions**: Avoid wrapper-around-wrapper patterns
2. **Don't Duplicate GraphRAG Logic**: Let GraphRAG handle its own concerns
3. **Don't Optimize Prematurely**: Only optimize based on actual performance issues
4. **Don't Cache Everything**: Cache only expensive operations with clear benefit

## Alternatives Considered

1. **Full GraphRAG Fork**: Fork and modify GraphRAG library directly
   - Rejected: Would require maintaining a parallel codebase

2. **Heavy Wrapper Layer**: Create comprehensive FileIntel-specific GraphRAG API
   - Rejected: Would duplicate GraphRAG functionality unnecessarily

3. **Plugin Architecture**: Create extensible plugin system for GraphRAG
   - Rejected: Over-engineering for current requirements

4. **No GraphRAG Integration**: Use only vector-based RAG
   - Rejected: GraphRAG provides valuable graph-based insights

## Success Metrics

### Code Quality:
- ✅ **834 lines of dead code removed**
- ✅ **11 critical issues resolved**
- ✅ **Clean architectural boundaries established**
- ✅ **SOLID principles compliance**

### Performance:
- ✅ **Efficient caching of large datasets**
- ✅ **Distributed processing via Celery**
- ✅ **Memory optimization for graph operations**

### Maintainability:
- ✅ **Clear separation of concerns**
- ✅ **Minimal integration surface area**
- ✅ **Easy to update GraphRAG library versions**

## References

- [GraphRAG Integration Audit Results](../../src/fileintel/graphrag-integration-todo.md)
- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
- [Microsoft GraphRAG Documentation](https://github.com/microsoft/graphrag)
- [Celery Task Architecture](../CELERY_ARCHITECTURE.md)
