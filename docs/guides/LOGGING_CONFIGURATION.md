# Logging Configuration Guide

## Overview

The logging system now supports configurable per-component log levels, allowing you to see progress updates for long-running operations while keeping verbose debug logs hidden.

## Configuration Files

### 1. **config/default.yaml**

```yaml
logging:
  level: WARNING  # Root logger level
  # Component-specific log levels (overrides)
  component_levels:
    # Application components (show progress)
    fileintel.rag.graph_rag.services: INFO
    fileintel.tasks.graphrag_tasks: INFO
    fileintel.document_processing: INFO
    # GraphRAG library (show workflow progress)
    graphrag.index.workflows: INFO
    # Keep these quiet (too verbose at INFO)
    graphrag.index.operations: WARNING
    graphrag.index.text_splitting: WARNING
    graphrag.query: WARNING
```

## Log Levels Explained

### WARNING (Default Root Level)
- Only warnings, errors, and critical messages
- Minimal output, suitable for production
- **Does not show progress updates**

### INFO (Component-specific)
- Shows important progress milestones
- Periodic updates for long operations (e.g., every 5% of 40k chunks)
- Example: `chunker progress: 2000/40000 (5.0%)`
- **Recommended for monitoring long-running jobs**

### DEBUG
- Shows every single operation
- Very verbose, only for troubleshooting
- Example: logs every single chunk being processed
- **Not recommended for production**

## Usage Examples

### Production (Minimal Logging)
```yaml
logging:
  level: WARNING
  component_levels: {}  # Empty, all inherit WARNING
```

### Development (See Progress)
```yaml
logging:
  level: WARNING  # Keep root quiet
  component_levels:
    fileintel.rag.graph_rag.services: INFO  # See GraphRAG progress
    fileintel.tasks.graphrag_tasks: INFO    # See task progress
    graphrag.index.workflows: INFO          # See workflow steps
```

### Full Debugging
```yaml
logging:
  level: DEBUG  # Everything in detail
  component_levels: {}  # Optional overrides
```

## Progress Update Behavior

### Chunking Progress (40k documents example)
- **INFO level**: Updates every 2000 docs (5%)
  ```
  [INFO] Starting chunking process for 40000 documents
  [INFO] chunker progress: 2000/40000 (5.0%)
  [INFO] chunker progress: 4000/40000 (10.0%)
  ...
  [INFO] chunker progress: 40000/40000 (100.0%)
  ```

- **DEBUG level**: Updates for every document
  ```
  [DEBUG] chunker progress: 1/40000
  [DEBUG] chunker progress: 2/40000
  ...
  [DEBUG] chunker progress: 40000/40000
  ```

### GraphRAG Index Build
- **INFO**: Major milestones only
  ```
  [INFO] Starting GraphRAG index build for collection abc123
  [INFO] Processing 40000 document chunks
  [INFO] GraphRAG index build completed in 3600.5 seconds
  ```

- **DEBUG**: All internal operations
  ```
  [DEBUG] GRAPHRAG DEBUG: Chat base URL: http://...
  [DEBUG] GRAPHRAG DEBUG: Embedding base URL: http://...
  ...
  ```

### Vector RAG Embedding Generation (40k chunks example)
- **INFO**: Workflow-level progress only
  ```
  [INFO] Found 40000 chunks to process for collection abc123
  [INFO] Started chord workflow: 40000 embeddings → completion callback
  ```

- **DEBUG**: Per-chunk and per-batch details
  ```
  [DEBUG] Successfully stored embedding for chunk chunk_001
  [DEBUG] Embedding request: 10 texts, token range: 200-450, avg: 325.5
  [DEBUG] Sending to vLLM: 10 texts, total chars: 3250
  ...
  [DEBUG] Successfully stored embedding for chunk chunk_40000
  ```

## Environment Variables

Override via environment:
```bash
export LOG_LEVEL=INFO  # Changes root level
# Component levels come from config file
```

## Adding New Component Loggers

To add logging for a new component:

1. Add to `config/default.yaml`:
```yaml
component_levels:
  my.new.component: INFO
```

2. Use in your code:
```python
import logging
logger = logging.getLogger(__name__)  # Will be my.new.component

logger.info("Important progress update")
logger.debug("Detailed diagnostic info")
```

## Troubleshooting

### Not seeing progress updates?
- Check `config/default.yaml` has the component at INFO level
- Verify root level isn't DEBUG (would flood with noise)
- Check logs for `[INFO]` level messages

### Too much output?
- Set component to WARNING in config
- Or set root level to WARNING and remove component override

### Need detailed debugging?
- Set specific component to DEBUG
- Or set `LOG_LEVEL=DEBUG` environment variable
