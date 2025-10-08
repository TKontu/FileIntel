`fileintel version`
`fileintel health`
`fileintel status`
`fileintel quickstart`
`fileintel collections create <name>`
`fileintel collections list` - Verify collection listing
`fileintel collections get <identifier>` - Check collection retrieval
`fileintel collections delete <identifier>` - Test collection deletion
`fileintel collections process <identifier>` - Validate collection processing
`fileintel collections status <identifier>` - Check processing status
`fileintel collections system-status` - Verify system status
`fileintel collections upload-and-process` - Test upload+process workflow

`fileintel documents upload <collection> <file>` - Validate document upload
`fileintel documents batch-upload <collection> <directory>` - Test batch upload
`fileintel documents list <collection>` - Check document listing
`fileintel documents get <document_id>` - Verify document retrieval
`fileintel documents delete <document_id>` - Test document deletion
`fileintel documents system-status` - Check document system status

`fileintel tasks list` - Validate task listing with pagination
`fileintel tasks get <task_id>` - Check task status retrieval
`fileintel tasks cancel <task_id>` - Test task cancellation
`fileintel tasks result <task_id>` - Verify result retrieval
`fileintel tasks wait <task_id>` - Test progress monitoring
`fileintel tasks metrics` - Check metrics collection
`fileintel tasks batch-cancel` - Test batch operations
`fileintel tasks system-status` - Verify task system status

`fileintel query collection <identifier> <question>` - Validate RAG querying
`fileintel query <collection_name_or_id> "your question here" --search-type vector`
    Search Type Options:
  - "vector" - Pure vector/semantic similarity search
  - "graph" - GraphRAG search
  - "adaptive" - Automatically chooses best method
  - "global" - GraphRAG global community search
  - "local" - GraphRAG local entity search
`fileintel query document <collection> <doc_id> <question>` - Test document queries
`fileintel query system-status` - Check query system status
`fileintel query test <collection>` - Validate system testing

`fileintel graphrag index <collection>` - Validate GraphRAG indexing
`fileintel graphrag query <collection> <question>` - Test GraphRAG queries
`fileintel graphrag status <collection>` - Check GraphRAG status
`fileintel graphrag entities <collection>`
`fileintel graphrag system-status`
