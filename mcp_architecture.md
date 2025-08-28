# FileIntel MCP Integration Architecture

## Overview

This document outlines the architecture for integrating FileIntel as a Model Context Protocol (MCP) server, enabling any MCP-compatible AI agent to leverage FileIntel's document analysis capabilities as standardized tools.

## Architecture Goals

- **Standardization**: Expose FileIntel functionality through the MCP protocol
- **Agent Accessibility**: Enable any MCP-compatible agent to use document intelligence features
- **Composability**: Allow agents to combine FileIntel with other MCP tools
- **Scalability**: Support multiple concurrent agent connections
- **Maintainability**: Keep MCP layer separate from core FileIntel functionality

## High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AI Agent A    │    │   AI Agent B    │    │   AI Agent C    │
│   (Claude Code) │    │   (Custom)      │    │   (Other)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────────┐
                    │   MCP Protocol Layer    │
                    │   (JSON-RPC over        │
                    │   stdio/HTTP/WebSocket) │
                    └─────────────────────────┘
                                 │
                    ┌─────────────────────────┐
                    │  FileIntel MCP Server   │
                    │  - Tool Registration    │
                    │  - Request Routing      │
                    │  - Response Formatting  │
                    │  - Job Management       │
                    └─────────────────────────┘
                                 │
                    ┌─────────────────────────┐
                    │   FileIntel Core API    │
                    │   - Collections         │
                    │   - Documents           │
                    │   - RAG Queries         │
                    │   - Metadata            │
                    └─────────────────────────┘
                                 │
                    ┌─────────────────────────┐
                    │   FileIntel Backend     │
                    │   - PostgreSQL + Redis  │
                    │   - Worker Processes    │
                    │   - LLM Integration     │
                    └─────────────────────────┘
```

## Component Architecture

### 1. MCP Server Layer (`src/document_analyzer/mcp/`)

#### Core Components:

**`server.py`** - Main MCP server implementation

```python
class FileIntelMCPServer(Server):
    - Tool registration and discovery
    - Request routing and validation
    - Response formatting
    - Error handling and logging
```

**`tools.py`** - Tool definitions and implementations

```python
# Core document tools
- create_collection
- upload_document
- query_collection
- get_document_metadata
- update_document_metadata

# Advanced analysis tools
- analyze_document_set
- extract_citations
- smart_document_summary
- comparative_analysis
```

**`client.py`** - FileIntel API client wrapper

```python
class MCPFileIntelClient:
    - Async API client for FileIntel REST API
    - Job polling and status management
    - Result formatting and error handling
```

**`schemas.py`** - MCP tool schemas

```python
# JSON Schema definitions for all MCP tools
- Input validation schemas
- Output format specifications
- Error response schemas
```

### 2. Tool Categories

#### **Core Document Management**

- `create_collection` - Create document collections
- `list_collections` - List available collections
- `upload_document` - Upload and index documents
- `delete_document` - Remove documents
- `get_document_details` - Retrieve document information

#### **Metadata Operations**

- `get_document_metadata` - Extract clean, structured metadata
- `update_document_metadata` - Manual metadata corrections
- `batch_update_metadata` - Bulk metadata operations
- `validate_metadata` - Metadata quality checks

#### **RAG Query Operations**

- `query_collection` - Ask questions using RAG
- `query_document` - Query specific documents
- `analyze_collection` - Template-driven analysis
- `multi_query` - Batch question processing

#### **Advanced Analysis Tools**

- `analyze_document_set` - Multi-document comparative analysis
- `extract_citations` - Find and format citations with sources
- `smart_document_summary` - Intelligent summarization
- `identify_themes` - Extract common themes across documents
- `fact_check_claims` - Verify claims against reference documents

#### **Workflow Automation**

- `process_document_pipeline` - End-to-end document processing
- `generate_research_report` - Automated report generation
- `create_bibliography` - Generate formatted bibliographies

### 3. Communication Patterns

#### **Synchronous Operations**

```json
{
  "method": "tools/call",
  "params": {
    "name": "get_document_metadata",
    "arguments": { "document_id": "doc-123" }
  }
}
```

#### **Asynchronous Operations**

```json
{
  "method": "tools/call",
  "params": {
    "name": "upload_document",
    "arguments": {
      "collection": "research",
      "file_path": "/path/to/doc.pdf"
    }
  }
}
// Returns job_id, agent can poll for completion
```

#### **Streaming Operations** (Future)

```json
{
  "method": "tools/call",
  "params": {
    "name": "stream_document_analysis",
    "arguments": {
      "collection": "papers",
      "analysis_type": "progressive_summary"
    }
  }
}
// Returns incremental results as processing completes
```

### 4. Job Management

#### **Async Job Handling**

```python
class JobManager:
    async def submit_job(self, tool_name: str, arguments: dict) -> str
    async def poll_job_status(self, job_id: str) -> JobStatus
    async def get_job_result(self, job_id: str) -> dict
    async def cancel_job(self, job_id: str) -> bool
```

#### **Job Status Types**

- `pending` - Job queued for processing
- `running` - Job currently being processed
- `completed` - Job finished successfully
- `failed` - Job encountered error
- `cancelled` - Job cancelled by user

### 5. Error Handling

#### **MCP Error Categories**

```python
class FileIntelMCPErrors:
    INVALID_COLLECTION = "Collection not found or invalid"
    DOCUMENT_NOT_FOUND = "Document not found"
    UPLOAD_FAILED = "Document upload failed"
    QUERY_FAILED = "RAG query processing failed"
    METADATA_INVALID = "Invalid metadata format"
    JOB_TIMEOUT = "Job processing timeout"
    API_UNAVAILABLE = "FileIntel API unavailable"
```

### 6. Configuration

#### **MCP Server Configuration**

```yaml
mcp:
  server:
    name: "fileintel"
    version: "1.0.0"
    transport: "stdio" # or "http", "websocket"

  api:
    base_url: "http://localhost:8000/api/v1"
    timeout: 300
    max_retries: 3

  jobs:
    poll_interval: 2.0
    max_poll_duration: 600

  tools:
    enable_advanced: true
    max_concurrent_uploads: 5
    max_document_size: "300MB"
```

## Integration Patterns

### 1. Agent Usage Examples

#### **Research Assistant Pattern**

```python
# Agent workflow for literature review
collections = await call_tool("list_collections")
if "literature_review" not in collections:
    await call_tool("create_collection", {"name": "literature_review"})

# Upload multiple papers
for paper_path in paper_paths:
    await call_tool("upload_document", {
        "collection": "literature_review",
        "file_path": paper_path
    })

# Analyze themes
themes = await call_tool("identify_themes", {
    "collection": "literature_review",
    "max_themes": 10
})

# Generate summary
summary = await call_tool("smart_document_summary", {
    "collection": "literature_review",
    "summary_type": "academic",
    "focus_areas": themes
})
```

#### **Document QA Pattern**

```python
# Agent answers questions about uploaded documents
answer = await call_tool("query_collection", {
    "collection": "company_docs",
    "question": "What is our current remote work policy?"
})

# Get source citations
citations = await call_tool("extract_citations", {
    "collection": "company_docs",
    "query": "remote work policy",
    "citation_style": "internal"
})
```

### 2. Composite Tool Operations

#### **Research Pipeline Tool**

```python
# High-level tool that orchestrates multiple operations
await call_tool("process_research_pipeline", {
    "documents": ["/path/to/paper1.pdf", "/path/to/paper2.pdf"],
    "research_questions": [
        "What are the main methodologies?",
        "What are the key findings?",
        "What are the limitations?"
    ],
    "output_format": "structured_report",
    "include_citations": true
})
```

### 3. Agent Collaboration

#### **Multi-Agent Document Analysis**

```python
# Agent A uploads and processes documents
collection_id = await agent_a.call_tool("create_collection", {...})
await agent_a.call_tool("upload_document", {...})

# Agent B performs analysis
analysis = await agent_b.call_tool("analyze_collection", {
    "collection": collection_id,
    "analysis_type": "technical_review"
})

# Agent C generates report
report = await agent_c.call_tool("generate_research_report", {
    "collection": collection_id,
    "analysis_results": analysis,
    "format": "executive_summary"
})
```

## Security Considerations

### 1. Authentication & Authorization

- **API Key Management**: Secure storage and rotation of FileIntel API keys
- **Agent Authentication**: Verify agent identity before tool access
- **Resource Isolation**: Ensure agents can only access their own collections
- **Rate Limiting**: Prevent abuse through request throttling

### 2. Data Privacy

- **Document Isolation**: Ensure documents are only accessible to authorized agents
- **Metadata Sanitization**: Remove sensitive information from metadata
- **Audit Logging**: Track all agent interactions for security monitoring
- **Data Retention**: Configurable document and result retention policies

### 3. Input Validation

- **Schema Validation**: Strict validation of all tool inputs
- **File Type Validation**: Verify uploaded documents are safe
- **Path Sanitization**: Prevent directory traversal attacks
- **Content Scanning**: Optional malware scanning for uploads

## Performance Considerations

### 1. Scalability

- **Connection Pooling**: Efficient management of API connections
- **Request Queuing**: Handle multiple concurrent agent requests
- **Resource Limits**: Prevent individual agents from monopolizing resources
- **Load Balancing**: Distribute requests across FileIntel instances

### 2. Caching

- **Result Caching**: Cache expensive query results
- **Metadata Caching**: Cache document metadata for faster access
- **Collection Caching**: Cache collection information
- **TTL Management**: Configurable cache expiration policies

### 3. Optimization

- **Batch Operations**: Group related operations for efficiency
- **Lazy Loading**: Load data only when needed
- **Background Processing**: Use async processing for long operations
- **Resource Monitoring**: Track and optimize resource usage

## Deployment Patterns

### 1. Standalone MCP Server

```bash
# Run as independent service
fileintel-mcp --config mcp_config.yaml --transport stdio
```

### 2. Integrated with FileIntel

```bash
# Run as part of FileIntel deployment
docker-compose up fileintel-with-mcp
```

### 3. Cloud Deployment

```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fileintel-mcp
spec:
  replicas: 3
  template:
    spec:
      containers:
        - name: mcp-server
          image: fileintel/mcp:latest
          env:
            - name: FILEINTEL_API_URL
              value: "https://fileintel-api.internal"
```

This architecture enables FileIntel to become a powerful, reusable document intelligence service that any MCP-compatible AI agent can leverage for sophisticated document analysis tasks.
