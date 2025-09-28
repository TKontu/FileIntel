"""Test document fixtures for FileIntel testing."""

import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
import json
import hashlib
from datetime import datetime
import uuid


class TestDocumentFixtures:
    """Manager for test document fixtures and sample data."""

    def __init__(self, fixture_dir: Path = None):
        """Initialize fixture manager."""
        self.fixture_dir = fixture_dir or Path(__file__).parent
        self.temp_dirs = []

    def cleanup(self):
        """Clean up temporary directories."""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def get_sample_text_document(self) -> Dict[str, Any]:
        """Get sample plain text document."""
        content = """
# Sample Document for Testing

This is a sample text document used for FileIntel testing purposes.

## Section 1: Introduction

FileIntel is a document analysis and RAG (Retrieval-Augmented Generation) system
that processes various document formats and enables intelligent querying.

## Section 2: Features

Key features include:
- Document upload and processing
- Text chunking and embedding generation
- Vector and Graph RAG implementations
- Job queue processing
- API endpoints for document management

## Section 3: Testing

This document serves as a test fixture for:
- Document reader functionality
- Text chunking algorithms
- Metadata extraction
- Content deduplication
- Search and retrieval testing

The quick brown fox jumps over the lazy dog. This pangram contains
all letters of the English alphabet and is commonly used for testing.
""".strip()

        return {
            "filename": "sample_document.txt",
            "content": content,
            "mime_type": "text/plain",
            "size": len(content.encode("utf-8")),
            "hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "metadata": {
                "title": "Sample Document for Testing",
                "sections": 3,
                "word_count": len(content.split()),
                "character_count": len(content),
                "language": "en",
            },
        }

    def get_sample_markdown_document(self) -> Dict[str, Any]:
        """Get sample Markdown document."""
        content = """# FileIntel Technical Documentation

## Overview

FileIntel is an enterprise-grade document intelligence platform that combines:

- **Document Processing**: Advanced OCR, format conversion, and content extraction
- **Vector RAG**: Traditional embedding-based retrieval and generation
- **Graph RAG**: Knowledge graph construction and relationship-aware queries
- **Job Management**: Asynchronous processing with retry mechanisms

## Architecture

```mermaid
graph TD
    A[Document Upload] --> B[Processing Queue]
    B --> C[Format Detection]
    C --> D[Content Extraction]
    D --> E[Chunking & Embedding]
    E --> F[Vector Storage]
    E --> G[Graph Construction]
```

## API Endpoints

### Document Management
- `POST /api/v1/documents` - Upload document
- `GET /api/v1/documents/{id}` - Retrieve document
- `DELETE /api/v1/documents/{id}` - Remove document

### Query Interface
- `POST /api/v1/query/vector` - Vector-based search
- `POST /api/v1/query/graph` - Graph-based search
- `POST /api/v1/query/hybrid` - Combined approach

## Configuration

```yaml
llm:
  provider: openai
  model: gpt-4
  temperature: 0.1

storage:
  type: postgresql
  host: localhost
  database: fileintel

processing:
  chunk_size: 1000
  overlap: 200
  batch_size: 10
```

## Testing Framework

The system includes comprehensive testing:

1. **Unit Tests**: Individual component validation
2. **Integration Tests**: End-to-end workflow verification
3. **Performance Tests**: Load and scalability assessment
4. **API Tests**: Endpoint functionality and security

### Sample Test Data

This document itself serves as test data for:
- Markdown parsing and rendering
- Code block extraction
- Table processing
- Link resolution
- Metadata generation

## Troubleshooting

Common issues and solutions:

| Issue | Cause | Solution |
|-------|--------|----------|
| Import errors | Missing dependencies | Run `poetry install` |
| Database connection | Incorrect config | Check connection string |
| Processing timeout | Large document | Increase timeout limits |

## References

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [PostgreSQL pgvector](https://github.com/pgvector/pgvector)
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
"""

        return {
            "filename": "technical_documentation.md",
            "content": content,
            "mime_type": "text/markdown",
            "size": len(content.encode("utf-8")),
            "hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "metadata": {
                "title": "FileIntel Technical Documentation",
                "format": "markdown",
                "sections": [
                    "Overview",
                    "Architecture",
                    "API Endpoints",
                    "Configuration",
                    "Testing Framework",
                    "Troubleshooting",
                ],
                "code_blocks": 3,
                "tables": 1,
                "links": 3,
                "word_count": len(content.split()),
            },
        }

    def get_sample_json_document(self) -> Dict[str, Any]:
        """Get sample JSON document."""
        content = {
            "document_type": "api_response",
            "timestamp": "2024-01-15T10:30:00Z",
            "api_version": "v1.2.0",
            "data": {
                "collections": [
                    {
                        "id": "coll-001",
                        "name": "Research Papers",
                        "description": "Academic research papers on AI and ML",
                        "document_count": 45,
                        "created_at": "2024-01-01T00:00:00Z",
                        "tags": ["research", "ai", "machine-learning"],
                    },
                    {
                        "id": "coll-002",
                        "name": "Technical Manuals",
                        "description": "Software and hardware documentation",
                        "document_count": 23,
                        "created_at": "2024-01-05T00:00:00Z",
                        "tags": ["documentation", "technical", "manuals"],
                    },
                ],
                "statistics": {
                    "total_documents": 68,
                    "total_collections": 2,
                    "processing_queue_size": 3,
                    "active_jobs": 1,
                    "system_health": "healthy",
                },
                "recent_activity": [
                    {
                        "action": "document_uploaded",
                        "document_id": "doc-123",
                        "timestamp": "2024-01-15T10:25:00Z",
                        "user_id": "user-456",
                    },
                    {
                        "action": "query_executed",
                        "query_id": "query-789",
                        "timestamp": "2024-01-15T10:28:00Z",
                        "user_id": "user-789",
                    },
                ],
            },
        }

        content_str = json.dumps(content, indent=2)

        return {
            "filename": "api_response_sample.json",
            "content": content_str,
            "mime_type": "application/json",
            "size": len(content_str.encode("utf-8")),
            "hash": hashlib.sha256(content_str.encode("utf-8")).hexdigest(),
            "metadata": {
                "json_structure": "nested_object",
                "top_level_keys": list(content.keys()),
                "collections_count": len(content["data"]["collections"]),
                "activity_count": len(content["data"]["recent_activity"]),
                "format_version": content.get("api_version", "unknown"),
            },
        }

    def create_test_pdf_content(self) -> bytes:
        """Create minimal test PDF content."""
        # Minimal PDF structure for testing (not a real PDF parser)
        pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 58
>>
stream
BT
/F1 12 Tf
100 700 Td
(Test PDF Document Content) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000206 00000 n
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
314
%%EOF"""
        return pdf_content

    def get_sample_documents(self) -> List[Dict[str, Any]]:
        """Get all sample documents."""
        return [
            self.get_sample_text_document(),
            self.get_sample_markdown_document(),
            self.get_sample_json_document(),
        ]

    def write_fixtures_to_disk(self, base_dir: Path = None) -> Path:
        """Write test fixtures to disk and return directory path."""
        if base_dir is None:
            temp_dir = Path(tempfile.mkdtemp(prefix="fileintel_test_fixtures_"))
            self.temp_dirs.append(temp_dir)
        else:
            temp_dir = base_dir / "test_fixtures"
            temp_dir.mkdir(exist_ok=True)

        # Write text documents
        for doc_data in self.get_sample_documents():
            file_path = temp_dir / doc_data["filename"]
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(doc_data["content"])

        # Write binary PDF
        pdf_content = self.create_test_pdf_content()
        pdf_path = temp_dir / "sample_document.pdf"
        with open(pdf_path, "wb") as f:
            f.write(pdf_content)

        # Write manifest file
        manifest = {
            "fixture_set": "fileintel_test_documents",
            "created_at": datetime.utcnow().isoformat(),
            "documents": [],
        }

        for doc_data in self.get_sample_documents():
            manifest["documents"].append(
                {
                    "filename": doc_data["filename"],
                    "size": doc_data["size"],
                    "hash": doc_data["hash"],
                    "mime_type": doc_data["mime_type"],
                    "metadata": doc_data["metadata"],
                }
            )

        # Add PDF to manifest
        manifest["documents"].append(
            {
                "filename": "sample_document.pdf",
                "size": len(pdf_content),
                "hash": hashlib.sha256(pdf_content).hexdigest(),
                "mime_type": "application/pdf",
                "metadata": {"type": "test_pdf", "pages": 1},
            }
        )

        manifest_path = temp_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return temp_dir

    def get_test_collection_data(self) -> Dict[str, Any]:
        """Get test collection configuration."""
        return {
            "id": str(uuid.uuid4()),
            "name": "Test Collection",
            "description": "Collection for automated testing",
            "settings": {
                "chunk_size": 500,
                "overlap": 100,
                "enable_ocr": False,
                "language": "en",
            },
            "expected_documents": 4,  # txt, md, json, pdf
            "expected_chunks": 15,  # Approximate
            "test_queries": [
                "What is FileIntel?",
                "How does document processing work?",
                "What are the API endpoints?",
                "Explain the architecture",
            ],
        }

    def get_test_job_data(self) -> List[Dict[str, Any]]:
        """Get sample job data for testing."""
        collection_id = str(uuid.uuid4())
        document_id = str(uuid.uuid4())

        return [
            {
                "id": str(uuid.uuid4()),
                "job_type": "indexing",
                "status": "pending",
                "priority": 1,
                "job_data": {
                    "collection_id": collection_id,
                    "document_id": document_id,
                    "file_path": "/test/sample_document.txt",
                },
                "retry_count": 0,
                "max_retries": 3,
            },
            {
                "id": str(uuid.uuid4()),
                "job_type": "analysis_separate",
                "status": "completed",
                "priority": 0,
                "job_data": {
                    "collection_id": collection_id,
                    "query": "What is this document about?",
                },
                "result": {
                    "answer": "This document is about FileIntel testing",
                    "sources": [{"chunk_id": "chunk-123", "score": 0.95}],
                },
                "retry_count": 0,
                "max_retries": 3,
            },
            {
                "id": str(uuid.uuid4()),
                "job_type": "graphrag_index_collection",
                "status": "failed",
                "priority": 2,
                "job_data": {"collection_id": collection_id},
                "error_message": "GraphRAG indexing timeout",
                "retry_count": 2,
                "max_retries": 3,
            },
        ]


# Pytest fixtures for easy access
import pytest


@pytest.fixture
def test_documents():
    """Pytest fixture for test documents."""
    fixtures = TestDocumentFixtures()
    yield fixtures
    fixtures.cleanup()


@pytest.fixture
def sample_text_doc():
    """Pytest fixture for sample text document."""
    fixtures = TestDocumentFixtures()
    return fixtures.get_sample_text_document()


@pytest.fixture
def sample_markdown_doc():
    """Pytest fixture for sample markdown document."""
    fixtures = TestDocumentFixtures()
    return fixtures.get_sample_markdown_document()


@pytest.fixture
def test_document_files(tmp_path):
    """Pytest fixture that creates test files on disk."""
    fixtures = TestDocumentFixtures()
    return fixtures.write_fixtures_to_disk(tmp_path)


@pytest.fixture
def test_collection_data():
    """Pytest fixture for test collection data."""
    fixtures = TestDocumentFixtures()
    return fixtures.get_test_collection_data()


@pytest.fixture
def test_job_data():
    """Pytest fixture for test job data."""
    fixtures = TestDocumentFixtures()
    return fixtures.get_test_job_data()
