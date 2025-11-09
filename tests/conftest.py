"""
Pytest configuration and shared fixtures for FileIntel tests.
"""
import os
import pytest
from unittest.mock import MagicMock, Mock, patch
from pathlib import Path
import tempfile


# Load environment variables from .env file at project root
def load_env_file():
    """Load environment variables from .env file"""
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


# Load .env file before imports
load_env_file()

# Set fallback test environment variables if not in .env
os.environ.setdefault("OPENAI_API_KEY", "test_key")
os.environ.setdefault("FILEINTEL_API_KEY", "test_api_key")

from fileintel.storage.models import (
    Collection,
    Document,
    DocumentChunk,
)

# from fileintel.worker.job_manager import JobManager  # Removed - migrated to Celery
from fileintel.llm_integration.openai_provider import OpenAIProvider
from fileintel.llm_integration.embedding_provider import (
    OpenAIEmbeddingProvider,
)
# Commented out - these modules no longer exist after refactoring
# from fileintel.prompt_management.composer import PromptComposer
# from fileintel.prompt_management.loader import PromptLoader
# from fileintel.document_processing.factory import ReaderFactory
# from fileintel.document_processing.chunking import TextChunker
# from fileintel.document_processing.metadata_extractor import MetadataExtractor

# from fileintel.worker.batch_manager import BatchProcessor  # Removed - migrated to Celery
# from fileintel.worker.job_processor_registry import JobProcessorRegistry  # Removed - migrated to Celery
import uuid
from datetime import datetime

# For real database integration testing, set up a proper PostgreSQL test database


@pytest.fixture
def mock_storage():
    """Mock storage interface for unit tests."""
    storage = MagicMock()
    storage.create_collection.return_value = Collection(
        id=str(uuid.uuid4()), name="test_collection"
    )
    storage.get_all_collections.return_value = []
    storage.get_collection.return_value = None
    storage.get_document.return_value = None
    return storage


# JobManager fixture removed - migrated to Celery task system


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider"""
    provider = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Test response"
    mock_response.usage_info = {"tokens": 100}
    mock_response._asdict.return_value = {
        "content": "Test response",
        "usage_info": {"tokens": 100},
    }
    provider.generate_response.return_value = mock_response
    return provider


@pytest.fixture
def mock_embedding_provider():
    """Mock embedding provider"""
    provider = MagicMock()
    provider.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
    return provider


@pytest.fixture
def mock_composer():
    """Mock prompt composer"""
    composer = MagicMock()
    composer.compose.return_value = "Test prompt"
    return composer


@pytest.fixture
def mock_loader():
    """Mock prompt loader"""
    loader = MagicMock()
    loader.load_prompt_components.return_value = {
        "system": "Test system prompt",
        "user": "Test user prompt",
        "question": "Test question prompt",
    }
    return loader


@pytest.fixture
def sample_collection():
    """Sample collection for testing"""
    return Collection(
        id=str(uuid.uuid4()), name="test_collection", created_at=datetime.now()
    )


@pytest.fixture
def sample_document(sample_collection):
    """Sample document for testing"""
    return Document(
        id=str(uuid.uuid4()),
        collection_id=sample_collection.id,
        filename="test_doc.pdf",
        original_filename="test_document.pdf",
        content_hash="abc123",
        file_size=1024,
        mime_type="application/pdf",
        document_metadata={"title": "Test Document"},
        created_at=datetime.now(),
    )


# Sample job fixture removed - replaced by Celery task testing patterns


@pytest.fixture
def sample_chunks(sample_document):
    """Sample chunks for testing"""
    chunks = []
    for i in range(3):
        chunk = DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=sample_document.id,
            collection_id=sample_document.collection_id,
            position=i,
            chunk_text=f"This is test chunk {i} content.",
            embedding=[0.1 * i, 0.2 * i, 0.3 * i],
            created_at=datetime.now(),
        )
        chunks.append(chunk)
    return chunks


@pytest.fixture
def mock_celery_task():
    """Mock Celery task for testing"""
    task = MagicMock()
    task.id = str(uuid.uuid4())
    task.state = "PENDING"
    task.result = None
    task.info = {}
    task.get = MagicMock(return_value={"status": "completed"})
    task.ready = MagicMock(return_value=False)
    task.successful = MagicMock(return_value=False)
    task.failed = MagicMock(return_value=False)
    return task


@pytest.fixture
def celery_test_dependencies(mock_llm_provider, mock_embedding_provider, mock_composer):
    """Complete set of dependencies for Celery task testing"""
    return {
        "llm_provider": mock_llm_provider,
        "embedding_provider": mock_embedding_provider,
        "composer": mock_composer,
        "reader_factory": MagicMock(),
        "text_chunker": MagicMock(),
        "metadata_extractor": MagicMock(),
    }


@pytest.fixture
def mock_celery_app():
    """Mock Celery application for testing"""
    app = MagicMock()
    app.control.inspect.return_value.active.return_value = {}
    app.control.inspect.return_value.reserved.return_value = {}
    app.control.inspect.return_value.stats.return_value = {}
    return app


@pytest.fixture
def mock_celery_result():
    """Mock Celery AsyncResult for testing"""
    result = MagicMock()
    result.id = str(uuid.uuid4())
    result.state = "SUCCESS"
    result.result = {"status": "completed", "processed_items": 5}
    result.info = {"current": 10, "total": 10, "description": "Completed"}
    result.ready.return_value = True
    result.successful.return_value = True
    result.failed.return_value = False
    result.get.return_value = {"status": "completed", "processed_items": 5}
    return result


@pytest.fixture
def sample_task_data():
    """Sample task data for testing Celery tasks"""
    return {
        "task_id": str(uuid.uuid4()),
        "task_name": "process_document",
        "args": ["/test/document.pdf"],
        "kwargs": {
            "document_id": str(uuid.uuid4()),
            "collection_id": str(uuid.uuid4()),
        },
        "queue": "document_processing",
        "routing_key": "document.process",
    }


@pytest.fixture
def mock_document_processing_task():
    """Mock document processing task result"""
    return {
        "task_id": str(uuid.uuid4()),
        "status": "completed",
        "document_id": str(uuid.uuid4()),
        "collection_id": str(uuid.uuid4()),
        "chunks": [
            "This is chunk 1 of the document",
            "This is chunk 2 of the document",
            "This is chunk 3 of the document",
        ],
        "metadata": {"title": "Test Document", "pages": 3, "file_size": 1024},
        "processing_time": 5.2,
    }


@pytest.fixture
def mock_graphrag_task():
    """Mock GraphRAG task result"""
    return {
        "task_id": str(uuid.uuid4()),
        "status": "completed",
        "collection_id": str(uuid.uuid4()),
        "workspace_path": "/test/graphrag/workspace",
        "entities_count": 25,
        "communities_count": 5,
        "processing_time": 45.7,
    }


@pytest.fixture
def mock_llm_task():
    """Mock LLM task result"""
    return {
        "task_id": str(uuid.uuid4()),
        "status": "completed",
        "query": "What is this document about?",
        "answer": "This document discusses machine learning algorithms.",
        "sources": ["doc1.pdf", "doc2.pdf"],
        "confidence": 0.85,
        "processing_time": 2.1,
    }


@pytest.fixture
def relevant_chunks_result(sample_chunks, sample_document):
    """Mock result for find_relevant_chunks methods"""
    results = []
    for chunk in sample_chunks:
        results.append(
            {
                "chunk": chunk,
                "filename": sample_document.original_filename,
                "document_metadata": sample_document.document_metadata,
                "similarity_score": 0.85,
            }
        )
    return results
