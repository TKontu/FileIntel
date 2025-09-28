"""
Unit tests for GraphRAG workflow module.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncio

from src.fileintel.rag.graph_rag.workflows.graphrag_workflow import GraphRAGWorkflow
from src.fileintel.storage.postgresql_storage import PostgreSQLStorage
from src.fileintel.rag.graph_rag.services.graphrag_service import GraphRAGService


class TestGraphRAGWorkflow:
    """Test suite for GraphRAGWorkflow class."""

    @pytest.fixture
    def mock_storage(self):
        """Create mock PostgreSQL storage."""
        storage = Mock(spec=PostgreSQLStorage)
        storage.get_documents_in_collection = Mock(return_value=[])
        storage.get_all_chunks_for_document = Mock(return_value=[])
        storage.get_document = Mock(return_value=None)
        storage.save_graphrag_index_info = Mock()
        storage.save_result = Mock()
        return storage

    @pytest.fixture
    def mock_llm_provider(self):
        """Create mock LLM provider."""
        provider = Mock()
        provider.generate_response = Mock()
        return provider

    @pytest.fixture
    def mock_graphrag_service(self):
        """Create mock GraphRAG service."""
        service = Mock(spec=GraphRAGService)
        service.build_collection_index = AsyncMock(return_value="/fake/index/path")
        service.global_query = AsyncMock(
            return_value={
                "answer": "Test answer",
                "sources": ["source1", "source2"],
                "communities_used": ["community1"],
                "confidence": 0.8,
            }
        )
        service.local_query = AsyncMock(
            return_value={
                "answer": "Local answer",
                "sources": ["source1"],
                "entities_used": ["entity1"],
                "relationships": ["rel1"],
                "confidence": 0.9,
            }
        )
        service.extract_entities = AsyncMock(
            return_value=[
                {"name": "Entity1", "type": "PERSON", "confidence": 0.9},
                {"name": "Entity2", "type": "ORGANIZATION", "confidence": 0.8},
            ]
        )
        service.find_entity_context = AsyncMock(
            return_value=[{"entity_name": "Entity1", "context": "Some context"}]
        )
        service.is_collection_indexed = Mock(return_value=True)
        return service

    @pytest.fixture
    def graphrag_workflow(self, mock_storage, mock_llm_provider):
        """Create GraphRAG workflow instance."""
        return GraphRAGWorkflow(mock_storage, mock_llm_provider)

    @pytest.fixture
    def graphrag_workflow_with_service(
        self, mock_storage, mock_llm_provider, mock_graphrag_service
    ):
        """Create GraphRAG workflow with mocked service."""
        workflow = GraphRAGWorkflow(mock_storage, mock_llm_provider)
        workflow.graphrag_service = mock_graphrag_service
        return workflow

    @pytest.fixture
    def mock_job(self):
        """Create mock job object."""
        job = Mock()
        job.id = "test_job_id"
        job.data = {"collection_id": "test_collection", "question": "Test question?"}
        return job

    @pytest.fixture
    def mock_adapter(self):
        """Create mock logging adapter."""
        adapter = Mock()
        adapter.info = Mock()
        adapter.error = Mock()
        adapter.warning = Mock()
        return adapter

    def test_init(self, mock_storage, mock_llm_provider):
        """Test GraphRAGWorkflow initialization."""
        workflow = GraphRAGWorkflow(mock_storage, mock_llm_provider)

        assert workflow.storage == mock_storage
        assert workflow.llm_provider == mock_llm_provider
        assert isinstance(workflow.graphrag_service, GraphRAGService)

    def test_get_supported_job_types(self, graphrag_workflow):
        """Test supported job types."""
        expected_types = [
            "graphrag_index_collection",
            "graphrag_index_document",
            "global_query",
            "local_query",
            "hybrid_citation_search",
            "comparative_analysis_enhanced",
        ]

        assert graphrag_workflow.get_supported_job_types() == expected_types

    @pytest.mark.asyncio
    async def test_process_async_graphrag_index_collection(
        self, graphrag_workflow_with_service, mock_job, mock_adapter
    ):
        """Test async processing of collection indexing."""
        mock_job.job_type = "graphrag_index_collection"

        # Mock documents and chunks
        mock_doc = Mock()
        mock_doc.id = "doc1"
        mock_doc.document_metadata = {"title": "Test Doc"}

        mock_chunk = Mock()
        mock_chunk.chunk_text = "This is test content"

        graphrag_workflow_with_service.storage.get_documents_in_collection.return_value = [
            mock_doc
        ]
        graphrag_workflow_with_service.storage.get_all_chunks_for_document.return_value = [
            mock_chunk
        ]

        await graphrag_workflow_with_service.process_async(mock_job, mock_adapter)

        # Verify GraphRAG service was called
        graphrag_workflow_with_service.graphrag_service.build_collection_index.assert_called_once()

        # Verify storage operations
        graphrag_workflow_with_service.storage.save_graphrag_index_info.assert_called_once()
        graphrag_workflow_with_service.storage.save_result.assert_called_once()

        # Check saved result
        save_call = graphrag_workflow_with_service.storage.save_result.call_args
        result_data = save_call[0][1]
        assert result_data["status"] == "success"
        assert result_data["collection_id"] == "test_collection"

    @pytest.mark.asyncio
    async def test_process_async_global_query(
        self, graphrag_workflow_with_service, mock_job, mock_adapter
    ):
        """Test async processing of global query."""
        mock_job.job_type = "global_query"

        await graphrag_workflow_with_service.process_async(mock_job, mock_adapter)

        # Verify GraphRAG service was called
        graphrag_workflow_with_service.graphrag_service.global_query.assert_called_once_with(
            "test_collection", "Test question?"
        )

        # Verify result was saved
        graphrag_workflow_with_service.storage.save_result.assert_called_once()
        save_call = graphrag_workflow_with_service.storage.save_result.call_args
        result_data = save_call[0][1]
        assert result_data["content"] == "Test answer"
        assert result_data["query_type"] == "global"
        assert result_data["method"] == "microsoft_graphrag_global"

    @pytest.mark.asyncio
    async def test_process_async_local_query(
        self, graphrag_workflow_with_service, mock_job, mock_adapter
    ):
        """Test async processing of local query."""
        mock_job.job_type = "local_query"

        await graphrag_workflow_with_service.process_async(mock_job, mock_adapter)

        # Verify GraphRAG service was called
        graphrag_workflow_with_service.graphrag_service.local_query.assert_called_once_with(
            "test_collection", "Test question?"
        )

        # Verify result was saved
        graphrag_workflow_with_service.storage.save_result.assert_called_once()
        save_call = graphrag_workflow_with_service.storage.save_result.call_args
        result_data = save_call[0][1]
        assert result_data["content"] == "Local answer"
        assert result_data["query_type"] == "local"
        assert result_data["method"] == "microsoft_graphrag_local"

    @pytest.mark.asyncio
    async def test_process_graphrag_index_collection_job_success(
        self, graphrag_workflow_with_service, mock_adapter
    ):
        """Test successful collection indexing."""
        # Create mock job
        mock_job = Mock()
        mock_job.id = "test_job"
        mock_job.data = {"collection_id": "test_collection"}

        # Mock documents and chunks
        mock_doc = Mock()
        mock_doc.id = "doc1"
        mock_doc.document_metadata = {"title": "Test Doc"}

        mock_chunk = Mock()
        mock_chunk.chunk_text = "Test content"

        graphrag_workflow_with_service.storage.get_documents_in_collection.return_value = [
            mock_doc
        ]
        graphrag_workflow_with_service.storage.get_all_chunks_for_document.return_value = [
            mock_chunk
        ]

        await graphrag_workflow_with_service.process_graphrag_index_collection_job(
            mock_job, mock_adapter
        )

        # Verify calls
        graphrag_workflow_with_service.graphrag_service.build_collection_index.assert_called_once()
        graphrag_workflow_with_service.storage.save_graphrag_index_info.assert_called_once_with(
            "test_collection", "/fake/index/path"
        )

        # Verify successful result
        save_call = graphrag_workflow_with_service.storage.save_result.call_args
        result_data = save_call[0][1]
        assert result_data["status"] == "success"
        assert result_data["documents_indexed"] == 1

    @pytest.mark.asyncio
    async def test_process_graphrag_index_collection_job_error(
        self, graphrag_workflow_with_service, mock_adapter
    ):
        """Test collection indexing error handling."""
        mock_job = Mock()
        mock_job.id = "test_job"
        mock_job.data = {"collection_id": "test_collection"}

        # Mock error
        graphrag_workflow_with_service.storage.get_documents_in_collection.side_effect = Exception(
            "Storage error"
        )

        with pytest.raises(Exception):
            await graphrag_workflow_with_service.process_graphrag_index_collection_job(
                mock_job, mock_adapter
            )

        # Verify error result was saved
        save_call = graphrag_workflow_with_service.storage.save_result.call_args
        result_data = save_call[0][1]
        assert result_data["status"] == "error"
        assert "Storage error" in result_data["error"]

    @pytest.mark.asyncio
    async def test_process_hybrid_citation_search_job(
        self, graphrag_workflow_with_service, mock_adapter
    ):
        """Test hybrid citation search."""
        mock_job = Mock()
        mock_job.id = "test_job"
        mock_job.data = {
            "document_id": "doc1",
            "reference_source": {"collection_id": "ref_collection"},
        }

        # Mock chunks
        mock_chunk = Mock()
        mock_chunk.id = "chunk1"
        mock_chunk.chunk_text = "This is test content"

        graphrag_workflow_with_service.storage.get_all_chunks_for_document.return_value = [
            mock_chunk
        ]

        await graphrag_workflow_with_service.process_hybrid_citation_search_job(
            mock_job, mock_adapter
        )

        # Verify entity extraction was called
        graphrag_workflow_with_service.graphrag_service.extract_entities.assert_called()

        # Verify result was saved
        save_call = graphrag_workflow_with_service.storage.save_result.call_args
        result_data = save_call[0][1]
        assert "citations" in result_data
        assert result_data["method"] == "hybrid_graphrag_vector"

    @pytest.mark.asyncio
    async def test_process_comparative_analysis_enhanced_job(
        self, graphrag_workflow_with_service, mock_adapter
    ):
        """Test enhanced comparative analysis."""
        mock_job = Mock()
        mock_job.id = "test_job"
        mock_job.data = {
            "collection_id": "test_collection",
            "analysis_text": "Compare these entities",
            "question": "What are the differences?",
        }

        await graphrag_workflow_with_service.process_comparative_analysis_enhanced_job(
            mock_job, mock_adapter
        )

        # Verify entity extraction and context finding
        graphrag_workflow_with_service.graphrag_service.extract_entities.assert_called_with(
            "Compare these entities"
        )
        graphrag_workflow_with_service.graphrag_service.find_entity_context.assert_called()

        # Verify result was saved
        save_call = graphrag_workflow_with_service.storage.save_result.call_args
        result_data = save_call[0][1]
        assert result_data["method"] == "graphrag_enhanced_comparative"
        assert "entities_found" in result_data
        assert "relationships" in result_data

    @pytest.mark.asyncio
    async def test_unsupported_job_type(
        self, graphrag_workflow_with_service, mock_adapter
    ):
        """Test handling of unsupported job type."""
        mock_job = Mock()
        mock_job.job_type = "unsupported_job_type"

        with pytest.raises(ValueError, match="Unsupported GraphRAG job type"):
            await graphrag_workflow_with_service.process_async(mock_job, mock_adapter)

    def test_create_citation_chunks(self, graphrag_workflow):
        """Test citation chunk creation."""
        mock_chunk = Mock()
        mock_chunk.id = "chunk1"
        mock_chunk.chunk_text = "Test content"

        chunks = [mock_chunk]
        result = graphrag_workflow._create_citation_chunks(chunks)

        assert len(result) == 1
        assert result[0]["content"] == "Test content"
        assert result[0]["chunk_id"] == "chunk1"

    @pytest.mark.asyncio
    async def test_get_collection_status(self, graphrag_workflow_with_service):
        """Test getting collection status."""
        collection_id = "test_collection"
        expected_status = {"indexed": True, "status": "ready"}

        graphrag_workflow_with_service.graphrag_service.get_index_status = AsyncMock(
            return_value=expected_status
        )

        result = await graphrag_workflow_with_service.get_collection_status(
            collection_id
        )

        assert result == expected_status
        graphrag_workflow_with_service.graphrag_service.get_index_status.assert_called_once_with(
            collection_id
        )
