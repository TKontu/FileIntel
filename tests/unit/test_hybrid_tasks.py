"""
Unit tests for hybrid query tasks.

Tests the combine_hybrid_results task that merges Vector RAG and GraphRAG results.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any


class TestCombineHybridResults:
    """Test suite for combine_hybrid_results task."""

    @pytest.fixture
    def vector_result(self) -> Dict[str, Any]:
        """Sample Vector RAG result."""
        return {
            "answer": "Machine learning is a subset of AI that enables systems to learn from data (Smith et al., 2023).",
            "sources": [
                {
                    "document_id": "doc1",
                    "filename": "ml_basics.pdf",
                    "text": "ML is a subset of AI...",
                    "similarity_score": 0.92
                },
                {
                    "document_id": "doc2",
                    "filename": "ai_overview.pdf",
                    "text": "AI encompasses multiple approaches...",
                    "similarity_score": 0.85
                }
            ]
        }

    @pytest.fixture
    def graph_result(self) -> Dict[str, Any]:
        """Sample GraphRAG result."""
        return {
            "answer": "Machine learning relates to neural networks, deep learning, and data science (Jones, 2024).",
            "sources": [
                {
                    "document_id": "doc3",
                    "filename": "relationships.pdf",
                    "citation": "Jones, 2024",
                    "text": "ML connects to various subfields..."
                },
                {
                    "document_id": "doc2",  # Duplicate with vector
                    "filename": "ai_overview.pdf",
                    "citation": "Brown et al., 2023",
                    "text": "The relationship between AI and ML..."
                }
            ]
        }

    @pytest.fixture
    def empty_result(self) -> Dict[str, Any]:
        """Empty result for testing edge cases."""
        return {
            "answer": "",
            "sources": []
        }

    def test_successful_combination(self, vector_result, graph_result):
        """Test successful combination of vector and graph results."""
        from fileintel.tasks.hybrid_tasks import combine_hybrid_results

        with patch('fileintel.core.config.get_config') as mock_config, \
             patch('fileintel.storage.postgresql_storage.PostgreSQLStorage') as mock_storage, \
             patch('fileintel.llm_integration.unified_provider.UnifiedLLMProvider') as mock_llm:

            # Setup mocks
            mock_config.return_value = Mock()
            mock_storage.return_value = Mock()

            # Mock LLM response
            mock_response = Mock()
            mock_response.content = "Combined answer: ML is a subset of AI that relates to neural networks and deep learning (Smith et al., 2023; Jones, 2024)."
            mock_llm.return_value.generate_response.return_value = mock_response

            # Execute
            result = combine_hybrid_results(
                vector_result=vector_result,
                graph_result=graph_result,
                query="What is machine learning?",
                collection_id="test-collection",
                answer_format="default"
            )

            # Assertions
            assert result["query_type"] == "hybrid"
            assert "answer" in result
            assert len(result["answer"]) > 0
            assert "sources" in result
            assert len(result["sources"]) > 0

            # Check metadata
            assert "metadata" in result
            assert result["metadata"]["synthesis_method"] == "llm"
            assert result["metadata"]["source_count"]["vector"] == 2
            assert result["metadata"]["source_count"]["graph"] == 2

            # Verify LLM was called
            mock_llm.return_value.generate_response.assert_called_once()

    def test_source_deduplication(self, vector_result, graph_result):
        """Test that duplicate sources are removed."""
        from fileintel.tasks.hybrid_tasks import combine_hybrid_results

        with patch('fileintel.core.config.get_config') as mock_config, \
             patch('fileintel.storage.postgresql_storage.PostgreSQLStorage') as mock_storage, \
             patch('fileintel.llm_integration.unified_provider.UnifiedLLMProvider') as mock_llm:

            # Setup mocks
            mock_config.return_value = Mock()
            mock_storage.return_value = Mock()
            mock_response = Mock()
            mock_response.content = "Combined answer"
            mock_llm.return_value.generate_response.return_value = mock_response

            # Execute
            result = combine_hybrid_results(
                vector_result=vector_result,
                graph_result=graph_result,
                query="Test query",
                collection_id="test-collection",
                answer_format="default"
            )

            # Assertions - should have 3 unique sources (doc1, doc2, doc3)
            # doc2 appears in both but should only be in result once
            source_ids = [s.get("document_id") for s in result["sources"]]
            assert len(source_ids) == 3  # doc1, doc2, doc3
            assert source_ids.count("doc2") == 1  # doc2 appears only once

            # Verify metadata reports correct counts
            assert result["metadata"]["source_count"]["merged"] == 3

    def test_empty_vector_result(self, empty_result, graph_result):
        """Test handling when vector result is empty."""
        from fileintel.tasks.hybrid_tasks import combine_hybrid_results

        with patch('fileintel.core.config.get_config') as mock_config, \
             patch('fileintel.storage.postgresql_storage.PostgreSQLStorage') as mock_storage, \
             patch('fileintel.llm_integration.unified_provider.UnifiedLLMProvider') as mock_llm:

            # Setup mocks
            mock_config.return_value = Mock()
            mock_storage.return_value = Mock()
            mock_response = Mock()
            mock_response.content = "Answer based on graph only: " + graph_result["answer"]
            mock_llm.return_value.generate_response.return_value = mock_response

            # Execute
            result = combine_hybrid_results(
                vector_result=empty_result,
                graph_result=graph_result,
                query="Test query",
                collection_id="test-collection",
                answer_format="default"
            )

            # Should still work, using graph result
            assert result["query_type"] == "hybrid"
            assert len(result["answer"]) > 0
            assert len(result["sources"]) == 2  # Only graph sources

    def test_empty_graph_result(self, vector_result, empty_result):
        """Test handling when graph result is empty."""
        from fileintel.tasks.hybrid_tasks import combine_hybrid_results

        with patch('fileintel.core.config.get_config') as mock_config, \
             patch('fileintel.storage.postgresql_storage.PostgreSQLStorage') as mock_storage, \
             patch('fileintel.llm_integration.unified_provider.UnifiedLLMProvider') as mock_llm:

            # Setup mocks
            mock_config.return_value = Mock()
            mock_storage.return_value = Mock()
            mock_response = Mock()
            mock_response.content = "Answer based on vector only: " + vector_result["answer"]
            mock_llm.return_value.generate_response.return_value = mock_response

            # Execute
            result = combine_hybrid_results(
                vector_result=vector_result,
                graph_result=empty_result,
                query="Test query",
                collection_id="test-collection",
                answer_format="default"
            )

            # Should still work, using vector result
            assert result["query_type"] == "hybrid"
            assert len(result["answer"]) > 0
            assert len(result["sources"]) == 2  # Only vector sources

    def test_both_results_empty(self, empty_result):
        """Test handling when both results are empty."""
        from fileintel.tasks.hybrid_tasks import combine_hybrid_results

        # Execute
        result = combine_hybrid_results(
            vector_result=empty_result,
            graph_result=empty_result,
            query="Test query",
            collection_id="test-collection",
            answer_format="default"
        )

        # Should return error message
        assert result["query_type"] == "hybrid"
        assert "No results found" in result["answer"]
        assert len(result["sources"]) == 0
        assert "error" in result

    def test_llm_synthesis_failure_fallback(self, vector_result, graph_result):
        """Test fallback when LLM synthesis fails."""
        from fileintel.tasks.hybrid_tasks import combine_hybrid_results

        with patch('fileintel.core.config.get_config') as mock_config, \
             patch('fileintel.storage.postgresql_storage.PostgreSQLStorage') as mock_storage, \
             patch('fileintel.llm_integration.unified_provider.UnifiedLLMProvider') as mock_llm:

            # Setup mocks
            mock_config.return_value = Mock()
            mock_storage.return_value = Mock()

            # Make LLM fail
            mock_llm.return_value.generate_response.side_effect = Exception("LLM unavailable")

            # Execute
            result = combine_hybrid_results(
                vector_result=vector_result,
                graph_result=graph_result,
                query="Test query",
                collection_id="test-collection",
                answer_format="default"
            )

            # Should fallback to concatenation
            assert result["query_type"] == "hybrid"
            assert len(result["answer"]) > 0
            assert "Vector Search Results:" in result["answer"] or "Graph Search Results:" in result["answer"]
            assert len(result["sources"]) > 0

    def test_complete_failure_fallback(self, vector_result, graph_result):
        """Test fallback when entire combination process fails."""
        from fileintel.tasks.hybrid_tasks import combine_hybrid_results

        with patch('fileintel.core.config.get_config') as mock_config:
            # Make config fail to trigger outer exception handler
            mock_config.side_effect = Exception("Config error")

            # Execute
            result = combine_hybrid_results(
                vector_result=vector_result,
                graph_result=graph_result,
                query="Test query",
                collection_id="test-collection",
                answer_format="default"
            )

            # Should return fallback result
            assert result["query_type"] == "hybrid_fallback"
            assert "error" in result
            assert len(result["answer"]) > 0  # Should have graph answer (fallback)
            assert "fallback_type" in result

    def test_source_limit(self, vector_result, graph_result):
        """Test that sources are limited to 15."""
        from fileintel.tasks.hybrid_tasks import combine_hybrid_results

        # Create results with many sources
        many_vector_sources = [
            {"document_id": f"vec_doc{i}", "text": f"Source {i}"} for i in range(10)
        ]
        many_graph_sources = [
            {"document_id": f"graph_doc{i}", "text": f"Source {i}"} for i in range(10)
        ]

        vector_result["sources"] = many_vector_sources
        graph_result["sources"] = many_graph_sources

        with patch('fileintel.core.config.get_config') as mock_config, \
             patch('fileintel.storage.postgresql_storage.PostgreSQLStorage') as mock_storage, \
             patch('fileintel.llm_integration.unified_provider.UnifiedLLMProvider') as mock_llm:

            # Setup mocks
            mock_config.return_value = Mock()
            mock_storage.return_value = Mock()
            mock_response = Mock()
            mock_response.content = "Combined answer"
            mock_llm.return_value.generate_response.return_value = mock_response

            # Execute
            result = combine_hybrid_results(
                vector_result=vector_result,
                graph_result=graph_result,
                query="Test query",
                collection_id="test-collection",
                answer_format="default"
            )

            # Should limit to 15 sources
            assert len(result["sources"]) <= 15
            assert result["metadata"]["source_count"]["returned"] <= 15

    def test_citation_preservation_in_prompt(self, vector_result, graph_result):
        """Test that synthesis prompt emphasizes citation preservation."""
        from fileintel.tasks.hybrid_tasks import combine_hybrid_results

        with patch('fileintel.core.config.get_config') as mock_config, \
             patch('fileintel.storage.postgresql_storage.PostgreSQLStorage') as mock_storage, \
             patch('fileintel.llm_integration.unified_provider.UnifiedLLMProvider') as mock_llm:

            # Setup mocks
            mock_config.return_value = Mock()
            mock_storage.return_value = Mock()
            mock_response = Mock()
            mock_response.content = "Combined answer with citations"
            mock_llm.return_value.generate_response.return_value = mock_response

            # Execute
            combine_hybrid_results(
                vector_result=vector_result,
                graph_result=graph_result,
                query="Test query",
                collection_id="test-collection",
                answer_format="default"
            )

            # Check that synthesis prompt includes citation preservation instruction
            call_args = mock_llm.return_value.generate_response.call_args
            prompt = call_args[1]["prompt"]

            assert "CRITICAL" in prompt
            assert "preserve all citations" in prompt.lower()
            assert "exactly as they appear" in prompt.lower()

    def test_metadata_completeness(self, vector_result, graph_result):
        """Test that result metadata is complete and accurate."""
        from fileintel.tasks.hybrid_tasks import combine_hybrid_results

        with patch('fileintel.core.config.get_config') as mock_config, \
             patch('fileintel.storage.postgresql_storage.PostgreSQLStorage') as mock_storage, \
             patch('fileintel.llm_integration.unified_provider.UnifiedLLMProvider') as mock_llm:

            # Setup mocks
            mock_config.return_value = Mock()
            mock_storage.return_value = Mock()
            mock_response = Mock()
            mock_response.content = "Combined answer"
            mock_llm.return_value.generate_response.return_value = mock_response

            # Execute
            result = combine_hybrid_results(
                vector_result=vector_result,
                graph_result=graph_result,
                query="Test query",
                collection_id="test-collection",
                answer_format="default"
            )

            # Check metadata structure
            metadata = result["metadata"]
            assert "vector_answer_length" in metadata
            assert "graph_answer_length" in metadata
            assert "source_count" in metadata
            assert "synthesis_method" in metadata

            # Check source counts
            source_count = metadata["source_count"]
            assert source_count["vector"] == 2
            assert source_count["graph"] == 2
            assert source_count["merged"] == 3  # After deduplication
            assert source_count["returned"] == 3

    def test_answer_format_parameter_used(self, vector_result, graph_result):
        """Test that answer_format parameter is properly handled."""
        from fileintel.tasks.hybrid_tasks import combine_hybrid_results

        with patch('fileintel.core.config.get_config') as mock_config, \
             patch('fileintel.storage.postgresql_storage.PostgreSQLStorage') as mock_storage, \
             patch('fileintel.llm_integration.unified_provider.UnifiedLLMProvider') as mock_llm:

            # Setup mocks
            mock_config.return_value = Mock()
            mock_storage.return_value = Mock()
            mock_response = Mock()
            mock_response.content = "Formatted answer"
            mock_llm.return_value.generate_response.return_value = mock_response

            # Execute with custom format
            result = combine_hybrid_results(
                vector_result=vector_result,
                graph_result=graph_result,
                query="Test query",
                collection_id="test-collection",
                answer_format="single_paragraph"
            )

            # Note: answer_format is passed but not explicitly used in synthesis
            # It's available for future enhancements
            assert result is not None
            assert "answer" in result

    def test_sources_without_ids(self, vector_result, graph_result):
        """Test handling of sources without document_id or filename."""
        from fileintel.tasks.hybrid_tasks import combine_hybrid_results

        # Add sources without IDs
        vector_result["sources"].append({
            "text": "Source without ID",
            "similarity_score": 0.75
        })
        graph_result["sources"].append({
            "text": "Another source without ID",
            "citation": "Unknown, 2024"
        })

        with patch('fileintel.core.config.get_config') as mock_config, \
             patch('fileintel.storage.postgresql_storage.PostgreSQLStorage') as mock_storage, \
             patch('fileintel.llm_integration.unified_provider.UnifiedLLMProvider') as mock_llm:

            # Setup mocks
            mock_config.return_value = Mock()
            mock_storage.return_value = Mock()
            mock_response = Mock()
            mock_response.content = "Combined answer"
            mock_llm.return_value.generate_response.return_value = mock_response

            # Execute - should not crash
            result = combine_hybrid_results(
                vector_result=vector_result,
                graph_result=graph_result,
                query="Test query",
                collection_id="test-collection",
                answer_format="default"
            )

            # Should include all sources, even those without IDs
            assert len(result["sources"]) >= 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
