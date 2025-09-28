"""
Unit tests for GraphRAG CLI commands.
"""
import pytest
from unittest.mock import Mock, patch
from typer.testing import CliRunner

from src.fileintel.cli.graphrag import app as graphrag_app


class TestGraphRAGCLI:
    """Test GraphRAG CLI commands."""

    @pytest.fixture
    def cli_runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_api_responses(self):
        """Mock API responses for CLI tests."""
        return {
            "index_success": {
                "job_id": "test_job_123",
                "status": "started",
                "collection_id": "test_collection",
            },
            "status_indexed": {
                "indexed": True,
                "status": "ready",
                "size_mb": 15.5,
                "documents_count": 10,
            },
            "status_not_indexed": {"indexed": False, "status": "not_indexed"},
            "global_query_success": {
                "content": "This is a comprehensive analysis of the collection.",
                "communities_used": ["community1", "community2"],
                "confidence": 0.85,
                "method": "microsoft_graphrag_global",
            },
            "local_query_success": {
                "content": "Entity-specific information found.",
                "entities_used": ["Entity1", "Entity2"],
                "relationships": ["rel1"],
                "confidence": 0.90,
                "method": "microsoft_graphrag_local",
            },
            "entities_success": {
                "entities": [
                    {
                        "name": "John Doe",
                        "type": "PERSON",
                        "description": "CEO",
                        "degree": 5,
                    },
                    {
                        "name": "ACME Corp",
                        "type": "ORGANIZATION",
                        "description": "Technology company",
                        "degree": 8,
                    },
                ]
            },
            "communities_success": {
                "communities": [
                    {
                        "id": "1",
                        "title": "Tech Community",
                        "level": 0,
                        "size": 15,
                        "entities": ["e1", "e2"],
                    },
                    {
                        "id": "2",
                        "title": "Business Community",
                        "level": 1,
                        "size": 8,
                        "entities": ["e3"],
                    },
                ]
            },
        }


@patch("src.fileintel.cli.graphrag.api.graphrag_index_collection")
def test_index_command_success(self, mock_index, cli_runner, mock_api_responses):
    """Test successful collection indexing command."""
    mock_index.return_value = mock_api_responses["index_success"]

    result = cli_runner.invoke(graphrag_app, ["index", "test_collection"])

    assert result.exit_code == 0
    assert "GraphRAG indexing started" in result.output
    mock_index.assert_called_once_with("test_collection")


@patch("src.fileintel.cli.graphrag.api.graphrag_index_collection")
def test_index_command_error(self, mock_index, cli_runner):
    """Test collection indexing command error handling."""
    mock_index.side_effect = Exception("API Error")

    result = cli_runner.invoke(graphrag_app, ["index", "test_collection"])

    assert result.exit_code == 1
    assert "Error indexing collection" in result.output


@patch("src.fileintel.cli.graphrag.api.graphrag_collection_status")
def test_status_command_indexed(self, mock_status, cli_runner, mock_api_responses):
    """Test status command for indexed collection."""
    mock_status.return_value = mock_api_responses["status_indexed"]

    result = cli_runner.invoke(graphrag_app, ["status", "test_collection"])

    assert result.exit_code == 0
    assert "Collection is indexed" in result.output
    assert "Status: ready" in result.output
    assert "15.50 MB" in result.output
    mock_status.assert_called_once_with("test_collection")


@patch("src.fileintel.cli.graphrag.api.graphrag_collection_status")
def test_status_command_not_indexed(self, mock_status, cli_runner, mock_api_responses):
    """Test status command for non-indexed collection."""
    mock_status.return_value = mock_api_responses["status_not_indexed"]

    result = cli_runner.invoke(graphrag_app, ["status", "test_collection"])

    assert result.exit_code == 0
    assert "Collection is not indexed" in result.output


@patch("src.fileintel.cli.graphrag.api.graphrag_global_query")
def test_global_query_command(self, mock_query, cli_runner, mock_api_responses):
    """Test global query command."""
    mock_query.return_value = mock_api_responses["global_query_success"]

    result = cli_runner.invoke(
        graphrag_app,
        ["global-query", "test_collection", "What are the main themes?"],
    )

    assert result.exit_code == 0
    assert "Global Query: What are the main themes?" in result.output
    assert "comprehensive analysis" in result.output
    assert "Communities used: 2" in result.output
    assert "Confidence: 0.85" in result.output
    mock_query.assert_called_once_with("test_collection", "What are the main themes?")


@patch("src.fileintel.cli.graphrag.api.graphrag_local_query")
def test_local_query_command(self, mock_query, cli_runner, mock_api_responses):
    """Test local query command."""
    mock_query.return_value = mock_api_responses["local_query_success"]

    result = cli_runner.invoke(
        graphrag_app, ["local-query", "test_collection", "Tell me about John Doe"]
    )

    assert result.exit_code == 0
    assert "Local Query: Tell me about John Doe" in result.output
    assert "Entity-specific information" in result.output
    assert "Entities used: 2" in result.output
    assert "Relationships found: 1" in result.output
    mock_query.assert_called_once_with("test_collection", "Tell me about John Doe")


@patch("src.fileintel.cli.graphrag.api.graphrag_entities")
def test_entities_command(self, mock_entities, cli_runner, mock_api_responses):
    """Test entities listing command."""
    mock_entities.return_value = mock_api_responses["entities_success"]

    result = cli_runner.invoke(
        graphrag_app, ["entities", "test_collection", "--limit", "10"]
    )

    assert result.exit_code == 0
    assert "John Doe" in result.output
    assert "ACME Corp" in result.output
    assert "PERSON" in result.output
    assert "ORGANIZATION" in result.output
    mock_entities.assert_called_once_with("test_collection", 10)


@patch("src.fileintel.cli.graphrag.api.graphrag_entities")
def test_entities_command_no_results(self, mock_entities, cli_runner):
    """Test entities command with no results."""
    mock_entities.return_value = {"entities": []}

    result = cli_runner.invoke(graphrag_app, ["entities", "test_collection"])

    assert result.exit_code == 0
    assert "No entities found" in result.output


@patch("src.fileintel.cli.graphrag.api.graphrag_communities")
def test_communities_command(self, mock_communities, cli_runner, mock_api_responses):
    """Test communities listing command."""
    mock_communities.return_value = mock_api_responses["communities_success"]

    result = cli_runner.invoke(graphrag_app, ["communities", "test_collection"])

    assert result.exit_code == 0
    assert "Tech Community" in result.output
    assert "Business Community" in result.output
    mock_communities.assert_called_once_with("test_collection")


@patch("src.fileintel.cli.graphrag.api.graphrag_communities")
def test_communities_command_no_results(self, mock_communities, cli_runner):
    """Test communities command with no results."""
    mock_communities.return_value = {"communities": []}

    result = cli_runner.invoke(graphrag_app, ["communities", "test_collection"])

    assert result.exit_code == 0
    assert "No communities found" in result.output


@patch("src.fileintel.cli.graphrag.api.graphrag_global_query")
def test_query_command_error_handling(self, mock_query, cli_runner):
    """Test query command error handling."""
    mock_query.side_effect = Exception("Service unavailable")

    result = cli_runner.invoke(
        graphrag_app, ["global-query", "test_collection", "What is this?"]
    )

    assert result.exit_code == 1
    assert "Error with global query" in result.output


def test_command_help(self, cli_runner):
    """Test that help text is available for commands."""
    result = cli_runner.invoke(graphrag_app, ["--help"])

    assert result.exit_code == 0
    assert "Manage GraphRAG operations and queries" in result.output
    assert "index" in result.output
    assert "status" in result.output
    assert "global-query" in result.output
    assert "local-query" in result.output
    assert "entities" in result.output
    assert "communities" in result.output


def test_index_command_help(self, cli_runner):
    """Test index command help."""
    result = cli_runner.invoke(graphrag_app, ["index", "--help"])

    assert result.exit_code == 0
    assert "Build GraphRAG index for a collection" in result.output
