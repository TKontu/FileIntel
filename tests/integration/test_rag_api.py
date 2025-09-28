import pytest
from fastapi.testclient import TestClient
from src.fileintel.api.main import app
import os
import uuid

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="function")
def client():
    """Create a test client with unique collection names per test."""
    return TestClient(app)


def unique_collection_name(base_name: str) -> str:
    """Generate unique collection name to avoid conflicts."""
    return f"{base_name}_{uuid.uuid4().hex[:8]}"


def test_create_collection(client):
    collection_name = unique_collection_name("Test Collection")
    response = client.post("/api/v1/collections", json={"name": collection_name})
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["name"] == collection_name


def test_list_collections(client):
    collection1_name = unique_collection_name("Collection 1")
    collection2_name = unique_collection_name("Collection 2")

    client.post("/api/v1/collections", json={"name": collection1_name})
    client.post("/api/v1/collections", json={"name": collection2_name})
    response = client.get("/api/v1/collections")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 2


def test_upload_document_and_query(client):
    # Create a collection
    collection_name = unique_collection_name("E2E Test Collection")
    response = client.post("/api/v1/collections", json={"name": collection_name})
    assert response.status_code == 201
    collection_id = response.json()["id"]

    # Use real PDF from fixtures
    from pathlib import Path

    pdf_path = Path(__file__).parent.parent / "fixtures" / "test.pdf"

    # Upload the document
    with open(pdf_path, "rb") as f:
        response = client.post(
            f"/api/v1/collections/{collection_id}/documents",
            files={"file": ("test.pdf", f, "application/pdf")},
        )
    assert response.status_code == 200
    document_id = response.json()["document_id"]
    assert document_id is not None

    # Query the collection
    response = client.post(
        f"/api/v1/collections/{collection_id}/question",
        json={"question": "What is this document about?"},
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert job_id is not None
