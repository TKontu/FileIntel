import pytest
from fastapi.testclient import TestClient
from src.document_analyzer.api.main import app
import os


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_create_collection(client):
    response = client.post("/api/v1/collections", params={"name": "Test Collection"})
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["name"] == "Test Collection"


def test_list_collections(client):
    client.post("/api/v1/collections", params={"name": "Collection 1"})
    client.post("/api/v1/collections", params={"name": "Collection 2"})
    response = client.get("/api/v1/collections")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 2


def test_upload_document_and_query(client):
    # Create a collection
    response = client.post(
        "/api/v1/collections", params={"name": "E2E Test Collection"}
    )
    assert response.status_code == 201
    collection_id = response.json()["id"]

    # Create a dummy file
    dummy_file_path = "test_document.txt"
    with open(dummy_file_path, "w") as f:
        f.write("This is a test document for the RAG pipeline.")

    # Upload the document
    with open(dummy_file_path, "rb") as f:
        response = client.post(
            f"/api/v1/collections/{collection_id}/documents",
            files={"file": (dummy_file_path, f, "text/plain")},
        )
    assert response.status_code == 200
    document_id = response.json()["document_id"]
    assert document_id is not None

    # Clean up the dummy file
    os.remove(dummy_file_path)

    # Query the collection
    response = client.post(
        f"/api/v1/collections/{collection_id}/query",
        json={"question": "What is this document about?"},
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert job_id is not None
