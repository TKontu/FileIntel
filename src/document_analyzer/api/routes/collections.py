from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import List
from ..dependencies import get_storage
from ...storage.base import StorageInterface
from ..models import QueryRequest
from ...batch_processing.job_manager import JobManager
import hashlib
import os

router = APIRouter()


@router.post("/collections", status_code=201)
def create_collection(name: str, storage: StorageInterface = Depends(get_storage)):
    collection = storage.create_collection(name)
    return {"id": collection.id, "name": collection.name}


@router.get("/collections", response_model=List[dict])
def list_collections(storage: StorageInterface = Depends(get_storage)):
    collections = storage.get_all_collections()
    return [{"id": c.id, "name": c.name} for c in collections]


@router.get("/collections/{collection_id}")
def get_collection(
    collection_id: str, storage: StorageInterface = Depends(get_storage)
):
    collection = storage.get_collection(collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    return {
        "id": collection.id,
        "name": collection.name,
        "documents": [
            {"id": d.id, "filename": d.filename} for d in collection.documents
        ],
    }


@router.delete("/collections/{collection_id}", status_code=204)
def delete_collection(
    collection_id: str, storage: StorageInterface = Depends(get_storage)
):
    storage.delete_collection(collection_id)
    return


@router.post("/collections/{collection_id}/documents")
async def upload_document_to_collection(
    collection_id: str,
    file: UploadFile = File(...),
    storage: StorageInterface = Depends(get_storage),
):
    collection = storage.get_collection(collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    contents = await file.read()
    file_hash = hashlib.sha256(contents).hexdigest()

    # Check for duplicates
    existing_doc = storage.get_document_by_hash(file_hash)
    if existing_doc:
        return {"message": "Document already exists", "document_id": existing_doc.id}

    # Save file to disk
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)

    # Create document record
    document = storage.create_document(
        filename=file.filename,
        content_hash=file_hash,
        file_size=len(contents),
        mime_type=file.content_type,
        collection_id=collection_id,
    )

    # Create indexing job
    job_manager = JobManager(storage)
    job_data = {
        "file_path": file_path,
    }
    job_manager.submit_job(
        job_type="indexing",
        data=job_data,
        document_id=document.id,
        collection_id=collection_id,
    )

    return {
        "message": "Document uploaded and indexing job created",
        "document_id": document.id,
    }


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str, storage: StorageInterface = Depends(get_storage)):
    storage.delete_document(document_id)
    return


@router.post("/collections/{collection_id}/query")
def query_collection(
    collection_id: str,
    request: QueryRequest,
    storage: StorageInterface = Depends(get_storage),
):
    collection = storage.get_collection(collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    job_manager = JobManager(storage)
    job_data = {
        "question": request.question,
        "task_name": request.task_name,
    }
    job_id = job_manager.submit_job(
        job_type="query",
        data=job_data,
        collection_id=collection_id,
    )

    return {"job_id": job_id}
