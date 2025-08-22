from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import List
from ..dependencies import (
    get_storage,
    get_collection_by_id_or_name,
    get_document_by_id_or_filename,
)
from ...storage.base import StorageInterface
from ...storage.models import Collection, Document
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


@router.get("/collections/{collection_identifier}")
def get_collection(collection: Collection = Depends(get_collection_by_id_or_name)):
    return {
        "id": collection.id,
        "name": collection.name,
        "documents": [
            {"id": d.id, "filename": d.filename} for d in collection.documents
        ],
    }


@router.delete("/collections/{collection_identifier}", status_code=204)
def delete_collection(
    collection: Collection = Depends(get_collection_by_id_or_name),
    storage: StorageInterface = Depends(get_storage),
):
    storage.delete_collection(collection.id)
    return


@router.post("/collections/{collection_identifier}/documents")
async def upload_document_to_collection(
    file: UploadFile = File(...),
    collection: Collection = Depends(get_collection_by_id_or_name),
    storage: StorageInterface = Depends(get_storage),
):
    contents = await file.read()
    file_hash = hashlib.sha256(contents).hexdigest()

    # Check for duplicates in the specific collection
    existing_doc = storage.get_document_by_hash_and_collection(file_hash, collection.id)
    if existing_doc:
        raise HTTPException(
            status_code=409,
            detail=f"Document with same content already exists in this collection with ID: {existing_doc.id}",
        )

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
        collection_id=collection.id,
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
        collection_id=collection.id,
    )

    return {
        "message": "Document uploaded and indexing job created",
        "document_id": document.id,
    }


@router.get("/collections/{collection_identifier}/documents")
def list_documents_in_collection(
    collection: Collection = Depends(get_collection_by_id_or_name),
):
    return [
        {"id": doc.id, "filename": doc.filename, "mime_type": doc.mime_type}
        for doc in collection.documents
    ]


@router.delete(
    "/collections/{collection_identifier}/documents/{document_identifier}",
    status_code=204,
)
def delete_document_from_collection(
    document: Document = Depends(get_document_by_id_or_filename),
    storage: StorageInterface = Depends(get_storage),
):
    storage.delete_document(document.id)
    return


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str, storage: StorageInterface = Depends(get_storage)):
    storage.delete_document(document_id)
    return


@router.post("/collections/{collection_identifier}/query")
def query_collection(
    request: QueryRequest,
    collection: Collection = Depends(get_collection_by_id_or_name),
    storage: StorageInterface = Depends(get_storage),
):
    job_manager = JobManager(storage)
    job_data = {
        "question": request.question,
        "task_name": request.task_name,
    }
    job_id = job_manager.submit_job(
        job_type="query",
        data=job_data,
        collection_id=collection.id,
    )

    return {"job_id": job_id}
