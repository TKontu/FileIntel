from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import List
from ..dependencies import (
    get_storage,
    get_collection_by_id_or_name,
    get_document_by_id_or_filename,
    get_api_key,
)
from ...storage.base import StorageInterface
from ...storage.models import Collection, Document
from ..models import QueryRequest, AnalysisRequest, DocumentMetadataUpdate
from ...worker.job_manager import JobManager
from ...core.config import get_config
import hashlib
import mimetypes
import os
import aiofiles
import uuid
import re

router = APIRouter(dependencies=[Depends(get_api_key)])


@router.post("/collections", status_code=201)
def create_collection(
    name: str = Form(...), storage: StorageInterface = Depends(get_storage)
):
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
            {
                "id": d.id,
                "filename": d.filename,
                "original_filename": d.original_filename,
            }
            for d in collection.documents
        ],
    }


@router.delete("/collections/{collection_identifier}", status_code=204)
def delete_collection(
    collection: Collection = Depends(get_collection_by_id_or_name),
    storage: StorageInterface = Depends(get_storage),
):
    storage.delete_collection(collection.id)
    return


def _parse_file_size(size_str: str) -> int:
    """Convert file size string like '100MB' to bytes"""
    match = re.match(r"^(\d+(?:\.\d+)?)\s*([KMGT]?B)$", size_str.upper())
    if not match:
        raise ValueError(f"Invalid file size format: {size_str}")

    size, unit = match.groups()
    size = float(size)

    multipliers = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
    }

    return int(size * multipliers[unit])


@router.post("/collections/{collection_identifier}/documents/batch")
async def upload_documents_batch_to_collection(
    files: List[UploadFile] = File(...),
    collection: Collection = Depends(get_collection_by_id_or_name),
    storage: StorageInterface = Depends(get_storage),
):
    """Upload multiple documents to a collection in batch."""
    config = get_config()
    max_file_size = _parse_file_size(config.document_processing.max_file_size)

    upload_dir = config.paths.uploads
    os.makedirs(upload_dir, exist_ok=True)

    processed_files = []
    skipped_files = []

    for file in files:
        try:
            # Generate secure filename
            original_filename = file.filename
            file_extension = os.path.splitext(original_filename)[1]
            secure_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = os.path.join(upload_dir, secure_filename)

            # Stream file to disk and calculate hash
            file_hash = hashlib.sha256()
            file_size = 0

            async with aiofiles.open(file_path, "wb") as f:
                while contents := await file.read(1024 * 1024):
                    file_size += len(contents)

                    if file_size > max_file_size:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        skipped_files.append(
                            {
                                "filename": original_filename,
                                "reason": f"File too large. Maximum size is {config.document_processing.max_file_size}",
                            }
                        )
                        break

                    await f.write(contents)
                    file_hash.update(contents)

            # Skip if file was too large
            if file_size > max_file_size:
                continue

            content_hash_hex = file_hash.hexdigest()

            # Check for duplicates
            existing_doc = storage.get_document_by_hash_and_collection(
                content_hash_hex, collection.id
            )
            if existing_doc:
                os.remove(file_path)
                skipped_files.append(
                    {
                        "filename": original_filename,
                        "reason": f"Document with same content already exists with ID: {existing_doc.id}",
                    }
                )
                continue

            # Create database record
            mime_type, _ = mimetypes.guess_type(original_filename)
            if mime_type is None:
                mime_type = "application/octet-stream"

            document = storage.create_document(
                filename=secure_filename,
                original_filename=original_filename,
                content_hash=content_hash_hex,
                file_size=file_size,
                mime_type=mime_type,
                collection_id=collection.id,
                document_metadata={},
            )

            processed_files.append(
                {
                    "document_id": document.id,
                    "filename": original_filename,
                    "file_path": file_path,
                }
            )

        except Exception as e:
            # Clean up file if it exists
            if "file_path" in locals() and os.path.exists(file_path):
                os.remove(file_path)
            skipped_files.append(
                {"filename": original_filename, "reason": f"Processing error: {str(e)}"}
            )

    # Create batch indexing job if we have files to process
    if processed_files:
        job_manager = JobManager(storage)
        job_data = {"files": processed_files, "collection_id": collection.id}
        job_id = job_manager.submit_job(
            job_type="batch_indexing",
            data=job_data,
            collection_id=collection.id,
        )

        return {
            "message": f"Batch upload completed. {len(processed_files)} files processed, {len(skipped_files)} skipped",
            "job_id": job_id,
            "processed_count": len(processed_files),
            "skipped_count": len(skipped_files),
            "processed_files": [
                {"document_id": f["document_id"], "filename": f["filename"]}
                for f in processed_files
            ],
            "skipped_files": skipped_files,
        }
    else:
        return {
            "message": "No files were processed successfully",
            "job_id": None,
            "processed_count": 0,
            "skipped_count": len(skipped_files),
            "processed_files": [],
            "skipped_files": skipped_files,
        }


@router.post("/collections/{collection_identifier}/documents")
async def upload_document_to_collection(
    file: UploadFile = File(...),
    collection: Collection = Depends(get_collection_by_id_or_name),
    storage: StorageInterface = Depends(get_storage),
):
    config = get_config()
    max_file_size = _parse_file_size(config.document_processing.max_file_size)

    # --- Secure File Handling ---
    upload_dir = config.paths.uploads
    os.makedirs(upload_dir, exist_ok=True)

    # Generate a secure, unique filename
    original_filename = file.filename
    file_extension = os.path.splitext(original_filename)[1]
    secure_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(upload_dir, secure_filename)

    # --- Stream to Disk and Calculate Hash Simultaneously ---
    file_hash = hashlib.sha256()
    file_size = 0
    try:
        async with aiofiles.open(file_path, "wb") as f:
            while contents := await file.read(1024 * 1024):  # Read in 1MB chunks
                file_size += len(contents)

                # Check file size limit during streaming
                if file_size > max_file_size:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size is {config.document_processing.max_file_size}",
                    )

                await f.write(contents)
                file_hash.update(contents)
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        # Clean up partially written file if something goes wrong
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    content_hash_hex = file_hash.hexdigest()

    # --- Check for Duplicates ---
    existing_doc = storage.get_document_by_hash_and_collection(
        content_hash_hex, collection.id
    )
    if existing_doc:
        os.remove(file_path)  # Clean up the newly uploaded file
        raise HTTPException(
            status_code=409,
            detail=f"Document with same content already exists in this collection with ID: {existing_doc.id}",
        )

    # --- Create Database Record ---
    mime_type, _ = mimetypes.guess_type(original_filename)
    if mime_type is None:
        mime_type = "application/octet-stream"

    document = storage.create_document(
        filename=secure_filename,
        original_filename=original_filename,
        content_hash=content_hash_hex,
        file_size=file_size,
        mime_type=mime_type,
        collection_id=collection.id,
        document_metadata={},
    )

    # --- Create Indexing Job ---
    job_manager = JobManager(storage)
    job_data = {"file_path": file_path}
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
        {
            "id": doc.id,
            "filename": doc.filename,
            "original_filename": doc.original_filename,
            "mime_type": doc.mime_type,
        }
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


@router.post("/collections/{collection_identifier}/analyze")
def analyze_collection(
    request: AnalysisRequest,
    collection: Collection = Depends(get_collection_by_id_or_name),
    storage: StorageInterface = Depends(get_storage),
):
    job_manager = JobManager(storage)
    job_data = {
        "task_name": request.task_name,
    }
    job_id = job_manager.submit_job(
        job_type="analysis",
        data=job_data,
        collection_id=collection.id,
    )

    return {"job_id": job_id}


@router.post(
    "/collections/{collection_identifier}/documents/{document_identifier}/query"
)
def query_document_in_collection(
    request: QueryRequest,
    document: Document = Depends(get_document_by_id_or_filename),
    storage: StorageInterface = Depends(get_storage),
):
    job_manager = JobManager(storage)
    job_data = {
        "question": request.question,
        "task_name": request.task_name,
    }
    job_id = job_manager.submit_job(
        job_type="document_query",
        data=job_data,
        document_id=document.id,
        collection_id=document.collection_id,
    )
    return {"job_id": job_id}


@router.post(
    "/collections/{collection_identifier}/documents/{document_identifier}/analyze"
)
def analyze_document_in_collection(
    request: AnalysisRequest,
    document: Document = Depends(get_document_by_id_or_filename),
    storage: StorageInterface = Depends(get_storage),
):
    job_manager = JobManager(storage)
    job_data = {
        "task_name": request.task_name,
    }
    job_id = job_manager.submit_job(
        job_type="document_analysis",
        data=job_data,
        document_id=document.id,
        collection_id=document.collection_id,
    )
    return {"job_id": job_id}


@router.get("/documents/{document_id}")
def get_document_details(
    document_id: str, storage: StorageInterface = Depends(get_storage)
):
    """Get detailed information about a document including its metadata."""
    document = storage.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "id": document.id,
        "collection_id": document.collection_id,
        "filename": document.filename,
        "original_filename": document.original_filename,
        "content_hash": document.content_hash,
        "file_size": document.file_size,
        "mime_type": document.mime_type,
        "document_metadata": document.document_metadata,
        "created_at": document.created_at,
        "updated_at": document.updated_at,
    }


@router.put("/documents/{document_id}/metadata")
def update_document_metadata(
    document_id: str,
    metadata_update: DocumentMetadataUpdate,
    storage: StorageInterface = Depends(get_storage),
):
    """Manually update metadata for a document."""
    document = storage.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Get current metadata
    current_metadata = document.document_metadata or {}

    # Update only the fields provided in the request
    update_data = metadata_update.dict(exclude_unset=True)

    # Filter out None values
    update_data = {k: v for k, v in update_data.items() if v is not None}

    # Merge with existing metadata
    updated_metadata = current_metadata.copy()
    updated_metadata.update(update_data)

    # Mark as manually updated
    updated_metadata["manually_updated"] = True
    updated_metadata["manual_update_fields"] = list(update_data.keys())

    # Update in storage
    storage.update_document_metadata(document_id, updated_metadata)

    return {
        "message": "Metadata updated successfully",
        "updated_fields": list(update_data.keys()),
        "document_id": document_id,
    }
