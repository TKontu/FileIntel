import json
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session
from src.document_analyzer.output_management.formatters.markdown import MarkdownFormatter
from ..models import AnalyzeResponse, JobStatusResponse, JobResultResponse
from ...batch_processing.job_manager import JobManager
from ...storage.postgresql_storage import PostgreSQLStorage
from ...storage.base import StorageInterface
from ...api.dependencies import get_db
import hashlib
import uuid
import os

router = APIRouter()

def get_storage(db: Session = Depends(get_db)) -> StorageInterface:
    return PostgreSQLStorage(db)

def get_job_manager(storage: StorageInterface = Depends(get_storage)) -> JobManager:
    return JobManager(storage=storage)

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_document(
    file: UploadFile = File(...),
    job_manager: JobManager = Depends(get_job_manager),
    storage: StorageInterface = Depends(get_storage),
):
    """
    Submits a document for analysis. If the document has been analyzed before,
    it returns the existing job ID.
    """
    # Read file content and calculate hash
    content = await file.read()
    content_hash = hashlib.sha256(content).hexdigest()

    # Check if document already exists
    existing_document = storage.get_document_by_hash(content_hash)
    if existing_document:
        # If it exists, find the associated job
        existing_job = storage.get_job_by_document_id(existing_document.id)
        if existing_job:
            return {"job_id": existing_job.id}

    # Save the file to the uploads directory
    file_path = os.path.join("uploads", f"{content_hash}_{file.filename}")
    with open(file_path, "wb") as f:
        f.write(content)

    # Create new document record
    document = storage.create_document(
        filename=file.filename,
        content_hash=content_hash,
        file_size=len(content),
        mime_type=file.content_type,
    )

    # Submit new job
    job_id = job_manager.submit_file_job(document_id=document.id, data={"file_path": str(file_path)})
    return {"job_id": job_id}

@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str, storage: StorageInterface = Depends(get_storage)):
    """
    Retrieves the full details and status of a job.
    """
    job = storage.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job.id, "status": job.status, "job_type": job.job_type, "data": job.data, "created_at": job.created_at}

@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
def get_job_status(job_id: str, storage: StorageInterface = Depends(get_storage)):
    """
    Retrieves the status of a job.
    """
    job = storage.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job.id, "status": job.status}

@router.get("/jobs/{job_id}/result", response_model=JobResultResponse)
def get_job_result(job_id: str, storage: StorageInterface = Depends(get_storage)):
    """

    Retrieves the result of a completed job.
    """
    job = storage.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job is not complete. Current status: {job.status}")

    result = storage.get_result(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return {"job_id": job_id, "result": json.loads(result.data)}

@router.get("/jobs/{job_id}/result/markdown", response_class=PlainTextResponse)
def get_job_result_markdown(job_id: str, storage: StorageInterface = Depends(get_storage)):
    """
    Retrieves the result of a completed job in markdown format.
    """
    job = storage.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job is not complete. Current status: {job.status}")

    result = storage.get_result(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")

    formatter = MarkdownFormatter()
    formatted_result = formatter.format(json.loads(result.data))
    
    return PlainTextResponse(content=formatted_result, media_type="text/markdown")