from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from ..models import AnalyzeResponse, JobStatusResponse, JobResultResponse
from ...batch_processing.job_manager import JobManager
from ...storage.postgresql_storage import PostgreSQLStorage
from ...storage.base import StorageInterface
from ...api.dependencies import get_db
import hashlib
import uuid

router = APIRouter()

def get_job_manager():
    return JobManager()

def get_storage(db: Session = Depends(get_db)) -> StorageInterface:
    return PostgreSQLStorage(db)

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_document(
    file: UploadFile = File(...),
    job_manager: JobManager = Depends(get_job_manager),
    storage: StorageInterface = Depends(get_storage),
):
    """
    Submits a document for analysis.
    """
    # Read file content and calculate hash
    content = await file.read()
    content_hash = hashlib.sha256(content).hexdigest()

    # Create document record
    document = storage.create_document(
        filename=file.filename,
        content_hash=content_hash,
        file_size=len(content),
        mime_type=file.content_type,
    )

    # Submit job
    job_id = str(uuid.uuid4())
    job_manager.submit_job({"document_id": document.id, "job_id": job_id})
    storage.save_job({
        "id": job_id,
        "document_id": document.id,
        "status": "pending",
    })
    return {"job_id": job_id}

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
    return {"job_id": job_id, "result": result.data}