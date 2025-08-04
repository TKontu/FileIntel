from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
import hashlib
import uuid

from ..dependencies import get_db
from ...storage.postgresql_storage import PostgreSQLStorage
from ...batch_processing.job_manager import JobManager
from ..models import BatchAnalyzeResponse

router = APIRouter()

@router.post("/analyze/batch", response_model=BatchAnalyzeResponse)
async def analyze_batch(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """
    Submits a batch of documents for analysis.
    """
    storage = PostgreSQLStorage(db)
    job_manager = JobManager(storage)
    
    batch_id = str(uuid.uuid4())
    # A more robust implementation would create a Batch record in the database
    # storage.create_batch(batch_id=batch_id, status="pending")

    job_ids = []
    for file in files:
        content = await file.read()
        content_hash = hashlib.sha256(content).hexdigest()

        document = storage.create_document(
            filename=file.filename,
            content_hash=content_hash,
            file_size=len(content),
            mime_type=file.content_type,
        )

        job_id = job_manager.submit_job(
            document_id=document.id,
            data={"filename": file.filename} # Add any other job-specific data here
        )
        job_ids.append(job_id)

    return {"batch_id": batch_id, "job_ids": job_ids}
