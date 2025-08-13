from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from document_analyzer.batch_processing.job_manager import JobManager
from document_analyzer.core.config import settings
from document_analyzer.storage.base import StorageInterface
from document_analyzer.storage.postgresql_storage import PostgreSQLStorage
from ..dependencies import get_db
import os

router = APIRouter()

def get_storage(db: Session = Depends(get_db)) -> StorageInterface:
    return PostgreSQLStorage(db)

def get_job_manager(storage: StorageInterface = Depends(get_storage)) -> JobManager:
    return JobManager(storage=storage)

@router.post("/batch", status_code=202)
async def create_batch_job(job_manager: JobManager = Depends(get_job_manager)):
    """
    Submits a batch processing job using pre-configured directories.
    """
    input_dir = settings.get('batch_processing.directory_input', 'input')
    
    if not os.path.isdir(input_dir) or not os.listdir(input_dir):
        raise HTTPException(status_code=400, detail=f"Input directory '{input_dir}' is empty or does not exist.")

    job_data = {
        "input_dir": input_dir,
        "output_dir": settings.get('batch_processing.directory_output', 'output'),
        "output_format": settings.get('batch_processing.default_format', 'json')
    }
    job_id = job_manager.submit_batch_job(data=job_data)
    return {"message": "Batch processing job submitted.", "job_id": job_id}
