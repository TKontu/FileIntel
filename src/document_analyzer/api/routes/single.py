from fastapi import APIRouter, Depends, UploadFile, File, Form
from pydantic import BaseModel
from sqlalchemy.orm import Session
from document_analyzer.batch_processing.job_manager import JobManager
from document_analyzer.core.config import settings
from document_analyzer.storage.base import StorageInterface
from document_analyzer.storage.postgresql_storage import PostgreSQLStorage
from ..dependencies import get_db
import shutil
from pathlib import Path

router = APIRouter()

def get_storage(db: Session = Depends(get_db)) -> StorageInterface:
    return PostgreSQLStorage(db)

def get_job_manager(storage: StorageInterface = Depends(get_storage)) -> JobManager:
    return JobManager(storage=storage)

@router.post("/single", status_code=202)
async def create_single_file_job(
    file: UploadFile = File(...),
    output_format: str = Form(settings.get('batch_processing.default_format', 'json')),
    job_manager: JobManager = Depends(get_job_manager)
):
    """
    Submits a single file processing job.
    """
    input_dir = Path(settings.get('batch_processing.directory_input', 'input'))
    input_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = input_dir / file.filename
    
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    job_data = {
        "file_path": str(file_path),
        "output_format": output_format
    }
    job_id = job_manager.submit_job(job_type='single_file', data=job_data)
    return {"message": "Single file processing job submitted.", "job_id": job_id}
