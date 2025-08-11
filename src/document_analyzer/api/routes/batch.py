from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from document_analyzer.batch_processing.batch_manager import BatchProcessor
from document_analyzer.core.config import settings
import os

router = APIRouter()

from document_analyzer.core.config import settings
class BatchRequest(BaseModel):
    input_dir: str = settings.get('batch_processing.directory_input', 'input')
    output_dir: str = settings.get('batch_processing.directory_output', 'output')
    output_format: str = settings.get('batch_processing.default_format', 'json')

def run_batch_processing(input_dir: str, output_dir: str, output_format: str):
    processor = BatchProcessor()
    processor.process_files(input_dir, output_dir, output_format)

@router.post("/batch", status_code=202)
async def create_batch_job(request: BatchRequest, background_tasks: BackgroundTasks):
    """
    Starts a batch processing job.
    """
    # Security check: Ensure directories are within the app's scope
    base_dir = "/home/appuser/app"
    input_dir = os.path.abspath(os.path.join(base_dir, request.input_dir))
    output_dir = os.path.abspath(os.path.join(base_dir, request.output_dir))

    if not input_dir.startswith(base_dir) or not output_dir.startswith(base_dir):
        raise HTTPException(status_code=400, detail="Invalid directory path.")

    if not os.path.isdir(input_dir):
        raise HTTPException(status_code=400, detail=f"Input directory does not exist: {input_dir}")

    background_tasks.add_task(run_batch_processing, input_dir, output_dir, request.output_format)
    return {"message": "Batch processing job started."}