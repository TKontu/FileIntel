from fastapi import APIRouter, Depends, HTTPException
from ..models import AnalyzeRequest, AnalyzeResponse, JobStatusResponse, JobResultResponse
from ...batch_processing.job_manager import JobManager
from ...storage.redis_storage import RedisStorage
from ...storage.base import StorageInterface

router = APIRouter()

# In a real app, you'd use a more robust dependency injection system
# to manage the lifecycle of these objects.
def get_job_manager():
    return JobManager()

def get_storage():
    return RedisStorage()

@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_document(
    request: AnalyzeRequest,
    job_manager: JobManager = Depends(get_job_manager),
    storage: StorageInterface = Depends(get_storage),
):
    """
    Submits a document for analysis.
    """
    job_id = job_manager.submit_job(request.dict())
    storage.save_job({"id": job_id, "data": request.dict()})
    return {"job_id": job_id}

@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
def get_job_status(job_id: str, storage: RedisStorage = Depends(get_storage)):
    """
    Retrieves the status of a job.
    """
    status = storage.get_job_status(job_id)
    if status == "not_found":
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "status": status}

@router.get("/jobs/{job_id}/result", response_model=JobResultResponse)
def get_job_result(job_id: str, storage: RedisStorage = Depends(get_storage)):
    """
    Retrieves the result of a completed job.
    """
    status = storage.get_job_status(job_id)
    if status == "not_found":
        raise HTTPException(status_code=404, detail="Job not found")
    if status != "completed":
        raise HTTPException(status_code=400, detail=f"Job is not complete. Current status: {status}")

    result = storage.get_result(job_id)
    return {"job_id": job_id, "result": result}
