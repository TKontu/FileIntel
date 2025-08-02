from fastapi import APIRouter, Depends
from ..models import AnalyzeRequest, AnalyzeResponse
from ...batch_processing.job_manager import JobManager

router = APIRouter()

def get_job_manager():
    # This is a simple dependency injection for the JobManager.
    # In a real application, you might have a more sophisticated setup.
    return JobManager()

@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_document(
    request: AnalyzeRequest,
    job_manager: JobManager = Depends(get_job_manager),
):
    """
    Submits a document for analysis.
    """
    job_id = job_manager.submit_job(request.dict())
    return {"job_id": job_id}
