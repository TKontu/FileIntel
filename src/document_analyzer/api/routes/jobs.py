import json
from fastapi import APIRouter, Depends, HTTPException
from ..dependencies import get_storage
from ...storage.base import StorageInterface
from ..models import JobStatusResponse, JobResultResponse
from ...output_management.factory import FormatterFactory
from fastapi.responses import PlainTextResponse

router = APIRouter()


@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
def get_job_status(job_id: str, storage: StorageInterface = Depends(get_storage)):
    job = storage.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        job_type=job.job_type,
        data=job.data,
        created_at=job.created_at,
    )


@router.get("/jobs/{job_id}/result", response_model=JobResultResponse)
def get_job_result(job_id: str, storage: StorageInterface = Depends(get_storage)):
    job = storage.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed. Current status: {job.status}",
        )
    result = storage.get_result(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found for this job")
    return JobResultResponse(job_id=job.id, result=json.loads(result.data))


@router.get("/jobs/{job_id}/result/markdown", response_class=PlainTextResponse)
def get_job_result_markdown(
    job_id: str, storage: StorageInterface = Depends(get_storage)
):
    job = storage.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed. Current status: {job.status}",
        )
    result = storage.get_result(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found for this job")

    formatter = FormatterFactory.get_formatter("markdown")
    markdown_result = formatter.format(json.loads(result.data))
    return PlainTextResponse(content=markdown_result)
