from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime


class AnalyzeRequest(BaseModel):
    file_path: str
    question: str


class AnalyzeResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    job_type: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None


class JobResultResponse(BaseModel):
    job_id: str
    result: Dict[str, Any]


class URLAnalyzeRequest(BaseModel):
    url: str
    question: str


class BatchAnalyzeResponse(BaseModel):
    batch_id: str
    job_ids: List[str]


class BatchJobCreate(BaseModel):
    task_name: str = "default_analysis"


class BatchJobResponse(BaseModel):
    message: str
    job_id: str


class QueryRequest(BaseModel):
    question: str
    task_name: str = "default_analysis"


class AnalysisRequest(BaseModel):
    task_name: str = "default_analysis"
