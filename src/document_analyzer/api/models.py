from pydantic import BaseModel
from typing import List, Dict, Any

class AnalyzeRequest(BaseModel):
    file_path: str
    question: str

class AnalyzeResponse(BaseModel):
    job_id: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str

class JobResultResponse(BaseModel):
    job_id: str
    result: Dict[str, Any]

class URLAnalyzeRequest(BaseModel):
    url: str
    question: str

class BatchAnalyzeResponse(BaseModel):
    batch_id: str
    job_ids: List[str]
