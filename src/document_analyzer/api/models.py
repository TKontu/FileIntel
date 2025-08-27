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


class DocumentMetadataUpdate(BaseModel):
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    publication_date: Optional[str] = None
    publisher: Optional[str] = None
    doi: Optional[str] = None
    source_url: Optional[str] = None
    language: Optional[str] = None
    document_type: Optional[str] = None
    keywords: Optional[List[str]] = None
    abstract: Optional[str] = None
    harvard_citation: Optional[str] = None
