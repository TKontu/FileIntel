from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
    file_path: str
    user_question: str
    desired_format: str

class AnalyzeResponse(BaseModel):
    job_id: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str

class JobResultResponse(BaseModel):
    job_id: str
    result: dict
