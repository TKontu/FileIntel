from .base import StorageInterface
from .models import Job, Result, Document
from sqlalchemy.orm import Session
import json
import uuid

class PostgreSQLStorage(StorageInterface):
    def __init__(self, db_session: Session):
        self.db = db_session

    def create_document(self, filename: str, content_hash: str, file_size: int, mime_type: str) -> Document:
        doc_id = str(uuid.uuid4())
        new_document = Document(
            id=doc_id,
            filename=filename,
            content_hash=content_hash,
            file_size=file_size,
            mime_type=mime_type,
        )
        self.db.add(new_document)
        self.db.commit()
        self.db.refresh(new_document)
        return new_document

    def get_document_by_hash(self, content_hash: str) -> Document:
        return self.db.query(Document).filter(Document.content_hash == content_hash).first()

    def save_job(self, job_data: dict) -> str:
        job_id = job_data.get("id")
        if not job_id:
            raise ValueError("Job data must contain an 'id'")

        new_job = Job(
            id=job_id,
            document_id=job_data.get("document_id"),
            status=job_data.get("status", "pending"),
            data=job_data.get("data")
        )
        self.db.add(new_job)
        self.db.commit()
        return job_id

    def get_job(self, job_id: str) -> Job:
        return self.db.query(Job).filter(Job.id == job_id).first()

    def get_job_by_document_id(self, document_id: str) -> Job:
        return self.db.query(Job).filter(Job.document_id == document_id).first()

    def get_pending_job(self) -> Job:
        return self.db.query(Job).filter(Job.status == "pending").first()

    def update_job_status(self, job_id: str, status: str):
        job = self.get_job(job_id)
        if job:
            job.status = status
            self.db.commit()

    def save_result(self, job_id: str, result_data: dict):
        result_json = json.dumps(result_data)
        new_result = Result(job_id=job_id, data=result_json)
        self.db.add(new_result)
        self.db.commit()

    def get_result(self, job_id: str) -> Result:
        return self.db.query(Result).filter(Result.job_id == job_id).first()
