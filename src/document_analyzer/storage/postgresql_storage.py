from .base import StorageInterface
from .models import Job, Result, Document, DocumentChunk, Collection
from sqlalchemy.orm import Session
import json
import uuid
from typing import List, Dict, Any


class PostgreSQLStorage(StorageInterface):
    def __init__(self, db_session: Session):
        self.db = db_session

    def create_collection(self, name: str) -> Collection:
        collection_id = str(uuid.uuid4())
        new_collection = Collection(id=collection_id, name=name)
        self.db.add(new_collection)
        self.db.commit()
        self.db.refresh(new_collection)
        return new_collection

    def get_collection(self, collection_id: str) -> Collection:
        return self.db.query(Collection).filter(Collection.id == collection_id).first()

    def get_all_collections(self) -> List[Collection]:
        return self.db.query(Collection).all()

    def delete_collection(self, collection_id: str):
        collection = self.get_collection(collection_id)
        if collection:
            self.db.delete(collection)
            self.db.commit()

    def create_document(
        self,
        filename: str,
        content_hash: str,
        file_size: int,
        mime_type: str,
        collection_id: str,
    ) -> Document:
        doc_id = str(uuid.uuid4())
        new_document = Document(
            id=doc_id,
            filename=filename,
            content_hash=content_hash,
            file_size=file_size,
            mime_type=mime_type,
            collection_id=collection_id,
        )
        self.db.add(new_document)
        self.db.commit()
        self.db.refresh(new_document)
        return new_document

    def get_document(self, document_id: str) -> Document:
        return self.db.query(Document).filter(Document.id == document_id).first()

    def get_document_by_hash(self, content_hash: str) -> Document:
        return (
            self.db.query(Document)
            .filter(Document.content_hash == content_hash)
            .first()
        )

    def delete_document(self, document_id: str):
        document = self.get_document(document_id)
        if document:
            self.db.delete(document)
            self.db.commit()

    def add_document_chunks(
        self, document_id: str, collection_id: str, chunks: List[Dict[str, Any]]
    ):
        for chunk_data in chunks:
            chunk_id = str(uuid.uuid4())
            new_chunk = DocumentChunk(
                id=chunk_id,
                document_id=document_id,
                collection_id=collection_id,
                chunk_text=chunk_data["text"],
                embedding=chunk_data["embedding"],
                chunk_metadata=chunk_data.get("metadata", {}),
            )
            self.db.add(new_chunk)
        self.db.commit()

    def find_relevant_chunks_in_collection(
        self, collection_id: str, query_embedding: List[float], top_k: int = 5
    ) -> List[DocumentChunk]:
        return (
            self.db.query(DocumentChunk)
            .filter(DocumentChunk.collection_id == collection_id)
            .order_by(DocumentChunk.embedding.l2_distance(query_embedding))
            .limit(top_k)
            .all()
        )

    def save_job(self, job_data: dict) -> str:
        job_id = job_data.get("id")
        if not job_id:
            raise ValueError("Job data must contain an 'id'")

        new_job = Job(
            id=job_id,
            document_id=job_data.get("document_id"),
            collection_id=job_data.get("collection_id"),
            job_type=job_data.get("job_type", "single_file"),
            status=job_data.get("status", "pending"),
            data=job_data.get("data"),
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
