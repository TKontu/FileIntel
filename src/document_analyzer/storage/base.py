from abc import ABC, abstractmethod
from typing import List, Dict
from .models import Document, Job, Result

class StorageInterface(ABC):
    @abstractmethod
    def create_document(self, filename: str, content_hash: str, file_size: int, mime_type: str) -> Document:
        pass

    @abstractmethod
    def get_document_by_hash(self, content_hash: str) -> Document:
        pass

    @abstractmethod
    def save_job(self, job_data: dict) -> str:
        pass

    @abstractmethod
    def get_job(self, job_id: str) -> Job:
        pass

    @abstractmethod
    def get_job_by_document_id(self, document_id: str) -> Job:
        pass

    @abstractmethod
    def get_pending_job(self) -> Job:
        pass

    @abstractmethod
    def update_job_status(self, job_id: str, status: str):
        pass

    @abstractmethod
    def save_result(self, job_id: str, result_data: dict):
        pass

    @abstractmethod
    def get_result(self, job_id: str) -> Result:
        pass

