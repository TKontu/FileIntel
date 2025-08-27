from abc import ABC, abstractmethod
from typing import List, Dict
from .models import Document, Job, Result, Collection


class StorageInterface(ABC):
    @abstractmethod
    def create_collection(self, name: str) -> Collection:
        pass

    @abstractmethod
    def get_collection(self, collection_id: str) -> Collection:
        pass

    @abstractmethod
    def get_collection_by_name(self, name: str) -> Collection:
        pass

    @abstractmethod
    def get_all_collections(self) -> List[Collection]:
        pass

    @abstractmethod
    def delete_collection(self, collection_id: str):
        pass

    @abstractmethod
    def create_document(
        self,
        filename: str,
        content_hash: str,
        file_size: int,
        mime_type: str,
        collection_id: str,
    ) -> Document:
        pass

    @abstractmethod
    def get_document(self, document_id: str) -> Document:
        pass

    @abstractmethod
    def get_document_by_hash(self, content_hash: str) -> Document:
        pass

    @abstractmethod
    def get_document_by_hash_and_collection(
        self, content_hash: str, collection_id: str
    ) -> Document:
        pass

    @abstractmethod
    def get_document_by_filename_and_collection(
        self, filename: str, collection_id: str
    ) -> Document:
        pass

    @abstractmethod
    def delete_document(self, document_id: str):
        pass

    @abstractmethod
    def create_job(
        self,
        job_type: str,
        data: Dict,
        document_id: str = None,
        collection_id: str = None,
    ) -> Job:
        pass

    @abstractmethod
    def get_job(self, job_id: str) -> Job:
        pass

    @abstractmethod
    def get_pending_job(self, job_type: str = None) -> Job:
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
