from ..storage.base import StorageInterface
from uuid import uuid4


class JobManager:
    def __init__(self, storage: StorageInterface):
        self.storage = storage

    def submit_job(
        self,
        job_type: str,
        data: dict,
        document_id: str = None,
        collection_id: str = None,
    ) -> str:
        """
        Submits a new job and returns a job ID.
        """
        job = self.storage.create_job(
            job_type=job_type,
            data=data,
            document_id=document_id,
            collection_id=collection_id,
        )
        return job.id

    def submit_file_job(self, document_id: str, data: dict) -> str:
        """
        Submits a new single-file job and returns a job ID.
        """
        return self.submit_job(
            job_type="single_file", data=data, document_id=document_id
        )

    def submit_batch_job(self, data: dict) -> str:
        """
        Submits a new batch job and returns a job ID.
        """
        return self.submit_job(job_type="batch", data=data)

    def get_next_job(self) -> dict:
        """
        Gets the next pending job from storage.
        """
        # This is a simplified implementation. A real implementation would
        # have a more robust way of finding and locking the next pending job.
        return self.storage.get_pending_job()

    def update_job_status(self, job_id: str, status: str):
        """
        Updates the status of a job.
        """
        self.storage.update_job_status(job_id, status)
