from ..storage.base import StorageInterface
from uuid import uuid4

class JobManager:
    def __init__(self, storage: StorageInterface):
        self.storage = storage

    def submit_job(self, document_id: str, data: dict) -> str:
        """
        Submits a new job and returns a job ID.
        """
        job_id = str(uuid4())
        job_data = {
            "id": job_id,
            "document_id": document_id,
            "status": "pending",
            "data": data
        }
        self.storage.save_job(job_data)
        return job_id

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

