from abc import ABC, abstractmethod

class StorageInterface(ABC):
    @abstractmethod
    def save_job(self, job_data: dict) -> str:
        """Save a new job and return its ID."""
        pass

    @abstractmethod
    def get_job(self, job_id: str) -> dict:
        """Retrieve a job by its ID."""
        pass

    @abstractmethod
    def update_job_status(self, job_id: str, status: str):
        """Update the status of a job."""
        pass

    @abstractmethod
    def save_result(self, job_id: str, result_data: dict):
        """Save the result of a job."""
        pass

    @abstractmethod
    def get_result(self, job_id: str) -> dict:
        """Retrieve the result of a job."""
        pass
