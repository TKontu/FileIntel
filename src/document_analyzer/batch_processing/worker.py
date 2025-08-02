from document_analyzer.batch_processing.job_manager import JobManager
from document_analyzer.document_processing.unified_processor import UnifiedDocumentProcessor
from document_analyzer.llm_integration.base import LLMProvider
from document_analyzer.storage.base import StorageInterface

class Worker:
    def __init__(
        self,
        job_manager: JobManager,
        document_processor: UnifiedDocumentProcessor,
        llm_provider: LLMProvider,
        storage: StorageInterface,
    ):
        self.job_manager = job_manager
        self.document_processor = document_processor
        self.llm_provider = llm_provider
        self.storage = storage

    def run(self):
        """
        Continuously processes jobs from the job manager.
        """
        while True:
            job = self.job_manager.get_next_job()
            if job is None:
                break

            job_id = job["id"]
            self.storage.update_job_status(job_id, "running")

            try:
                # For now, we assume the job data contains the file path
                file_path = job["data"]["file_path"]
                processed_content = self.document_processor.process(file_path)
                response = self.llm_provider.get_response(processed_content)
                self.storage.save_result(job_id, {"response": response})
                self.storage.update_job_status(job_id, "completed")
            except Exception as e:
                self.storage.update_job_status(job_id, "failed")
                print(f"Error processing job {job_id}: {e}") # Replace with proper logging
            finally:
                self.job_manager.job_done()
