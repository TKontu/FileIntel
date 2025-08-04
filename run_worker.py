import time
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from document_analyzer.storage.postgresql_storage import PostgreSQLStorage
from document_analyzer.batch_processing.job_manager import JobManager
from document_analyzer.batch_processing.worker import Worker
from document_analyzer.storage.models import SessionLocal

def main():
    print("Worker starting...")
    db_session = SessionLocal()
    storage = PostgreSQLStorage(db_session)
    job_manager = JobManager(storage)
    worker = Worker(job_manager)

    print("Worker started. Waiting for jobs...")
    while True:
        try:
            job = job_manager.get_next_job()
            if job:
                print(f"Processing job: {job.id}")
                worker.process_job(job)
            else:
                # If no job is found, wait for a bit before checking again
                time.sleep(5)
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(10) # Wait longer after an error

if __name__ == "__main__":
    main()