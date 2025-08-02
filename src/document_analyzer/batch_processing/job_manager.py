from queue import Queue
from uuid import uuid4

class JobManager:
    def __init__(self):
        self.queue = Queue()

    def submit_job(self, data: dict) -> str:
        """
        Submits a new job to the queue and returns a job ID.
        """
        job_id = str(uuid4())
        self.queue.put({"id": job_id, "data": data})
        return job_id

    def get_next_job(self) -> dict:
        """
        Gets the next job from the queue.
        """
        return self.queue.get()

    def job_done(self):
        """
        Marks a job as done.
        """
        self.queue.task_done()
