import redis
import json
from uuid import uuid4
from ..core.config import settings

class JobManager:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.storage.redis_host,
            port=settings.storage.redis_port,
            db=settings.storage.redis_db,
            decode_responses=True
        )
        self.queue_name = settings.storage.job_queue_name

    def submit_job(self, data: dict) -> str:
        """
        Submits a new job to the Redis queue and returns a job ID.
        """
        job_id = str(uuid4())
        job_data = {"id": job_id, "data": data}
        self.redis_client.lpush(self.queue_name, json.dumps(job_data))
        return job_id

    def get_next_job(self) -> dict:
        """
        Gets the next job from the Redis queue (blocking).
        """
        _, job_json = self.redis_client.brpop(self.queue_name)
        return json.loads(job_json)

    def job_done(self):
        """
        In a Redis list-based queue, this is a no-op,
        as brpop removes the item.
        """
        pass

