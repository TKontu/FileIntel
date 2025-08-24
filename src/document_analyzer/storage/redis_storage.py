import redis
import json
from .base import StorageInterface
from ..core.config import settings


class RedisStorage(StorageInterface):
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.storage.redis_host,
            port=settings.storage.redis_port,
            db=settings.storage.redis_db,
            decode_responses=True,
        )

    def _get_job_key(self, job_id: str) -> str:
        return f"job:{job_id}"

    def save_job(self, job_data: dict) -> str:
        job_id = job_data["id"]
        key = self._get_job_key(job_id)
        # Store job details in a Redis hash
        self.redis_client.hset(
            key, mapping={"status": "pending", "data": json.dumps(job_data)}
        )
        return job_id

    def get_job(self, job_id: str) -> dict:
        key = self._get_job_key(job_id)
        job_data = self.redis_client.hgetall(key)
        if job_data and "data" in job_data:
            return json.loads(job_data["data"])
        return {}

    def update_job_status(self, job_id: str, status: str):
        key = self._get_job_key(job_id)
        self.redis_client.hset(key, "status", status)

    def save_result(self, job_id: str, result_data: dict):
        key = self._get_job_key(job_id)
        self.redis_client.hset(key, "result", json.dumps(result_data))

    def get_result(self, job_id: str) -> dict:
        key = self._get_job_key(job_id)
        result = self.redis_client.hget(key, "result")
        if result:
            return json.loads(result)
        return {}

    def get_job_status(self, job_id: str) -> str:
        key = self._get_job_key(job_id)
        status = self.redis_client.hget(key, "status")
        return status if status else "not_found"
