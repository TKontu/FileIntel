from abc import ABC, abstractmethod
import redis
from ..core.config import settings

class CacheInterface(ABC):
    @abstractmethod
    def get(self, key: str) -> str:
        pass

    @abstractmethod
    def set(self, key: str, value: str, ttl: int = None):
        pass

class RedisCache(CacheInterface):
    def __init__(self):
        self.redis = redis.Redis(
            host=settings.get("redis.host", "localhost"),
            port=settings.get("redis.port", 6379),
            db=0
        )
        self.ttl = settings.get("storage.cache_ttl", 3600)

    def get(self, key: str) -> str:
        return self.redis.get(key)

    def set(self, key: str, value: str, ttl: int = None):
        self.redis.set(key, value, ex=ttl or self.ttl)
