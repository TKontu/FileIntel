"""
Simple Redis cache client for FileIntel.

Replaces over-engineered cache_manager.py with basic Redis operations.
"""

import json
import logging
from typing import Any, Optional
import redis
from ..core.config import get_config

logger = logging.getLogger(__name__)

# Constants for cache configuration
DEFAULT_TTL_SECONDS = 3600  # 1 hour default TTL
QUERY_CACHE_TTL_SECONDS = 1800  # 30 minutes for query results
EMBEDDING_CACHE_TTL_SECONDS = 86400  # 24 hours for embeddings


class SimpleCache:
    """
    Simple Redis cache client focused on essential operations.

    Eliminates unnecessary abstractions and complexity from cache_manager.py.
    """

    def __init__(self):
        config = get_config()
        redis_config = config.rag.cache

        self.client = redis.Redis(
            host=redis_config.redis_host,
            port=redis_config.redis_port,
            db=redis_config.redis_db,
            password=getattr(redis_config, "redis_password", None),
            decode_responses=True,
        )

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            value = self.client.get(key)
            if value is None:
                return None
            return json.loads(value)
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.warning(f"Cache get failed for key {key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = DEFAULT_TTL_SECONDS) -> bool:
        """Set value in cache with TTL."""
        try:
            serialized = json.dumps(value)
            return self.client.setex(key, ttl, serialized)
        except (redis.RedisError, TypeError) as e:
            logger.warning(f"Cache set failed for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            return bool(self.client.delete(key))
        except redis.RedisError as e:
            logger.warning(f"Cache delete failed for key {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            return bool(self.client.exists(key))
        except redis.RedisError as e:
            logger.warning(f"Cache exists check failed for key {key}: {e}")
            return False

    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except redis.RedisError as e:
            logger.warning(f"Cache clear pattern failed for {pattern}: {e}")
            return 0

    def ping(self) -> bool:
        """Check if Redis is available."""
        try:
            return self.client.ping()
        except redis.RedisError:
            return False


# Global cache instance
_cache_instance = None


def get_cache() -> SimpleCache:
    """Get the cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SimpleCache()
    return _cache_instance


# Wrapper functions removed - use cache = get_cache() and call methods directly
# This eliminates unnecessary abstraction layers and improves code clarity
