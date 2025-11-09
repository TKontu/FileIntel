import asyncio
from cachetools import LRUCache
from typing import Any, Optional

import pandas as pd
import redis.asyncio as redis

from fileintel.core.config import Settings


class GraphRAGDataFrameCache:
    """A cache for GraphRAG DataFrames, with Redis and in-memory LRU fallback."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.redis_client = None
        self.cache_hits = 0
        self.cache_misses = 0

        # Estimate maxsize for LRU cache based on average dataframe size
        # This is a rough estimation and can be tuned
        avg_df_size_mb = 5  # Assuming an average of 5MB per DataFrame
        max_lru_size = settings.cache.max_size_mb // avg_df_size_mb
        self.lru_cache = LRUCache(maxsize=max_lru_size)

        if settings.cache.enabled and settings.cache.redis_host:
            try:
                self.redis_client = redis.from_url(
                    f"redis://{settings.cache.redis_host}:{settings.cache.redis_port}/{settings.cache.redis_db}",
                    decode_responses=True
                )
            except Exception as e:
                print(f"Could not connect to Redis: {e}")
                self.redis_client = None

    async def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get a DataFrame from the cache."""
        if key in self.lru_cache:
            self.cache_hits += 1
            return self.lru_cache[key]

        if self.redis_client:
            try:
                data = await self.redis_client.get(key)
                if data:
                    self.cache_hits += 1
                    # CRITICAL FIX: Run blocking pandas deserialization in thread pool for gevent compatibility
                    # pd.read_json() blocks for large DataFrames (e.g., 109MB community_clusters)
                    df = await asyncio.to_thread(pd.read_json, data)
                    self.lru_cache[key] = df
                    return df
            except Exception as e:
                print(f"Redis get failed: {e}")

        self.cache_misses += 1
        return None

    async def set(self, key: str, df: pd.DataFrame, ttl: int):
        """Set a DataFrame in the cache."""
        self.lru_cache[key] = df
        if self.redis_client:
            try:
                # CRITICAL FIX: Run blocking pandas serialization in thread pool for gevent compatibility
                # df.to_json() blocks for large DataFrames (e.g., 109MB community_clusters)
                json_data = await asyncio.to_thread(df.to_json)
                await self.redis_client.set(key, json_data, ex=ttl)
            except Exception as e:
                print(f"Redis set failed: {e}")

    async def clear(self, key: str):
        """Clear a key from the cache."""
        if key in self.lru_cache:
            del self.lru_cache[key]
        if self.redis_client:
            try:
                await self.redis_client.delete(key)
            except Exception as e:
                print(f"Redis delete failed: {e}")

    def clear_all(self):
        """Clear all entries from LRU cache to free memory."""
        self.lru_cache.clear()

    def get_cache_stats(self) -> dict:
        """Get cache statistics for monitoring."""
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0
            else 0,
            "cache_size": len(self.lru_cache),
            "max_cache_size": self.lru_cache.maxsize,
        }

    async def warmup_cache(self, storage, parquet_loader=None):
        """Pre-load frequently used collections into the cache."""
        import asyncio

        if not self.settings.cache.warmup_collections:
            return

        if parquet_loader is None:
            # Import here to avoid circular dependency
            from .parquet_loader import ParquetLoader

            parquet_loader = ParquetLoader(self)

        for collection_id in self.settings.cache.warmup_collections:
            try:
                # Run blocking storage call in thread pool to avoid blocking event loop
                loop = asyncio.get_event_loop()
                index_info = await loop.run_in_executor(
                    None, storage.get_graphrag_index_info, collection_id
                )

                if not index_info or not index_info.get("index_path"):
                    print(
                        f"Could not warm up cache for collection {collection_id}: index not found"
                    )
                    continue

                workspace_path = index_info["index_path"]
                await parquet_loader.load_parquet_files(workspace_path, collection_id)
                print(f"Warmed up cache for collection {collection_id}")
            except Exception as e:
                print(f"Could not warm up cache for collection {collection_id}: {e}")
