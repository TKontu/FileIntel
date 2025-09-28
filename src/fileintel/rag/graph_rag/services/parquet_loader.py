import asyncio
import os
from datetime import datetime
from typing import Dict, Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .dataframe_cache import GraphRAGDataFrameCache


class ParquetLoader:
    """A loader for Parquet files, with caching."""

    def __init__(self, cache: "GraphRAGDataFrameCache"):
        self.cache = cache

    async def load_parquet_files(
        self, workspace_path: str, collection_id: str
    ) -> Dict[str, pd.DataFrame]:
        """Load all necessary parquet files for a collection."""
        files_to_load = {
            "entities": "output/entities.parquet",
            "communities": "output/communities.parquet",
            "community_reports": "output/community_reports.parquet",
        }

        dataframes = {}

        for name, path in files_to_load.items():
            full_path = os.path.join(workspace_path, path)
            cache_key = await self._get_cache_key(collection_id, full_path, name)

            df = await self.cache.get(cache_key)
            if df is not None:
                dataframes[name] = df
                continue

            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Parquet file not found: {full_path}")

            df = pd.read_parquet(full_path)
            await self.cache.set(
                cache_key, df, ttl=self.cache.settings.rag.cache.ttl_seconds
            )
            dataframes[name] = df

        return dataframes

    async def _get_cache_key(
        self, collection_id: str, file_path: str, file_type: str
    ) -> str:
        """Generate a cache key based on collection_id and file modification time."""
        try:
            mod_time = os.path.getmtime(file_path)
            mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y%m%d%H%M%S")
            return f"graphrag:{collection_id}:{file_type}:{mod_time_str}"
        except OSError:
            return f"graphrag:{collection_id}:{file_type}:no-file"
