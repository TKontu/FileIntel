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
        """Load all necessary parquet files for a collection.

        Note: workspace_path is expected to be the output directory already
        (e.g., /data/graphrag_indices/<collection_id>/output), so files are
        referenced directly without additional 'output/' prefix.
        """
        files_to_load = {
            "entities": "entities.parquet",
            "communities": "communities.parquet",
            "community_reports": "community_reports.parquet",
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

            # Use asyncio.to_thread to prevent blocking the event loop
            df = await asyncio.to_thread(pd.read_parquet, full_path)
            await self.cache.set(
                cache_key, df, ttl=self.cache.settings.cache.ttl_seconds
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
