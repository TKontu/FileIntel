"""
MinerU output caching system for fingerprint-based content reuse.

Enables:
- Reusing MinerU processing results for identical files
- Avoiding redundant API calls for duplicate content
- Faster reprocessing with different chunking/embedding settings
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import zipfile
import io

from .elements import TextElement

logger = logging.getLogger(__name__)


class MinerUCache:
    """
    Manages cached MinerU outputs keyed by content fingerprint.

    Directory structure:
        output_directory/
            {fingerprint}/
                {fingerprint}.zip
                {fingerprint}.md
                {fingerprint}_content_list.json
                {fingerprint}_model.json
                {fingerprint}_middle.json
                images/...
    """

    def __init__(self, output_directory: str):
        """
        Initialize cache manager.

        Args:
            output_directory: Base directory for MinerU outputs
        """
        self.output_directory = Path(output_directory)
        # Create directory if it doesn't exist
        self.output_directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Initialized MinerU cache at {self.output_directory}")

    def get_cache_dir(self, fingerprint: str) -> Path:
        """Get cache directory path for fingerprint."""
        return self.output_directory / fingerprint

    def has_cache(self, fingerprint: str) -> bool:
        """
        Check if MinerU output exists in cache for this fingerprint.

        Args:
            fingerprint: Content fingerprint UUID

        Returns:
            True if cached output exists, False otherwise
        """
        cache_dir = self.get_cache_dir(fingerprint)

        # Check for essential files (at minimum, need content_list or markdown)
        content_list_exists = (cache_dir / f"{fingerprint}_content_list.json").exists()
        markdown_exists = (cache_dir / f"{fingerprint}.md").exists()
        zip_exists = (cache_dir / f"{fingerprint}.zip").exists()

        has_cache = content_list_exists or markdown_exists or zip_exists

        if has_cache:
            logger.info(f"MinerU cache HIT for fingerprint {fingerprint}")
        else:
            logger.debug(f"MinerU cache MISS for fingerprint {fingerprint}")

        return has_cache

    def load_cached_output(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        """
        Load cached MinerU output from disk.

        Args:
            fingerprint: Content fingerprint UUID

        Returns:
            Dict with 'response_type' and cached data, or None if cache invalid

        Structure:
            {
                'response_type': 'json' or 'zip',
                'json_content': {...} if JSON,
                'zip_content': bytes if ZIP,
                'from_cache': True
            }
        """
        if not self.has_cache(fingerprint):
            return None

        cache_dir = self.get_cache_dir(fingerprint)

        try:
            # Try loading ZIP first (most complete)
            zip_path = cache_dir / f"{fingerprint}.zip"
            if zip_path.exists():
                logger.info(f"Loading cached ZIP output for {fingerprint}")
                with open(zip_path, 'rb') as f:
                    return {
                        'response_type': 'zip',
                        'zip_content': f.read(),
                        'processing_time': None,
                        'from_cache': True
                    }

            # Fallback: reconstruct from individual JSON files
            content_list_path = cache_dir / f"{fingerprint}_content_list.json"
            markdown_path = cache_dir / f"{fingerprint}.md"

            if content_list_path.exists() or markdown_path.exists():
                logger.info(f"Loading cached JSON output for {fingerprint}")

                json_response = {'results': {}}
                doc_key = fingerprint

                doc_data = {}

                # Load markdown
                if markdown_path.exists():
                    with open(markdown_path, 'r', encoding='utf-8') as f:
                        doc_data['md_content'] = f.read()

                # Load content_list
                if content_list_path.exists():
                    with open(content_list_path, 'r', encoding='utf-8') as f:
                        doc_data['content_list'] = json.load(f)

                # Load model output
                model_path = cache_dir / f"{fingerprint}_model.json"
                if model_path.exists():
                    with open(model_path, 'r', encoding='utf-8') as f:
                        doc_data['model_output'] = json.load(f)

                # Load middle JSON
                middle_path = cache_dir / f"{fingerprint}_middle.json"
                if middle_path.exists():
                    with open(middle_path, 'r', encoding='utf-8') as f:
                        doc_data['middle_json'] = json.load(f)

                json_response['results'][doc_key] = doc_data

                return {
                    'response_type': 'json',
                    'json_content': json_response,
                    'processing_time': None,
                    'from_cache': True
                }

            logger.warning(f"Cache directory exists but no valid files found for {fingerprint}")
            return None

        except Exception as e:
            logger.error(f"Failed to load cached MinerU output for {fingerprint}: {e}")
            return None

    def save_to_cache(
        self,
        fingerprint: str,
        mineru_results: Dict[str, Any],
        markdown_content: str,
        json_data: Dict[str, Any]
    ) -> None:
        """
        Save MinerU output to cache keyed by fingerprint.

        This is called after successful MinerU processing to store
        results for future reuse.

        Args:
            fingerprint: Content fingerprint UUID
            mineru_results: Raw MinerU API response
            markdown_content: Extracted markdown
            json_data: Extracted JSON data (content_list, model_data, etc.)
        """
        cache_dir = self.get_cache_dir(fingerprint)

        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Saving MinerU output to cache: {cache_dir}")

            if mineru_results['response_type'] == 'zip':
                # Save ZIP file
                zip_path = cache_dir / f"{fingerprint}.zip"
                with open(zip_path, 'wb') as f:
                    f.write(mineru_results['zip_content'])
                logger.debug(f"Cached ZIP: {zip_path}")

                # Extract for easier access
                with zipfile.ZipFile(io.BytesIO(mineru_results['zip_content'])) as zip_file:
                    zip_file.extractall(cache_dir)

            else:
                # Save individual JSON components
                if markdown_content:
                    md_path = cache_dir / f"{fingerprint}.md"
                    with open(md_path, 'w', encoding='utf-8') as f:
                        f.write(markdown_content)
                    logger.debug(f"Cached markdown: {md_path}")

                if json_data.get('content_list'):
                    cl_path = cache_dir / f"{fingerprint}_content_list.json"
                    with open(cl_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data['content_list'], f, indent=2, ensure_ascii=False)
                    logger.debug(f"Cached content_list: {cl_path}")

                if json_data.get('model_data'):
                    model_path = cache_dir / f"{fingerprint}_model.json"
                    with open(model_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data['model_data'], f, indent=2, ensure_ascii=False)
                    logger.debug(f"Cached model_output: {model_path}")

                if json_data.get('middle_data'):
                    middle_path = cache_dir / f"{fingerprint}_middle.json"
                    with open(middle_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data['middle_data'], f, indent=2, ensure_ascii=False)
                    logger.debug(f"Cached middle_json: {middle_path}")

            logger.info(f"Successfully cached MinerU output for fingerprint {fingerprint}")

        except Exception as e:
            # Don't fail processing if caching fails
            logger.warning(f"Failed to cache MinerU output for {fingerprint}: {e}")

    def clear_cache(self, fingerprint: Optional[str] = None) -> None:
        """
        Clear cached outputs.

        Args:
            fingerprint: Specific fingerprint to clear, or None to clear all
        """
        if fingerprint:
            cache_dir = self.get_cache_dir(fingerprint)
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
                logger.info(f"Cleared cache for fingerprint {fingerprint}")
        else:
            # Clear entire cache directory
            if self.output_directory.exists():
                import shutil
                shutil.rmtree(self.output_directory)
                self.output_directory.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared entire MinerU cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about cache contents.

        Returns:
            Dict with cache statistics
        """
        if not self.output_directory.exists():
            return {
                'total_cached_documents': 0,
                'total_size_bytes': 0,
                'cache_directory': str(self.output_directory)
            }

        cached_fingerprints = [
            d for d in self.output_directory.iterdir()
            if d.is_dir()
        ]

        total_size = sum(
            sum(f.stat().st_size for f in fp_dir.rglob('*') if f.is_file())
            for fp_dir in cached_fingerprints
        )

        return {
            'total_cached_documents': len(cached_fingerprints),
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'cache_directory': str(self.output_directory),
            'cached_fingerprints': [d.name for d in cached_fingerprints]
        }
