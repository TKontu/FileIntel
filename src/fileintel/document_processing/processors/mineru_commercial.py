"""
Enhanced MinerU PDF processor with JSON-first approach for maximum data preservation.

This implementation extracts and preserves ALL MinerU output including:
- Rich JSON metadata with perfect page mapping via page_idx
- Element type classification (text, header, footer, table)
- Coordinate/bounding box information
- Image file references

Designed as a complete replacement for the broken markdown-only implementation.
"""

import requests
import zipfile
import json
import time
import shutil
import uuid
import io
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import logging

from ..elements import DocumentElement, TextElement
from .traditional_pdf import validate_file_for_processing, DocumentProcessingError

logger = logging.getLogger(__name__)


class MinerUEnhancedProcessor:
    """
    Enhanced MinerU PDF processor using JSON-first approach.

    Extracts and preserves ALL MinerU output including rich JSON metadata,
    perfect page mapping, element types, and coordinate information.
    """

    def __init__(self, config=None):
        from fileintel.core.config import get_config
        self.config = config or get_config()

        # Validate API token for commercial API
        if not self.config.document_processing.mineru.api_token:
            raise DocumentProcessingError(
                "MinerU API token is required for commercial API. "
                "Set MINERU_API_TOKEN environment variable or configure api_token in settings."
            )

        self.shared_folder = Path(self.config.document_processing.mineru.shared_folder_path)
        self._validate_shared_folder()

    def _validate_shared_folder(self) -> None:
        """Validate shared folder with proper error handling and permission checks."""
        try:
            # Create directory if it doesn't exist
            self.shared_folder.mkdir(exist_ok=True, parents=True)

            # Test write permissions with a temporary file
            test_file = self.shared_folder / f"test_{uuid.uuid4()}.tmp"
            test_file.write_text("permission_test")
            test_file.unlink()

        except PermissionError as e:
            raise DocumentProcessingError(
                f"Cannot write to shared folder {self.shared_folder}: {e}. "
                f"Please ensure the folder exists and is writable by the application."
            )
        except Exception as e:
            raise DocumentProcessingError(f"Shared folder validation failed: {e}")

    def read(self, file_path: Path, adapter: logging.LoggerAdapter = None) -> Tuple[List[DocumentElement], Dict[str, Any]]:
        """
        Process PDF using MinerU API with JSON-first approach.

        Returns DocumentElements with rich metadata preservation and perfect page mapping.
        Falls back to traditional processor on any failure.
        """
        log = adapter or logger
        validate_file_for_processing(file_path, ".pdf")

        shared_file_path = None
        try:
            # Copy file to shared folder for MinerU access
            file_url, shared_file_path = self._copy_to_shared_folder(file_path)
            log.info(f"File copied to shared folder: {shared_file_path}")

            # Process with MinerU API
            mineru_results = self._process_with_mineru(file_url, file_path, log)

            # Extract ALL data from ZIP (markdown + JSON)
            markdown_content, json_data = self._extract_all_results(mineru_results['zip_content'])

            # Create elements using JSON-first approach
            elements = self._create_elements_from_json(
                json_data, markdown_content, file_path, log
            )

            # Build comprehensive metadata
            metadata = self._build_metadata(json_data, mineru_results, file_path)

            log.info(f"MinerU processing successful: {len(elements)} elements created")
            return elements, metadata

        except (requests.RequestException, DocumentProcessingError) as e:
            log.error(f"MinerU processing failed for {file_path}: {e}")
            return self._fallback_to_traditional(file_path, adapter)
        except Exception as e:
            log.error(f"Unexpected error in MinerU processing for {file_path}: {e}")
            return self._fallback_to_traditional(file_path, adapter)
        finally:
            self._cleanup_shared_file(shared_file_path)

    def _copy_to_shared_folder(self, file_path: Path) -> Tuple[str, Path]:
        """Copy file to shared folder and return URL and shared path."""
        # Generate unique filename to avoid conflicts
        unique_name = f"{uuid.uuid4()}_{file_path.name}"
        shared_path = self.shared_folder / unique_name

        try:
            shutil.copy2(file_path, shared_path)
        except Exception as e:
            raise DocumentProcessingError(f"Failed to copy file to shared folder: {e}")

        # Build URL for MinerU access
        file_url = f"{self.config.document_processing.mineru.shared_folder_url_prefix}/{unique_name}"
        return file_url, shared_path

    def _process_with_mineru(self, file_url: str, file_path: Path, log) -> Dict[str, Any]:
        """Process PDF with MinerU API and return complete results."""
        # Submit processing task
        task_id = self._submit_task(file_url)
        log.info(f"MinerU task {task_id} submitted for {file_path.name}")

        # Poll for completion with configured timeout and interval
        max_polls = self.config.document_processing.mineru.timeout // self.config.document_processing.mineru.poll_interval

        for poll_count in range(max_polls):
            status_data = self._get_task_status(task_id)

            if status_data["state"] == "done":
                log.info(f"MinerU task {task_id} completed after {poll_count + 1} polls")
                zip_content = self._download_results(status_data["full_zip_url"])
                return {
                    'zip_content': zip_content,
                    'task_status': status_data,
                    'processing_time': (poll_count + 1) * self.config.document_processing.mineru.poll_interval
                }
            elif status_data["state"] == "failed":
                error_msg = status_data.get('err_msg', 'Unknown error')
                raise DocumentProcessingError(f"MinerU processing failed: {error_msg}")

            time.sleep(self.config.document_processing.mineru.poll_interval)

        raise DocumentProcessingError(
            f"MinerU processing timed out after {self.config.document_processing.mineru.timeout}s"
        )

    def _submit_task(self, file_url: str) -> str:
        """Submit processing task to MinerU API with retry logic."""
        url = f"{self.config.document_processing.mineru.base_url}/extract/task"
        headers = {
            "Authorization": f"Bearer {self.config.document_processing.mineru.api_token}",
            "Content-Type": "application/json"
        }
        data = {
            "url": file_url,
            "is_ocr": True,
            "enable_table": self.config.document_processing.mineru.enable_table,
            "enable_formula": self.config.document_processing.mineru.enable_formula,
            "language": self.config.document_processing.mineru.language,
            "model_version": self.config.document_processing.mineru.model_version
        }

        last_exception = None
        for attempt in range(self.config.document_processing.mineru.max_retries):
            try:
                response = requests.post(url, headers=headers, json=data, timeout=30)
                response.raise_for_status()

                result = response.json()
                if result.get("code") != 0:
                    raise DocumentProcessingError(f"MinerU API error: {result.get('msg')}")

                return result["data"]["task_id"]

            except requests.RequestException as e:
                last_exception = e
                if attempt < self.config.document_processing.mineru.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"MinerU task submission attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                continue

        raise DocumentProcessingError(
            f"Failed to submit MinerU task after {self.config.document_processing.mineru.max_retries} attempts: {last_exception}"
        )

    def _get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status from MinerU API."""
        url = f"{self.config.document_processing.mineru.base_url}/extract/task/{task_id}"
        headers = {"Authorization": f"Bearer {self.config.document_processing.mineru.api_token}"}

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        result = response.json()
        if result.get("code") != 0:
            raise DocumentProcessingError(f"MinerU API error: {result.get('msg')}")

        return result["data"]

    def _download_results(self, zip_url: str) -> bytes:
        """Download ZIP file containing results."""
        response = requests.get(zip_url, timeout=300)
        response.raise_for_status()
        return response.content

    def _extract_all_results(self, zip_content: bytes) -> Tuple[str, Dict[str, Any]]:
        """
        Extract ALL content from MinerU ZIP file.

        Returns:
            Tuple of (markdown_content, json_data_dict)
        """
        results = {
            'markdown': None,
            'content_list': None,
            'model_data': None,
            'middle_data': None,
            'images': []
        }

        try:
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:
                for filename in zip_file.namelist():
                    try:
                        if filename.endswith('.md'):
                            results['markdown'] = zip_file.read(filename).decode('utf-8')
                        elif filename.endswith('_content_list.json'):
                            content = zip_file.read(filename).decode('utf-8')
                            results['content_list'] = json.loads(content)
                        elif filename.endswith('_model.json'):
                            content = zip_file.read(filename).decode('utf-8')
                            results['model_data'] = json.loads(content)
                        elif filename.endswith('_middle.json'):
                            content = zip_file.read(filename).decode('utf-8')
                            results['middle_data'] = json.loads(content)
                        elif filename.startswith('images/') and not filename.endswith('/'):
                            results['images'].append(filename)
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        logger.warning(f"Failed to parse file {filename} from MinerU ZIP: {e}")
                        continue

        except zipfile.BadZipFile as e:
            raise DocumentProcessingError(f"Invalid ZIP file from MinerU: {e}")

        # Ensure we have at least markdown or content_list
        if not results['markdown'] and not results['content_list']:
            raise DocumentProcessingError("No usable content found in MinerU results")

        return results['markdown'] or "", results

    def _create_elements_from_json(
        self,
        json_data: Dict[str, Any],
        markdown_content: str,
        file_path: Path,
        log
    ) -> List[TextElement]:
        """
        Create TextElements using JSON-first approach with perfect page mapping.

        Uses content_list.json for accurate page_idx values instead of guessing
        from markdown patterns.
        """
        content_list = json_data.get('content_list')

        if not content_list:
            log.warning("No content_list.json found, falling back to markdown-only approach")
            return self._create_elements_from_markdown_fallback(markdown_content, file_path)

        # Group elements by actual page_idx from JSON
        elements_by_page = {}
        for item in content_list:
            page_idx = item.get('page_idx', 0)
            if page_idx not in elements_by_page:
                elements_by_page[page_idx] = []

            # Preserve all element information from JSON
            element_info = {
                'text': item.get('text', ''),
                'type': item.get('type', 'text'),
                'bbox': item.get('bbox', [])
            }
            elements_by_page[page_idx].append(element_info)

        # Create TextElements with perfect page mapping
        text_elements = []
        for page_idx in sorted(elements_by_page.keys()):
            page_elements = elements_by_page[page_idx]

            # Combine text from all elements on this page
            page_text_parts = []
            element_types = {}
            total_elements = len(page_elements)
            elements_with_coords = 0

            for elem_info in page_elements:
                text = elem_info['text'].strip()
                if text:  # Only include non-empty text
                    page_text_parts.append(text)

                # Count element types
                elem_type = elem_info['type']
                element_types[elem_type] = element_types.get(elem_type, 0) + 1

                # Count elements with coordinates
                if elem_info['bbox']:
                    elements_with_coords += 1

            if not page_text_parts:
                continue  # Skip pages with no text

            page_text = '\n'.join(page_text_parts)

            # Build rich metadata
            metadata = {
                'source': str(file_path),
                'page_number': page_idx + 1,  # Convert 0-based to 1-based
                'extraction_method': 'mineru_json',
                'format': 'structured_json',
                'element_count': total_elements,
                'element_types': element_types,
                'has_coordinates': elements_with_coords > 0,
                'coordinate_coverage': elements_with_coords / total_elements if total_elements > 0 else 0.0
            }

            text_elements.append(TextElement(text=page_text, metadata=metadata))

        log.info(f"Created {len(text_elements)} elements from JSON data across {len(elements_by_page)} pages")
        return text_elements

    def _create_elements_from_markdown_fallback(
        self,
        markdown_content: str,
        file_path: Path
    ) -> List[TextElement]:
        """Fallback to single markdown element when JSON data is unavailable."""
        if not markdown_content.strip():
            markdown_content = "[No content extracted from PDF]"

        metadata = {
            'source': str(file_path),
            'page_number': 1,
            'extraction_method': 'mineru_markdown_fallback',
            'format': 'markdown'
        }

        return [TextElement(text=markdown_content, metadata=metadata)]

    def _build_metadata(
        self,
        json_data: Dict[str, Any],
        mineru_results: Dict[str, Any],
        file_path: Path
    ) -> Dict[str, Any]:
        """Build comprehensive metadata from all available sources."""
        content_list = json_data.get('content_list', [])
        model_data = json_data.get('model_data', [])
        middle_data = json_data.get('middle_data', {})

        # Calculate statistics
        total_elements = len(content_list)
        pages_found = len(set(item.get('page_idx', 0) for item in content_list))

        # Count element types
        element_type_counts = {}
        for item in content_list:
            elem_type = item.get('type', 'unknown')
            element_type_counts[elem_type] = element_type_counts.get(elem_type, 0) + 1

        metadata = {
            'processor': 'mineru_enhanced',
            'file_path': str(file_path),
            'processing_time': mineru_results.get('processing_time', 0),
            'total_pages': len(model_data) if model_data else pages_found,
            'total_elements': total_elements,
            'element_types': element_type_counts,
            'has_images': len(json_data.get('images', [])) > 0,
            'image_count': len(json_data.get('images', [])),
            'json_files_extracted': [
                key for key in ['content_list', 'model_data', 'middle_data']
                if json_data.get(key) is not None
            ]
        }

        # Add middle.json metadata if available
        if middle_data:
            metadata['ocr_confidence'] = middle_data.get('confidence')
            metadata['language_detected'] = middle_data.get('language')

        return metadata

    def _fallback_to_traditional(
        self,
        file_path: Path,
        adapter: logging.LoggerAdapter = None
    ) -> Tuple[List[DocumentElement], Dict[str, Any]]:
        """Fallback to traditional PDF processor with error context."""
        log = adapter or logger
        log.info(f"Falling back to traditional PDF processor for {file_path}")

        try:
            from .traditional_pdf import PDFProcessor
            fallback_processor = PDFProcessor()
            elements, metadata = fallback_processor.read(file_path, adapter)

            # Add fallback context to metadata
            metadata['mineru_fallback'] = True
            metadata['processor'] = 'traditional_pdf_fallback'

            return elements, metadata
        except Exception as e:
            log.error(f"Traditional PDF fallback also failed for {file_path}: {e}")
            # Return minimal element to prevent complete failure
            return [TextElement(
                text="[PDF processing failed - unable to extract content]",
                metadata={
                    'source': str(file_path),
                    'extraction_failed': True,
                    'mineru_failed': True,
                    'traditional_failed': True
                }
            )], {'processing_failed': True}

    def _cleanup_shared_file(self, shared_file_path: Optional[Path]) -> None:
        """Clean up shared file with error handling."""
        if shared_file_path and shared_file_path.exists():
            try:
                shared_file_path.unlink()
                logger.debug(f"Cleaned up shared file: {shared_file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup shared file {shared_file_path}: {e}")


# Backward compatibility alias for drop-in replacement
MinerUPDFProcessor = MinerUEnhancedProcessor