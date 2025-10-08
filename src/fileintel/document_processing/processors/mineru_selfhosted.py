"""
Self-hosted MinerU PDF processor for FastAPI implementation.

This implementation works with the self-hosted MinerU FastAPI server that uses
direct file uploads and synchronous processing, as opposed to the commercial
async task-based API.

Key differences from commercial API:
- Direct file upload via multipart/form-data
- Synchronous processing (immediate response)
- Different parameter names and structure
- ZIP file or JSON response options
"""

import requests
import zipfile
import json
import io
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import logging

from ..elements import DocumentElement, TextElement
from .traditional_pdf import validate_file_for_processing, DocumentProcessingError

logger = logging.getLogger(__name__)


class MinerUSelfHostedProcessor:
    """
    Self-hosted MinerU PDF processor using FastAPI direct upload.

    Designed for the open-source self-hosted MinerU implementation that
    uses /file_parse endpoint with direct file uploads.
    """

    def __init__(self, config=None):
        from fileintel.core.config import get_config
        self.config = config or get_config()

        # Validate configuration for self-hosted API
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration for self-hosted MinerU API."""
        mineru_config = self.config.document_processing.mineru

        # Validate base_url
        if not mineru_config.base_url:
            raise DocumentProcessingError("MinerU base_url is required for self-hosted API")

        if not mineru_config.base_url.startswith(('http://', 'https://')):
            raise DocumentProcessingError(
                f"MinerU base_url must be a valid HTTP/HTTPS URL, got: {mineru_config.base_url}"
            )

        # Validate timeout
        if mineru_config.timeout <= 0:
            raise DocumentProcessingError(
                f"MinerU timeout must be positive, got: {mineru_config.timeout}"
            )

        # Validate language (warn for unknown languages but don't fail)
        supported_languages = ['en', 'zh', 'ch', 'de', 'fr', 'es', 'ja', 'ko']
        if mineru_config.language not in supported_languages:
            logger.warning(
                f"Unknown language '{mineru_config.language}' for MinerU. "
                f"Supported: {supported_languages}. Proceeding anyway."
            )

        # Validate API type for consistency
        if mineru_config.api_type != "selfhosted":
            raise DocumentProcessingError(
                f"Self-hosted processor requires api_type='selfhosted', got: '{mineru_config.api_type}'"
            )

        logger.info(f"Configured for self-hosted MinerU API at {mineru_config.base_url}")

    def read(self, file_path: Path, adapter: logging.LoggerAdapter = None) -> Tuple[List[DocumentElement], Dict[str, Any]]:
        """
        Process PDF using self-hosted MinerU FastAPI.

        Returns DocumentElements with rich metadata preservation from JSON data.
        Falls back to traditional processor on any failure.
        """
        log = adapter or logger
        validate_file_for_processing(file_path, ".pdf")

        try:
            # Process with self-hosted MinerU API
            log.info(f"Processing {file_path.name} with self-hosted MinerU API")

            mineru_results = self._process_with_selfhosted_api(file_path, log)

            # Extract data from response
            markdown_content, json_data = self._extract_results_from_response(mineru_results)

            # Create elements using JSON-first approach
            elements = self._create_elements_from_json(
                json_data, markdown_content, file_path, log
            )

            # Build comprehensive metadata
            metadata = self._build_metadata(json_data, mineru_results, file_path)

            log.info(f"Self-hosted MinerU processing successful: {len(elements)} elements created")
            return elements, metadata

        except (requests.RequestException, DocumentProcessingError) as e:
            log.error(f"Self-hosted MinerU processing failed for {file_path}: {e}")
            return self._fallback_to_traditional(file_path, adapter)
        except Exception as e:
            log.error(f"Unexpected error in self-hosted MinerU processing for {file_path}: {e}")
            return self._fallback_to_traditional(file_path, adapter)

    def _process_with_selfhosted_api(self, file_path: Path, log) -> Dict[str, Any]:
        """Process PDF with self-hosted MinerU FastAPI."""
        url = f"{self.config.document_processing.mineru.base_url}/file_parse"

        # Prepare form data for self-hosted API
        form_data = self._build_form_data()

        # Use context manager for proper file handle management
        try:
            log.info(f"Uploading {file_path.name} to self-hosted MinerU API")

            with open(file_path, 'rb') as file_handle:
                files = {
                    'files': (file_path.name, file_handle, 'application/pdf')
                }

                # Make request to self-hosted API
                response = requests.post(
                    url,
                    data=form_data,
                    files=files,
                    timeout=self.config.document_processing.mineru.timeout
                )
                response.raise_for_status()

            # Handle response based on format - moved outside context manager
            content_type = response.headers.get('content-type', '').lower().split(';')[0].strip()

            if content_type == 'application/zip':
                # ZIP file response
                log.info("Received ZIP file response from self-hosted API")
                return {
                    'response_type': 'zip',
                    'zip_content': response.content,
                    'processing_time': None  # Not available in self-hosted API
                }
            else:
                # JSON response - validate before processing
                log.info("Received JSON response from self-hosted API")
                try:
                    json_response = response.json()
                    if not isinstance(json_response, dict):
                        raise DocumentProcessingError("Invalid JSON response format from self-hosted API")
                    if 'results' not in json_response:
                        raise DocumentProcessingError("Missing 'results' field in self-hosted API response")
                except (ValueError, KeyError) as e:
                    raise DocumentProcessingError(f"Invalid JSON response from self-hosted API: {e}")

                return {
                    'response_type': 'json',
                    'json_content': json_response,
                    'processing_time': None
                }

        except requests.RequestException as e:
            raise DocumentProcessingError(f"Self-hosted MinerU API request failed: {e}")

    def _build_form_data(self) -> Dict[str, str]:
        """
        Build form data for self-hosted API request.

        FastAPI multipart form data requires all values to be strings.
        Lists and booleans must be converted appropriately.
        """
        mineru_config = self.config.document_processing.mineru

        return {
            'lang_list': mineru_config.language,  # Single string, not list
            'backend': 'pipeline',  # or 'vlm' based on model_version
            'parse_method': 'auto',
            'formula_enable': str(mineru_config.enable_formula).lower(),  # Convert boolean to string
            'table_enable': str(mineru_config.enable_table).lower(),
            'return_md': 'true',
            'return_content_list': 'true',
            'return_model_output': 'true',
            'return_middle_json': 'true',
            'return_images': 'true',
            'response_format_zip': 'true',  # Prefer ZIP format for consistency
            'start_page_id': '0',  # Convert integers to strings
            'end_page_id': '99999'
        }

    def _extract_results_from_response(self, mineru_results: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Extract markdown content and JSON data from API response."""
        if mineru_results['response_type'] == 'zip':
            # Handle ZIP file response (same as commercial API)
            return self._extract_from_zip(mineru_results['zip_content'])
        else:
            # Handle JSON response
            return self._extract_from_json_response(mineru_results['json_content'])

    def _extract_from_zip(self, zip_content: bytes) -> Tuple[str, Dict[str, Any]]:
        """Extract content from ZIP file response."""
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
                        elif 'images/' in filename and not filename.endswith('/'):
                            results['images'].append(filename)
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        logger.warning(f"Failed to parse file {filename} from self-hosted MinerU ZIP: {e}")
                        continue

        except zipfile.BadZipFile as e:
            raise DocumentProcessingError(f"Invalid ZIP file from self-hosted MinerU: {e}")

        return results['markdown'] or "", results

    def _extract_from_json_response(self, json_response: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Extract content from JSON response."""
        results = {
            'markdown': None,
            'content_list': None,
            'model_data': None,
            'middle_data': None,
            'images': []
        }

        # Extract data from JSON response structure
        response_results = json_response.get('results', {})

        # Get the first (and typically only) document result
        if response_results:
            doc_key = list(response_results.keys())[0]
            doc_data = response_results[doc_key]

            # Extract different data types
            results['markdown'] = doc_data.get('md_content', '')

            # Parse JSON strings if they exist - handle both string and pre-parsed data
            if doc_data.get('content_list'):
                try:
                    if isinstance(doc_data['content_list'], str):
                        results['content_list'] = json.loads(doc_data['content_list'])
                    else:
                        # Already parsed JSON data
                        results['content_list'] = doc_data['content_list']
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in content_list: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error parsing content_list: {e}")

            if doc_data.get('model_output'):
                try:
                    if isinstance(doc_data['model_output'], str):
                        results['model_data'] = json.loads(doc_data['model_output'])
                    else:
                        # Already parsed JSON data
                        results['model_data'] = doc_data['model_output']
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in model_output: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error parsing model_output: {e}")

            if doc_data.get('middle_json'):
                try:
                    if isinstance(doc_data['middle_json'], str):
                        results['middle_data'] = json.loads(doc_data['middle_json'])
                    else:
                        # Already parsed JSON data
                        results['middle_data'] = doc_data['middle_json']
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in middle_json: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error parsing middle_json: {e}")

            # Handle base64 encoded images
            if doc_data.get('images'):
                results['images'] = list(doc_data['images'].keys())

        return results['markdown'] or "", results

    def _extract_markdown_headers(self, markdown_content: str) -> List[Dict[str, Any]]:
        """
        Extract markdown headers with their hierarchy and text.

        Returns list of headers with level, text, and line number.
        """
        if not markdown_content:
            return []

        headers = []
        lines = markdown_content.split('\n')

        for line_num, line in enumerate(lines):
            # Match markdown headers (# Header, ## Subheader, etc.)
            match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if match:
                level = len(match.group(1))  # Number of # symbols
                text = match.group(2).strip()
                headers.append({
                    'level': level,
                    'text': text,
                    'line_number': line_num,
                    'type': f'h{level}'
                })

        logger.debug(f"Extracted {len(headers)} markdown headers")
        return headers

    def _map_headers_to_pages(
        self,
        headers: List[Dict[str, Any]],
        markdown_content: str,
        content_list: List[Dict[str, Any]]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Map markdown headers to page numbers using content_list correlation.

        Returns dict mapping page_idx to list of headers relevant to that page.
        """
        if not headers or not content_list:
            return {}

        # Build a mapping of header text to page by finding matches in content_list
        header_to_page = {}

        for header in headers:
            header_text = header['text']
            # Find this header text in content_list to determine its page
            for item in content_list:
                if header_text in item.get('text', '') or item.get('text', '') in header_text:
                    page_idx = item.get('page_idx', 0)
                    header_to_page[header_text] = page_idx
                    break

        # Group headers by page and build hierarchy
        pages_to_headers = {}
        current_section = None  # Track the current top-level section

        for header in headers:
            header_text = header['text']
            page_idx = header_to_page.get(header_text, 0)

            if page_idx not in pages_to_headers:
                pages_to_headers[page_idx] = []

            # Track section hierarchy
            if header['level'] == 1:
                current_section = header_text

            header_info = {
                'level': header['level'],
                'text': header_text,
                'type': header['type'],
                'section': current_section  # Top-level section this belongs to
            }
            pages_to_headers[page_idx].append(header_info)

        logger.debug(f"Mapped headers to {len(pages_to_headers)} pages")
        return pages_to_headers

    def _create_elements_from_json(
        self,
        json_data: Dict[str, Any],
        markdown_content: str,
        file_path: Path,
        log
    ) -> List[TextElement]:
        """
        Create TextElements using JSON-first approach with perfect page mapping.

        Same logic as commercial API processor for consistency.
        """
        content_list = json_data.get('content_list')

        if not content_list:
            log.warning("No content_list found in self-hosted API response, falling back to markdown")
            return self._create_elements_from_markdown_fallback(markdown_content, file_path)

        # Extract markdown headers and map to pages for enhanced metadata
        markdown_headers = self._extract_markdown_headers(markdown_content)
        headers_by_page = self._map_headers_to_pages(markdown_headers, markdown_content, content_list)
        log.info(f"Extracted {len(markdown_headers)} markdown headers across {len(headers_by_page)} pages")

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
                'extraction_method': 'mineru_selfhosted_json',
                'format': 'structured_json',
                'element_count': total_elements,
                'element_types': element_types,
                'has_coordinates': elements_with_coords > 0,
                'coordinate_coverage': elements_with_coords / total_elements if total_elements > 0 else 0.0
            }

            # Add markdown header context if available for this page
            if page_idx in headers_by_page:
                metadata['markdown_headers'] = headers_by_page[page_idx]

                # Extract section title from highest-level header on page
                top_level_headers = [h for h in headers_by_page[page_idx] if h['level'] <= 2]
                if top_level_headers:
                    metadata['section_title'] = top_level_headers[0]['text']

                # Add hierarchical section path
                section_path = []
                for h in headers_by_page[page_idx]:
                    if h.get('section'):
                        section_path.append(h['section'])
                if section_path:
                    metadata['section_path'] = ' > '.join(dict.fromkeys(section_path))  # Remove duplicates while preserving order

            text_elements.append(TextElement(text=page_text, metadata=metadata))

        log.info(f"Created {len(text_elements)} elements from self-hosted JSON data across {len(elements_by_page)} pages")
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
            'extraction_method': 'mineru_selfhosted_markdown_fallback',
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
        total_elements = len(content_list) if content_list else 0
        pages_found = len(set(item.get('page_idx', 0) for item in content_list)) if content_list else 0

        # Count element types
        element_type_counts = {}
        if content_list:
            for item in content_list:
                elem_type = item.get('type', 'unknown')
                element_type_counts[elem_type] = element_type_counts.get(elem_type, 0) + 1

        metadata = {
            'processor': 'mineru_selfhosted',
            'api_type': 'selfhosted_fastapi',
            'response_type': mineru_results.get('response_type'),
            'file_path': str(file_path),
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
            metadata['mineru_selfhosted_fallback'] = True
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
                    'mineru_selfhosted_failed': True,
                    'traditional_failed': True
                }
            )], {'processing_failed': True}


# Alias for compatibility
MinerUPDFProcessor = MinerUSelfHostedProcessor