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
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import logging

from ..elements import DocumentElement, TextElement
from .traditional_pdf import validate_file_for_processing, DocumentProcessingError
from ..element_detection import detect_semantic_type

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

        # Validate model_version (used as backend selector for selfhosted)
        valid_backends = ['pipeline', 'vlm']
        backend = getattr(mineru_config, 'model_version', 'pipeline')
        if backend not in valid_backends:
            raise DocumentProcessingError(
                f"Invalid model_version '{backend}'. Must be one of: {valid_backends}"
            )

        logger.info(f"Configured for self-hosted MinerU API at {mineru_config.base_url} (backend: {backend})")

        if backend == 'vlm':
            logger.info(
                "VLM backend selected: First request may take 30-60s for model loading. "
                "Subsequent requests will be faster."
            )

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

            # Save outputs if enabled (for debugging)
            self._save_mineru_outputs(mineru_results, file_path, log)

            # Extract data from response
            markdown_content, json_data = self._extract_results_from_response(mineru_results)

            # Create elements using JSON-first approach
            elements = self._create_elements_from_json(
                json_data, markdown_content, file_path, log
            )

            # Apply element filtering if enabled (Phase 2)
            mineru_config = self.config.document_processing.mineru
            if (hasattr(mineru_config, 'use_element_level_types') and
                mineru_config.use_element_level_types and
                hasattr(mineru_config, 'enable_element_filtering') and
                mineru_config.enable_element_filtering):

                from ..element_filter import filter_elements_for_rag

                log.info(f"Element filtering enabled: filtering {len(elements)} elements")
                filtered_elements, extracted_structure = filter_elements_for_rag(elements)

                log.info(f"Filtered: {len(filtered_elements)} elements to embed (removed {len(elements) - len(filtered_elements)})")
                elements = filtered_elements
            else:
                # Filtering disabled, no structure extraction
                extracted_structure = None

            # Build comprehensive metadata
            metadata = self._build_metadata(json_data, mineru_results, file_path)

            # Add extracted structure to metadata (Phase 4)
            if extracted_structure:
                metadata['document_structure'] = extracted_structure

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
        mineru_config = self.config.document_processing.mineru

        # Prepare form data for self-hosted API
        form_data = self._build_form_data()

        # Adjust timeout for VLM backend (first request loads models)
        backend = getattr(mineru_config, 'model_version', 'pipeline')
        timeout = mineru_config.timeout
        if backend == 'vlm' and timeout < 180:
            # VLM first request needs at least 3 minutes for model loading
            log.info(f"Extending timeout from {timeout}s to 180s for VLM backend first request")
            timeout = 180

        # Use context manager for proper file handle management
        try:
            log.info(f"Uploading {file_path.name} to self-hosted MinerU API (backend: {backend})")

            with open(file_path, 'rb') as file_handle:
                files = {
                    'files': (file_path.name, file_handle, 'application/pdf')
                }

                # Make request to self-hosted API
                response = requests.post(
                    url,
                    data=form_data,
                    files=files,
                    timeout=timeout
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

    def _save_mineru_outputs(self, mineru_results: Dict[str, Any], file_path: Path, log) -> None:
        """
        Save MinerU outputs to disk for debugging and inspection.

        Only saves if config.document_processing.mineru.save_outputs is True.
        Organizes outputs by document name in the configured output directory.
        """
        mineru_config = self.config.document_processing.mineru

        if not mineru_config.save_outputs:
            return  # Output saving disabled

        # Create output directory for this document
        output_base = Path(mineru_config.output_directory)
        doc_name = file_path.stem  # Filename without extension
        doc_output_dir = output_base / doc_name

        try:
            doc_output_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"Saving MinerU outputs to {doc_output_dir}")

            if mineru_results['response_type'] == 'zip':
                # Extract and save ZIP contents
                zip_content = mineru_results['zip_content']

                # Save the raw ZIP file
                zip_path = doc_output_dir / f"{doc_name}.zip"
                with open(zip_path, 'wb') as f:
                    f.write(zip_content)
                log.info(f"Saved raw ZIP: {zip_path}")

                # Extract ZIP contents
                with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:
                    zip_file.extractall(doc_output_dir)
                    extracted_files = zip_file.namelist()
                    log.info(f"Extracted {len(extracted_files)} files from ZIP")

            else:
                # JSON response - save the JSON data
                json_response = mineru_results['json_content']
                json_path = doc_output_dir / f"{doc_name}_response.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_response, f, indent=2, ensure_ascii=False)
                log.info(f"Saved JSON response: {json_path}")

                # Extract and save individual components if available
                if 'results' in json_response:
                    results = json_response['results']
                    if results:
                        doc_key = list(results.keys())[0]
                        doc_data = results[doc_key]

                        # Save markdown
                        if doc_data.get('md_content'):
                            md_path = doc_output_dir / f"{doc_name}.md"
                            with open(md_path, 'w', encoding='utf-8') as f:
                                f.write(doc_data['md_content'])
                            log.info(f"Saved markdown: {md_path}")

                        # Save content_list JSON
                        if doc_data.get('content_list'):
                            content_list = doc_data['content_list']
                            if isinstance(content_list, str):
                                content_list = json.loads(content_list)
                            cl_path = doc_output_dir / f"{doc_name}_content_list.json"
                            with open(cl_path, 'w', encoding='utf-8') as f:
                                json.dump(content_list, f, indent=2, ensure_ascii=False)
                            log.info(f"Saved content_list: {cl_path}")

                        # Save model_output JSON
                        if doc_data.get('model_output'):
                            model_output = doc_data['model_output']
                            if isinstance(model_output, str):
                                model_output = json.loads(model_output)
                            mo_path = doc_output_dir / f"{doc_name}_model.json"
                            with open(mo_path, 'w', encoding='utf-8') as f:
                                json.dump(model_output, f, indent=2, ensure_ascii=False)
                            log.info(f"Saved model_output: {mo_path}")

                        # Save middle_json
                        if doc_data.get('middle_json'):
                            middle_json = doc_data['middle_json']
                            if isinstance(middle_json, str):
                                middle_json = json.loads(middle_json)
                            mj_path = doc_output_dir / f"{doc_name}_middle.json"
                            with open(mj_path, 'w', encoding='utf-8') as f:
                                json.dump(middle_json, f, indent=2, ensure_ascii=False)
                            log.info(f"Saved middle_json: {mj_path}")

            log.info(f"Successfully saved MinerU outputs for {file_path.name}")

        except Exception as e:
            # Don't fail processing if output saving fails
            log.warning(f"Failed to save MinerU outputs for {file_path.name}: {e}")

    def _build_form_data(self) -> Dict[str, str]:
        """
        Build form data for self-hosted API request.

        FastAPI multipart form data requires all values to be strings.
        Lists and booleans must be converted appropriately.
        """
        mineru_config = self.config.document_processing.mineru

        # Map model_version to API backend values
        # 'vlm' -> 'vlm-vllm-async-engine' (VLM backend with async vLLM inference engine)
        # 'pipeline' -> 'pipeline' (OCR + Layout detection pipeline)
        backend = getattr(mineru_config, 'model_version', 'pipeline')
        backend_api_values = {
            'pipeline': 'pipeline',
            'vlm': 'vlm-vllm-async-engine'
        }
        api_backend = backend_api_values.get(backend, 'pipeline')

        # Base form data common to all backends
        form_data = {
            'lang_list': mineru_config.language,  # Single string, not list
            'backend': api_backend,
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

        # Pipeline backend requires parse_method parameter
        # VLM backend doesn't use it (uses vision-language model for all documents)
        if backend == 'pipeline':
            form_data['parse_method'] = 'auto'

        return form_data

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
        Create TextElements using JSON-first approach.

        Dispatches to element-level or page-level processing based on config flag.
        """
        mineru_config = self.config.document_processing.mineru

        content_list = json_data.get('content_list')
        if not content_list:
            backend = getattr(mineru_config, 'model_version', 'pipeline')
            log.warning("No content_list found in self-hosted API response, falling back to markdown")
            return self._create_elements_from_markdown_fallback(markdown_content, file_path, backend)

        # Check feature flag for element-level vs page-level processing
        if mineru_config.use_element_level_types:
            log.info("Using element-level type preservation (EXPERIMENTAL)")
            return self._create_elements_element_level(json_data, markdown_content, file_path, log)
        else:
            return self._create_elements_page_level(json_data, markdown_content, file_path, log)

    def _create_elements_page_level(
        self,
        json_data: Dict[str, Any],
        markdown_content: str,
        file_path: Path,
        log
    ) -> List[TextElement]:
        """
        Create TextElements by grouping all elements per page (backward compatible).

        This is the original behavior: concatenates all text from a page into one TextElement.
        """
        mineru_config = self.config.document_processing.mineru
        backend = getattr(mineru_config, 'model_version', 'pipeline')

        content_list = json_data.get('content_list')

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
                'extraction_method': f'mineru_selfhosted_{backend}_json',
                'backend': backend,
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

    def _create_elements_element_level(
        self,
        json_data: Dict[str, Any],
        markdown_content: str,
        file_path: Path,
        log
    ) -> List[TextElement]:
        """
        Create one TextElement per content_list item (preserves element boundaries).

        Extracts MinerU fields: text_level, table_body, image_caption
        Creates two-layer type system: layout_type (from MinerU) + semantic_type (detected)
        """
        mineru_config = self.config.document_processing.mineru
        backend = getattr(mineru_config, 'model_version', 'pipeline')
        content_list = json_data.get('content_list', [])

        text_elements = []
        element_index = 0

        for item in content_list:
            # Get layout type from MinerU (Layer 1: layout-level)
            layout_type = item.get('type', 'text')
            text = item.get('text', '').strip()
            page_idx = item.get('page_idx', 0)

            # Build base metadata with MinerU fields
            metadata = {
                'source': str(file_path),
                'page_number': page_idx + 1,  # Convert 0-based to 1-based
                'layout_type': layout_type,  # L1: text/table/image from MinerU
                'extraction_method': f'mineru_selfhosted_{backend}_element_level',
                'backend': backend,
                'format': 'structured_json',
                'element_index': element_index,
                'bbox': item.get('bbox', [])
            }

            # Extract type-specific MinerU fields
            text_level = 0

            if layout_type == 'text':
                # Capture header level (0 = not a header, 1-6 = header levels)
                text_level = item.get('text_level', 0)
                metadata['text_level'] = text_level
                metadata['is_header'] = text_level > 0

            elif layout_type == 'table':
                # Extract table-specific MinerU fields
                metadata['table_body'] = item.get('table_body', '')  # HTML table structure
                metadata['table_caption'] = item.get('table_caption', [])
                metadata['table_footnote'] = item.get('table_footnote', [])

                # Use caption as text if main text field is empty
                # (MinerU often has empty .text for tables, content is in table_body)
                if not text and metadata['table_caption']:
                    text = ' '.join(metadata['table_caption'])

            elif layout_type == 'image':
                # Extract image-specific MinerU fields
                metadata['img_path'] = item.get('img_path', '')
                metadata['image_caption'] = item.get('image_caption', [])
                metadata['image_footnote'] = item.get('image_footnote', [])

                # Use caption as text (for embedding - makes figures searchable)
                # Images themselves can't be embedded, but captions can
                if metadata['image_caption']:
                    text = ' '.join(metadata['image_caption'])

            # Detect semantic type (Layer 2: semantic classification)
            # Uses pattern matching + text_level + layout_type
            semantic_type = detect_semantic_type(text, layout_type, text_level)
            metadata['semantic_type'] = semantic_type

            # Add header_level for headers (convenience field)
            if semantic_type == 'header' and text_level > 0:
                metadata['header_level'] = text_level

            # Only create TextElement if we have text content
            if text:
                text_elements.append(TextElement(text=text, metadata=metadata))
                element_index += 1

        log.info(f"Created {len(text_elements)} elements from {len(content_list)} content_list items (element-level mode)")
        return text_elements

    def _create_elements_from_markdown_fallback(
        self,
        markdown_content: str,
        file_path: Path,
        backend: str = 'pipeline'
    ) -> List[TextElement]:
        """Fallback to single markdown element when JSON data is unavailable."""
        if not markdown_content.strip():
            markdown_content = "[No content extracted from PDF]"

        metadata = {
            'source': str(file_path),
            'page_number': 1,
            'extraction_method': f'mineru_selfhosted_{backend}_markdown_fallback',
            'backend': backend,
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
        mineru_config = self.config.document_processing.mineru
        backend = getattr(mineru_config, 'model_version', 'pipeline')

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
            'backend': backend,  # Track which backend was used
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