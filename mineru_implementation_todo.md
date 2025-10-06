# MinerU OCR Integration Implementation Plan

## 🎯 **Project Overview**

Replace the current poor-quality OCR system with MinerU's high-quality cloud OCR API to convert scanned PDFs to markdown for improved indexing and chunking.

### **Key Objectives**
- **Quality**: High-accuracy OCR with proper text structure
- **Format**: Output markdown for better chunking and semantic understanding
- **Integration**: Seamless replacement of existing OCR pipeline
- **Performance**: Efficient processing of large scanned PDFs (up to 200MB)
- **Reliability**: Robust error handling and fallback mechanisms

### **MinerU API Configuration**
- **Service**: Cloud-based API at https://mineru.net/api/v4/
- **Authentication**: Bearer token required
- **File Limits**: 200MB max, 600 pages max
- **Processing**: Async with polling or webhook callbacks
- **Output**: ZIP file containing markdown, JSON, and optional formats
- **Rate Limits**: 2000 pages/day priority quota
- **Local Alternative**: http://192.168.0.247:8000 (if self-hosted version)

### **Critical API Requirements**
- ❗ **Token authentication** required for all requests
- ❗ **File hosting** needed (URLs only, no direct upload)
- ❗ **ZIP file processing** for result extraction
- ❗ **Async polling** for job completion
- ❗ **Rate limiting** awareness (2000 pages/day)

---

## 📋 **Phase 1: Authentication & File Hosting Setup (4-6 hours)**

### **1.1 MinerU Authentication Setup**
**Priority**: 🔴 **CRITICAL**
**Complexity**: Low

**Requirements**:
- [ ] Obtain MinerU API token
- [ ] Test token validity and permissions
- [ ] Configure secure token storage
- [ ] Implement token refresh mechanism (if needed)

**Configuration**:
```yaml
# config/default.yaml
ocr:
  mineru:
    api_token: ${MINERU_API_TOKEN}  # Environment variable
    base_url: "https://mineru.net/api/v4"
    timeout: 600  # 10 minutes
    max_retries: 3
    model_version: "vlm"  # or "pipeline"
    enable_formula: false
    enable_table: true
    language: "en"  # or "ch" for Chinese
```

### **1.2 File Hosting Strategy**
**Priority**: 🔴 **CRITICAL**
**Complexity**: Medium

**Problem**: MinerU requires publicly accessible URLs, not direct file uploads.

**Solution Options**:

#### **Option A: Local HTTP Server (Recommended)**
```python
# src/fileintel/ocr/file_server.py
"""
Simple HTTP file server for MinerU file hosting.
"""

import threading
import http.server
import socketserver
import os
from pathlib import Path
from typing import Optional
import uuid

class MinerUFileServer:
    """Local HTTP server to host files for MinerU processing."""

    def __init__(self, port: int = 8080, upload_dir: str = "./temp_uploads"):
        self.port = port
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        self.server = None
        self.server_thread = None

    def start(self):
        """Start the file server in background thread."""
        os.chdir(self.upload_dir.parent)

        handler = http.server.SimpleHTTPRequestHandler
        self.server = socketserver.TCPServer(("", self.port), handler)

        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()

        print(f"File server started on port {self.port}")

    def stop(self):
        """Stop the file server."""
        if self.server:
            self.server.shutdown()
            self.server_thread.join()

    def host_file(self, file_path: str) -> str:
        """
        Copy file to hosting directory and return public URL.

        Returns:
            Public URL accessible by MinerU
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Generate unique filename to avoid conflicts
        unique_name = f"{uuid.uuid4()}_{file_path.name}"
        hosted_path = self.upload_dir / unique_name

        # Copy file to hosting directory
        import shutil
        shutil.copy2(file_path, hosted_path)

        # Return public URL
        # TODO: Replace with your server's external IP
        public_url = f"http://192.168.0.247:{self.port}/temp_uploads/{unique_name}"
        return public_url

    def cleanup_file(self, url: str):
        """Clean up hosted file after processing."""
        try:
            filename = url.split('/')[-1]
            file_path = self.upload_dir / filename
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            print(f"Failed to cleanup file: {e}")
```

#### **Option B: External File Hosting**
- Use existing web server or cloud storage
- Configure nginx/apache to serve temp files
- Use S3/MinIO for file hosting

### **1.3 Network Configuration**
**Priority**: 🔴 **HIGH**

**Requirements**:
- [ ] Ensure MinerU cloud service can access local file server
- [ ] Configure firewall rules if necessary
- [ ] Test connectivity from external networks
- [ ] Set up reverse proxy if behind NAT

---

## 📋 **Phase 2: MinerU Client Implementation (1-2 days)**

### **2.1 Core MinerU Client**
**Priority**: 🔴 **HIGH**
**Files**: `src/fileintel/ocr/mineru_client.py`

```python
"""
MinerU OCR Client for high-quality PDF to Markdown conversion.
Based on official MinerU API v4 documentation.
"""

import logging
import requests
import time
import zipfile
import io
import json
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CONVERTING = "converting"


@dataclass
class OCRResult:
    """Result of MinerU OCR processing."""
    task_id: str
    status: ProcessingStatus
    markdown_content: Optional[str] = None
    json_content: Optional[Dict] = None
    zip_url: Optional[str] = None
    error_message: Optional[str] = None
    extracted_pages: Optional[int] = None
    total_pages: Optional[int] = None
    start_time: Optional[str] = None
    data_id: Optional[str] = None


class MinerUClient:
    """High-quality OCR client using MinerU Cloud API."""

    def __init__(self, api_token: str, base_url: str = "https://mineru.net/api/v4", timeout: int = 600):
        self.api_token = api_token
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()

        # Set default headers
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Accept": "*/*"
        })

        # Configure retries
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504, 429],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def submit_task(self, file_url: str, **kwargs) -> str:
        """
        Submit PDF processing task to MinerU.

        Args:
            file_url: Publicly accessible URL to PDF file
            **kwargs: Additional processing options

        Returns:
            task_id: Unique identifier for tracking the OCR job
        """
        endpoint = f"{self.base_url}/extract/task"

        # Default processing options
        data = {
            "url": file_url,
            "is_ocr": True,
            "enable_formula": kwargs.get("enable_formula", False),
            "enable_table": kwargs.get("enable_table", True),
            "language": kwargs.get("language", "en"),
            "model_version": kwargs.get("model_version", "vlm"),
            "extra_formats": kwargs.get("extra_formats", []),
        }

        # Optional parameters
        if "data_id" in kwargs:
            data["data_id"] = kwargs["data_id"]
        if "page_ranges" in kwargs:
            data["page_ranges"] = kwargs["page_ranges"]
        if "callback" in kwargs:
            data["callback"] = kwargs["callback"]
            data["seed"] = kwargs.get("seed", "fileintel_ocr")

        try:
            response = self.session.post(endpoint, json=data, timeout=30)
            response.raise_for_status()

            result = response.json()

            if result.get("code") != 0:
                raise ValueError(f"MinerU API error: {result.get('msg', 'Unknown error')}")

            task_id = result["data"]["task_id"]
            logger.info(f"MinerU task submitted successfully: {task_id}")
            return task_id

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to submit MinerU task: {e}")
            raise

    def get_task_status(self, task_id: str) -> OCRResult:
        """Get current status and results of OCR task."""
        endpoint = f"{self.base_url}/extract/task/{task_id}"

        try:
            response = self.session.get(endpoint, timeout=30)
            response.raise_for_status()

            result = response.json()

            if result.get("code") != 0:
                raise ValueError(f"MinerU API error: {result.get('msg', 'Unknown error')}")

            data = result["data"]
            status_str = data.get("state", "pending").lower()

            # Map MinerU status to our enum
            status_mapping = {
                "pending": ProcessingStatus.PENDING,
                "running": ProcessingStatus.RUNNING,
                "done": ProcessingStatus.DONE,
                "failed": ProcessingStatus.FAILED,
                "converting": ProcessingStatus.CONVERTING,
            }
            status = status_mapping.get(status_str, ProcessingStatus.PENDING)

            # Extract progress information
            progress = data.get("extract_progress", {})

            return OCRResult(
                task_id=task_id,
                status=status,
                zip_url=data.get("full_zip_url"),
                error_message=data.get("err_msg"),
                extracted_pages=progress.get("extracted_pages"),
                total_pages=progress.get("total_pages"),
                start_time=progress.get("start_time"),
                data_id=data.get("data_id")
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get MinerU task status: {e}")
            return OCRResult(
                task_id=task_id,
                status=ProcessingStatus.FAILED,
                error_message=str(e)
            )

    def download_and_extract_results(self, zip_url: str) -> Dict[str, Any]:
        """
        Download ZIP file and extract markdown content.

        Returns:
            Dict containing markdown, json, and other extracted content
        """
        try:
            # Download ZIP file
            response = self.session.get(zip_url, timeout=self.timeout)
            response.raise_for_status()

            # Extract ZIP content
            extracted_content = {}

            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                for file_info in zip_file.filelist:
                    filename = file_info.filename

                    if filename.endswith('.md'):
                        # Markdown content
                        extracted_content['markdown'] = zip_file.read(filename).decode('utf-8')
                    elif filename.endswith('.json'):
                        # JSON metadata
                        extracted_content['json'] = json.loads(zip_file.read(filename).decode('utf-8'))
                    elif filename.endswith(('.docx', '.html', '.latex')):
                        # Additional formats
                        format_name = filename.split('.')[-1]
                        extracted_content[format_name] = zip_file.read(filename)

            return extracted_content

        except Exception as e:
            logger.error(f"Failed to download/extract MinerU results: {e}")
            raise

    def process_pdf_sync(self, file_url: str, poll_interval: int = 10, max_wait: int = 1800, **kwargs) -> OCRResult:
        """
        Process PDF with MinerU and wait for results (synchronous).

        Args:
            file_url: Publicly accessible URL to PDF file
            poll_interval: Seconds between status checks (default: 10s)
            max_wait: Maximum time to wait for completion (default: 30min)
            **kwargs: Processing options

        Returns:
            OCRResult with markdown content or error
        """
        start_time = time.time()

        try:
            # Submit task
            task_id = self.submit_task(file_url, **kwargs)
            logger.info(f"MinerU task {task_id} submitted, polling every {poll_interval}s")

            # Poll for completion
            while (time.time() - start_time) < max_wait:
                result = self.get_task_status(task_id)

                if result.status == ProcessingStatus.DONE:
                    # Download and extract results
                    if result.zip_url:
                        try:
                            extracted = self.download_and_extract_results(result.zip_url)
                            result.markdown_content = extracted.get('markdown')
                            result.json_content = extracted.get('json')
                            logger.info(f"MinerU task {task_id} completed successfully")
                        except Exception as e:
                            result.error_message = f"Failed to extract results: {e}"
                            result.status = ProcessingStatus.FAILED
                    return result

                elif result.status == ProcessingStatus.FAILED:
                    logger.error(f"MinerU task {task_id} failed: {result.error_message}")
                    return result

                # Log progress
                if result.extracted_pages and result.total_pages:
                    progress = (result.extracted_pages / result.total_pages) * 100
                    logger.info(f"MinerU task {task_id} progress: {progress:.1f}% ({result.extracted_pages}/{result.total_pages} pages)")
                else:
                    logger.info(f"MinerU task {task_id} status: {result.status.value}")

                time.sleep(poll_interval)

            # Timeout
            logger.warning(f"MinerU task {task_id} timed out after {max_wait}s")
            return OCRResult(
                task_id=task_id,
                status=ProcessingStatus.FAILED,
                error_message=f"Processing timed out after {max_wait} seconds"
            )

        except Exception as e:
            logger.error(f"MinerU processing failed: {e}")
            return OCRResult(
                task_id="",
                status=ProcessingStatus.FAILED,
                error_message=str(e)
            )

    async def process_pdf_async(self, file_url: str, **kwargs) -> str:
        """
        Process PDF asynchronously (returns task_id for later retrieval).

        Returns:
            task_id for tracking the async processing
        """
        return self.submit_task(file_url, **kwargs)


def create_mineru_client(api_token: str = None, base_url: str = None) -> MinerUClient:
    """Create MinerU client with configuration."""
    from fileintel.core.config import get_config

    config = get_config()

    token = api_token or getattr(config.ocr.mineru, 'api_token', None)
    if not token:
        raise ValueError("MinerU API token is required")

    url = base_url or getattr(config.ocr.mineru, 'base_url', 'https://mineru.net/api/v4')

    return MinerUClient(api_token=token, base_url=url)
```

### **2.2 Configuration Updates**
**Priority**: 🟡 **MEDIUM**
**Files**: `src/fileintel/core/config.py`

```python
# Add MinerU-specific settings
class MinerUSettings(BaseModel):
    api_token: str = Field(default="")
    base_url: str = Field(default="https://mineru.net/api/v4")
    timeout: int = Field(default=1800)  # 30 minutes
    poll_interval: int = Field(default=10)  # 10 seconds
    max_retries: int = Field(default=3)

    # Processing options
    model_version: str = Field(default="vlm")  # "vlm" or "pipeline"
    enable_formula: bool = Field(default=False)
    enable_table: bool = Field(default=True)
    language: str = Field(default="en")  # "en" or "ch"
    extra_formats: List[str] = Field(default_factory=list)  # ["docx", "html", "latex"]

class OCRSettings(BaseModel):
    primary_engine: str = Field(default="mineru")
    fallback_engines: List[str] = Field(default_factory=lambda: ["tesseract", "google_vision"])

    # MinerU configuration
    mineru: MinerUSettings = Field(default_factory=MinerUSettings)

    # File hosting for MinerU
    file_server_port: int = Field(default=8080)
    file_server_host: str = Field(default="192.168.0.247")
    temp_upload_dir: str = Field(default="./temp_uploads")
```

---

## 📋 **Phase 3: OCR Pipeline Integration (1 day)**

### **3.1 Enhanced OCR Provider with File Hosting**
**Priority**: 🔴 **HIGH**
**Files**: `src/fileintel/ocr/ocr_provider.py`

```python
"""
Enhanced OCR provider with MinerU integration and file hosting.
"""

import logging
from typing import Dict, Any, Union, Optional
from pathlib import Path

from .mineru_client import MinerUClient, OCRResult, ProcessingStatus, create_mineru_client
from .file_server import MinerUFileServer

logger = logging.getLogger(__name__)


class MinerUOCRProvider:
    """High-quality OCR provider using MinerU cloud service."""

    def __init__(self, config=None):
        from fileintel.core.config import get_config

        self.config = config or get_config()
        self.client = create_mineru_client()

        # Initialize file server for hosting local files
        self.file_server = MinerUFileServer(
            port=self.config.ocr.file_server_port,
            upload_dir=self.config.ocr.temp_upload_dir
        )
        self.file_server.start()
        logger.info(f"MinerU file server started on port {self.config.ocr.file_server_port}")

    def __del__(self):
        """Cleanup file server on destruction."""
        if hasattr(self, 'file_server'):
            self.file_server.stop()

    def is_available(self) -> bool:
        """Check if MinerU service is available."""
        try:
            # Test with a simple API call (could be a health check if available)
            # For now, we'll assume it's available if client can be created
            return True
        except Exception as e:
            logger.error(f"MinerU service unavailable: {e}")
            return False

    def process_pdf(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process PDF using MinerU OCR.

        Returns:
            Dict with extracted content and metadata
        """
        file_path = Path(file_path)
        hosted_url = None

        logger.info(f"Processing PDF with MinerU: {file_path}")

        try:
            # Host file for MinerU access
            hosted_url = self.file_server.host_file(file_path)
            logger.info(f"File hosted at: {hosted_url}")

            # Process with MinerU
            processing_options = {
                "enable_formula": self.config.ocr.mineru.enable_formula,
                "enable_table": self.config.ocr.mineru.enable_table,
                "language": self.config.ocr.mineru.language,
                "model_version": self.config.ocr.mineru.model_version,
                "extra_formats": self.config.ocr.mineru.extra_formats,
                "data_id": f"fileintel_{file_path.stem}"
            }

            result = self.client.process_pdf_sync(
                file_url=hosted_url,
                poll_interval=self.config.ocr.mineru.poll_interval,
                max_wait=self.config.ocr.mineru.timeout,
                **processing_options
            )

            if result.status == ProcessingStatus.DONE:
                logger.info(f"MinerU processing completed for {file_path}")

                return {
                    'success': True,
                    'content': result.markdown_content,
                    'format': 'markdown',
                    'json_metadata': result.json_content,
                    'total_pages': result.total_pages,
                    'extracted_pages': result.extracted_pages,
                    'provider': 'mineru',
                    'task_id': result.task_id,
                    'data_id': result.data_id,
                    'zip_url': result.zip_url
                }
            else:
                logger.error(f"MinerU processing failed: {result.error_message}")
                return {
                    'success': False,
                    'error': result.error_message,
                    'provider': 'mineru',
                    'task_id': result.task_id
                }

        except Exception as e:
            logger.error(f"MinerU processing exception: {e}")
            return {
                'success': False,
                'error': str(e),
                'provider': 'mineru'
            }
        finally:
            # Cleanup hosted file
            if hosted_url:
                try:
                    self.file_server.cleanup_file(hosted_url)
                except Exception as e:
                    logger.warning(f"Failed to cleanup hosted file: {e}")
```

### **3.2 Document Processing Integration**
**Priority**: 🔴 **HIGH**
**Files**: `src/fileintel/document_processing/processors/mineru_pdf.py`

```python
"""
PDF processor using MinerU for high-quality OCR and markdown output.
"""

from typing import List, Dict, Any, Tuple
from pathlib import Path
import logging
import re

from ..base import DocumentProcessor
from fileintel.ocr.ocr_provider import MinerUOCRProvider

logger = logging.getLogger(__name__)


class MinerUPDFProcessor(DocumentProcessor):
    """PDF processor using MinerU for high-quality OCR and markdown output."""

    def __init__(self, config=None):
        super().__init__(config)
        self.ocr_provider = MinerUOCRProvider(config)

    def can_process(self, file_path: str) -> bool:
        """Check if this processor can handle the file."""
        return Path(file_path).suffix.lower() == '.pdf'

    def extract_content(self, file_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extract content from PDF using MinerU OCR.

        Returns:
            Tuple of (markdown_content, page_mappings)
        """
        file_path = Path(file_path)

        logger.info(f"Processing PDF with MinerU OCR: {file_path}")

        try:
            # Process with MinerU OCR
            ocr_result = self.ocr_provider.process_pdf(file_path)

            if not ocr_result.get('success', False):
                raise ValueError(f"OCR processing failed: {ocr_result.get('error', 'Unknown error')}")

            markdown_content = ocr_result.get('content', '')
            total_pages = ocr_result.get('total_pages', 1)

            if not markdown_content:
                raise ValueError("OCR returned empty content")

            logger.info(f"MinerU OCR completed: {len(markdown_content)} chars, {total_pages} pages")

            # Create page mappings from markdown content
            page_mappings = self._create_page_mappings_from_markdown(
                markdown_content, total_pages, ocr_result.get('json_metadata', {})
            )

            # Store OCR metadata for analysis
            self._store_ocr_metadata(file_path, ocr_result)

            return markdown_content, page_mappings

        except Exception as e:
            logger.error(f"MinerU PDF processing failed for {file_path}: {e}")
            raise

    def _create_page_mappings_from_markdown(self, content: str, page_count: int, json_metadata: Dict) -> List[Dict[str, Any]]:
        """
        Create page mappings from markdown content and JSON metadata.

        MinerU JSON output may contain page information that helps with mapping.
        """
        if page_count <= 1:
            return [{
                'start_pos': 0,
                'end_pos': len(content),
                'page_number': 1,
                'extraction_method': 'mineru_ocr'
            }]

        # Try to extract page info from JSON metadata
        if json_metadata and 'pages' in json_metadata:
            return self._extract_page_mappings_from_json(content, json_metadata)

        # Fallback: detect page breaks in markdown
        page_breaks = self._detect_page_breaks_in_markdown(content)

        if len(page_breaks) == page_count - 1:
            # Found correct number of page breaks
            page_mappings = []
            prev_pos = 0

            for i, break_pos in enumerate(page_breaks + [len(content)]):
                page_mappings.append({
                    'start_pos': prev_pos,
                    'end_pos': break_pos,
                    'page_number': i + 1,
                    'extraction_method': 'mineru_ocr_pagebreak'
                })
                prev_pos = break_pos

            return page_mappings
        else:
            # Estimate equal distribution
            return self._estimate_page_mappings(content, page_count)

    def _extract_page_mappings_from_json(self, content: str, json_metadata: Dict) -> List[Dict[str, Any]]:
        """Extract page mappings from MinerU JSON metadata if available."""
        # TODO: Implement based on actual MinerU JSON structure
        # This will depend on what page information MinerU provides in the JSON
        pass

    def _detect_page_breaks_in_markdown(self, content: str) -> List[int]:
        """
        Detect page breaks in MinerU markdown output.

        MinerU might insert page break markers or patterns.
        """
        page_break_patterns = [
            r'<!--\s*page\s*break\s*-->',
            r'---\s*PAGE\s*BREAK\s*---',
            r'\n\n---\n\n',
            r'\n\n\*\*\*\n\n',
            r'\\pagebreak',
            r'\\newpage',
            r'\n\n#{1,6}\s+Page\s+\d+',  # Headers like "# Page 2"
        ]

        breaks = []
        for pattern in page_break_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                breaks.append(match.start())

        return sorted(list(set(breaks)))

    def _estimate_page_mappings(self, content: str, page_count: int) -> List[Dict[str, Any]]:
        """Estimate page boundaries based on content length."""
        chars_per_page = len(content) // page_count
        page_mappings = []

        for i in range(page_count):
            start_pos = i * chars_per_page
            end_pos = (i + 1) * chars_per_page if i < page_count - 1 else len(content)

            page_mappings.append({
                'start_pos': start_pos,
                'end_pos': end_pos,
                'page_number': i + 1,
                'extraction_method': 'mineru_ocr_estimated'
            })

        return page_mappings

    def _store_ocr_metadata(self, file_path: Path, ocr_result: Dict[str, Any]):
        """Store OCR processing metadata for analysis."""
        metadata = {
            'file_path': str(file_path),
            'provider': 'mineru',
            'task_id': ocr_result.get('task_id'),
            'data_id': ocr_result.get('data_id'),
            'total_pages': ocr_result.get('total_pages'),
            'extracted_pages': ocr_result.get('extracted_pages'),
            'content_length': len(ocr_result.get('content', '')),
            'has_json_metadata': bool(ocr_result.get('json_metadata')),
            'zip_url': ocr_result.get('zip_url')
        }

        logger.info(f"MinerU OCR metadata for {file_path.name}: {metadata}")
```

---

## 📋 **Phase 4: Testing & Validation (1-2 days)**

### **4.1 API Testing Script**
**Priority**: 🔴 **HIGH**
**Files**: `scripts/test_mineru_api.py`

```python
"""
Test script for MinerU API integration.
"""

import os
import sys
sys.path.append('/home/tuomo/code/fileintel/src')

from fileintel.ocr.mineru_client import create_mineru_client
from fileintel.ocr.file_server import MinerUFileServer
import logging

logging.basicConfig(level=logging.INFO)

def test_mineru_integration():
    """Test complete MinerU integration with a sample PDF."""

    # Check for API token
    api_token = os.getenv('MINERU_API_TOKEN')
    if not api_token:
        print("ERROR: MINERU_API_TOKEN environment variable not set")
        return False

    try:
        # Initialize client
        client = create_mineru_client(api_token=api_token)
        print("✓ MinerU client created successfully")

        # Start file server
        file_server = MinerUFileServer(port=8080)
        file_server.start()
        print("✓ File server started")

        # Test with sample PDF (create a small test PDF first)
        test_pdf_path = "test_sample.pdf"
        if not os.path.exists(test_pdf_path):
            print(f"ERROR: Test PDF not found: {test_pdf_path}")
            print("Create a small test PDF file and run again.")
            return False

        # Host file
        hosted_url = file_server.host_file(test_pdf_path)
        print(f"✓ File hosted at: {hosted_url}")

        # Process PDF
        print("Starting OCR processing...")
        result = client.process_pdf_sync(
            file_url=hosted_url,
            poll_interval=5,
            max_wait=300,  # 5 minutes for testing
            language="en",
            enable_table=True
        )

        if result.markdown_content:
            print(f"✓ OCR completed successfully!")
            print(f"Content length: {len(result.markdown_content)} characters")
            print(f"Pages processed: {result.extracted_pages}/{result.total_pages}")
            print("First 200 characters:")
            print(result.markdown_content[:200])
            return True
        else:
            print(f"✗ OCR failed: {result.error_message}")
            return False

    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        return False
    finally:
        if 'file_server' in locals():
            file_server.stop()

if __name__ == "__main__":
    success = test_mineru_integration()
    sys.exit(0 if success else 1)
```

### **4.2 Performance Benchmarks**
**Priority**: 🟡 **MEDIUM**

**Test Cases**:
- [ ] Small PDF (1-5 pages, <1MB)
- [ ] Medium PDF (10-20 pages, 5-10MB)
- [ ] Large PDF (50-100 pages, 50-100MB)
- [ ] Complex layouts (tables, images, multiple columns)
- [ ] Poor quality scans
- [ ] Non-English content

### **4.3 Integration Tests**
**Priority**: 🔴 **HIGH**

```python
# tests/integration/test_mineru_pipeline.py
import pytest
from pathlib import Path

def test_end_to_end_pdf_processing():
    """Test complete pipeline from PDF to chunked content."""
    # TODO: Implement comprehensive integration test
    pass

def test_markdown_chunking_quality():
    """Test that markdown from MinerU chunks well."""
    # TODO: Test chunking of MinerU markdown output
    pass

def test_error_handling():
    """Test error handling and fallback mechanisms."""
    # TODO: Test various error scenarios
    pass
```

---

## 📋 **Phase 5: Deployment & Monitoring (4-6 hours)**

### **5.1 Environment Configuration**
**Priority**: 🔴 **HIGH**

**Required Environment Variables**:
```bash
# .env file
MINERU_API_TOKEN=your_mineru_token_here
MINERU_BASE_URL=https://mineru.net/api/v4  # or local server URL
```

**Configuration Updates**:
```yaml
# config/default.yaml
ocr:
  primary_engine: "mineru"
  fallback_engines: ["tesseract", "google_vision"]

  mineru:
    api_token: ${MINERU_API_TOKEN}
    base_url: ${MINERU_BASE_URL:-https://mineru.net/api/v4}
    timeout: 1800  # 30 minutes
    poll_interval: 10
    model_version: "vlm"
    enable_formula: false
    enable_table: true
    language: "en"

  file_server_port: 8080
  file_server_host: "192.168.0.247"
  temp_upload_dir: "./temp_uploads"
```

### **5.2 Security Considerations**
**Priority**: 🔴 **HIGH**

**Security Measures**:
- [ ] Secure API token storage (environment variables)
- [ ] File server access restrictions (local network only)
- [ ] Temporary file cleanup after processing
- [ ] Network security (firewall rules)
- [ ] Rate limiting awareness

### **5.3 Monitoring & Logging**
**Priority**: 🟡 **MEDIUM**

**Metrics to Track**:
- OCR success/failure rates
- Processing times per page
- API rate limit usage
- File hosting server performance
- Quality assessments

---

## 🎯 **Success Criteria & Testing Checklist**

### **Functional Requirements**
- [ ] API token authentication works
- [ ] File hosting server operates correctly
- [ ] PDF processing completes successfully
- [ ] Markdown output quality is high
- [ ] Page mappings are accurate
- [ ] Error handling works correctly
- [ ] Fallback mechanisms activate when needed

### **Performance Requirements**
- [ ] <5 minutes for 10-page PDF
- [ ] <20 minutes for 50-page PDF
- [ ] <2000 pages/day usage (within rate limits)
- [ ] 95%+ success rate for quality PDFs

### **Integration Requirements**
- [ ] Seamless replacement of existing OCR
- [ ] Markdown content chunks correctly
- [ ] All existing tests pass
- [ ] No regression in document processing pipeline

---

## 🚨 **Risk Assessment & Mitigation**

### **High Risk Items**

#### **1. API Rate Limits (2000 pages/day)**
- **Risk**: Exceeding daily quota
- **Mitigation**:
  - Monitor daily usage
  - Implement usage tracking
  - Fallback to alternative OCR when quota exceeded
  - Consider multiple API accounts if needed

#### **2. File Hosting Security**
- **Risk**: Exposing sensitive documents publicly
- **Mitigation**:
  - Local network file server only
  - Temporary hosting with cleanup
  - Unique filenames to prevent guessing
  - Consider HTTPS/authentication for file server

#### **3. Network Dependency**
- **Risk**: MinerU cloud service unavailability
- **Mitigation**:
  - Robust fallback to local OCR
  - Retry mechanisms with exponential backoff
  - Health monitoring and alerts

### **Medium Risk Items**
- Processing time variability
- Large file handling (200MB limit)
- JSON metadata structure changes
- Token expiration handling

---

## 📅 **Implementation Timeline**

### **Week 1: Foundation & Setup**
- **Day 1**: Authentication setup and file hosting strategy
- **Day 2**: MinerU client implementation
- **Day 3**: OCR provider integration
- **Day 4**: Document processor implementation
- **Day 5**: Basic testing and validation

### **Week 2: Testing & Production**
- **Day 1**: Comprehensive testing with real PDFs
- **Day 2**: Performance optimization and monitoring
- **Day 3**: Security hardening and error handling
- **Day 4**: Documentation and deployment
- **Day 5**: Production rollout and monitoring

---

## 🔧 **Quick Start Guide**

### **1. Get MinerU API Token**
1. Sign up at https://mineru.net/
2. Generate API token
3. Set environment variable: `export MINERU_API_TOKEN="your_token"`

### **2. Test Basic Functionality**
```bash
cd /home/tuomo/code/fileintel
python scripts/test_mineru_api.py
```

### **3. Configure System**
1. Update `config/default.yaml` with MinerU settings
2. Ensure firewall allows file server port (8080)
3. Test with sample PDF documents

### **4. Monitor Usage**
- Track daily page usage to stay within 2000 page limit
- Monitor processing success rates
- Watch for API errors and rate limiting

This implementation plan will provide a robust, high-quality OCR solution using MinerU's cloud API while maintaining proper security, error handling, and fallback mechanisms.