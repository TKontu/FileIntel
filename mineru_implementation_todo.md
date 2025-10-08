# MinerU OCR Integration Implementation Plan - UPDATED

## 🚨 **CRITICAL UPDATE: CURRENT IMPLEMENTATION FUNDAMENTALLY BROKEN**

**Status**: REQUIRES COMPLETE REWRITE
**Reason**: Analysis of actual MinerU output reveals massive data loss and wrong assumptions

### **🔍 Key Discoveries from Sample Data**
- **Rich JSON Metadata**: MinerU provides perfect page mapping via `page_idx` (0-11 for 12 pages)
- **Element Classification**: Text, headers, footers, tables with precise coordinates
- **Image Extraction**: Separate images folder with referenced files
- **Multiple Formats**: content_list.json, model.json, middle.json, layout.pdf
- **Current Loss**: Implementation throws away 80% of MinerU's value

## 🎯 **REVISED Project Overview**

Replace current OCR with MinerU API, leveraging **ALL** of its rich structured output including JSON metadata for precise page mapping and element classification.

### **Key Objectives - UPDATED**
- **Quality**: High-accuracy OCR with **full JSON metadata preservation**
- **Integration**: **JSON-first** approach using MinerU's structured data
- **Architecture**: Leverage MinerU's element classification and page mapping
- **Performance**: **No more guesswork** - use actual page indices and coordinates
- **Reliability**: Robust error handling with shared folder approach

### **MinerU Output Structure - ACTUAL**
- **Service**: Local instance at http://192.168.0.136/api/v4
- **Authentication**: Bearer token required
- **Processing**: Async with polling for completion status
- **Output**: ZIP containing:
  - `*.md` - Clean markdown (302 lines for 12-page doc)
  - `*_content_list.json` - 166 elements with page_idx, type, bbox
  - `*_model.json` - Page-structured layout data (12 pages)
  - `*_middle.json` - Processing metadata
  - `images/` - Extracted image files

### **Architecture Integration Points - REVISED**
- 🔄 **Processor Interface**: JSON-first processing with markdown fallback
- 🔄 **Page Mapping**: Use `page_idx` from content_list.json (perfect accuracy)
- 🔄 **Element Types**: Preserve text/header/footer/table classification
- 🔄 **Storage Layer**: Enhanced metadata storage with element types
- ✅ **Configuration**: Shared folder approach (implemented)
- ✅ **Fallback Strategy**: Graceful degradation (implemented)

---

## 🚨 **IMMEDIATE FIXES REQUIRED (Before Any Testing)**

### **CRITICAL ISSUE #1: Complete JSON Extraction Rewrite**
**Priority**: 🔴 **BLOCKER**
**Files**: `src/fileintel/document_processing/processors/mineru_pdf.py`
**Problem**: Current implementation only extracts markdown, ignoring 80% of MinerU's value

**REQUIRED CHANGES**:
```python
def _extract_results(self, zip_content: bytes) -> Tuple[str, Dict[str, Any]]:
    """Extract both markdown and ALL JSON metadata from ZIP."""
    results = {
        'markdown': None,
        'content_list': None,
        'model_data': None,
        'middle_data': None,
        'images': []
    }

    with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:
        for filename in zip_file.namelist():
            if filename.endswith('.md'):
                results['markdown'] = zip_file.read(filename).decode('utf-8')
            elif filename.endswith('_content_list.json'):
                results['content_list'] = json.loads(zip_file.read(filename).decode('utf-8'))
            elif filename.endswith('_model.json'):
                results['model_data'] = json.loads(zip_file.read(filename).decode('utf-8'))
            elif filename.endswith('_middle.json'):
                results['middle_data'] = json.loads(zip_file.read(filename).decode('utf-8'))
            elif filename.startswith('images/'):
                results['images'].append(filename)

    return results['markdown'], results
```

### **CRITICAL ISSUE #2: Complete Page Mapping Rewrite**
**Priority**: 🔴 **BLOCKER**
**Files**: `src/fileintel/document_processing/processors/mineru_pdf.py`
**Problem**: Page splitting based on non-existent markdown patterns instead of using `page_idx`

**REQUIRED CHANGES**:
```python
def _create_elements_from_content_list(self, content_list: List[Dict], file_path: Path) -> List[TextElement]:
    """Create TextElements using ACTUAL page indices from MinerU."""
    if not content_list:
        # Fallback to markdown-only approach
        return [TextElement(text=self.markdown_content, metadata={'page_number': 1})]

    # Group elements by actual page_idx
    elements_by_page = {}
    for item in content_list:
        page_idx = item['page_idx']
        if page_idx not in elements_by_page:
            elements_by_page[page_idx] = []

        # Include element type information
        element_info = {
            'text': item['text'],
            'type': item.get('type', 'text'),
            'bbox': item.get('bbox', [])
        }
        elements_by_page[page_idx].append(element_info)

    # Create TextElements with accurate page mapping
    elements = []
    for page_idx in sorted(elements_by_page.keys()):
        page_elements = elements_by_page[page_idx]

        # Combine text from all elements on this page
        page_text = '\n'.join([elem['text'] for elem in page_elements])

        # Count element types
        element_types = {}
        for elem in page_elements:
            elem_type = elem['type']
            element_types[elem_type] = element_types.get(elem_type, 0) + 1

        elements.append(TextElement(
            text=page_text,
            metadata={
                'source': str(file_path),
                'page_number': page_idx + 1,  # Convert 0-based to 1-based
                'extraction_method': 'mineru_json',
                'element_count': len(page_elements),
                'element_types': element_types,
                'has_coordinates': bool(any(elem['bbox'] for elem in page_elements))
            }
        ))

    return elements
```

### **CRITICAL ISSUE #3: Shared Folder Error Handling**
**Priority**: 🔴 **BLOCKER**
**Files**: `src/fileintel/document_processing/processors/mineru_pdf.py`
**Problem**: Will fail on most systems due to permission errors

**REQUIRED CHANGES**:
```python
def __init__(self, config=None):
    from fileintel.core.config import get_config
    self.config = config or get_config()
    self.shared_folder = Path(self.config.document_processing.mineru.shared_folder_path)

    # Validate shared folder with proper error handling
    try:
        self.shared_folder.mkdir(exist_ok=True, parents=True)
        # Test write permissions
        test_file = self.shared_folder / f"test_{uuid.uuid4()}.tmp"
        test_file.write_text("test")
        test_file.unlink()
    except PermissionError as e:
        raise DocumentProcessingError(f"Cannot create/write to shared folder {self.shared_folder}: {e}")
    except Exception as e:
        raise DocumentProcessingError(f"Shared folder setup failed: {e}")
```

---

## 📋 **REVISED Phase 1: JSON-First Implementation (6-8 hours)**

### **1.1 ✅ Configuration** (COMPLETED)
**Status**: Already implemented with shared folder support

### **1.2 🔄 JSON Extraction Engine** (CRITICAL)
**Priority**: 🔴 **BLOCKER**
**Files**: `src/fileintel/document_processing/processors/mineru_pdf.py`

**Implementation Strategy**:
- 🔄 **JSON-First**: Extract and parse ALL JSON files from ZIP
- 🔄 **Perfect Page Mapping**: Use `page_idx` from content_list.json
- 🔄 **Element Preservation**: Maintain text/header/footer/table types
- 🔄 **Coordinate Data**: Store bounding box information
- 🔄 **Image Handling**: Track extracted images and references

```python
import requests
import zipfile
import json
import time
import shutil
import uuid
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging

from ..elements import DocumentElement, TextElement
from .traditional_pdf import validate_file_for_processing, DocumentProcessingError

logger = logging.getLogger(__name__)

class MinerUPDFProcessor:
    """PDF processor using MinerU API for high-quality OCR and markdown output."""

    def __init__(self, config=None):
        from fileintel.core.config import get_config
        self.config = config or get_config()
        self.temp_dir = Path("/tmp/mineru_uploads")
        self.temp_dir.mkdir(exist_ok=True)

    def read(self, file_path: Path, adapter: logging.LoggerAdapter = None) -> Tuple[List[DocumentElement], Dict[str, Any]]:
        """Process PDF using MinerU API and return DocumentElements."""
        log = adapter or logger
        validate_file_for_processing(file_path, ".pdf")

        try:
            # Host file temporarily
            file_url = self._host_file_temporarily(file_path)

            # Process with MinerU
            markdown_content, metadata = self._process_with_mineru(file_url, file_path, log)

            # Convert markdown to TextElements
            elements = [TextElement(
                text=markdown_content,
                metadata={
                    "source": str(file_path),
                    "extraction_method": "mineru_ocr",
                    "format": "markdown"
                }
            )]

            return elements, metadata

        except Exception as e:
            log.error(f"MinerU processing failed for {file_path}: {e}")
            # Fallback to traditional processor
            from .traditional_pdf import PDFProcessor
            fallback = PDFProcessor()
            return fallback.read(file_path, adapter)
        finally:
            self._cleanup_temp_file(file_path)

    def _host_file_temporarily(self, file_path: Path) -> str:
        """Copy file to temp location and return URL."""
        unique_name = f"{uuid.uuid4()}_{file_path.name}"
        hosted_path = self.temp_dir / unique_name
        shutil.copy2(file_path, hosted_path)
        return f"http://192.168.0.247:8080/uploads/{unique_name}"

    def _process_with_mineru(self, file_url: str, file_path: Path, log) -> Tuple[str, Dict]:
        """Process PDF with MinerU API and return markdown content."""
        # Submit task
        task_id = self._submit_task(file_url)
        log.info(f"MinerU task {task_id} submitted for {file_path.name}")

        # Poll for completion
        for _ in range(180):  # 30 minutes max
            status_data = self._get_task_status(task_id)

            if status_data["state"] == "done":
                zip_content = self._download_results(status_data["full_zip_url"])
                return self._extract_markdown(zip_content), status_data
            elif status_data["state"] == "failed":
                raise DocumentProcessingError(f"MinerU processing failed: {status_data.get('err_msg')}")

            time.sleep(10)

        raise DocumentProcessingError("MinerU processing timed out")

    def _submit_task(self, file_url: str) -> str:
        """Submit processing task to MinerU API."""
        url = f"{self.config.document_processing.mineru.base_url}/api/v4/extract/task"
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

        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()

        result = response.json()
        if result.get("code") != 0:
            raise DocumentProcessingError(f"MinerU API error: {result.get('msg')}")

        return result["data"]["task_id"]

    def _get_task_status(self, task_id: str) -> Dict:
        """Get task status from MinerU API."""
        url = f"{self.config.document_processing.mineru.base_url}/api/v4/extract/task/{task_id}"
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

    def _extract_markdown(self, zip_content: bytes) -> str:
        """Extract markdown content from ZIP file."""
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:
            for filename in zip_file.namelist():
                if filename.endswith('.md'):
                    return zip_file.read(filename).decode('utf-8')
        raise DocumentProcessingError("No markdown file found in MinerU results")

    def _cleanup_temp_file(self, original_path: Path):
        """Clean up temporary hosted file."""
        # Cleanup logic here
        pass
```

### **2.2 Task Integration**
**Priority**: 🔴 **HIGH**
**Files**: `src/fileintel/tasks/document_tasks.py`

**Update processor mapping**:
```python
# Line 58-62: Update processor mapping
processors = {
    ".pdf": MinerUPDFProcessor if config.document_processing.primary_pdf_processor == "mineru" else TraditionalPDFProcessor,
    ".epub": EPUBProcessor,
    ".mobi": MOBIProcessor,
}
```

**Import statement**:
```python
# Add after line 35
from fileintel.document_processing.processors.mineru_pdf import MinerUPDFProcessor
```

---

## 📋 **Phase 3: Testing & Environment Setup (2-4 hours)**

### **3.1 Environment Configuration**
**Priority**: 🔴 **HIGH**

**Required Environment Variables**:
```bash
# .env file
MINERU_API_TOKEN=your_mineru_token_here
```

**Configuration in config/default.yaml**:
```yaml
document_processing:
  primary_pdf_processor: "mineru"  # or "traditional"
  fallback_enabled: true

  mineru:
    api_token: ${MINERU_API_TOKEN}
    base_url: "http://192.168.0.136/api/v4"
    timeout: 600
    poll_interval: 10
    max_retries: 3
    model_version: "vlm"
    enable_formula: false
    enable_table: true
    language: "en"
```

### **3.2 Simple Test Script**
**Priority**: 🔴 **HIGH**
**Files**: `scripts/test_mineru_integration.py`

```python
#!/usr/bin/env python3
"""Test MinerU integration with a sample PDF."""

import sys
import os
sys.path.append('src')

from pathlib import Path
from fileintel.document_processing.processors.mineru_pdf import MinerUPDFProcessor
from fileintel.core.config import get_config
import logging

logging.basicConfig(level=logging.INFO)

def test_mineru_processing():
    """Test MinerU processor with sample PDF."""
    # Check for test PDF
    test_pdf = Path("test_document.pdf")
    if not test_pdf.exists():
        print(f"Error: Test PDF not found: {test_pdf}")
        return False

    # Check API token
    config = get_config()
    if not config.document_processing.mineru.api_token:
        print("Error: MINERU_API_TOKEN not configured")
        return False

    try:
        # Test processor
        processor = MinerUPDFProcessor()
        elements, metadata = processor.read(test_pdf)

        print(f"✓ Processing successful!")
        print(f"  Elements: {len(elements)}")
        print(f"  Content length: {len(elements[0].text) if elements else 0}")
        print(f"  Metadata keys: {list(metadata.keys())}")

        return True

    except Exception as e:
        print(f"✗ Processing failed: {e}")
        return False

if __name__ == "__main__":
    success = test_mineru_processing()
    sys.exit(0 if success else 1)
```

---

## 🎯 **REVISED Implementation Timeline**

### **IMMEDIATE (Day 1): Critical Fixes**
1. **Rewrite JSON Extraction** (2-3 hours)
   - Replace `_extract_markdown()` with `_extract_results()`
   - Parse content_list.json, model.json, middle.json
   - Handle image file references

2. **Rewrite Page Mapping** (2-3 hours)
   - Replace `_split_markdown_by_pages()` with `_create_elements_from_content_list()`
   - Use actual `page_idx` values from JSON
   - Preserve element types and coordinates

3. **Fix Shared Folder Handling** (1-2 hours)
   - Add proper permission error handling
   - Test write permissions during initialization
   - Clear error messages for setup failures

### **Day 2: Enhanced Features**
1. **Element Type Preservation** (2-3 hours)
   - Distinguish headers, footers, tables from regular text
   - Store element type metadata for chunking
   - Preserve coordinate information

2. **Image Handling** (2-3 hours)
   - Extract and store image references
   - Handle image files in ZIP
   - Link images to text elements

3. **Testing & Validation** (2-3 hours)
   - Test with actual MinerU output samples
   - Validate page mapping accuracy
   - Verify metadata preservation

### **Day 3: Integration & Polish**
1. **Storage Integration** (2-3 hours)
   - Enhanced chunk metadata storage
   - Element type indexing
   - Coordinate data preservation

2. **Error Handling & Monitoring** (2-3 hours)
   - Comprehensive error handling
   - Processing metrics
   - Quality validation

3. **Documentation & Deployment** (2-3 hours)
   - Updated configuration guide
   - Testing documentation
   - Production deployment checklist

## 🚨 **CRITICAL STATUS**

**Current Implementation**: FUNDAMENTALLY BROKEN
**Blocks Testing**: YES - Will fail immediately
**Data Loss**: 80% of MinerU's value ignored
**Page Mapping**: Completely wrong approach

**CANNOT PROCEED** with any testing until the three critical fixes above are implemented.

## 🔧 **Success Criteria - UPDATED**

### **Functional Requirements**
- [ ] Extract and parse ALL JSON files from MinerU ZIP
- [ ] Use `page_idx` for perfect page mapping (not markdown patterns)
- [ ] Preserve element types (text, header, footer, table)
- [ ] Store coordinate/bounding box data
- [ ] Handle extracted images and references
- [ ] Robust shared folder permission handling

### **Quality Requirements**
- [ ] **Zero data loss** from MinerU output
- [ ] **Perfect page boundaries** using JSON data
- [ ] **Element type preservation** for enhanced chunking
- [ ] **Proper error handling** for all failure modes
- [ ] **Clean fallback** to traditional OCR when needed

The implementation requires **complete rewrite** of core extraction and page mapping logic before any testing can occur.