# Document Processing Integration Architecture

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import asyncio
import aiohttp
import json


class DocumentType(Enum):
    TEXT_BASED_PDF = "text_pdf"
    SCANNED_PDF = "scanned_pdf"
    MIXED_PDF = "mixed_pdf"
    EPUB = "epub"
    MOBI = "mobi"
    IMAGE = "image"


@dataclass
class DocumentElement:
    """Unified representation of document content"""

    type: str  # text, table, image, heading, etc.
    content: Union[str, Dict[str, Any]]
    bbox: Optional[Dict[str, float]] = None  # bounding box
    page: Optional[int] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = None


@dataclass
class ProcessingResult:
    """Result from document processing"""

    elements: List[DocumentElement]
    metadata: Dict[str, Any]
    processing_method: str
    confidence: float


class DocumentProcessor(ABC):
    """Base class for all document processors"""

    @abstractmethod
    async def can_process(self, file_path: str, document_type: DocumentType) -> bool:
        """Check if this processor can handle the document"""
        pass

    @abstractmethod
    async def process(self, file_path: str) -> ProcessingResult:
        """Process the document and return structured result"""
        pass

    @abstractmethod
    def get_priority(self) -> int:
        """Priority for processor selection (lower = higher priority)"""
        pass


class PDFExtractKitProcessor(DocumentProcessor):
    """Processor using PDF-Extract-Kit for advanced layout understanding"""

    def __init__(self, api_endpoint: str = "http://localhost:8080"):
        self.api_endpoint = api_endpoint
        self.session = None

    async def can_process(self, file_path: str, document_type: DocumentType) -> bool:
        # PDF-Extract-Kit excels at complex PDFs with tables/images
        return document_type in [
            DocumentType.SCANNED_PDF,
            DocumentType.MIXED_PDF,
            DocumentType.IMAGE,
        ]

    def get_priority(self) -> int:
        return 1  # High priority for complex documents

    async def process(self, file_path: str) -> ProcessingResult:
        """Process document using PDF-Extract-Kit API"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        # Send document to PDF-Extract-Kit
        async with self.session.post(
            f"{self.api_endpoint}/extract", data={"file": open(file_path, "rb")}
        ) as response:
            result = await response.json()

        # Convert PDF-Extract-Kit output to unified format
        elements = []

        for page_num, page_data in enumerate(result.get("pages", [])):
            # Process text blocks
            for text_block in page_data.get("text_blocks", []):
                elements.append(
                    DocumentElement(
                        type="text",
                        content=text_block["text"],
                        bbox=text_block["bbox"],
                        page=page_num,
                        confidence=text_block.get("confidence", 1.0),
                        metadata={"reading_order": text_block.get("order", 0)},
                    )
                )

            # Process tables
            for table in page_data.get("tables", []):
                elements.append(
                    DocumentElement(
                        type="table",
                        content={
                            "headers": table["headers"],
                            "rows": table["rows"],
                            "html": table.get("html", ""),
                        },
                        bbox=table["bbox"],
                        page=page_num,
                        confidence=table.get("confidence", 1.0),
                    )
                )

            # Process images
            for image in page_data.get("images", []):
                elements.append(
                    DocumentElement(
                        type="image",
                        content=image.get("description", ""),
                        bbox=image["bbox"],
                        page=page_num,
                        metadata={
                            "image_path": image.get("extracted_path"),
                            "image_type": image.get("type"),
                        },
                    )
                )

        return ProcessingResult(
            elements=elements,
            metadata={
                "total_pages": len(result.get("pages", [])),
                "processing_time": result.get("processing_time"),
                "model_version": result.get("model_version"),
            },
            processing_method="pdf_extract_kit",
            confidence=result.get("overall_confidence", 0.9),
        )


class TraditionalPDFProcessor(DocumentProcessor):
    """Processor for text-based PDFs using traditional methods"""

    def __init__(self):
        pass

    async def can_process(self, file_path: str, document_type: DocumentType) -> bool:
        return document_type == DocumentType.TEXT_BASED_PDF

    def get_priority(self) -> int:
        return 2  # Lower priority, but faster for simple PDFs

    async def process(self, file_path: str) -> ProcessingResult:
        """Process using pdfplumber for text-based PDFs"""
        import pdfplumber

        elements = []

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract text
                text = page.extract_text()
                if text:
                    elements.append(
                        DocumentElement(
                            type="text", content=text, page=page_num, confidence=1.0
                        )
                    )

                # Extract tables
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        headers = table[0] if table else []
                        rows = table[1:] if len(table) > 1 else []
                        elements.append(
                            DocumentElement(
                                type="table",
                                content={"headers": headers, "rows": rows},
                                page=page_num,
                                confidence=0.8,
                            )
                        )

        return ProcessingResult(
            elements=elements,
            metadata={"total_pages": len(pdf.pages)},
            processing_method="pdfplumber",
            confidence=0.9,
        )


class FallbackOCRProcessor(DocumentProcessor):
    """Fallback processor using Tesseract OCR"""

    def __init__(self):
        pass

    async def can_process(self, file_path: str, document_type: DocumentType) -> bool:
        # Can process any document, but lowest priority
        return True

    def get_priority(self) -> int:
        return 10  # Lowest priority - fallback only

    async def process(self, file_path: str) -> ProcessingResult:
        """Fallback OCR processing using Tesseract"""
        import pytesseract
        from pdf2image import convert_from_path

        elements = []

        if file_path.lower().endswith(".pdf"):
            images = convert_from_path(file_path)
            for page_num, image in enumerate(images):
                text = pytesseract.image_to_string(image)
                if text.strip():
                    elements.append(
                        DocumentElement(
                            type="text",
                            content=text,
                            page=page_num,
                            confidence=0.7,  # Lower confidence for OCR
                        )
                    )

        return ProcessingResult(
            elements=elements,
            metadata={"ocr_engine": "tesseract"},
            processing_method="tesseract_ocr",
            confidence=0.7,
        )


class DocumentTypeDetector:
    """Detects document type to choose optimal processor"""

    @staticmethod
    async def detect_type(file_path: str) -> DocumentType:
        """Detect document type based on content analysis"""
        import pdfplumber

        if not file_path.lower().endswith(".pdf"):
            if file_path.lower().endswith(".epub"):
                return DocumentType.EPUB
            elif file_path.lower().endswith(".mobi"):
                return DocumentType.MOBI
            else:
                return DocumentType.IMAGE

        try:
            with pdfplumber.open(file_path) as pdf:
                text_pages = 0
                total_pages = len(pdf.pages)

                # Sample first few pages to determine type
                sample_pages = min(3, total_pages)

                for page in pdf.pages[:sample_pages]:
                    text = page.extract_text()
                    if text and len(text.strip()) > 100:
                        text_pages += 1

                text_ratio = text_pages / sample_pages

                if text_ratio > 0.8:
                    return DocumentType.TEXT_BASED_PDF
                elif text_ratio > 0.2:
                    return DocumentType.MIXED_PDF
                else:
                    return DocumentType.SCANNED_PDF

        except Exception:
            return DocumentType.SCANNED_PDF


class UnifiedDocumentProcessor:
    """Main processor that orchestrates different processing engines"""

    def __init__(self):
        self.processors = [
            PDFExtractKitProcessor(),
            TraditionalPDFProcessor(),
            FallbackOCRProcessor(),
        ]
        self.type_detector = DocumentTypeDetector()

    async def process_document(self, file_path: str) -> ProcessingResult:
        """Main entry point for document processing"""

        # 1. Detect document type
        doc_type = await self.type_detector.detect_type(file_path)

        # 2. Find suitable processors
        suitable_processors = []
        for processor in self.processors:
            if await processor.can_process(file_path, doc_type):
                suitable_processors.append(processor)

        # 3. Sort by priority
        suitable_processors.sort(key=lambda p: p.get_priority())

        # 4. Try processors in order until success
        last_error = None
        for processor in suitable_processors:
            try:
                result = await processor.process(file_path)

                # Quality check - ensure we got meaningful content
                if self._validate_result(result):
                    return result

            except Exception as e:
                last_error = e
                continue

        # If all processors failed
        raise Exception(f"All processors failed. Last error: {last_error}")

    def _validate_result(self, result: ProcessingResult) -> bool:
        """Validate that processing result contains meaningful content"""
        if not result.elements:
            return False

        # Check if we have some text content
        text_elements = [e for e in result.elements if e.type == "text"]
        if not text_elements:
            return False

        # Check total text length
        total_text = " ".join(
            [e.content for e in text_elements if isinstance(e.content, str)]
        )
        if len(total_text.strip()) < 50:  # Minimum meaningful content
            return False

        return True


class DocumentToLLMBridge:
    """Converts processed document elements to LLM-ready format"""

    def __init__(self):
        self.unified_processor = UnifiedDocumentProcessor()

    async def prepare_for_llm(
        self, file_path: str, preserve_structure: bool = True
    ) -> Dict[str, Any]:
        """Convert document to LLM-optimized format"""

        # Process document
        result = await self.unified_processor.process_document(file_path)

        if preserve_structure:
            return self._structured_format(result)
        else:
            return self._flattened_format(result)

    def _structured_format(self, result: ProcessingResult) -> Dict[str, Any]:
        """Preserve document structure for complex analysis"""

        formatted = {
            "metadata": result.metadata,
            "processing_method": result.processing_method,
            "confidence": result.confidence,
            "content": {"pages": {}, "tables": [], "images": []},
        }

        # Group elements by page
        for element in result.elements:
            page_num = element.page or 0

            if page_num not in formatted["content"]["pages"]:
                formatted["content"]["pages"][page_num] = {
                    "text_blocks": [],
                    "tables": [],
                    "images": [],
                }

            if element.type == "text":
                formatted["content"]["pages"][page_num]["text_blocks"].append(
                    {
                        "content": element.content,
                        "confidence": element.confidence,
                        "bbox": element.bbox,
                    }
                )
            elif element.type == "table":
                table_data = {
                    "content": element.content,
                    "page": page_num,
                    "confidence": element.confidence,
                }
                formatted["content"]["tables"].append(table_data)
                formatted["content"]["pages"][page_num]["tables"].append(table_data)
            elif element.type == "image":
                image_data = {
                    "description": element.content,
                    "page": page_num,
                    "metadata": element.metadata,
                }
                formatted["content"]["images"].append(image_data)
                formatted["content"]["pages"][page_num]["images"].append(image_data)

        return formatted

    def _flattened_format(self, result: ProcessingResult) -> Dict[str, Any]:
        """Create simple text format for basic analysis"""

        text_elements = [e for e in result.elements if e.type == "text"]
        full_text = "\n\n".join(
            [e.content for e in text_elements if isinstance(e.content, str)]
        )

        # Extract tables as text
        table_elements = [e for e in result.elements if e.type == "table"]
        table_text = ""
        for table in table_elements:
            if isinstance(table.content, dict):
                headers = table.content.get("headers", [])
                rows = table.content.get("rows", [])

                if headers:
                    table_text += "\n\nTable:\n"
                    table_text += " | ".join(headers) + "\n"
                    table_text += "-" * len(" | ".join(headers)) + "\n"

                    for row in rows:
                        if row:
                            table_text += " | ".join([str(cell) for cell in row]) + "\n"

        return {
            "text": full_text + table_text,
            "metadata": result.metadata,
            "processing_method": result.processing_method,
            "confidence": result.confidence,
        }


# Usage example
async def main():
    bridge = DocumentToLLMBridge()

    # Process a complex PDF
    result = await bridge.prepare_for_llm(
        "complex_document.pdf", preserve_structure=True
    )

    print(f"Processed with: {result['processing_method']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Found {len(result['content']['tables'])} tables")

    # For simple LLM analysis
    await bridge.prepare_for_llm("simple_document.pdf", preserve_structure=False)

    # Now ready to send to LLM
    """
    Analyze the following document:

    {simple_result['text']}

    [Your analysis instructions here]
    """
