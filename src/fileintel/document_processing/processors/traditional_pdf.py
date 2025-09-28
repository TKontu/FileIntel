import pdfplumber
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging
import os
import stat

# FileReader import removed - no longer using abstract base
from ..elements import DocumentElement, TextElement

# Removed custom exception import - using built-in exceptions

# Optional OCR imports with fallback
try:
    import pytesseract
    from PIL import Image

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

logger = logging.getLogger(__name__)

# Constants for text quality assessment
MIN_TEXT_THRESHOLD = 100  # Minimum characters to consider text extraction successful
MIN_TEXT_RATIO = 0.5  # Minimum ratio of non-whitespace characters

# File validation constants
MAX_FILE_SIZE_MB = 500  # Maximum file size in MB
MIN_FILE_SIZE_BYTES = 10  # Minimum file size in bytes (avoid empty files)


def validate_file_for_processing(
    file_path: Path, expected_extension: str = None
) -> None:
    """
    Comprehensive file validation for document processing.

    Args:
        file_path: Path to the file to validate
        expected_extension: Expected file extension (e.g., '.pdf')

    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file is not readable
        ValueError: If file validation fails (size, format, etc.)
    """
    # Check existence
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check if it's actually a file (not a directory)
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    # Check file permissions
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"File is not readable: {file_path}")

    # Check file size
    try:
        file_size = file_path.stat().st_size
        if file_size < MIN_FILE_SIZE_BYTES:
            raise ValueError(f"File is too small ({file_size} bytes): {file_path}")
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(
                f"File is too large ({file_size / (1024 * 1024):.1f}MB > {MAX_FILE_SIZE_MB}MB): {file_path}"
            )
    except OSError as e:
        raise ValueError(f"Could not access file stats: {file_path} - {e}")

    # Check file extension if specified
    if expected_extension:
        actual_extension = file_path.suffix.lower()
        if actual_extension != expected_extension.lower():
            raise ValueError(
                f"File extension mismatch: expected {expected_extension}, got {actual_extension} for {file_path}"
            )

    logger.debug(f"File validation passed for {file_path}")


class DocumentProcessingError(Exception):
    """Custom exception for document processing errors."""

    pass


class UnsupportedFileTypeError(ValueError):
    """Raised when file type is not supported for processing."""

    pass


class FileCorruptionError(DocumentProcessingError):
    """Raised when file appears to be corrupted or unreadable."""

    pass


class PDFProcessor:
    def read(
        self, file_path: Path, adapter: logging.LoggerAdapter = None
    ) -> Tuple[List[DocumentElement], Dict[str, Any]]:
        """
        Reads text content from a PDF file using traditional text extraction.
        Falls back to OCR if text extraction yields insufficient content.
        Returns a tuple containing a list of TextElements and a metadata dictionary.
        """
        log = adapter or logger

        # Comprehensive file validation
        validate_file_for_processing(file_path, ".pdf")

        elements = []
        doc_metadata = {}

        try:
            with pdfplumber.open(file_path) as pdf:
                doc_metadata = pdf.metadata or {}
                total_pages = len(pdf.pages)
                log.info(f"Processing PDF {file_path.name} ({total_pages} pages)")

                # First, try traditional text extraction
                extracted_elements = self._extract_text_traditional(pdf, file_path, log)

                # Check if text extraction was successful
                if self._is_text_extraction_sufficient(extracted_elements):
                    log.info(
                        f"Traditional text extraction successful for {file_path.name}"
                    )
                    return extracted_elements, doc_metadata

                # Fallback to OCR if traditional extraction failed and OCR is available
                if OCR_AVAILABLE:
                    log.warning(
                        f"Traditional text extraction insufficient for {file_path.name}, trying OCR fallback"
                    )
                    ocr_elements = self._extract_text_ocr(pdf, file_path, log)
                    doc_metadata["ocr_processed"] = True

                    if ocr_elements:
                        log.info(f"OCR extraction successful for {file_path.name}")
                        return ocr_elements, doc_metadata

                log.warning(
                    f"Both traditional and OCR extraction failed for {file_path.name}"
                )
                # Return whatever was extracted, even if minimal
                return (
                    extracted_elements
                    or [
                        TextElement(
                            text="[No readable text found in PDF]",
                            metadata={
                                "source": str(file_path),
                                "extraction_failed": True,
                            },
                        )
                    ],
                    doc_metadata,
                )

        except pdfplumber.exceptions.PDFSyntaxError as e:
            raise FileCorruptionError(
                f"PDF file appears to be corrupted: {file_path} - {e}"
            )
        except Exception as e:
            raise DocumentProcessingError(f"Error processing PDF file {file_path}: {e}")

    def _extract_text_traditional(
        self, pdf, file_path: Path, log
    ) -> List[DocumentElement]:
        """Extract text using traditional pdfplumber text extraction."""
        elements = []
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                page_metadata = {
                    "source": str(file_path),
                    "page_number": i + 1,
                    "extraction_method": "traditional",
                }
                elements.append(TextElement(text=page_text, metadata=page_metadata))
            log.debug(
                f"Extracted text from page {i + 1}/{len(pdf.pages)} of {file_path.name}"
            )
        return elements

    def _extract_text_ocr(self, pdf, file_path: Path, log) -> List[DocumentElement]:
        """Extract text using OCR as fallback."""
        if not OCR_AVAILABLE:
            return []

        elements = []
        for i, page in enumerate(pdf.pages):
            try:
                # Convert page to image for OCR
                img = page.to_image(resolution=300)
                text = pytesseract.image_to_string(img.original)

                if text and text.strip():
                    metadata = {
                        "source": str(file_path),
                        "page_number": i + 1,
                        "extraction_method": "ocr",
                        "ocr_engine": "tesseract",
                    }
                    elements.append(TextElement(text=text, metadata=metadata))
                log.debug(
                    f"OCR processed page {i + 1}/{len(pdf.pages)} of {file_path.name}"
                )
            except Exception as e:
                log.warning(f"OCR failed for page {i + 1} of {file_path.name}: {e}")
                continue
        return elements

    def _is_text_extraction_sufficient(self, elements: List[DocumentElement]) -> bool:
        """Check if traditional text extraction yielded sufficient content."""
        if not elements:
            return False

        total_text = " ".join(elem.text for elem in elements if hasattr(elem, "text"))
        total_chars = len(total_text)
        non_whitespace_chars = len(
            total_text.replace(" ", "").replace("\n", "").replace("\t", "")
        )

        # Check minimum character threshold and text quality ratio
        return (
            total_chars >= MIN_TEXT_THRESHOLD
            and non_whitespace_chars / max(total_chars, 1) >= MIN_TEXT_RATIO
        )


# Backward compatibility alias
TraditionalPDFProcessor = PDFProcessor
