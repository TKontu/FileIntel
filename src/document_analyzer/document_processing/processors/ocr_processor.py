from pathlib import Path
from typing import List
import pdfplumber
import pytesseract
from PIL import Image
import logging

from ..base import FileReader
from ..elements import DocumentElement, TextElement

logger = logging.getLogger(__name__)


class OCRProcessor(FileReader):
    """
    A processor for extracting text from image-based PDFs using Tesseract OCR.
    This serves as a fallback processor.
    """

    def read(
        self, file_path: Path, adapter: logging.LoggerAdapter = None
    ) -> List[DocumentElement]:
        """
        Reads an image-based PDF, performs OCR on each page, and
        returns the extracted text.

        Args:
            file_path: The path to the PDF file.
            adapter: A logger adapter for contextual logging.

        Returns:
            A list of TextElement objects, one for each page with
            successfully extracted text.
        """
        log = adapter or logger
        elements = []
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                log.info(
                    f"Starting OCR processing for {file_path.name} ({total_pages} pages)."
                )

                for i, page in enumerate(pdf.pages):
                    # Convert the page to an image. Higher resolution can improve OCR accuracy.
                    img = page.to_image(resolution=300)

                    # Perform OCR on the image
                    text = pytesseract.image_to_string(img.original)

                    if text and text.strip():
                        metadata = {
                            "source": str(file_path),
                            "page_number": i + 1,
                            "ocr_engine": "tesseract",
                        }
                        elements.append(TextElement(text=text, metadata=metadata))

                    log.info(
                        f"Processed page {i + 1}/{total_pages} of {file_path.name}."
                    )

            if not elements:
                log.warning(
                    f"OCR completed for {file_path.name}, but no text was found."
                )
                return [
                    TextElement(
                        text="[OCR completed, but no text was found.]",
                        metadata={"source": str(file_path)},
                    )
                ]

            log.info(f"Successfully finished OCR processing for {file_path.name}.")
            return elements
        except Exception as e:
            log.error(
                f"Error during Tesseract OCR processing of {file_path}: {e}",
                exc_info=True,
            )
            return [TextElement(text=f"[ERROR DURING OCR: {e}]")]
