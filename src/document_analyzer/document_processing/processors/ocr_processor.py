from pathlib import Path
from typing import List
import pdfplumber
import pytesseract
from PIL import Image
from io import BytesIO

from ..base import FileReader
from ..elements import DocumentElement, TextElement

class OCRProcessor(FileReader):
    """
    A processor for extracting text from image-based PDFs using Tesseract OCR.
    This serves as a fallback processor.
    """

    def read(self, file_path: Path) -> List[DocumentElement]:
        """
        Reads an image-based PDF, performs OCR on each page, and
        returns the extracted text.

        Args:
            file_path: The path to the PDF file.

        Returns:
            A list of TextElement objects, one for each page with
            successfully extracted text.
        """
        elements = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    # Convert the page to an image. Higher resolution can improve OCR accuracy.
                    img = page.to_image(resolution=300) 
                    
                    # Perform OCR on the image
                    text = pytesseract.image_to_string(img.original)

                    if text and text.strip():
                        metadata = {"source": str(file_path), "page_number": i + 1, "ocr_engine": "tesseract"}
                        elements.append(TextElement(text=text, metadata=metadata))
            
            if not elements:
                return [TextElement(text="[OCR completed, but no text was found.]", metadata={"source": str(file_path)})]

            return elements
        except Exception as e:
            print(f"Error during Tesseract OCR processing of {file_path}: {e}")
            return [TextElement(text=f"[ERROR DURING OCR: {e}]")]
