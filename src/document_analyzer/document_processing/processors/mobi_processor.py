from pathlib import Path
from typing import List
import mobi
from bs4 import BeautifulSoup
import os
import shutil
import tempfile
import logging

from ..base import FileReader
from ..elements import DocumentElement, TextElement
from ...core.exceptions import DocumentProcessingException

logger = logging.getLogger(__name__)


class MOBIReader(FileReader):
    def read(
        self, file_path: Path, adapter: logging.LoggerAdapter = None
    ) -> List[DocumentElement]:
        """
        Reads the text content from a MOBI file by unpacking it and
        processing the resulting HTML/EPUB content.
        """
        log = adapter or logger
        if not file_path.exists():
            raise DocumentProcessingException(f"File not found: {file_path}")

        temp_dir = tempfile.mkdtemp()
        elements = []
        try:
            log.info(f"Extracting MOBI file: {file_path.name}")
            # mobi.extract returns the path to the unpacked content
            unpacked_dir, _ = mobi.extract(str(file_path), temp_dir)
            log.info(f"Successfully extracted MOBI file to: {unpacked_dir}")

            # Walk through the unpacked directory and parse text from HTML files
            for root, _, files in os.walk(unpacked_dir):
                for file in sorted(files):  # Sort files to maintain order
                    if file.endswith((".html", ".htm", ".xhtml")):
                        full_path = os.path.join(root, file)
                        with open(full_path, "r", encoding="utf-8") as f:
                            soup = BeautifulSoup(f.read(), "html.parser")
                            text = soup.get_text()
                            if text and text.strip():
                                metadata = {"source": str(file_path), "part": file}
                                elements.append(
                                    TextElement(text=text, metadata=metadata)
                                )
            log.info(f"Successfully processed MOBI file: {file_path.name}")
            return elements
        except Exception as e:
            log.error(f"Error processing MOBI file {file_path}: {e}", exc_info=True)
            raise DocumentProcessingException(
                f"Error processing MOBI file {file_path}: {e}"
            )
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
