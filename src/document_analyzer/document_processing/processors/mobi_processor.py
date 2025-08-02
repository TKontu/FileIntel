from pathlib import Path
import mobi
from .base import FileReader
from ..core.exceptions import DocumentProcessingException
from bs4 import BeautifulSoup
import os
import shutil
import tempfile

class MOBIProcessor(FileReader):
    def read(self, file_path: Path) -> str:
        """
        Reads the text content from a MOBI file using the mobi library.
        This library unpacks the MOBI file into its source HTML/EPUB.
        """
        if not file_path.exists():
            raise DocumentProcessingException(f"File not found: {file_path}")

        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp()
            # mobi.extract returns the path to the unpacked epub/html
            unpacked_path, _ = mobi.extract(str(file_path), temp_dir)

            content = ""
            if unpacked_path.endswith('.html') or unpacked_path.endswith('.htm'):
                with open(unpacked_path, 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f.read(), "html.parser")
                    content += soup.get_text() + "\n"
            elif unpacked_path.endswith('.epub'):
                # If it's an epub, we can reuse our EPUBProcessor logic
                # For now, a simplified version:
                from .epub_processor import EPUBProcessor
                epub_processor = EPUBProcessor()
                content = epub_processor.read(Path(unpacked_path))
            else:
                 # Fallback for other unpacked formats if necessary
                for root, _, files in os.walk(unpacked_path):
                    for file in files:
                        if file.endswith(('.html', '.htm', '.txt', '.xhtml')):
                            file_path = os.path.join(root, file)
                            with open(file_path, 'r', encoding='utf-8') as f:
                                soup = BeautifulSoup(f.read(), "html.parser")
                                content += soup.get_text() + "\n"

            return content
        except Exception as e:
            raise DocumentProcessingException(f"Error processing MOBI file {file_path}: {e}")
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
