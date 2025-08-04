import magic
from pathlib import Path

class DocumentTypeDetector:
    def __init__(self):
        self.magic = magic.Magic(mime=True)

    def detect_mime_type(self, file_path: Path) -> str:
        """Detect the MIME type of a file."""
        return self.magic.from_file(str(file_path))
