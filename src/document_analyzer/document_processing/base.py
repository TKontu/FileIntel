from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from .elements import DocumentElement

class FileReader(ABC):
    @abstractmethod
    def read(self, file_path: Path) -> List[DocumentElement]:
        """Read the content of a file and return it as a list of DocumentElement objects."""
        pass
