from abc import ABC, abstractmethod
from pathlib import Path

class FileReader(ABC):
    @abstractmethod
    def read(self, file_path: Path) -> str:
        """Read the content of a file and return it as a string."""
        pass
