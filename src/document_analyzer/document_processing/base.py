from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging
from .elements import DocumentElement


class FileReader(ABC):
    @abstractmethod
    def read(
        self, file_path: Path, adapter: logging.LoggerAdapter = None
    ) -> Tuple[List[DocumentElement], Dict[str, Any]]:
        """
        Read the content and metadata of a file.
        Returns a tuple containing a list of DocumentElement objects and a dictionary of metadata.
        """
        pass
