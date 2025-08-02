from abc import ABC, abstractmethod

class OutputFormatter(ABC):
    @abstractmethod
    def format(self, data: dict) -> str:
        """
        Formats the data into a string.
        """
        pass
