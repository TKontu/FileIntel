from abc import ABC, abstractmethod


class OutputFormatter(ABC):
    @property
    @abstractmethod
    def file_extension(self) -> str:
        """
        Returns the file extension for the format.
        """
        pass

    @abstractmethod
    def format(self, data: dict) -> str:
        """
        Formats the data into a string.
        """
        pass
