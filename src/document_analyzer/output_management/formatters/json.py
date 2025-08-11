import json
from ..base import OutputFormatter

class JSONFormatter(OutputFormatter):
    @property
    def file_extension(self) -> str:
        return "json"

    def format(self, data: dict) -> str:
        """
        Formats the data as a JSON string.
        """
        return json.dumps(data, indent=4)
