import json
from ..base import OutputFormatter

class JSONFormatter(OutputFormatter):
    def format(self, data: dict) -> str:
        """
        Formats the data as a JSON string.
        """
        return json.dumps(data, indent=4)
