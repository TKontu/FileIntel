import json
from ..base import OutputFormatter
import re

class JSONFormatter(OutputFormatter):
    @property
    def file_extension(self) -> str:
        return "json"

    def format(self, data: dict) -> str:
        """
        Formats the data as a JSON string.
        """
        content = data.get("content", "")
        
        # Use regex to find the JSON block within the content string
        match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
        if match:
            json_string = match.group(1)
            try:
                # Parse and re-dump to ensure it's valid and nicely formatted
                parsed_json = json.loads(json_string)
                return json.dumps(parsed_json, indent=4)
            except json.JSONDecodeError:
                # If the extracted string is not valid JSON, return it as is.
                return json_string
        
        # If no JSON block is found, return the original content
        return content
