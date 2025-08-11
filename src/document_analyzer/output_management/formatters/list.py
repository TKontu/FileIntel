from ..base import OutputFormatter

class ListFormatter(OutputFormatter):
    @property
    def file_extension(self) -> str:
        return "txt"

    def format(self, data: dict) -> str:
        """
        Formats the data as a bulleted list.
        Assumes the data is a dictionary with a "response" key
        which is a list of strings.
        """
        items = data.get("response", [])
        if isinstance(items, list):
            return "\n".join(f"- {item}" for item in items)
        return str(items)

