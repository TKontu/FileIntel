from ..base import OutputFormatter

class MarkdownFormatter(OutputFormatter):
    def format(self, data: dict) -> str:
        """
        Formats the given data as a markdown file.
        """
        return data.get("content", "")
