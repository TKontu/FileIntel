from ..base import OutputFormatter


class MarkdownFormatter(OutputFormatter):
    @property
    def file_extension(self) -> str:
        return "md"

    def format(self, data: dict) -> str:
        """
        Formats the given data as a markdown file.
        """
        return data.get("content", "")
