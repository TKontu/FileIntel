from .formatters.json import JSONFormatter
from .formatters.markdown import MarkdownFormatter
from .formatters.essay import EssayFormatter
from .formatters.list import ListFormatter
from .formatters.table import TableFormatter
from .base import OutputFormatter

class FormatterFactory:
    formatters = {
        "json": JSONFormatter,
        "markdown": MarkdownFormatter,
        "essay": EssayFormatter,
        "list": ListFormatter,
        "table": TableFormatter,
    }

    @staticmethod
    def get_formatter(format_name: str) -> OutputFormatter:
        formatter = FormatterFactory.formatters.get(format_name.lower())
        if not formatter:
            raise ValueError(f"Unsupported format: {format_name}")
        return formatter()
