import csv
import json
from io import StringIO
from ..base import OutputFormatter

class TableFormatter(OutputFormatter):
    @property
    def file_extension(self) -> str:
        return "csv"

    def format(self, data: dict, format_type: str = "csv") -> str:
        """
        Formats the given data as a table (CSV or JSON).
        Assumes the data contains 'headers' and 'rows' keys.
        """
        headers = data.get("headers", [])
        rows = data.get("rows", [])

        if format_type == "csv":
            return self._format_csv(headers, rows)
        elif format_type == "json":
            return self._format_json(headers, rows)
        else:
            raise ValueError(f"Unsupported table format: {format_type}")

    def _format_csv(self, headers: list, rows: list) -> str:
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(headers)
        writer.writerows(rows)
        return output.getvalue()

    def _format_json(self, headers: list, rows: list) -> str:
        return json.dumps([dict(zip(headers, row)) for row in rows], indent=2)
