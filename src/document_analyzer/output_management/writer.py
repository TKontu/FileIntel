from pathlib import Path
from .base import OutputFormatter
from ..storage.base import StorageInterface

class OutputWriter:
    def __init__(self, storage: StorageInterface = None):
        self.storage = storage

    def write(self, data: str, destination: str, formatter: OutputFormatter, original_filename: str = None, **kwargs):
        """
        Writes the data to the specified destination using the given formatter.
        """
        formatted_data = formatter.format(data, **kwargs)

        if destination == "file":
            output_path = self._get_output_path(original_filename, formatter, **kwargs)
            self._write_to_file(formatted_data, output_path)
        elif destination == "database":
            if not self.storage:
                raise ValueError("Storage must be provided for database output.")
            self._write_to_database(formatted_data, kwargs.get("job_id"))
        else:
            raise ValueError(f"Unsupported output destination: {destination}")

    def _get_output_path(self, original_filename: str, formatter: OutputFormatter, **kwargs) -> Path:
        if not original_filename:
            raise ValueError("Original filename must be provided for file output.")
        
        base_name = Path(original_filename).stem
        format_extension = self._get_format_extension(formatter, **kwargs)
        
        # This is a simple naming convention. A more robust implementation
        # could include timestamps or other metadata.
        return Path(f"{base_name}_output.{format_extension}")

    def _get_format_extension(self, formatter: OutputFormatter, **kwargs) -> str:
        if isinstance(formatter, TableFormatter):
            return kwargs.get("format_type", "csv")
        elif isinstance(formatter, EssayFormatter):
            return "md"
        elif isinstance(formatter, JSONFormatter):
            return "json"
        elif isinstance(formatter, ListFormatter):
            return "txt"
        else:
            return "txt"

    def _write_to_file(self, data: str, output_path: Path):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(data)

    def _write_to_database(self, data: str, job_id: str):
        if not job_id:
            raise ValueError("Job ID must be provided for database output.")
        
        self.storage.save_result(job_id, {"result": data})
