import click
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from document_analyzer.batch_processing.batch_manager import BatchProcessor
from document_analyzer.core.config import settings


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    default=settings.get("batch_processing.directory_input", "input"),
    help="Directory containing files to process.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, resolve_path=True),
    default=settings.get("batch_processing.directory_output", "output"),
    help="Directory to save output files.",
)
@click.option(
    "--output-format",
    type=click.Choice(["json", "markdown", "essay", "list", "table"]),
    default=settings.get("batch_processing.default_format", "json"),
    help="The output format for the results.",
)
def batch_process_files(input_dir, output_dir, output_format):
    """
    Processes each file in the input directory and saves the analysis to the output directory.
    """
    processor = BatchProcessor()
    click.echo(
        f"Processing files from '{input_dir}' and saving to '{output_dir}' in '{output_format}' format."
    )
    processor.process_files(input_dir, output_dir, output_format)
    click.echo("Batch processing complete.")


if __name__ == "__main__":
    batch_process_files()
