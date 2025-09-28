#!/usr/bin/env python3
"""
Batch file processing script using Celery distributed task processing.

This script replaces the old batch manager with Celery-based distributed processing.
"""
import click
from pathlib import Path
import sys
import os
import asyncio
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fileintel.core.config import get_config
from fileintel.tasks.document_tasks import process_document
from fileintel.storage.postgresql_storage import PostgreSQLStorage
from fileintel.storage.models import SessionLocal


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    default="input",
    help="Directory containing files to process.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, resolve_path=True),
    default="output",
    help="Directory to save output files.",
)
@click.option(
    "--collection-name",
    default="batch_processed",
    help="Name of the collection to store processed documents.",
)
@click.option(
    "--output-format",
    type=click.Choice(["json", "markdown", "essay", "list", "table"]),
    default="json",
    help="The output format for the results.",
)
@click.option(
    "--max-concurrent",
    type=int,
    default=4,
    help="Maximum number of concurrent processing tasks.",
)
def batch_process_files(
    input_dir, output_dir, collection_name, output_format, max_concurrent
):
    """
    Processes each file in the input directory using Celery distributed tasks.
    """
    config = get_config()
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Get list of supported files
    supported_formats = config.document_processing.supported_formats
    files_to_process = []

    for ext in supported_formats:
        files_to_process.extend(input_path.glob(f"*.{ext}"))

    if not files_to_process:
        click.echo(f"No supported files found in {input_dir}")
        click.echo(f"Supported formats: {', '.join(supported_formats)}")
        return

    click.echo(f"Found {len(files_to_process)} files to process")
    click.echo(f"Processing with collection: {collection_name}")
    click.echo(f"Output format: {output_format}")
    click.echo(f"Max concurrent tasks: {max_concurrent}")

    # Create database session and collection
    db_session = SessionLocal()
    storage = PostgreSQLStorage(db_session)

    try:
        # Create or get collection
        collection = storage.get_collection_by_name(collection_name)
        if not collection:
            collection = storage.create_collection(
                name=collection_name,
                description=f"Batch processed files from {input_dir}",
            )
            click.echo(f"Created collection: {collection.name}")
        else:
            click.echo(f"Using existing collection: {collection.name}")

        # Submit Celery tasks for processing
        task_results = []
        for file_path in files_to_process:
            click.echo(f"Submitting task for: {file_path.name}")
            result = process_document.delay(
                file_path=str(file_path),
                collection_id=collection.id,
                output_format=output_format,
            )
            task_results.append((file_path, result))

        # Monitor task completion
        click.echo(f"Submitted {len(task_results)} tasks, monitoring progress...")
        completed = 0
        failed = 0

        for file_path, result in task_results:
            try:
                # Wait for task completion (with timeout)
                task_result = result.get(timeout=600)  # 10 minute timeout per file

                if task_result.get("status") == "completed":
                    completed += 1
                    click.echo(f"✓ Completed: {file_path.name}")

                    # Save result to output directory
                    output_file = (
                        output_path
                        / f"{file_path.stem}_{output_format}.{output_format}"
                    )
                    with open(output_file, "w", encoding="utf-8") as f:
                        if output_format == "json":
                            import json

                            json.dump(task_result.get("result", {}), f, indent=2)
                        else:
                            f.write(str(task_result.get("result", "")))
                else:
                    failed += 1
                    click.echo(
                        f"✗ Failed: {file_path.name} - {task_result.get('error', 'Unknown error')}"
                    )

            except Exception as e:
                failed += 1
                click.echo(f"✗ Task failed: {file_path.name} - {str(e)}")

        click.echo(f"\nBatch processing complete!")
        click.echo(f"Completed: {completed}")
        click.echo(f"Failed: {failed}")
        click.echo(f"Results saved to: {output_dir}")

    finally:
        db_session.close()


if __name__ == "__main__":
    batch_process_files()
