#!/usr/bin/env python3
"""
Prompt preview script for FileIntel.

This script replicates the prompt compilation process used by the current
Celery-based task system. It processes a document and composes the prompt
using the same logic as the live environment.
"""
import click
from pathlib import Path
import sys

# Add the src directory to the Python path to allow for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fileintel.core.config import get_config
from fileintel.document_processing.unified_processor import UnifiedDocumentProcessor
from fileintel.prompt_management.loader import PromptLoader
from fileintel.prompt_management.composer import PromptComposer


@click.command()
@click.argument("task_name", type=str)
@click.argument(
    "file_path", type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
def replicate_api_prompt(task_name, file_path):
    """
    Processes a document and compiles the exact prompt used by the API worker.

    TASK_NAME: The name of the prompt task directory (e.g., 'default_analysis').
    FILE_PATH: The absolute path to the document file (e.g., PDF, TXT).
    """
    click.echo(f"Using prompt task: '{task_name}'")
    click.echo(f"Processing file: {file_path}")

    try:
        # 1. Setup components using current FileIntel architecture
        config = get_config()
        prompts_dir = Path(config.paths.prompts)
        loader = PromptLoader(prompts_dir=prompts_dir)
        composer = PromptComposer(
            loader=loader,
            max_length=config.llm.context_length,
            model_name=config.llm.model,
        )
        processor = UnifiedDocumentProcessor(config)

        # 2. Process the document using the unified processor
        click.echo("--> Reading and extracting text from document...")
        result = processor.process_document(Path(file_path))
        document_text = result.get("content", "")
        click.echo(f"--> Extracted {len(document_text)} characters of text.")

        # 3. Compose the prompt using the prompt composer
        click.echo("--> Composing final prompt...")
        context = {
            "document_text": document_text,
            "metadata": result.get("metadata", {}),
        }
        final_prompt = composer.compose(task_name, context)

        # 4. Print the final prompt
        click.echo("\n" + "=" * 30)
        click.echo(f"   COMPILED PROMPT PREVIEW ({task_name})")
        click.echo("=" * 30 + "\n")
        click.echo(final_prompt)
        click.echo("\n" + "=" * 30)
        click.echo(f"Final Prompt Length: {len(final_prompt)} characters")
        click.echo("=" * 30)

    except Exception as e:
        click.echo(f"\nAn error occurred: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    replicate_api_prompt()
