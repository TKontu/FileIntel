"""This script replicates the exact prompt compilation process used by the API worker.

It takes a file path and a task name, processes the document,
and composes the prompt using the same refactored logic as the live environment.
This provides an exact preview of the prompt that would be sent to the LLM.
"""
import click
from pathlib import Path
import sys

# Add the src directory to the Python path to allow for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from document_analyzer.core.config import settings
from document_analyzer.document_processing.factory import ReaderFactory
from document_analyzer.prompt_management.loader import PromptLoader
from document_analyzer.prompt_management.composer import PromptComposer

@click.command()
@click.argument('task_name', type=str)
@click.argument('file_path', type=click.Path(exists=True, dir_okay=False, resolve_path=True))
def replicate_api_prompt(task_name, file_path):
    """
    Processes a document and compiles the exact prompt used by the API worker.

    TASK_NAME: The name of the prompt task directory (e.g., 'default_analysis').
    FILE_PATH: The absolute path to the document file (e.g., PDF, TXT).
    """
    click.echo(f"Using prompt task: '{task_name}'")
    click.echo(f"Processing file: {file_path}")

    try:
        # 1. Setup components exactly like the worker
        prompts_dir_str = settings.get('prompts.directory', str(Path(__file__).parent.parent / 'prompts'))
        prompts_dir = Path(prompts_dir_str)
        loader = PromptLoader(prompts_dir=prompts_dir)
        composer = PromptComposer(
            loader=loader,
            max_length=settings.get('llm.context_length'),
            model_name=settings.get('llm.model')
        )
        reader_factory = ReaderFactory()

        # 2. Process the document using the factory to get the correct reader
        click.echo("--> Reading and extracting text from document...")
        reader = reader_factory.get_reader(file_path)
        elements = reader.read(Path(file_path))
        document_text = "\n".join([el.text for el in elements if hasattr(el, 'text')])
        click.echo(f"--> Extracted {len(document_text)} characters of text.")

        # 3. Compose the prompt using the new refactored logic
        click.echo("--> Composing final prompt...")
        context = {
            "document_text": document_text,
        }
        final_prompt = composer.compose(task_name, context)

        # 4. Print the final prompt
        click.echo("\n" + "="*30)
        click.echo(f"   COMPILED PROMPT PREVIEW ({task_name})")
        click.echo("="*30 + "\n")
        click.echo(final_prompt)
        click.echo("\n" + "="*30)
        click.echo(f"Final Prompt Length: {len(final_prompt)} characters")
        click.echo("="*30)

    except Exception as e:
        click.echo(f"\nAn error occurred: {e}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    replicate_api_prompt()
