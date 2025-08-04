import click
from pathlib import Path
import sys

# Add the src directory to the Python path to allow for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from document_analyzer.prompt_management.loader import PromptLoader
from document_analyzer.prompt_management.composer import PromptComposer

@click.command()
@click.option('--doc', required=True, type=click.Path(exists=True), help='Path to the document file to use as context.')
@click.option('--question', required=True, help='The question to ask about the document.')
@click.option('--instruction', default='instruction', help='Name of the instruction template.')
@click.option('--format', 'answer_format', default='answer_format', help='Name of the answer format template.')
@click.option('--max-length', type=int, help='Optional max length for the prompt.')
def preview_prompt(doc, question, instruction, answer_format, max_length):
    """
    Composes and prints a final prompt for preview and debugging purposes.
    """
    # Setup the necessary components
    prompts_dir = Path(__file__).parent.parent / 'prompts' / 'templates'
    loader = PromptLoader(prompts_dir=prompts_dir)
    composer = PromptComposer(loader=loader)

    # Read the document content
    with open(doc, 'r', encoding='utf-8') as f:
        document_text = f.read()

    # Create the context dictionary
    context = {
        "document_text": document_text,
        "question": question,
    }

    # Compose the prompt
    try:
        final_prompt = composer.compose(
            context=context,
            instruction_template=instruction,
            question_template=question,
            answer_format_template=answer_format,
            max_length=max_length
        )

        click.echo("--- COMPOSED PROMPT PREVIEW ---")
        click.echo(final_prompt)
        click.echo("\n--- END OF PREVIEW ---")
        click.echo(f"\nPrompt Length: {len(final_prompt)}")

    except Exception as e:
        click.echo(f"Error composing prompt: {e}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    preview_prompt()
