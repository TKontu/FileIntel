from .loader import PromptLoader
from .template_engine import TemplateEngine
from ..core.exceptions import ConfigException
import tiktoken


class PromptComposer:
    def __init__(
        self, loader: PromptLoader, max_length: int, model_name: str = "gpt-4"
    ):
        self.loader = loader
        self.template_engine = TemplateEngine()
        self.max_length = max_length
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            print(f"Warning: Model {model_name} not found. Using cl100k_base encoding.")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def compose(self, task_name: str, context: dict) -> str:
        """
        Composes a prompt from a task directory, ensuring it fits within the
        specified token limit by truncating the document text if necessary.
        """
        # Load all templates for the given task
        templates = self.loader.load_task_templates(task_name)

        # The main frame must be present
        if "prompt" not in templates:
            raise ConfigException(
                f"Main 'prompt.md' template not found in task '{task_name}'"
            )

        prompt_frame = templates["prompt"]

        # Combine the static templates and the dynamic context for rendering
        render_context = {**templates, **context}

        # Calculate overhead tokens from the prompt frame and its components,
        # excluding the main document_text that we might need to truncate.
        overhead_context = render_context.copy()
        overhead_context["document_text"] = ""  # Placeholder

        prompt_shell = self.template_engine.render_string(
            prompt_frame, **overhead_context
        )
        overhead_tokens = len(self.tokenizer.encode(prompt_shell))

        # Calculate available token budget for the document text
        available_tokens = self.max_length - overhead_tokens

        if "document_text" in context:
            document_tokens = self.tokenizer.encode(context["document_text"])

            if len(document_tokens) > available_tokens:
                print(
                    f"Document text ({len(document_tokens)} tokens) exceeds available token budget ({available_tokens}). Truncating."
                )
                truncated_tokens = document_tokens[:available_tokens]
                truncated_text = self.tokenizer.decode(truncated_tokens)

                # Update context with truncated text
                render_context["document_text"] = (
                    truncated_text + "\n... [TRUNCATED DUE TO CONTEXT LIMIT] ..."
                )

        # Render the final prompt with the (potentially truncated) context
        final_prompt = self.template_engine.render_string(
            prompt_frame, **render_context
        )
        final_token_count = len(self.tokenizer.encode(final_prompt))

        print(f"Final prompt token count: {final_token_count}/{self.max_length}")
        return final_prompt
