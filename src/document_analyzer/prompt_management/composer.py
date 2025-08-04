from .loader import PromptLoader
from .template_engine import TemplateEngine
import json

class PromptComposer:
    def __init__(self, loader: PromptLoader):
        self.loader = loader
        self.template_engine = TemplateEngine()

    def compose(
        self, 
        context: dict,
        instruction_template: str = "instruction",
        question_template: str = "question",
        answer_format_template: str = "answer_format",
        instruction_version: str = None,
        question_version: str = None,
        answer_format_version: str = None,
        max_length: int = None,
    ) -> str:
        """
        Composes a prompt from the instruction, question, and answer format templates.
        """
        instruction = self.loader.load_prompt(instruction_template, version=instruction_version)
        question = self.loader.load_prompt(question_template, version=question_version)
        answer_format = self.loader.load_prompt(answer_format_template, version=answer_format_version)

        full_prompt_template = f"{instruction}\n\n{question}\n\n{answer_format}"

        # Initial render to see the length
        prompt = self.template_engine.render_string(full_prompt_template, **context)

        if max_length and len(prompt) > max_length:
            # This is a simple truncation strategy. A more sophisticated approach
            # would be to summarize the context or use a more intelligent
            # truncation method.
            print(f"Prompt length ({len(prompt)}) exceeds max_length ({max_length}). Truncating context.")
            
            # Calculate the length of the template without the context
            template_without_context = self.template_engine.render_string(full_prompt_template, **{k: '' for k in context})
            overhead = len(template_without_context)
            
            # Truncate the context (assuming context is a single string for now)
            # A better implementation would handle different context structures.
            if 'document_text' in context:
                allowed_context_len = max_length - overhead
                truncated_text = context['document_text'][:allowed_context_len]
                
                new_context = context.copy()
                new_context['document_text'] = truncated_text + "\n... [TRUNCATED] ..."
                
                prompt = self.template_engine.render_string(full_prompt_template, **new_context)

        return prompt
