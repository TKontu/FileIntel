from .loader import PromptLoader
from .template_engine import TemplateEngine
import tiktoken

class PromptComposer:
    def __init__(self, loader: PromptLoader, max_length: int, model_name: str = "gpt-4"):
        self.loader = loader
        self.template_engine = TemplateEngine()
        self.max_length = max_length
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            print(f"Warning: Model {model_name} not found. Using cl100k_base encoding.")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def compose(
        self,
        context: dict,
        instruction_template: str = "instruction",
        prompt_frame: str = "prompt_frame",
        answer_format_template: str = "answer_format",
        instruction_version: str = None,
        question_version: str = None,
        answer_format_version: str = None,
    ) -> str:
        """
        Composes a prompt, ensuring it fits within the specified token limit
        by truncating the document text if necessary.
        """
        instruction = self.loader.load_prompt(instruction_template, version=instruction_version)
        question = self.loader.load_prompt(prompt_frame, version=question_version)
        answer_format = self.loader.load_prompt(answer_format_template, version=answer_format_version)

        full_prompt_template = f"{instruction}\n\n{question}\n\n{answer_format}"

        # Calculate the token count of the template without the document_text
        template_context = context.copy()
        template_context['document_text'] = '' # Placeholder
        
        prompt_shell = self.template_engine.render_string(full_prompt_template, **template_context)
        overhead_tokens = len(self.tokenizer.encode(prompt_shell))

        # Calculate the available token budget for the document text
        available_tokens = self.max_length - overhead_tokens

        if 'document_text' in context:
            document_tokens = self.tokenizer.encode(context['document_text'])
            
            if len(document_tokens) > available_tokens:
                print(f"Document text ({len(document_tokens)} tokens) exceeds available token budget ({available_tokens}). Truncating.")
                truncated_tokens = document_tokens[:available_tokens]
                truncated_text = self.tokenizer.decode(truncated_tokens)
                
                # Update context with truncated text
                context['document_text'] = truncated_text + "\n... [TRUNCATED DUE TO CONTEXT LIMIT] ..."
            
        # Render the final prompt with the (potentially truncated) context
        final_prompt = self.template_engine.render_string(full_prompt_template, **context)
        final_token_count = len(self.tokenizer.encode(final_prompt))

        print(f"Final prompt token count: {final_token_count}/{self.max_length}")
        return final_prompt

