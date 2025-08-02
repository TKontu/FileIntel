from .loader import PromptLoader
from .template_engine import TemplateEngine

class PromptComposer:
    def __init__(self, loader: PromptLoader):
        self.loader = loader
        self.template_engine = TemplateEngine()

    def compose(self, context: dict, instruction_template: str = "instruction", question_template: str = "question", answer_format_template: str = "answer_format") -> str:
        """
        Composes a prompt from the instruction, question, and answer format templates.
        """
        instruction = self.loader.load_prompt(instruction_template)
        question = self.loader.load_prompt(question_template)
        answer_format = self.loader.load_prompt(answer_format_template)

        full_prompt_template = f"{instruction}\n\n{question}\n\n{answer_format}"

        return self.template_engine.render_string(full_prompt_template, **context)
