from pathlib import Path
from document_analyzer.document_processing.factory import ReaderFactory
from document_analyzer.llm_integration.openai_provider import OpenAIProvider
from document_analyzer.prompt_management.composer import PromptComposer
from document_analyzer.prompt_management.loader import PromptLoader
from document_analyzer.output_management.factory import FormatterFactory
from document_analyzer.core.config import settings

class BatchProcessor:
    def __init__(self):
        self.reader_factory = ReaderFactory()
        self.llm_provider = OpenAIProvider()
        prompts_dir_str = settings.get('prompts.directory', 'prompts/templates')
        prompts_dir = Path(prompts_dir_str)
        self.loader = PromptLoader(prompts_dir=prompts_dir)
        self.composer = PromptComposer(
            loader=self.loader,
            max_length=settings.get('llm.context_length'),
            model_name=settings.get('llm.model')
        )

    def process_files(self, input_dir: str, output_dir: str, output_format: str):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        formatter = FormatterFactory.get_formatter(output_format)

        for file_path in input_path.iterdir():
            if file_path.is_file():
                try:
                    # 1. Read document content
                    reader = self.reader_factory.get_reader(file_path)
                    elements = reader.read(file_path)
                    document_text = "\n".join([el.text for el in elements if hasattr(el, 'text')])

                    # 2. Compose the prompt
                    user_question = self.loader.load_prompt("user_question")
                    context = {
                        "document_text": document_text,
                        "question": user_question
                    }
                    prompt = self.composer.compose(context)

                    # 3. Get LLM response
                    response = self.llm_provider.generate_response(prompt)

                    # 4. Format and save the result
                    formatted_output = formatter.format(response._asdict())
                    output_filename = output_path / f"{file_path.stem}_output.{formatter.file_extension}"
                    with open(output_filename, 'w', encoding='utf-8') as f:
                        f.write(formatted_output)
                except Exception as e:
                    print(f"Error processing {file_path.name}: {e}")
