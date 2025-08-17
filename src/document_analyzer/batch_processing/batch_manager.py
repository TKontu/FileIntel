from pathlib import Path
from document_analyzer.document_processing.factory import ReaderFactory
from document_analyzer.llm_integration.base import LLMProvider
from document_analyzer.prompt_management.composer import PromptComposer
from document_analyzer.output_management.factory import FormatterFactory

class BatchProcessor:
    def __init__(self, composer: PromptComposer, llm_provider: LLMProvider):
        self.reader_factory = ReaderFactory()
        self.composer = composer
        self.llm_provider = llm_provider

    def process_files(self, input_dir: str, output_dir: str, output_format: str, task_name: str):
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
                    context = {
                        "document_text": document_text,
                    }
                    prompt = self.composer.compose(task_name, context)

                    # 3. Get LLM response
                    response = self.llm_provider.generate_response(prompt)

                    # 4. Format and save the result
                    formatted_output = formatter.format(response._asdict())
                    output_filename = output_path / f"{file_path.stem}_output.{formatter.file_extension}"
                    with open(output_filename, 'w', encoding='utf-8') as f:
                        f.write(formatted_output)
                except Exception as e:
                    print(f"Error processing {file_path.name}: {e}")
