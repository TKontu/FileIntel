from .job_manager import JobManager
from ..document_processing.factory import ReaderFactory
from ..llm_integration.openai_provider import OpenAIProvider # Using OpenAI for now
from ..prompt_management.composer import PromptComposer
from ..prompt_management.loader import PromptLoader
from pathlib import Path
from ..core.config import settings

class Worker:
    def __init__(self, job_manager: JobManager):
        self.job_manager = job_manager
        self.reader_factory = ReaderFactory()
        self.llm_provider = OpenAIProvider() # This should be configurable
        
        prompts_dir_str = settings.get('prompts.directory', '/home/appuser/app/prompts/templates')
        prompts_dir = Path(prompts_dir_str)
        self.loader = PromptLoader(prompts_dir=prompts_dir)
        self.composer = PromptComposer(
            loader=self.loader,
            max_length=settings.get('llm.context_length'),
            model_name=settings.get('llm.model')
        )


    def process_job(self, job):
        """
        Processes a single job.
        """
        job_id = job.id
        self.job_manager.update_job_status(job_id, "running")

        try:
            # 1. Get document content
            file_path = job.data.get("file_path") 
            reader = self.reader_factory.get_reader(file_path)
            elements = reader.read(Path(file_path))
            
            document_text = "\n".join([el.text for el in elements if hasattr(el, 'text')])

            # 2. Compose the prompt
            # Load the question from the dedicated markdown file
            user_question = self.loader.load_prompt("user_question")
            context = {
                "document_text": document_text,
                "question": user_question
            }
            prompt = self.composer.compose(context)

            # 3. Get LLM response
            response = self.llm_provider.generate_response(prompt)

            # 4. Save result
            self.job_manager.storage.save_result(job_id, response._asdict())
            self.job_manager.update_job_status(job_id, "completed")
            print(f"Job {job_id} completed successfully.")

        except Exception as e:
            self.job_manager.update_job_status(job_id, "failed")
            print(f"Error processing job {job_id}: {e}")
