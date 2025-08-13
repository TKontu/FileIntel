from .job_manager import JobManager
from ..document_processing.factory import ReaderFactory
from ..llm_integration.openai_provider import OpenAIProvider
from ..prompt_management.composer import PromptComposer
from ..prompt_management.loader import PromptLoader
from .batch_manager import BatchProcessor
from pathlib import Path
from ..core.config import settings
import logging

logger = logging.getLogger(__name__)

class Worker:
    def __init__(self, job_manager: JobManager):
        self.job_manager = job_manager
        self.reader_factory = ReaderFactory()
        self.llm_provider = OpenAIProvider()
        self.batch_processor = BatchProcessor()
        
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
        Processes a job based on its type.
        """
        job_id = job.id
        job_type = job.job_type

        logger.debug(f"--- Processing job {job_id} with type: {job_type}")

        self.job_manager.update_job_status(job_id, "running")

        try:
            if job_type == "batch":
                logger.info(f"Handling as BATCH job: {job_id}")
                self._process_batch_job(job)
            else: # Default to single_file
                logger.info(f"Handling as SINGLE FILE job: {job_id}")
                self._process_single_file_job(job)
            
            self.job_manager.update_job_status(job_id, "completed")
            logger.info(f"Job {job_id} ({job_type}) completed successfully.")

        except Exception as e:
            self.job_manager.update_job_status(job_id, "failed")
            logger.error(f"Error processing job {job_id} ({job_type}): {e}", exc_info=True)

    def _process_batch_job(self, job):
        """
        Processes a batch job.
        """
        job_data = job.data
        input_dir = job_data.get("input_dir")
        output_dir = job_data.get("output_dir")
        output_format = job_data.get("output_format")
        
        self.batch_processor.process_files(input_dir, output_dir, output_format)

    def _process_single_file_job(self, job):
        """
        Processes a single file job.
        """
        # 1. Get document content
        file_path = job.data.get("file_path") 
        reader = self.reader_factory.get_reader(file_path)
        elements = reader.read(Path(file_path))
        
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

        # 4. Save result
        self.job_manager.storage.save_result(job.id, response._asdict())
