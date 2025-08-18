from .job_manager import JobManager
from ..document_processing.factory import ReaderFactory
from ..document_processing.chunking import TextChunker
from ..llm_integration.openai_provider import OpenAIProvider
from ..llm_integration.embedding_provider import OpenAIEmbeddingProvider
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
        self.text_chunker = TextChunker()
        self.llm_provider = OpenAIProvider()
        self.embedding_provider = OpenAIEmbeddingProvider()

        prompts_dir_str = settings.get("prompts.directory", "/home/appuser/app/prompts")
        prompts_dir = Path(prompts_dir_str)
        self.loader = PromptLoader(prompts_dir=prompts_dir)
        self.composer = PromptComposer(
            loader=self.loader,
            max_length=settings.get("llm.context_length"),
            model_name=settings.get("llm.model"),
        )
        self.batch_processor = BatchProcessor(
            composer=self.composer, llm_provider=self.llm_provider
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
            elif job_type == "indexing":
                logger.info(f"Handling as INDEXING job: {job_id}")
                self._process_indexing_job(job)
            elif job_type == "query":
                logger.info(f"Handling as QUERY job: {job_id}")
                self._process_query_job(job)
            else:  # Default to single_file
                logger.info(f"Handling as SINGLE FILE job: {job_id}")
                self._process_single_file_job(job)

            self.job_manager.update_job_status(job_id, "completed")
            logger.info(f"Job {job_id} ({job_type}) completed successfully.")

        except Exception as e:
            self.job_manager.update_job_status(job_id, "failed")
            logger.error(
                f"Error processing job {job_id} ({job_type}): {e}", exc_info=True
            )

    def _process_batch_job(self, job):
        """
        Processes a batch job.
        """
        job_data = job.data
        input_dir = job_data.get("input_dir")
        output_dir = job_data.get("output_dir")
        output_format = job_data.get("output_format")
        task_name = job_data.get("task_name", "default_analysis")

        self.batch_processor.process_files(
            input_dir, output_dir, output_format, task_name
        )

    def _process_single_file_job(self, job):
        """
        Processes a single file job.
        """
        # 1. Get document content
        file_path = job.data.get("file_path")
        reader = self.reader_factory.get_reader(file_path)
        elements = reader.read(Path(file_path))

        document_text = "\n".join([el.text for el in elements if hasattr(el, "text")])

        # 2. Compose the prompt
        task_name = job.data.get("task_name", "default_analysis")
        logger.info(f"Using prompt task: '{task_name}' for job {job.id}")
        context = {
            "document_text": document_text,
        }
        prompt = self.composer.compose(task_name, context)

        # 3. Get LLM response
        response = self.llm_provider.generate_response(prompt)

        # 4. Save result
        self.job_manager.storage.save_result(job.id, response._asdict())

    def _process_indexing_job(self, job):
        """
        Processes an indexing job.
        """
        document_id = job.document_id
        collection_id = job.collection_id
        file_path = job.data.get("file_path")

        # 1. Extract text
        reader = self.reader_factory.get_reader(file_path)
        elements = reader.read(Path(file_path))
        document_text = "\n".join([el.text for el in elements if hasattr(el, "text")])

        # 2. Chunk text
        chunks = self.text_chunker.chunk_text(document_text)

        # 3. Embed chunks
        embeddings = self.embedding_provider.get_embeddings(chunks)

        # 4. Store chunks
        chunk_data = [
            {"text": chunk, "embedding": embedding}
            for chunk, embedding in zip(chunks, embeddings)
        ]
        self.job_manager.storage.add_document_chunks(
            document_id, collection_id, chunk_data
        )

    def _process_query_job(self, job):
        """
        Processes a query job.
        """
        collection_id = job.collection_id
        question = job.data.get("question")

        # 1. Embed the question
        query_embedding = self.embedding_provider.get_embeddings([question])[0]

        # 2. Find relevant chunks
        relevant_chunks = self.job_manager.storage.find_relevant_chunks_in_collection(
            collection_id, query_embedding
        )
        context_text = "\n".join([chunk.chunk_text for chunk in relevant_chunks])

        # 3. Compose the prompt
        task_name = job.data.get("task_name", "default_analysis")
        context = {
            "document_text": context_text,
            "question": question,
        }
        prompt = self.composer.compose(task_name, context)

        # 4. Get LLM response
        response = self.llm_provider.generate_response(prompt)

        # 5. Save result
        self.job_manager.storage.save_result(job.id, response._asdict())
