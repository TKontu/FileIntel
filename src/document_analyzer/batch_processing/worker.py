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
from logging import LoggerAdapter

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

        # Create a logger adapter with the job_id
        adapter = logging.LoggerAdapter(logger, {"job_id": job_id})

        adapter.info(f"Processing job with type: {job_type}")

        self.job_manager.update_job_status(job_id, "running")

        try:
            if job_type == "batch":
                adapter.info(f"Handling as BATCH job")
                self._process_batch_job(job, adapter)
            elif job_type == "indexing":
                adapter.info(f"Handling as INDEXING job")
                self._process_indexing_job(job, adapter)
            elif job_type == "query":
                adapter.info(f"Handling as QUERY job")
                self._process_query_job(job, adapter)
            else:  # Default to single_file
                adapter.info(f"Handling as SINGLE FILE job")
                self._process_single_file_job(job, adapter)

            self.job_manager.update_job_status(job_id, "completed")
            adapter.info(f"Job completed successfully.")

        except Exception as e:
            self.job_manager.update_job_status(job_id, "failed")
            adapter.error(f"Error processing job: {e}", exc_info=True)

    def _process_batch_job(self, job, adapter: LoggerAdapter):
        """
        Processes a batch job.
        """
        job_data = job.data
        input_dir = job_data.get("input_dir")
        output_dir = job_data.get("output_dir")
        output_format = job_data.get("output_format")
        task_name = job_data.get("task_name", "default_analysis")

        adapter.debug(
            f"Starting batch processing from '{input_dir}' to '{output_dir}'."
        )
        self.batch_processor.process_files(
            input_dir, output_dir, output_format, task_name
        )
        adapter.debug("Batch processing finished.")

    def _process_single_file_job(self, job, adapter: LoggerAdapter):
        """
        Processes a single file job.
        """
        file_path_str = job.data.get("file_path")
        file_path = Path(file_path_str)
        adapter.debug(f"Reading content from file: {file_path}")
        reader = self.reader_factory.get_reader(file_path)
        elements = reader.read(file_path, adapter)

        document_text = "\n".join([el.text for el in elements if hasattr(el, "text")])
        adapter.debug(f"Extracted {len(document_text)} characters of text.")

        task_name = job.data.get("task_name", "default_analysis")
        adapter.debug(f"Composing prompt using task: '{task_name}'")
        context = {
            "document_text": document_text,
        }
        prompt = self.composer.compose(task_name, context)

        adapter.debug("Sending prompt to LLM.")
        response = self.llm_provider.generate_response(prompt)
        adapter.debug("Received response from LLM.")

        self.job_manager.storage.save_result(job.id, response._asdict())
        adapter.debug("Saved result to storage.")

    def _process_indexing_job(self, job, adapter: LoggerAdapter):
        """
        Processes an indexing job.
        """
        document_id = job.document_id
        collection_id = job.collection_id
        file_path_str = job.data.get("file_path")
        file_path = Path(file_path_str)

        adapter.info(f"Starting indexing for document: {file_path.name}")

        adapter.info("Step 1/4: Reading and extracting text...")
        reader = self.reader_factory.get_reader(file_path)
        elements = reader.read(file_path, adapter)
        document_text = "\n".join([el.text for el in elements if hasattr(el, "text")])
        adapter.info(f"Extracted {len(document_text)} characters.")

        adapter.info("Step 2/4: Chunking text...")
        chunks = self.text_chunker.chunk_text(document_text)
        adapter.info(f"Created {len(chunks)} chunks.")

        adapter.info("Step 3/4: Generating embeddings for chunks...")
        embeddings = self.embedding_provider.get_embeddings(chunks)
        adapter.info("Embeddings generated.")

        chunk_data = [
            {"text": chunk, "embedding": embedding}
            for chunk, embedding in zip(chunks, embeddings)
        ]
        adapter.info(f"Step 4/4: Saving {len(chunks)} chunks to database...")
        self.job_manager.storage.add_document_chunks(
            document_id, collection_id, chunk_data
        )
        adapter.info("Saved chunks and embeddings to storage.")

    def _process_query_job(self, job, adapter: LoggerAdapter):
        """
        Processes a query job.
        """
        collection_id = job.collection_id
        question = job.data.get("question")
        adapter.debug(
            f"Querying collection {collection_id} with question: '{question}'"
        )

        adapter.debug("Embedding the question.")
        query_embedding = self.embedding_provider.get_embeddings([question])[0]

        adapter.debug("Finding relevant chunks.")
        relevant_chunks = self.job_manager.storage.find_relevant_chunks_in_collection(
            collection_id, query_embedding
        )
        adapter.debug(f"Found {len(relevant_chunks)} relevant chunks.")
        context_text = "\n".join([chunk.chunk_text for chunk in relevant_chunks])

        task_name = job.data.get("task_name", "default_analysis")
        adapter.debug(f"Composing prompt with task: '{task_name}'")
        context = {
            "document_text": context_text,
            "question": question,
        }
        prompt = self.composer.compose(task_name, context)

        adapter.debug("Sending prompt to LLM.")
        response = self.llm_provider.generate_response(prompt)
        adapter.debug("Received response from LLM.")

        self.job_manager.storage.save_result(job.id, response._asdict())
        adapter.debug("Saved result to storage.")
