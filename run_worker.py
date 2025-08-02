import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from document_analyzer.batch_processing.job_manager import JobManager
from document_analyzer.batch_processing.worker import Worker
from document_analyzer.document_processing.unified_processor import UnifiedDocumentProcessor
from document_analyzer.llm_integration.openai_provider import OpenAIProvider # Or another provider
from document_analyzer.storage.redis_storage import RedisStorage
from document_analyzer.core.logging import setup_logging, get_logger

def main():
    """
    Initializes all components and starts the worker.
    """
    setup_logging()
    logger = get_logger("worker")
    logger.info("Worker starting...")

    try:
        job_manager = JobManager()
        storage = RedisStorage()
        document_processor = UnifiedDocumentProcessor()
        llm_provider = OpenAIProvider() # You could make this configurable

        worker = Worker(
            job_manager=job_manager,
            document_processor=document_processor,
            llm_provider=llm_provider,
            storage=storage,
        )

        logger.info("Worker started. Waiting for jobs...")
        worker.run()

    except Exception as e:
        logger.error(f"A critical error occurred in the worker: {e}", exc_info=True)
    finally:
        logger.info("Worker shutting down.")

if __name__ == "__main__":
    main()
