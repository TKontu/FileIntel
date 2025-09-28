#!/usr/bin/env python3
"""
Celery worker startup script for FileIntel.

Replaces the old job management worker with Celery distributed processing.
"""

import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging before importing anything else
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Configure logging levels
logging.getLogger("fileintel.storage.postgresql_storage").setLevel(logging.INFO)
logging.getLogger("fileintel.llm_integration").setLevel(logging.INFO)
logging.getLogger("fileintel.rag.graph_rag.services.graphrag_service").setLevel(
    logging.INFO
)
logging.getLogger("celery").setLevel(logging.INFO)

# Reduce noise from third-party libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def main():
    """Start Celery worker."""
    logger.info("Starting Celery worker...")

    try:
        # Import Celery app after setting up logging
        from fileintel.celery_config import app

        # Start the worker with appropriate settings
        app.worker_main(
            [
                "worker",
                "--loglevel=info",
                "--concurrency=4",
                "--max-tasks-per-child=50",
                "--time-limit=600",
                "--soft-time-limit=480",
            ]
        )

    except ImportError as e:
        logger.error(f"Failed to import Celery configuration: {e}")
        logger.error(
            "Make sure Celery is installed and the celery_config module exists"
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Worker startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
