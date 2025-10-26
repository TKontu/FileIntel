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

# Import config to get logging settings
from fileintel.core.config import get_config

# Get configured log level from application settings
config = get_config()
log_level_str = config.logging.level.upper()
log_level = getattr(logging, log_level_str, logging.WARNING)

# Set up logging before importing anything else
logging.basicConfig(
    level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Apply component-specific log levels from configuration
for logger_name, level_str in config.logging.component_levels.items():
    component_level = getattr(logging, level_str.upper(), None)
    if component_level is not None:
        logging.getLogger(logger_name).setLevel(component_level)
        logger.debug(f"Set {logger_name} to {level_str}")

# Reduce noise from third-party libraries (these stay at WARNING regardless of app log level)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def main():
    """Start Celery worker."""
    logger.info("Starting Celery worker...")

    try:
        # Import Celery app after setting up logging
        from fileintel.celery_config import app

        # Use already-loaded config
        worker_loglevel = config.logging.level.lower()

        # Start the worker with appropriate settings
        app.worker_main(
            [
                "worker",
                f"--loglevel={worker_loglevel}",
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
