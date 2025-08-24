import logging
import sys
import json
from logging.handlers import RotatingFileHandler
from .config import settings


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "job_id"):
            log_record["job_id"] = record.job_id
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


def setup_logging():
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create a new handler with the JSON formatter
    handler = logging.StreamHandler(sys.stdout)
    formatter = JsonFormatter()
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Add file handler
    file_handler = RotatingFileHandler(
        "logs/document_analyzer.log", maxBytes=1024 * 1024 * 5, backupCount=5  # 5 MB
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
