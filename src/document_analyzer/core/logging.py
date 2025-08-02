import logging
import sys
from logging.handlers import RotatingFileHandler
from .config import settings

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            RotatingFileHandler(
                "logs/document_analyzer.log",
                maxBytes=1024 * 1024 * 5, # 5 MB
                backupCount=5
            ),
        ],
    )

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
