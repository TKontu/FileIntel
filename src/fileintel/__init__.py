"""
FileIntel - Intelligent Document Processing and Analysis System

A comprehensive system for document processing, analysis, and retrieval
using modern distributed processing with Celery and advanced RAG capabilities.
"""

__version__ = "2.0.0"
__author__ = "FileIntel Team"

# Core modules (always available)
from . import core
from . import storage
from . import document_processing

# Optional imports with fallback handling for Celery dependencies
try:
    from . import tasks
except ImportError:
    # Tasks module may not be available if Celery dependencies are missing
    pass

try:
    from . import celery_config
except ImportError:
    # Celery configuration may not be available in all environments
    pass

try:
    from . import rag
except ImportError:
    # RAG components may not be available if dependencies are missing
    pass

try:
    from . import llm_integration
except ImportError:
    # LLM integration may not be available if dependencies are missing
    pass

__all__ = [
    "core",
    "storage",
    "document_processing",
    "tasks",
    "celery_config",
    "rag",
    "llm_integration",
]
