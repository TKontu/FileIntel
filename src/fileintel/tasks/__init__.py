"""
FileIntel Celery Tasks Module

This module contains all Celery task definitions for distributed processing
of document analysis, RAG operations, and LLM integration tasks.

Task Categories:
- document_tasks: Document processing and analysis tasks
- rag_tasks: Vector and Graph RAG processing tasks
- llm_tasks: Language model integration tasks
- indexing_tasks: Search index building and maintenance tasks
"""

from .base import BaseFileIntelTask

# Import task modules for auto-discovery
from . import document_tasks
from . import llm_tasks
from . import graphrag_tasks
from . import workflow_tasks

__all__ = [
    "BaseFileIntelTask",
    "document_tasks",
    "llm_tasks",
    "graphrag_tasks",
    "workflow_tasks",
]
