"""RAG module models and data structures."""

from pydantic import BaseModel
from typing import List, Dict, Any


class DirectQueryResponse(BaseModel):
    """Response from direct query operations."""

    answer: str
    sources: List[Dict[str, Any]]
    query_type: str
    routing_explanation: str
