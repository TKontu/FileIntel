"""This module defines data adapters for converting FileIntel data to GraphRAG format."""

import pandas as pd
from typing import List
from fileintel.storage.models import DocumentChunk


class GraphRAGDataAdapter:
    """A class to adapt FileIntel data for GraphRAG."""

    def adapt_documents(self, documents: List[DocumentChunk]) -> pd.DataFrame:
        """Adapts FileIntel documents to the format expected by GraphRAG."""
        if not documents:
            return pd.DataFrame(
                columns=["id", "text", "title", "path", "creation_date"]
            )

        doc_records = []
        for chunk in documents:
            record = {
                "id": chunk.id,
                "text": chunk.chunk_text,
                "title": chunk.document.original_filename,
                "path": chunk.document.filename,
                "creation_date": chunk.document.created_at.isoformat()
                if chunk.document.created_at
                else None,
            }
            if chunk.chunk_metadata:
                for key, value in chunk.chunk_metadata.items():
                    record[key] = value
            doc_records.append(record)

        return pd.DataFrame.from_records(doc_records)

    def convert_response(self, response, context=None):
        """Converts a GraphRAG search response to the FileIntel format."""
        response_data = {}
        if isinstance(response, str):
            response_data["response"] = response
        else:
            # If the response is already a dict or some other structure, use it
            response_data = response

        if context:
            response_data["context"] = context

        return response_data
