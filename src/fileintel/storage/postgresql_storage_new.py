"""
Unified PostgreSQL storage implementation using composition.

Composes specialized storage classes to provide a complete storage interface
while maintaining clear separation of concerns and following SOLID principles.
"""

from .base import StorageInterface
from .document_storage import DocumentStorage
from .vector_search_storage import VectorSearchStorage
from .graphrag_storage import GraphRAGStorage
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class PostgreSQLStorage(StorageInterface):
    """
    Unified PostgreSQL storage interface using composition.

    Delegates operations to specialized storage components:
    - DocumentStorage: Collections, documents, chunks
    - VectorSearchStorage: Vector similarity search
    - GraphRAGStorage: GraphRAG entities, communities, relationships

    This approach follows the Single Responsibility Principle while
    maintaining backward compatibility with the existing interface.
    """

    def __init__(self, config_or_session):
        """Initialize composed storage components."""
        # Initialize specialized storage components
        self.document_storage = DocumentStorage(config_or_session)
        self.vector_storage = VectorSearchStorage(config_or_session)
        self.graphrag_storage = GraphRAGStorage(config_or_session)

        # Keep reference to database session for direct access if needed
        self.db = self.document_storage.db

    # Collection Operations (delegate to DocumentStorage)
    def create_collection(self, name: str, description: str = None):
        """Create a new collection."""
        return self.document_storage.create_collection(name, description)

    def get_collection(self, collection_id: str):
        """Get collection by ID."""
        return self.document_storage.get_collection(collection_id)

    def get_collection_by_name(self, name: str):
        """Get collection by name."""
        return self.document_storage.get_collection_by_name(name)

    def get_all_collections(self) -> List:
        """Get all collections."""
        return self.document_storage.get_all_collections()

    def delete_collection(self, collection_id: str):
        """Delete a collection."""
        return self.document_storage.delete_collection(collection_id)

    def update_collection_status(self, collection_id: str, status: str) -> bool:
        """Update collection processing status."""
        return self.document_storage.update_collection_status(collection_id, status)

    # Document Operations (delegate to DocumentStorage)
    def create_document(
        self,
        filename: str,
        content_hash: str,
        file_size: int,
        mime_type: str,
        collection_id: str,
        file_path: str = None,
        original_filename: str = None,
        metadata: Dict[str, Any] = None,
    ):
        """Create a new document."""
        return self.document_storage.create_document(
            filename,
            content_hash,
            file_size,
            mime_type,
            collection_id,
            file_path,
            original_filename,
            metadata,
        )

    def get_document(self, document_id: str):
        """Get document by ID."""
        return self.document_storage.get_document(document_id)

    def get_document_by_hash(self, content_hash: str):
        """Get document by content hash."""
        return self.document_storage.get_document_by_hash(content_hash)

    def get_document_by_hash_and_collection(
        self, content_hash: str, collection_id: str
    ):
        """Get document by hash and collection."""
        return self.document_storage.get_document_by_hash_and_collection(
            content_hash, collection_id
        )

    def get_document_by_filename_and_collection(
        self, filename: str, collection_id: str
    ):
        """Get document by filename and collection."""
        return self.document_storage.get_document_by_filename_and_collection(
            filename, collection_id
        )

    def get_document_by_original_filename_and_collection(
        self, original_filename: str, collection_id: str
    ):
        """Get document by original filename and collection."""
        return self.document_storage.get_document_by_original_filename_and_collection(
            original_filename, collection_id
        )

    def get_documents_by_collection(self, collection_id: str) -> List:
        """Get all documents in a collection."""
        return self.document_storage.get_documents_by_collection(collection_id)

    def delete_document(self, document_id: str):
        """Delete a document."""
        return self.document_storage.delete_document(document_id)

    def update_document_metadata(self, document_id: str, metadata: Dict[str, Any], replace: bool = False):
        """Update document metadata."""
        return self.document_storage.update_document_metadata(document_id, metadata, replace=replace)

    # Chunk Operations (delegate to DocumentStorage)
    def add_document_chunks(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]],
        chunk_metadata: Dict[str, Any] = None,
    ):
        """Add chunks for a document."""
        return self.document_storage.add_document_chunks(
            document_id, chunks, chunk_metadata
        )

    def get_all_chunks_for_document(self, document_id: str):
        """Get all chunks for a document."""
        return self.document_storage.get_all_chunks_for_document(document_id)

    def get_all_chunks_for_collection(self, collection_id: str):
        """Get all chunks for a collection."""
        return self.document_storage.get_all_chunks_for_collection(collection_id)

    def update_chunk_embedding(self, chunk_id: str, embedding: List[float]):
        """Update chunk with embedding vector."""
        return self.document_storage.update_chunk_embedding(chunk_id, embedding)

    # Vector Search Operations (delegate to VectorSearchStorage)
    def find_relevant_chunks_in_collection(
        self,
        collection_id: str,
        query_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
        exclude_chunks: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find relevant chunks using vector similarity."""
        return self.vector_storage.find_relevant_chunks_in_collection(
            collection_id, query_embedding, limit, similarity_threshold, exclude_chunks
        )

    def find_relevant_chunks_in_document(
        self,
        document_id: str,
        query_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Find relevant chunks in document using vector similarity."""
        return self.vector_storage.find_relevant_chunks_in_document(
            document_id, query_embedding, limit, similarity_threshold
        )

    def find_relevant_chunks_hybrid(
        self,
        collection_id: str,
        query_embedding: List[float],
        text_keywords: List[str] = None,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        keyword_boost: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining vector similarity and keyword matching."""
        return self.vector_storage.find_relevant_chunks_hybrid(
            collection_id,
            query_embedding,
            text_keywords,
            limit,
            similarity_threshold,
            keyword_boost,
        )

    def get_embedding_statistics(self, collection_id: str = None) -> Dict[str, Any]:
        """Get embedding statistics."""
        return self.vector_storage.get_embedding_statistics(collection_id)

    # GraphRAG Operations (delegate to GraphRAGStorage)
    def save_graphrag_index_info(
        self,
        collection_id: str,
        index_path: str,
        documents_count: int = 0,
        entities_count: int = 0,
        communities_count: int = 0,
    ) -> None:
        """Save GraphRAG index information."""
        return self.graphrag_storage.save_graphrag_index_info(
            collection_id,
            index_path,
            documents_count,
            entities_count,
            communities_count,
        )

    def get_graphrag_index_info(self, collection_id: str) -> Optional[Dict]:
        """Get GraphRAG index information."""
        return self.graphrag_storage.get_graphrag_index_info(collection_id)

    def remove_graphrag_index_info(self, collection_id: str) -> bool:
        """Remove GraphRAG index information."""
        return self.graphrag_storage.remove_graphrag_index_info(collection_id)

    def save_graphrag_entities(self, collection_id: str, entities: List[dict]) -> None:
        """Save GraphRAG entities."""
        return self.graphrag_storage.save_graphrag_entities(collection_id, entities)

    def get_graphrag_entities(self, collection_id: str) -> List[dict]:
        """Get GraphRAG entities."""
        return self.graphrag_storage.get_graphrag_entities(collection_id)

    def save_graphrag_communities(
        self, collection_id: str, communities: List[dict]
    ) -> None:
        """Save GraphRAG communities."""
        return self.graphrag_storage.save_graphrag_communities(
            collection_id, communities
        )

    def get_graphrag_communities(self, collection_id: str) -> List[dict]:
        """Get GraphRAG communities."""
        return self.graphrag_storage.get_graphrag_communities(collection_id)

    def get_graphrag_relationships(self, collection_id: str) -> List[dict]:
        """Get GraphRAG relationships."""
        return self.graphrag_storage.get_graphrag_relationships(collection_id)

    # Utility methods for accessing underlying storage components
    def get_document_storage(self) -> DocumentStorage:
        """Access the document storage component."""
        return self.document_storage

    def get_vector_storage(self) -> VectorSearchStorage:
        """Access the vector search storage component."""
        return self.vector_storage

    def get_graphrag_storage(self) -> GraphRAGStorage:
        """Access the GraphRAG storage component."""
        return self.graphrag_storage

    def close(self):
        """Close all storage connections."""
        self.document_storage.close()
        self.vector_storage.close()
        self.graphrag_storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
