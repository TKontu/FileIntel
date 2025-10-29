from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from .models import Document, Collection


class StorageInterface(ABC):
    """
    Unified interface for all storage operations.

    Combines document, collection, chunk, and GraphRAG storage operations
    into a single interface to eliminate unnecessary separation.
    """

    # Collection operations
    @abstractmethod
    def create_collection(self, name: str, description: str = None) -> Collection:
        pass

    @abstractmethod
    def get_collection(self, collection_id: str) -> Collection:
        pass

    @abstractmethod
    def get_collection_by_name(self, name: str) -> Collection:
        pass

    @abstractmethod
    def get_all_collections(self) -> List[Collection]:
        pass

    @abstractmethod
    def delete_collection(self, collection_id: str):
        pass

    # Document operations
    @abstractmethod
    def create_document(
        self,
        filename: str,
        content_hash: str,
        file_size: int,
        mime_type: str,
        file_path: str,
        original_filename: str = None,
        metadata: dict = None,
        content_fingerprint: str = None,
    ) -> Document:
        pass

    @abstractmethod
    def add_document_to_collection(self, document_id: str, collection_id: str) -> bool:
        pass

    @abstractmethod
    def get_document(self, document_id: str) -> Document:
        pass

    @abstractmethod
    def get_document_by_hash(self, content_hash: str) -> Document:
        pass

    @abstractmethod
    def get_document_by_hash_and_collection(
        self, content_hash: str, collection_id: str
    ) -> Document:
        pass

    @abstractmethod
    def get_document_by_filename_and_collection(
        self, filename: str, collection_id: str
    ) -> Document:
        pass

    @abstractmethod
    def delete_document(self, document_id: str):
        pass

    # Chunk operations
    @abstractmethod
    def get_all_chunks_for_document(self, document_id: str):
        pass

    @abstractmethod
    def get_all_chunks_for_collection(self, collection_id: str):
        pass

    @abstractmethod
    def get_chunks_by_type_for_collection(self, collection_id: str, chunk_type: str = None):
        pass

    @abstractmethod
    def update_chunk_embedding(self, chunk_id: str, embedding: List[float]):
        pass

    # GraphRAG operations
    @abstractmethod
    def save_graphrag_index_info(
        self,
        collection_id: str,
        index_path: str,
        documents_count: int = 0,
        entities_count: int = 0,
        communities_count: int = 0,
    ) -> None:
        pass

    @abstractmethod
    def get_graphrag_index_info(self, collection_id: str) -> Optional[Dict]:
        pass

    @abstractmethod
    def save_graphrag_entities(self, collection_id: str, entities: List[dict]) -> None:
        pass

    @abstractmethod
    def get_graphrag_entities(self, collection_id: str) -> List[dict]:
        pass

    @abstractmethod
    def save_graphrag_communities(
        self, collection_id: str, communities: List[dict]
    ) -> None:
        pass

    @abstractmethod
    def get_graphrag_communities(self, collection_id: str) -> List[dict]:
        pass

    @abstractmethod
    def get_graphrag_relationships(self, collection_id: str) -> List[dict]:
        pass


# Backward compatibility aliases - DEPRECATED
DocumentStorageInterface = StorageInterface
GraphRAGStorageInterface = StorageInterface
