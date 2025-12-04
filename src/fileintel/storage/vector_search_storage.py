"""
Vector search storage operations for similarity search and embeddings.

Handles pgvector-based similarity search operations separated from
core document storage functionality.
"""

import logging
from typing import List, Dict, Any, Optional
from sqlalchemy import text, and_
from .base_storage import BaseStorageInfrastructure
from .models import DocumentChunk, Document

logger = logging.getLogger(__name__)


class VectorSearchStorage:
    """
    Handles vector similarity search operations.

    Provides pgvector-based similarity search functionality
    separated from basic document storage operations.
    """

    def __init__(self, config_or_session):
        """Initialize with shared infrastructure."""
        self.base = BaseStorageInfrastructure(config_or_session)
        self.db = self.base.db
        self._ensure_pgvector_extension()

    def _ensure_pgvector_extension(self):
        """Ensure pgvector extension is available before performing vector operations."""
        try:
            # Check if pgvector extension is installed and enabled
            check_extension_sql = """
            SELECT EXISTS (
                SELECT 1 FROM pg_extension WHERE extname = 'vector'
            );
            """
            result = self.db.execute(text(check_extension_sql)).fetchone()

            if not result or not result[0]:
                # Extension doesn't exist, try to create it
                logger.info("pgvector extension not found, attempting to create it")
                self.db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                self.db.commit()
                logger.info("pgvector extension created successfully")
            else:
                logger.debug("pgvector extension is available")

        except Exception as e:
            logger.error(f"Failed to ensure pgvector extension: {e}")
            logger.warning("Vector similarity search operations may fail")

    def find_relevant_chunks_in_collection(
        self,
        collection_id: str,
        query_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.0,
        exclude_chunks: List[str] = None,
        min_chunk_length: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Find relevant chunks in collection using vector similarity.

        Args:
            collection_id: Collection to search in
            query_embedding: Query vector for similarity search
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            exclude_chunks: Chunk IDs to exclude from results
            min_chunk_length: Minimum chunk text length (filters short headlines/sentences)

        Returns:
            List of relevant chunks with similarity scores
        """
        try:
            # Build exclusion filter
            exclusion_filter = ""
            # Build min length filter
            min_length_filter = ""
            params = {
                "collection_id": collection_id,
                "query_embedding": query_embedding,
                "limit": limit,
                "similarity_threshold": similarity_threshold,
            }

            if exclude_chunks:
                placeholders = ",".join(
                    [f":exclude_{i}" for i in range(len(exclude_chunks))]
                )
                exclusion_filter = f"AND c.id NOT IN ({placeholders})"
                for i, chunk_id in enumerate(exclude_chunks):
                    params[f"exclude_{i}"] = chunk_id

            if min_chunk_length > 0:
                min_length_filter = "AND LENGTH(c.chunk_text) >= :min_chunk_length"
                params["min_chunk_length"] = min_chunk_length

            query = text(
                f"""
                SELECT
                    c.id as chunk_id,
                    c.chunk_text,
                    c.position,
                    c.chunk_metadata,
                    d.filename,
                    d.original_filename,
                    d.id as document_id,
                    d.document_metadata,
                    1 - (c.embedding <=> CAST(:query_embedding AS vector)) as similarity
                FROM document_chunks c
                JOIN documents d ON c.document_id = d.id
                JOIN collection_documents cd ON d.id = cd.document_id
                WHERE cd.collection_id = :collection_id
                    AND c.embedding IS NOT NULL
                    AND 1 - (c.embedding <=> CAST(:query_embedding AS vector)) >= :similarity_threshold
                    {exclusion_filter}
                    {min_length_filter}
                ORDER BY c.embedding <=> CAST(:query_embedding AS vector)
                LIMIT :limit
            """
            )

            result = self.db.execute(query, params)
            chunks = []

            for row in result:
                chunk_data = {
                    "chunk_id": row.chunk_id,
                    "text": self.base._clean_text(row.chunk_text),
                    "chunk_index": row.position,  # Backward compatibility
                    "token_count": (row.chunk_metadata or {}).get("token_count", 0),
                    "start_char": (row.chunk_metadata or {}).get("start_char", 0),
                    "end_char": (row.chunk_metadata or {}).get("end_char", 0),
                    "metadata": row.chunk_metadata or {},
                    "filename": row.filename,
                    "original_filename": row.original_filename,
                    "document_id": row.document_id,
                    "document_metadata": row.document_metadata or {},
                    "similarity": float(row.similarity),
                    "similarity_score": float(row.similarity),  # Backward compatibility
                }
                chunks.append(self.base._clean_result_data(chunk_data))

            logger.info(
                f"Found {len(chunks)} relevant chunks in collection {collection_id}"
            )
            return chunks

        except Exception as e:
            logger.error(f"Error finding relevant chunks in collection: {e}")
            return []

    def find_relevant_chunks_in_document(
        self,
        document_id: str,
        query_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.0,
        min_chunk_length: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Find relevant chunks in specific document using vector similarity.

        Args:
            document_id: Document to search in
            query_embedding: Query vector for similarity search
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            min_chunk_length: Minimum chunk text length (filters short headlines/sentences)

        Returns:
            List of relevant chunks with similarity scores
        """
        try:
            # Build min length filter
            min_length_filter = ""
            params = {
                "document_id": document_id,
                "query_embedding": query_embedding,
                "similarity_threshold": similarity_threshold,
                "limit": limit,
            }

            if min_chunk_length > 0:
                min_length_filter = "AND LENGTH(c.chunk_text) >= :min_chunk_length"
                params["min_chunk_length"] = min_chunk_length

            query = text(
                f"""
                SELECT
                    c.id as chunk_id,
                    c.chunk_text,
                    c.position,
                    c.chunk_metadata,
                    d.filename,
                    d.original_filename,
                    d.id as document_id,
                    d.document_metadata,
                    1 - (c.embedding <=> CAST(:query_embedding AS vector)) as similarity
                FROM document_chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.document_id = :document_id
                    AND c.embedding IS NOT NULL
                    AND 1 - (c.embedding <=> CAST(:query_embedding AS vector)) >= :similarity_threshold
                    {min_length_filter}
                ORDER BY c.embedding <=> CAST(:query_embedding AS vector)
                LIMIT :limit
            """
            )

            result = self.db.execute(query, params)

            chunks = []
            for row in result:
                chunk_data = {
                    "chunk_id": row.chunk_id,
                    "text": self.base._clean_text(row.chunk_text),
                    "chunk_index": row.position,  # Backward compatibility
                    "token_count": (row.chunk_metadata or {}).get("token_count", 0),
                    "start_char": (row.chunk_metadata or {}).get("start_char", 0),
                    "end_char": (row.chunk_metadata or {}).get("end_char", 0),
                    "metadata": row.chunk_metadata or {},
                    "filename": row.filename,
                    "original_filename": row.original_filename,
                    "document_id": row.document_id,
                    "document_metadata": row.document_metadata or {},
                    "similarity": float(row.similarity),
                    "similarity_score": float(row.similarity),  # Backward compatibility
                }
                chunks.append(self.base._clean_result_data(chunk_data))

            logger.info(
                f"Found {len(chunks)} relevant chunks in document {document_id}"
            )
            return chunks

        except Exception as e:
            logger.error(f"Error finding relevant chunks in document: {e}")
            return []

    def find_relevant_chunks_hybrid(
        self,
        collection_id: str,
        query_embedding: List[float],
        text_keywords: List[str] = None,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        keyword_boost: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector similarity and keyword matching.

        Args:
            collection_id: Collection to search in
            query_embedding: Query vector for similarity search
            text_keywords: Keywords for text-based boosting
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            keyword_boost: Boost factor for keyword matches

        Returns:
            List of relevant chunks with hybrid scores
        """
        try:
            # Start with base similarity search
            base_chunks = self.find_relevant_chunks_in_collection(
                collection_id, query_embedding, limit * 2, similarity_threshold
            )

            if not text_keywords or not base_chunks:
                return base_chunks[:limit]

            # Apply keyword boosting
            boosted_chunks = []
            keywords_lower = [kw.lower() for kw in text_keywords]

            for chunk in base_chunks:
                chunk_text_lower = chunk["text"].lower()
                keyword_matches = sum(
                    1 for keyword in keywords_lower if keyword in chunk_text_lower
                )

                # Apply keyword boost to similarity score
                boost = keyword_matches * keyword_boost
                chunk["hybrid_score"] = chunk["similarity"] + boost
                chunk["keyword_matches"] = keyword_matches

                boosted_chunks.append(chunk)

            # Sort by hybrid score and return top results
            boosted_chunks.sort(key=lambda x: x["hybrid_score"], reverse=True)
            result_chunks = boosted_chunks[:limit]

            logger.info(
                f"Hybrid search found {len(result_chunks)} chunks with keyword boosting"
            )
            return result_chunks

        except Exception as e:
            logger.error(f"Error in hybrid chunk search: {e}")
            return []

    def get_embedding_statistics(self, collection_id: str = None) -> Dict[str, Any]:
        """
        Get embedding statistics for collection or entire database.

        Args:
            collection_id: Optional collection to get stats for

        Returns:
            Dictionary with embedding statistics
        """
        try:
            if collection_id:
                # Collection-specific statistics
                query = text(
                    """
                    SELECT
                        COUNT(*) as total_chunks,
                        COUNT(c.embedding) as chunks_with_embeddings,
                        COUNT(*) - COUNT(c.embedding) as chunks_without_embeddings,
                        AVG(c.token_count) as avg_token_count,
                        AVG(CASE WHEN c.embedding IS NOT NULL THEN array_length(c.embedding, 1) END) as avg_embedding_dimension
                    FROM document_chunks c
                    JOIN documents d ON c.document_id = d.id
                    JOIN collection_documents cd ON d.id = cd.document_id
                    WHERE cd.collection_id = :collection_id
                """
                )

                result = self.db.execute(
                    query, {"collection_id": collection_id}
                ).fetchone()

                if result:
                    stats = {
                        "collection_id": collection_id,
                        "total_chunks": result.total_chunks or 0,
                        "chunks_with_embeddings": result.chunks_with_embeddings or 0,
                        "chunks_without_embeddings": result.chunks_without_embeddings
                        or 0,
                        "embedding_coverage": (
                            (result.chunks_with_embeddings / result.total_chunks * 100)
                            if result.total_chunks > 0
                            else 0.0
                        ),
                        "avg_token_count": float(result.avg_token_count or 0),
                        "avg_embedding_dimension": int(
                            result.avg_embedding_dimension or 0
                        ),
                    }
                else:
                    stats = {
                        "collection_id": collection_id,
                        "total_chunks": 0,
                        "chunks_with_embeddings": 0,
                        "chunks_without_embeddings": 0,
                        "embedding_coverage": 0.0,
                        "avg_token_count": 0.0,
                        "avg_embedding_dimension": 0,
                    }

            else:
                # Global statistics
                query = text(
                    """
                    SELECT
                        COUNT(*) as total_chunks,
                        COUNT(embedding) as chunks_with_embeddings,
                        COUNT(*) - COUNT(embedding) as chunks_without_embeddings,
                        AVG(token_count) as avg_token_count,
                        AVG(CASE WHEN embedding IS NOT NULL THEN array_length(embedding, 1) END) as avg_embedding_dimension
                    FROM document_chunks
                """
                )

                result = self.db.execute(query).fetchone()

                stats = {
                    "scope": "global",
                    "total_chunks": result.total_chunks or 0,
                    "chunks_with_embeddings": result.chunks_with_embeddings or 0,
                    "chunks_without_embeddings": result.chunks_without_embeddings or 0,
                    "embedding_coverage": (
                        (result.chunks_with_embeddings / result.total_chunks * 100)
                        if result.total_chunks > 0
                        else 0.0
                    ),
                    "avg_token_count": float(result.avg_token_count or 0),
                    "avg_embedding_dimension": int(result.avg_embedding_dimension or 0),
                }

            logger.info(f"Retrieved embedding statistics: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error getting embedding statistics: {e}")
            return {
                "total_chunks": 0,
                "chunks_with_embeddings": 0,
                "chunks_without_embeddings": 0,
                "embedding_coverage": 0.0,
                "error": str(e),
            }

    def close(self):
        """Close the storage connection."""
        self.base.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
