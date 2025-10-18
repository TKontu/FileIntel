from sqlalchemy import (
    create_engine,
    Column,
    String,
    JSON,
    Text,
    DateTime,
    Integer,
    ForeignKey,
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from enum import Enum
import os

Base = declarative_base()


class CollectionStatus(str, Enum):
    """Valid collection processing status values."""

    CREATED = "created"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Collection(Base):
    __tablename__ = "collections"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(String, nullable=True)
    processing_status = Column(
        String, nullable=False, default=CollectionStatus.CREATED.value
    )  # See CollectionStatus enum for valid values
    collection_metadata = Column(
        JSONB, nullable=True
    )  # For GraphRAG index info and other metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Task tracking fields for preventing stuck collections
    current_task_id = Column(String, nullable=True, index=True)  # Currently running task
    task_history = Column(JSONB, nullable=True)  # History of all tasks for this collection
    status_updated_at = Column(DateTime(timezone=True), nullable=True)  # When status last changed

    documents = relationship(
        "Document", back_populates="collection", cascade="all, delete-orphan"
    )
    chunks = relationship(
        "DocumentChunk", back_populates="collection", cascade="all, delete-orphan"
    )
    graphrag_indices = relationship(
        "GraphRAGIndex", cascade="all, delete-orphan", overlaps="collection"
    )
    graphrag_entities = relationship(
        "GraphRAGEntity", cascade="all, delete-orphan", overlaps="collection"
    )
    graphrag_communities = relationship(
        "GraphRAGCommunity", cascade="all, delete-orphan", overlaps="collection"
    )
    graphrag_relationships = relationship(
        "GraphRAGRelationship", cascade="all, delete-orphan", overlaps="collection"
    )


class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True)
    collection_id = Column(
        String, ForeignKey("collections.id"), nullable=False, index=True
    )
    filename = Column(
        String, nullable=False
    )  # This will store the secure, UUID-based filename
    original_filename = Column(
        String, nullable=False
    )  # This will store the user's original filename
    content_hash = Column(
        String, nullable=False, index=True
    )  # No longer unique across all collections
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String, nullable=False)
    document_metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    collection = relationship("Collection", back_populates="documents")
    chunks = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )
    structures = relationship(
        "DocumentStructure", back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)
    collection_id = Column(
        String, ForeignKey("collections.id"), nullable=False, index=True
    )
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector())  # Assuming OpenAI's text-embedding-3-small
    chunk_metadata = Column(JSON)
    position = Column(
        Integer, nullable=False, default=0
    )  # Position in original document
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    document = relationship("Document", back_populates="chunks")
    collection = relationship("Collection", back_populates="chunks")


class DocumentStructure(Base):
    """
    Store extracted document structure (TOC, LOF, LOT, headers).

    Phase 4 of MinerU structure utilization - enables structure-based
    navigation and querying without embedding TOC/LOF in vector store.
    """
    __tablename__ = "document_structures"
    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)
    structure_type = Column(String, nullable=False, index=True)  # 'toc', 'lof', 'lot', 'headers'
    data = Column(JSONB, nullable=False)  # Structured entries (format depends on type)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    document = relationship("Document", back_populates="structures")


class GraphRAGIndex(Base):
    """Store GraphRAG index metadata and status"""

    __tablename__ = "graphrag_indices"
    id = Column(String, primary_key=True)
    collection_id = Column(String, ForeignKey("collections.id"), nullable=False)
    index_path = Column(String, nullable=False)  # Path to GraphRAG index files
    index_status = Column(
        String, nullable=False, default="building"
    )  # building, ready, failed, updating
    index_metadata = Column(JSONB, nullable=True)  # GraphRAG-specific metadata
    documents_count = Column(Integer, nullable=False, default=0)
    entities_count = Column(Integer, nullable=False, default=0)
    communities_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    collection = relationship("Collection")


class GraphRAGEntity(Base):
    """Store extracted entities from GraphRAG processing"""

    __tablename__ = "graphrag_entities"
    id = Column(String, primary_key=True)
    collection_id = Column(String, ForeignKey("collections.id"), nullable=False)
    entity_name = Column(String, nullable=False, index=True)
    entity_type = Column(String, nullable=True)  # PERSON, ORGANIZATION, CONCEPT, etc.
    description = Column(Text, nullable=True)
    importance_score = Column(Integer, nullable=True)
    entity_metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    collection = relationship("Collection")


class GraphRAGCommunity(Base):
    """Store GraphRAG community detection results"""

    __tablename__ = "graphrag_communities"
    id = Column(String, primary_key=True)
    collection_id = Column(String, ForeignKey("collections.id"), nullable=False)
    community_id = Column(Integer, nullable=False)  # GraphRAG community ID
    level = Column(Integer, nullable=False, default=0)  # Hierarchy level
    title = Column(String, nullable=True)
    summary = Column(Text, nullable=True)
    entities = Column(JSONB, nullable=True)  # List of entity IDs in this community
    size = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    collection = relationship("Collection")


class GraphRAGRelationship(Base):
    """Store entity relationships from GraphRAG"""

    __tablename__ = "graphrag_relationships"
    id = Column(String, primary_key=True)
    collection_id = Column(String, ForeignKey("collections.id"), nullable=False)
    source_entity = Column(String, nullable=False, index=True)
    target_entity = Column(String, nullable=False, index=True)
    relationship_type = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    weight = Column(Integer, nullable=True)
    relationship_metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    collection = relationship("Collection")


from urllib.parse import quote_plus

# Construct the database URL from environment variables
db_user = os.getenv("DB_USER", "user")
db_password = os.getenv("DB_PASSWORD", "password")
db_host = os.getenv("DB_HOST", "postgres")


db_port = os.getenv("DB_PORT", "5432")
db_name = os.getenv("DB_NAME", "fileintel")

# URL-encode the username and password
encoded_user = quote_plus(db_user)
encoded_password = quote_plus(db_password)

DATABASE_URL = (
    f"postgresql://{encoded_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():
    """Create all database tables if they don't exist."""
    import logging
    import os

    logger = logging.getLogger(__name__)

    # Check if database is disabled
    if os.environ.get('DISABLE_DATABASE', 'false').lower() == 'true':
        logger.info("Database disabled via DISABLE_DATABASE environment variable - skipping table creation")
        return

    try:
        logger.info("Starting database table creation...")
        logger.info(f"Database URL: {DATABASE_URL.replace(db_password, '***')}")

        # Test database connection first
        with engine.connect() as conn:
            logger.info("Database connection successful")

        # Create all tables
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)

        # Verify tables were created
        from sqlalchemy import text

        with engine.connect() as conn:
            tables = conn.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname='public'")
            ).fetchall()
            table_names = [row[0] for row in tables]
            logger.info(f"Tables created: {table_names}")

        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        logger.error(
            f"Database URL (masked): {DATABASE_URL.replace(db_password, '***')}"
        )
        raise
