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
from pgvector.sqlalchemy import Vector
import os

Base = declarative_base()


class Collection(Base):
    __tablename__ = "collections"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    documents = relationship(
        "Document", back_populates="collection", cascade="all, delete-orphan"
    )
    chunks = relationship(
        "DocumentChunk", back_populates="collection", cascade="all, delete-orphan"
    )


class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True)
    collection_id = Column(String, ForeignKey("collections.id"), nullable=False)
    filename = Column(String, nullable=False)
    content_hash = Column(String, nullable=False, unique=True)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    collection = relationship("Collection", back_populates="documents")
    jobs = relationship("Job", back_populates="document")
    chunks = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    collection_id = Column(String, ForeignKey("collections.id"), nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(1536))  # Assuming OpenAI's text-embedding-3-small
    chunk_metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    document = relationship("Document", back_populates="chunks")
    collection = relationship("Collection", back_populates="chunks")


class Job(Base):
    __tablename__ = "jobs"
    id = Column(String, primary_key=True)
    document_id = Column(
        String, ForeignKey("documents.id"), nullable=True
    )  # For indexing jobs
    collection_id = Column(
        String, ForeignKey("collections.id"), nullable=True
    )  # For query jobs
    job_type = Column(String, nullable=False)  # 'indexing' or 'query'
    status = Column(String, nullable=False)
    data = Column(JSON)  # For query question or other metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    document = relationship("Document", back_populates="jobs")
    result = relationship(
        "Result", back_populates="job", uselist=False, cascade="all, delete-orphan"
    )


class Result(Base):
    __tablename__ = "results"
    job_id = Column(String, ForeignKey("jobs.id"), primary_key=True)
    data = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    job = relationship("Job", back_populates="result")


# Construct the database URL from environment variables
db_user = os.getenv("DB_USER", "user")
db_password = os.getenv("DB_PASSWORD", "password")
db_host = os.getenv("DB_HOST", "postgres")
db_port = os.getenv("DB_PORT", "5432")
db_name = os.getenv("DB_NAME", "fileintel")

DATABASE_URL = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():
    Base.metadata.create_all(bind=engine)
