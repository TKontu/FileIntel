from sqlalchemy import create_engine, Column, String, JSON, Text, DateTime, Integer, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.sql import func
import os

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    content_hash = Column(String, nullable=False, unique=True)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    jobs = relationship("Job", back_populates="document")

class Job(Base):
    __tablename__ = 'jobs'
    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey('documents.id')) # Can be null for batch jobs
    job_type = Column(String, nullable=False, default="single_file") # 'single_file' or 'batch'
    status = Column(String, nullable=False)
    data = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    document = relationship("Document", back_populates="jobs")
    result = relationship("Result", back_populates="job", uselist=False)

class Result(Base):
    __tablename__ = 'results'
    job_id = Column(String, ForeignKey('jobs.id'), primary_key=True)
    data = Column(Text) # Using Text to store larger JSON strings
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
