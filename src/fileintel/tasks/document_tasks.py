"""
Document processing Celery tasks.

Converts document processing workflows to distributed Celery tasks for multicore utilization.
Tasks are designed as pure functions with clear inputs/outputs and proper error handling.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from celery import group

from fileintel.celery_config import app
from .base import BaseFileIntelTask
from fileintel.core.config import get_config

logger = logging.getLogger(__name__)


def read_document_content(file_path: str) -> str:
    """
    Pure function to read document content based on file type.

    Args:
        file_path: Path to the document file

    Returns:
        Raw text content from the document

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file type is unsupported
    """
    from fileintel.document_processing.processors.traditional_pdf import (
        PDFProcessor as TraditionalPDFProcessor,
        validate_file_for_processing,
    )
    from fileintel.document_processing.processors.epub_processor import (
        EPUBReader as EPUBProcessor,
    )
    from fileintel.document_processing.processors.mobi_processor import (
        MOBIReader as MOBIProcessor,
    )

    path = Path(file_path)

    # Basic existence check first
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()

    # Validate file before processing
    validate_file_for_processing(path, extension)

    # Direct processor mapping - eliminates complex selection logic
    processors = {
        ".pdf": TraditionalPDFProcessor,
        ".epub": EPUBProcessor,
        ".mobi": MOBIProcessor,
    }

    processor_class = processors.get(extension)
    if not processor_class:
        supported_types = ", ".join(processors.keys())
        raise ValueError(
            f"Unsupported file type: {extension}. Supported types: {supported_types}"
        )

    # Process document and extract text
    processor = processor_class()
    elements, metadata = processor.read(path)
    return " ".join([elem.text for elem in elements if hasattr(elem, "text")])


def clean_and_chunk_text(
    text: str, chunk_size: int = None, overlap: int = None
) -> List[str]:
    """
    Pure function to clean and chunk text content.

    Args:
        text: Raw text content
        chunk_size: Size of each chunk (defaults from config)
        overlap: Overlap between chunks (defaults from config)

    Returns:
        List of text chunks
    """
    import re

    config = get_config()
    if chunk_size is None:
        chunk_size = config.rag.chunking.chunk_size
    if overlap is None:
        overlap = config.rag.chunking.chunk_overlap

    # Clean text
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = re.sub(r"\s+", " ", text).strip()

    # Chunk text
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


@app.task(base=BaseFileIntelTask, bind=True, queue="document_processing")
def process_document(
    self, file_path: str, document_id: str = None, collection_id: str = None, **kwargs
) -> Dict[str, Any]:
    """
    Process a single document: extract content, clean, and chunk it.

    Args:
        file_path: Path to the document file
        document_id: Unique identifier for the document
        collection_id: Collection to which the document belongs
        **kwargs: Additional processing parameters

    Returns:
        Dict containing document processing results
    """
    self.validate_input(["file_path"], file_path=file_path)

    try:
        # Update progress
        self.update_progress(0, 3, "Reading document content")

        # Read document content
        content = read_document_content(file_path)
        logger.info(f"Extracted {len(content)} characters from {file_path}")

        # Update progress
        self.update_progress(1, 3, "Cleaning and chunking text")

        # Clean and chunk text
        chunks = clean_and_chunk_text(content)
        logger.info(f"Created {len(chunks)} chunks from document")

        # Update progress
        self.update_progress(2, 3, "Storing chunks in database")

        # Store chunks in database
        if document_id and collection_id:
            from fileintel.storage.postgresql_storage import PostgreSQLStorage
            from fileintel.core.config import get_config
            import os
            import hashlib

            config = get_config()
            storage = PostgreSQLStorage(config)
            try:
                # Create document record first
                try:
                    # Get file information for document creation
                    file_size = os.path.getsize(file_path)
                    filename = os.path.basename(file_path)

                    # Generate content hash
                    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

                    # Determine MIME type based on file extension
                    mime_type = (
                        "application/pdf"
                        if file_path.lower().endswith(".pdf")
                        else "text/plain"
                    )

                    # Create the document record
                    document = storage.create_document(
                        filename=filename,
                        original_filename=filename,
                        content_hash=content_hash,
                        file_size=file_size,
                        mime_type=mime_type,
                        collection_id=collection_id,
                        metadata={"processed_by": "celery_task"},
                    )

                    # Update the document_id to use the one from the created document
                    actual_document_id = document.id
                    logger.info(
                        f"Created document record {actual_document_id} for {filename}"
                    )

                except Exception as e:
                    logger.warning(
                        f"Failed to create document record, using provided document_id {document_id}: {e}"
                    )
                    actual_document_id = document_id

                # Format chunks for storage
                chunk_data = [
                    {"text": chunk, "metadata": {"position": i}}
                    for i, chunk in enumerate(chunks)
                ]

                storage.add_document_chunks(
                    actual_document_id, collection_id, chunk_data
                )
                logger.info(
                    f"Stored {len(chunks)} chunks in database for document {actual_document_id}"
                )
            finally:
                storage.close()

        result = {
            "document_id": actual_document_id
            if "actual_document_id" in locals()
            else document_id,
            "collection_id": collection_id,
            "file_path": file_path,
            "content_length": len(content),
            "chunks_count": len(chunks),
            "chunks_stored": len(chunks) if document_id and collection_id else 0,
            "status": "completed",
        }

        self.update_progress(3, 3, "Document processing completed")
        return result

    except Exception as e:
        logger.error(f"Error processing document {file_path}: {e}")
        return {
            "document_id": document_id,
            "collection_id": collection_id,
            "file_path": file_path,
            "error": str(e),
            "status": "failed",
        }


@app.task(base=BaseFileIntelTask, bind=True, queue="document_processing")
def process_collection(
    self, collection_id: str, file_paths: List[str], **kwargs
) -> Dict[str, Any]:
    """
    Process multiple documents in a collection using parallel Celery tasks.

    Args:
        collection_id: Unique identifier for the collection
        file_paths: List of file paths to process
        **kwargs: Additional processing parameters

    Returns:
        Dict containing collection processing results
    """
    self.validate_input(
        ["collection_id", "file_paths"],
        collection_id=collection_id,
        file_paths=file_paths,
    )

    try:
        # Update collection status to processing
        from fileintel.storage.postgresql_storage import PostgreSQLStorage
        from fileintel.core.config import get_config

        config = get_config()
        storage = PostgreSQLStorage(config)
        try:
            storage.update_collection_status(collection_id, "processing")

            self.update_progress(
                0,
                len(file_paths),
                f"Starting batch processing of {len(file_paths)} documents",
            )

            # Create a group of parallel document processing tasks
            job = group(
                process_document.s(
                    file_path=file_path,
                    document_id=f"{collection_id}_{i}",
                    collection_id=collection_id,
                    **kwargs,
                )
                for i, file_path in enumerate(file_paths)
            )

            # Execute the group without blocking
            result = job.apply_async()

            # Return task information instead of blocking for results
            return {
                "collection_id": collection_id,
                "total_files": len(file_paths),
                "processing_task_id": result.id,
                "status": "processing",
                "message": f"Started processing {len(file_paths)} documents",
            }
        finally:
            storage.close()

    except Exception as e:
        logger.error(f"Error processing collection {collection_id}: {e}")

        # Update collection status to failed
        try:
            storage = PostgreSQLStorage(config)
            try:
                storage.update_collection_status(collection_id, "failed")
            finally:
                storage.close()
        except:
            pass  # Don't fail the task if status update fails

        return {"collection_id": collection_id, "error": str(e), "status": "failed"}


@app.task(base=BaseFileIntelTask, bind=True, queue="document_processing")
def extract_document_metadata(
    self, file_path: str, content_chunks: List[str] = None, **kwargs
) -> Dict[str, Any]:
    """
    Extract structured metadata from document using sophisticated LLM analysis.

    Args:
        file_path: Path to the document file
        content_chunks: Pre-processed content chunks (optional)
        **kwargs: Additional extraction parameters

    Returns:
        Dict containing clean, structured metadata
    """
    self.validate_input(["file_path"], file_path=file_path)

    try:
        self.update_progress(0, 3, "Preparing metadata extraction")

        # Get chunks if not provided
        if content_chunks is None:
            content = read_document_content(file_path)
            content_chunks = clean_and_chunk_text(content)

        self.update_progress(1, 3, "Preparing metadata extraction")

        # Create sync LLM provider directly
        from fileintel.llm_integration.unified_provider import UnifiedLLMProvider
        from fileintel.storage.postgresql_storage import PostgreSQLStorage
        from fileintel.core.config import get_config

        config = get_config()
        storage = PostgreSQLStorage(config)
        try:
            llm_provider = UnifiedLLMProvider(config, storage)
            prompts_dir = Path("prompts/templates")

            self.update_progress(2, 3, "Extracting metadata with MetadataExtractor")

            # Use MetadataExtractor with proper configuration
            from fileintel.document_processing.metadata_extractor import (
                MetadataExtractor,
            )

            extractor = MetadataExtractor(
                llm_provider=llm_provider,
                prompts_dir=prompts_dir,
                max_length=4000,
                max_chunks_for_extraction=3,
            )

            # Extract basic file metadata
            path = Path(file_path)
            file_metadata = {
                "file_name": path.name,
                "file_size": path.stat().st_size if path.exists() else 0,
                "file_path": str(path),
            }

            # Run extraction synchronously (no event loop needed)
            try:
                metadata = extractor.extract_metadata(content_chunks, file_metadata)
            except Exception as e:
                logger.warning(f"MetadataExtractor failed: {e}, using basic metadata")
                metadata = file_metadata
        finally:
            storage.close()

        self.update_progress(3, 3, "Metadata extraction completed")

        return {
            "metadata": metadata,
            "chunks_analyzed": min(len(content_chunks), 3),
            "status": "completed",
        }

    except Exception as e:
        logger.error(f"Error extracting metadata from {file_path}: {e}")
        return {"file_path": file_path, "error": str(e), "status": "failed"}


@app.task(base=BaseFileIntelTask, bind=True, queue="document_processing")
def chunk_existing_document(
    self, document_text: str, chunk_size: int = None, overlap: int = None, **kwargs
) -> Dict[str, Any]:
    """
    Re-chunk an already processed document with new parameters.

    Args:
        document_text: Full text content of the document
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        **kwargs: Additional chunking parameters

    Returns:
        Dict containing re-chunked document data
    """
    self.validate_input(["document_text"], document_text=document_text)

    try:
        self.update_progress(0, 1, "Re-chunking document")

        chunks = clean_and_chunk_text(document_text, chunk_size, overlap)

        result = {
            "original_length": len(document_text),
            "chunks_count": len(chunks),
            "chunks": chunks,
            "chunk_size": chunk_size or get_config().rag.chunking.chunk_size,
            "overlap": overlap or get_config().rag.chunking.chunk_overlap,
            "status": "completed",
        }

        self.update_progress(1, 1, "Re-chunking completed")
        return result

    except Exception as e:
        logger.error(f"Error re-chunking document: {e}")
        return {"error": str(e), "status": "failed"}
