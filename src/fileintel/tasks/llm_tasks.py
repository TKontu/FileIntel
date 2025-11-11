"""
LLM integration Celery tasks.

Converts LLM operations to distributed Celery tasks for parallel processing
with proper rate limiting, retry logic, and error handling.
"""

import logging
import os
from typing import List, Dict, Any, Optional
from celery import group, chain

from fileintel.celery_config import app
from .base import BaseFileIntelTask
from fileintel.core.config import get_config

# Get configurable rate limit for LLM tasks
# Default: 60/m (60 requests per minute) - appropriate for local LLMs
# For cloud APIs: set FILEINTEL_LLM_RATE_LIMIT="10/m" or "3/m" depending on your plan
# For powerful local LLMs: set FILEINTEL_LLM_RATE_LIMIT="120/m" or higher
# To disable rate limiting: set FILEINTEL_LLM_RATE_LIMIT="" (empty string)
LLM_RATE_LIMIT = os.getenv("FILEINTEL_LLM_RATE_LIMIT", "60/m")

logger = logging.getLogger(__name__)


def create_embedding_client():
    """Pure function to create OpenAI embedding client."""
    import openai

    config = get_config()

    # Use separate embedding server if configured
    embedding_base_url = config.get("llm.openai.embedding_base_url") or config.get(
        "llm.openai.base_url"
    )

    return openai.OpenAI(
        base_url=embedding_base_url, api_key=config.get("llm.openai.api_key")
    )


def prepare_text_for_embedding(text: str, max_tokens: int = 400) -> str:
    """
    Pure function to prepare text for embedding by cleaning and truncating if needed.

    Args:
        text: Input text
        max_tokens: Maximum number of tokens allowed

    Returns:
        Cleaned and truncated text, or None if text is too poor quality
    """
    import tiktoken
    import re

    # Clean the text first
    cleaned_text = text.strip()

    # Remove excessive whitespace and normalize
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    # Fix tokenizer divergence patterns that cause vLLM failures
    # Replace excessive dots (table of contents artifacts) with single spaces
    cleaned_text = re.sub(r'\.{4,}', ' ', cleaned_text)
    # Clean up page number artifacts (dots followed by numbers)
    cleaned_text = re.sub(r'\.{3,}\s*\d+\s*$', '', cleaned_text, flags=re.MULTILINE)
    # Remove standalone sequences of dots and dashes
    cleaned_text = re.sub(r'^\s*[.\-]{4,}\s*$', '', cleaned_text, flags=re.MULTILINE)

    # Remove lines that are mostly dots or punctuation (table of contents artifacts)
    lines = cleaned_text.split('\n')
    clean_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip lines that are >70% punctuation/dots/spaces
        non_punct_chars = len(re.sub(r'[^\w\s]', '', line))
        total_chars = len(line)

        if total_chars > 0 and (non_punct_chars / total_chars) >= 0.3:
            clean_lines.append(line)

    cleaned_text = ' '.join(clean_lines).strip()

    # Additional OCR artifact cleaning
    # Remove excessive single character sequences (OCR artifacts)
    cleaned_text = re.sub(r'\b[a-zA-Z0-9]\s+[a-zA-Z0-9]\s+[a-zA-Z0-9]\s+', ' ', cleaned_text)

    # Clean up number/letter mixtures that look like OCR errors (e.g., "9 F5 a7")
    cleaned_text = re.sub(r'\b\d+\s+[A-Za-z]+\d+\s+[a-zA-Z]+\d+\b', ' ', cleaned_text)

    # Remove standalone single characters followed by punctuation
    cleaned_text = re.sub(r'\s+[a-zA-Z]\s*[,\.]\s*', ' ', cleaned_text)

    # If text is too short or empty after cleaning, return original
    if len(cleaned_text) < 10:
        cleaned_text = text.strip()

    # Final quality check - if still too short or looks garbled, skip embedding
    if len(cleaned_text) < 5:
        return ""

    # Check if text still looks like OCR artifacts after cleaning
    words = cleaned_text.split()
    if len(words) > 5:
        single_chars = sum(1 for word in words if len(word) == 1)
        single_char_ratio = single_chars / len(words)
        avg_word_length = sum(len(word) for word in words) / len(words)

        # Skip embedding if text quality is still poor
        if single_char_ratio > 0.4 or avg_word_length < 1.5:
            logger.warning(f"Skipping low-quality text (likely OCR artifacts): '{cleaned_text[:100]}...'")
            return ""

    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    except Exception:
        tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")

    tokens = tokenizer.encode(cleaned_text)
    if len(tokens) <= max_tokens:
        return cleaned_text

    # Truncate and decode back to text
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens)


@app.task(
    base=BaseFileIntelTask,
    bind=True,
    queue="embedding_processing",
    rate_limit="30/m",
    max_retries=3,
    default_retry_delay=60,
)
def generate_text_embedding(
    self, text: str, model: str = None, **kwargs
) -> Dict[str, Any]:
    """
    Generate embedding for a single text using OpenAI API.

    Args:
        text: Text to embed
        model: Embedding model to use (defaults from config)
        **kwargs: Additional parameters

    Returns:
        Dict containing embedding and metadata
    """
    self.validate_input(["text"], text=text)

    config = get_config()
    model = model or config.rag.embedding_model

    try:
        self.update_progress(0, 2, "Preparing text for embedding")

        # Prepare text (clean and truncate if needed)
        prepared_text = prepare_text_for_embedding(text)

        # Skip embedding if text is too poor quality
        if not prepared_text or len(prepared_text.strip()) < 5:
            logger.warning(f"Skipping embedding for low-quality text: '{text[:100]}...'")
            return {
                "text": text,
                "text_length": len(text),
                "prepared_text_length": 0,
                "embedding": None,
                "embedding_dimension": 0,
                "model": model,
                "tokens_used": 0,
                "status": "skipped",
                "skip_reason": "poor_quality_text"
            }

        self.update_progress(1, 2, "Generating embedding")

        # Create client and generate embedding
        client = create_embedding_client()
        response = client.embeddings.create(input=prepared_text, model=model)

        embedding = response.data[0].embedding

        result = {
            "text": text,
            "text_length": len(text),
            "prepared_text_length": len(prepared_text),
            "embedding": embedding,
            "embedding_dimension": len(embedding),
            "model": model,
            "tokens_used": response.usage.total_tokens
            if hasattr(response, "usage")
            else None,
            "status": "completed",
        }

        self.update_progress(2, 2, "Embedding generated successfully")
        return result

    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        # Retry for API errors
        if "rate limit" in str(e).lower() or "timeout" in str(e).lower():
            raise self.retry(exc=e, countdown=60)

        return {"text": text, "error": str(e), "status": "failed"}


# REMOVED: generate_collection_embeddings - unnecessary wrapper function
# Use generate_collection_embeddings_simple in workflow_tasks.py instead


# REMOVED: generate_collection_embeddings_with_completion - unnecessary wrapper function
# Use generate_collection_embeddings_simple in workflow_tasks.py instead


# REMOVED: complete_embeddings_and_collection - unnecessary wrapper function
# Completion handling is now integrated into generate_collection_embeddings_simple


@app.task(base=BaseFileIntelTask, bind=True, queue="embedding_processing")
def generate_and_store_chunk_embedding(
    self, chunk_id: str, text: str, model: str = None, **kwargs
) -> Dict[str, Any]:
    """
    Generate embedding for a single chunk and store it.

    Args:
        chunk_id: Chunk ID to update
        text: Text to generate embedding for
        model: Embedding model to use
        **kwargs: Additional parameters

    Returns:
        Dict containing result
    """
    try:
        from fileintel.llm_integration.embedding_provider import OpenAIEmbeddingProvider
        from fileintel.celery_config import get_shared_storage
        from fileintel.core.config import get_config
        import gc

        config = get_config()

        # Store embedding using shared storage
        storage = get_shared_storage()
        try:
            # Generate embedding with shared storage to avoid duplicate connections
            embedding_provider = OpenAIEmbeddingProvider(storage=storage, settings=config)
            embeddings = embedding_provider.get_embeddings([text])

            if not embeddings or len(embeddings) == 0:
                raise ValueError("No embedding generated")

            embedding = embeddings[0]  # Get the first (and only) embedding

            success = storage.update_chunk_embedding(chunk_id, embedding)

            if success:
                # Log at DEBUG level for each chunk (visible only when debugging)
                logger.debug(f"Successfully stored embedding for chunk {chunk_id}")

                result = {
                    "chunk_id": chunk_id,
                    "embedding_dimensions": len(embedding),
                    "status": "completed",
                }

                # Explicit memory cleanup to prevent accumulation
                del embedding
                del embeddings
                del embedding_provider
                gc.collect()

                return result
            else:
                raise ValueError(f"Failed to update chunk {chunk_id} in database")
        finally:
            storage.close()
            # Force garbage collection after storage cleanup
            gc.collect()

    except Exception as e:
        # Get chunk and document info for better error logging
        try:
            from fileintel.celery_config import get_shared_storage
            storage = get_shared_storage()
            try:
                chunk = storage.get_chunk(chunk_id)
                if chunk:
                    document_id = chunk.document_id
                    chunk_index = chunk.metadata.get('position', 'unknown') if chunk.metadata else 'unknown'
                    text_preview = text[:500] + "..." if len(text) > 500 else text
                    logger.error(
                        f"Error generating and storing embedding | "
                        f"chunk_id={chunk_id} | "
                        f"document_id={document_id} | "
                        f"chunk_index={chunk_index} | "
                        f"text_length={len(text)} chars | "
                        f"error={str(e)} | "
                        f"chunk_text:\n{text_preview}"
                    )
                else:
                    logger.error(
                        f"Error generating and storing embedding | "
                        f"chunk_id={chunk_id} (chunk not found in DB) | "
                        f"text_length={len(text)} chars | "
                        f"error={str(e)}"
                    )
            finally:
                storage.close()
        except Exception as log_error:
            # Fallback logging if we can't get chunk details
            logger.error(
                f"Error generating and storing embedding for chunk {chunk_id}: {e} "
                f"(Failed to retrieve chunk details: {log_error})"
            )

        return {"chunk_id": chunk_id, "error": str(e), "status": "failed"}


@app.task(base=BaseFileIntelTask, bind=True, queue="embedding_processing")
def generate_and_store_chunk_embeddings_batch(
    self, chunk_data: List[Dict[str, str]], model: str = None, **kwargs
) -> Dict[str, Any]:
    """
    Generate embeddings for multiple chunks in a single API call (batched).

    This is 10-25x more efficient than individual calls:
    - Reduces HTTP connection overhead (1 connection vs N connections)
    - Allows vLLM to process multiple texts in parallel on GPU
    - Reduces database connection overhead (1 connection vs N connections)
    - Reduces Celery task serialization overhead (1 task vs N tasks)

    Args:
        chunk_data: List of dicts with 'chunk_id' and 'text' keys
        model: Embedding model to use (optional, uses config default)
        **kwargs: Additional parameters

    Returns:
        Dict containing batch results with success/failure counts

    Example:
        chunk_data = [
            {"chunk_id": "abc123", "text": "First chunk text..."},
            {"chunk_id": "def456", "text": "Second chunk text..."},
            {"chunk_id": "ghi789", "text": "Third chunk text..."},
        ]
        result = generate_and_store_chunk_embeddings_batch(chunk_data)
        # result = {
        #     "batch_size": 3,
        #     "success_count": 3,
        #     "failed_count": 0,
        #     "failed_chunks": [],
        #     "status": "completed"
        # }
    """
    try:
        from fileintel.llm_integration.embedding_provider import OpenAIEmbeddingProvider
        from fileintel.celery_config import get_shared_storage
        from fileintel.core.config import get_config
        import gc

        config = get_config()
        batch_size = len(chunk_data)

        logger.info(f"Starting batch embedding generation for {batch_size} chunks")

        # Extract texts and chunk IDs
        texts = [item['text'] for item in chunk_data]
        chunk_ids = [item['chunk_id'] for item in chunk_data]

        # Validate input
        if not texts or not chunk_ids:
            raise ValueError("Empty chunk_data provided to batch task")

        if len(texts) != len(chunk_ids):
            raise ValueError(f"Mismatched texts ({len(texts)}) and chunk_ids ({len(chunk_ids)})")

        # Store embeddings using shared storage
        storage = get_shared_storage()
        try:
            # Generate ALL embeddings in a single API call
            # This is the key performance optimization - vLLM processes them in parallel
            embedding_provider = OpenAIEmbeddingProvider(storage=storage, settings=config)

            logger.debug(f"Calling API with batch of {batch_size} texts")
            embeddings = embedding_provider.get_embeddings(texts)

            if not embeddings:
                raise ValueError(f"No embeddings returned from API for batch of {batch_size}")

            if len(embeddings) != batch_size:
                raise ValueError(
                    f"Expected {batch_size} embeddings, got {len(embeddings)}. "
                    f"This indicates an API/provider issue."
                )

            # Store each embedding in the database
            success_count = 0
            failed_chunks = []

            for chunk_id, embedding in zip(chunk_ids, embeddings):
                try:
                    success = storage.update_chunk_embedding(chunk_id, embedding)
                    if success:
                        success_count += 1
                    else:
                        failed_chunks.append(chunk_id)
                        logger.warning(f"Failed to store embedding for chunk {chunk_id}")
                except Exception as e:
                    failed_chunks.append(chunk_id)
                    logger.error(f"Error storing embedding for chunk {chunk_id}: {e}")

            logger.info(
                f"Batch embedding complete: {success_count}/{batch_size} succeeded, "
                f"{len(failed_chunks)} failed"
            )

            result = {
                "batch_size": batch_size,
                "success_count": success_count,
                "failed_count": len(failed_chunks),
                "failed_chunks": failed_chunks,
                "status": "completed" if success_count == batch_size else "partial",
            }

            # Explicit memory cleanup to prevent accumulation
            del embeddings
            del embedding_provider
            gc.collect()

            return result

        finally:
            storage.close()
            # Force garbage collection after storage cleanup
            gc.collect()

    except Exception as e:
        logger.error(f"Error in batch embedding generation: {e}")

        # Return detailed error information for debugging
        return {
            "batch_size": len(chunk_data),
            "success_count": 0,
            "failed_count": len(chunk_data),
            "error": str(e),
            "status": "failed",
            "failed_chunks": [item['chunk_id'] for item in chunk_data],
        }


# REMOVED: generate_batch_embeddings - unnecessary wrapper function
# Use generate_text_embedding directly in groups for batch processing


@app.task(
    base=BaseFileIntelTask,
    bind=True,
    queue="llm_processing",
    rate_limit=LLM_RATE_LIMIT,
    max_retries=3,
    default_retry_delay=120,
)
def analyze_with_llm(
    self, prompt: str, model: str = None, max_tokens: int = None, **kwargs
) -> Dict[str, Any]:
    """
    Generate text analysis using LLM API.

    Args:
        prompt: Prompt for LLM analysis
        model: LLM model to use (defaults from config)
        max_tokens: Maximum tokens to generate
        **kwargs: Additional LLM parameters

    Returns:
        Dict containing LLM response and metadata
    """
    self.validate_input(["prompt"], prompt=prompt)

    config = get_config()
    model = model or config.llm.model
    max_tokens = max_tokens or config.llm.max_tokens

    try:
        self.update_progress(0, 2, "Preparing LLM request")

        # Use sync UnifiedLLMProvider directly (no event loop needed)
        from fileintel.llm_integration.unified_provider import UnifiedLLMProvider
        from fileintel.celery_config import get_shared_storage

        storage = get_shared_storage()
        try:
            llm_provider = UnifiedLLMProvider(config, storage)

            self.update_progress(1, 2, "Generating LLM response")

            # Generate response synchronously
            response = llm_provider.generate_response(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=config.llm.temperature,
                **kwargs,
            )

            result = {
                "prompt": prompt,
                "response": response.content,
                "model": response.model,
                "tokens_used": response.usage.get("total_tokens")
                if response.usage
                else None,
                "prompt_tokens": response.usage.get("prompt_tokens")
                if response.usage
                else None,
                "completion_tokens": response.usage.get("completion_tokens")
                if response.usage
                else None,
                "status": "completed",
            }

            self.update_progress(2, 2, "LLM analysis completed")
            return result
        finally:
            storage.close()

    except Exception as e:
        logger.error(f"Error in LLM analysis: {e}")
        # Retry for API errors
        if "rate limit" in str(e).lower() or "timeout" in str(e).lower():
            raise self.retry(exc=e, countdown=120)

        return {"prompt": prompt, "error": str(e), "status": "failed"}


@app.task(base=BaseFileIntelTask, bind=True, queue="llm_processing")
def summarize_content(
    self, text: str, summary_type: str = "brief", max_length: int = None, **kwargs
) -> Dict[str, Any]:
    """
    Generate content summary using LLM.

    Args:
        text: Text content to summarize
        summary_type: Type of summary (brief, detailed, bullet_points)
        max_length: Maximum length of summary
        **kwargs: Additional parameters

    Returns:
        Dict containing summary and metadata
    """
    self.validate_input(["text"], text=text)

    try:
        self.update_progress(0, 2, "Preparing content for summarization")

        # Create summarization prompt based on type
        if summary_type == "bullet_points":
            prompt_template = (
                "Summarize the following text as bullet points:\n\n{text}\n\nSummary:"
            )
        elif summary_type == "detailed":
            prompt_template = "Provide a detailed summary of the following text:\n\n{text}\n\nDetailed Summary:"
        else:  # brief
            prompt_template = "Provide a brief summary of the following text:\n\n{text}\n\nBrief Summary:"

        # Truncate text if too long
        max_text_length = 3000  # Leave room for prompt and response
        if len(text) > max_text_length:
            text = text[:max_text_length] + "..."

        prompt = prompt_template.format(text=text)

        self.update_progress(1, 2, "Generating summary")

        # Generate summary using LLM (avoid blocking with .get())
        analysis_task = analyze_with_llm.apply_async(
            args=[prompt], kwargs={"max_tokens": max_length or 300, **kwargs}
        )
        analysis_result = {"status": "processing", "task_id": analysis_task.id}

        # Return processing status instead of blocking for completion
        result = {
            "original_text_length": len(text),
            "summary_type": summary_type,
            "analysis_task_id": analysis_result["task_id"],
            "status": "processing",
            "message": "Summary generation task started",
        }

        self.update_progress(2, 2, "Content summarization completed")
        return result

    except Exception as e:
        logger.error(f"Error summarizing content: {e}")
        return {
            "original_text_length": len(text),
            "summary_type": summary_type,
            "error": str(e),
            "status": "failed",
        }


@app.task(
    base=BaseFileIntelTask,
    bind=True,
    queue="llm_processing",
    rate_limit=LLM_RATE_LIMIT,
    max_retries=3,
    default_retry_delay=120,
)
def extract_document_metadata(
    self,
    document_id: str,
    text_chunks: List[str],
    file_metadata: Optional[Dict[str, Any]] = None,
    max_chunks: int = 3,
    **kwargs
) -> Dict[str, Any]:
    """
    Extract structured metadata from document chunks using LLM analysis.

    Args:
        document_id: Document ID to extract metadata for
        text_chunks: List of text chunks from the document
        file_metadata: Existing metadata from file properties (optional)
        max_chunks: Maximum number of chunks to use for extraction (default: 3)
        **kwargs: Additional parameters

    Returns:
        Dict containing extracted metadata and processing info
    """
    self.validate_input(["document_id", "text_chunks"], document_id=document_id, text_chunks=text_chunks)

    config = get_config()

    try:
        self.update_progress(0, 3, "Initializing metadata extraction")

        from fileintel.llm_integration.unified_provider import UnifiedLLMProvider
        from fileintel.celery_config import get_shared_storage
        from fileintel.document_processing.metadata_extractor import MetadataExtractor
        from pathlib import Path

        storage = get_shared_storage()
        try:
            llm_provider = UnifiedLLMProvider(config, storage)

            # Get prompts directory - use environment variable or fallback to relative path
            import os
            prompts_base = os.getenv("PROMPTS_DIR", "./prompts")
            prompts_dir = Path(prompts_base) / "templates"

            # Debug logging
            logger.debug(f"Prompts base directory: {prompts_base}")
            logger.debug(f"Prompts templates directory: {prompts_dir}")
            logger.debug(f"Templates directory exists: {prompts_dir.exists()}")

            metadata_extraction_dir = prompts_dir / "metadata_extraction"
            logger.debug(f"Metadata extraction directory: {metadata_extraction_dir}")
            logger.debug(f"Metadata extraction directory exists: {metadata_extraction_dir.exists()}")

            if metadata_extraction_dir.exists():
                prompt_file = metadata_extraction_dir / "prompt.md"
                logger.debug(f"Prompt file path: {prompt_file}")
                logger.debug(f"Prompt file exists: {prompt_file.exists()}")

            metadata_extractor = MetadataExtractor(
                llm_provider=llm_provider,
                prompts_dir=prompts_dir,
                max_length=config.llm.context_length,
                max_chunks_for_extraction=max_chunks,
            )

            self.update_progress(1, 3, "Extracting metadata with LLM")

            # Extract metadata
            extracted_metadata = metadata_extractor.extract_metadata(text_chunks, file_metadata)

            self.update_progress(2, 3, "Storing metadata in database")

            # Store extracted metadata in document
            if extracted_metadata:
                document = storage.get_document(document_id)
                if document:
                    # Check if this is a force re-extraction (indicated by existing llm_extracted metadata)
                    existing_metadata = document.document_metadata or {}
                    has_existing_llm_metadata = (
                        existing_metadata.get("llm_extracted", False) or
                        existing_metadata.get("extraction_method") == "llm_analysis"
                    )

                    # If force re-extracting, replace entirely. Otherwise merge.
                    replace = has_existing_llm_metadata
                    storage.update_document_metadata(document_id, extracted_metadata, replace=replace)

                    action = "Replaced" if replace else "Updated"
                    logger.info(f"{action} document {document_id} with extracted metadata")
                else:
                    logger.warning(f"Document {document_id} not found, cannot store metadata")

            result = {
                "document_id": document_id,
                "extracted_metadata": extracted_metadata,
                "chunks_processed": len(text_chunks),
                "file_metadata_provided": file_metadata is not None,
                "fields_extracted": len(extracted_metadata) if extracted_metadata else 0,
                "status": "completed",
            }

            self.update_progress(3, 3, "Metadata extraction completed")
            return result
        finally:
            storage.close()

    except Exception as e:
        logger.error(f"Error extracting metadata for document {document_id}: {e}")
        # Retry for API errors
        if "rate limit" in str(e).lower() or "timeout" in str(e).lower():
            raise self.retry(exc=e, countdown=120)

        return {
            "document_id": document_id,
            "chunks_processed": len(text_chunks),
            "error": str(e),
            "status": "failed",
        }


@app.task(
    bind=True,
    base=BaseFileIntelTask,
    name="fileintel.tasks.generate_citation",
    max_retries=3,
    default_retry_delay=60,
    rate_limit=LLM_RATE_LIMIT,
    acks_late=True,
)
def generate_citation(
    self,
    text_segment: str,
    collection_id: str,
    document_id: Optional[str] = None,
    min_similarity: Optional[float] = None,
    include_llm_analysis: bool = False,
    top_k: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate citation for a text segment using vector similarity search.

    Args:
        text_segment: The text that needs citation (10-5000 chars)
        collection_id: Collection to search in
        document_id: Optional specific document to search within
        min_similarity: Minimum similarity threshold (0.0-1.0)
        include_llm_analysis: Use LLM for relevance analysis
        top_k: Number of candidates to retrieve
        **kwargs: Additional parameters

    Returns:
        Dict containing:
        - citation: {in_text, full, style}
        - source: {document_id, chunk_id, similarity_score, text_excerpt, ...}
        - confidence: "high"|"medium"|"low"
        - relevance_note: str (if include_llm_analysis)
        - warning: str (if applicable)
        - status: "completed" or "failed"
    """
    self.validate_input(
        ["text_segment", "collection_id"],
        text_segment=text_segment,
        collection_id=collection_id
    )

    config = get_config()

    try:
        self.update_progress(0, 3, "Initializing citation generation")

        from fileintel.celery_config import get_shared_storage
        from fileintel.services.citation_service import CitationGenerationService

        # Get storage
        storage = get_shared_storage()

        try:
            self.update_progress(1, 3, "Searching for matching source")

            # Initialize citation service
            citation_service = CitationGenerationService(config, storage)

            # Generate citation
            result = citation_service.generate_citation(
                text_segment=text_segment,
                collection_id=collection_id,
                document_id=document_id,
                min_similarity=min_similarity,
                include_llm_analysis=include_llm_analysis,
                top_k=top_k
            )

            self.update_progress(2, 3, "Citation generated successfully")

            # Add status to result
            result["status"] = "completed"
            result["text_segment"] = text_segment[:100] + "..." if len(text_segment) > 100 else text_segment

            self.update_progress(3, 3, "Citation generation completed")
            return result

        finally:
            storage.close()

    except ValueError as e:
        # Input validation errors
        logger.error(f"Citation generation validation error: {e}")
        return {
            "text_segment": text_segment[:100] + "..." if len(text_segment) > 100 else text_segment,
            "collection_id": collection_id,
            "error": str(e),
            "error_type": "validation_error",
            "status": "failed",
        }

    except RuntimeError as e:
        # No source found errors
        logger.warning(f"Citation generation: No source found: {e}")
        return {
            "text_segment": text_segment[:100] + "..." if len(text_segment) > 100 else text_segment,
            "collection_id": collection_id,
            "error": str(e),
            "error_type": "no_source_found",
            "status": "failed",
        }

    except Exception as e:
        logger.error(f"Error generating citation: {e}", exc_info=True)

        # Retry for API errors
        if "rate limit" in str(e).lower() or "timeout" in str(e).lower():
            raise self.retry(exc=e, countdown=60)

        return {
            "text_segment": text_segment[:100] + "..." if len(text_segment) > 100 else text_segment,
            "collection_id": collection_id,
            "error": str(e),
            "error_type": "internal_error",
            "status": "failed",
        }


@app.task(
    base=BaseFileIntelTask,
    name="fileintel.tasks.inject_citation",
    max_retries=3,
    default_retry_delay=60,
    rate_limit=LLM_RATE_LIMIT,
    acks_late=True,
)
def inject_citation_task(
    self,
    text_segment: str,
    collection_id: str,
    document_id: Optional[str] = None,
    min_similarity: Optional[float] = None,
    top_k: Optional[int] = None,
    insertion_style: str = "footnote",
    include_full_citation: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Inject citation into text segment.

    Finds the best matching source and injects a Harvard citation into the text
    using the specified insertion style.

    Args:
        text_segment: Text to annotate (10-10000 chars)
        collection_id: Collection to search in
        document_id: Optional specific document to search within
        min_similarity: Minimum similarity threshold (0.0-1.0)
        top_k: Number of candidates to retrieve
        insertion_style: 'inline', 'footnote', 'endnote', or 'markdown_link'
        include_full_citation: Include full citation text
        **kwargs: Additional parameters

    Returns:
        Dict containing:
        - annotated_text: Text with citation injected
        - original_text: Original input text
        - citation: {in_text, full, style}
        - source: {document_id, chunk_id, similarity_score, ...}
        - confidence: "high"|"medium"|"low"
        - insertion_style: Style used
        - character_positions: {start, end}
        - status: "completed" or "failed"
    """
    self.validate_input(
        ["text_segment", "collection_id"],
        text_segment=text_segment,
        collection_id=collection_id
    )

    config = get_config()

    try:
        self.update_progress(0, 4, "Initializing citation injection")

        from fileintel.celery_config import get_shared_storage
        from fileintel.services.citation_service import CitationGenerationService

        storage = get_shared_storage()

        try:
            self.update_progress(1, 4, "Finding source for citation")

            # Use existing citation service to find source
            citation_service = CitationGenerationService(config, storage)
            citation_result = citation_service.generate_citation(
                text_segment=text_segment,
                collection_id=collection_id,
                document_id=document_id,
                min_similarity=min_similarity,
                top_k=top_k,
                include_llm_analysis=False  # Not needed for injection
            )

            # Check if citation generation failed
            if citation_result.get("status") == "failed":
                return {
                    "text_segment": text_segment[:100] + "..." if len(text_segment) > 100 else text_segment,
                    "error": citation_result.get("error", "Citation generation failed"),
                    "error_type": citation_result.get("error_type", "citation_failed"),
                    "status": "failed"
                }

            self.update_progress(2, 4, "Formatting citation injection")

            # Extract citation and source info
            citation = citation_result["citation"]
            source = citation_result["source"]
            confidence = citation_result["confidence"]

            # Inject citation based on style
            annotated_text, start_pos, end_pos = _inject_citation_into_text(
                text_segment,
                citation,
                insertion_style,
                include_full_citation
            )

            self.update_progress(3, 4, "Building response")

            result = {
                "annotated_text": annotated_text,
                "original_text": text_segment,
                "citation": citation,
                "source": source,
                "confidence": confidence,
                "insertion_style": insertion_style,
                "character_positions": {
                    "start": start_pos,
                    "end": end_pos
                },
                "status": "completed"
            }

            # Add warning if present
            if "warning" in citation_result:
                result["warning"] = citation_result["warning"]

            self.update_progress(4, 4, "Citation injection completed")
            return result

        finally:
            storage.close()

    except ValueError as e:
        # Input validation errors
        logger.error(f"Citation injection validation error: {e}")
        return {
            "text_segment": text_segment[:100] + "..." if len(text_segment) > 100 else text_segment,
            "collection_id": collection_id,
            "error": str(e),
            "error_type": "validation_error",
            "status": "failed",
        }

    except RuntimeError as e:
        # No source found errors
        logger.warning(f"Citation injection: No source found: {e}")
        return {
            "text_segment": text_segment[:100] + "..." if len(text_segment) > 100 else text_segment,
            "collection_id": collection_id,
            "error": str(e),
            "error_type": "no_source_found",
            "status": "failed",
        }

    except Exception as e:
        logger.error(f"Error injecting citation: {e}", exc_info=True)

        # Retry for API errors
        if "rate limit" in str(e).lower() or "timeout" in str(e).lower():
            raise self.retry(exc=e, countdown=60)

        return {
            "text_segment": text_segment[:100] + "..." if len(text_segment) > 100 else text_segment,
            "collection_id": collection_id,
            "error": str(e),
            "error_type": "internal_error",
            "status": "failed",
        }


def _inject_citation_into_text(
    text: str,
    citation: Dict[str, str],
    style: str,
    include_full: bool
) -> tuple[str, int, int]:
    """
    Inject citation into text based on style.

    Args:
        text: Original text
        citation: Citation dict with 'in_text' and 'full' keys
        style: Injection style (inline, footnote, endnote, markdown_link)
        include_full: Whether to include full citation

    Returns:
        Tuple of (annotated_text, start_position, end_position)
    """
    in_text = citation.get("in_text", "")
    full = citation.get("full", "")

    if style == "inline":
        # Inject at end: "Text." → "Text. (Author, Year)"
        # Strip trailing whitespace and periods, then add period + citation
        clean_text = text.rstrip()
        if clean_text.endswith('.'):
            clean_text = clean_text[:-1]
        annotated = f"{clean_text}. {in_text}"
        start_pos = len(clean_text) + 2  # After ". "
        end_pos = len(annotated)

    elif style == "footnote":
        # Add superscript footnote: "Text.[1]" + "\n\n[1] Full citation"
        annotated = f"{text}¹"
        if include_full:
            annotated += f"\n\n[1] {full}"
        start_pos = len(text)
        end_pos = len(annotated)

    elif style == "endnote":
        # Similar to footnote but marked for endnote section
        annotated = f"{text}[dn1]"
        if include_full:
            annotated += f"\n\n[dn1] {full}"
        start_pos = len(text)
        end_pos = len(annotated)

    elif style == "markdown_link":
        # Markdown link: "Text [(Author, Year)](#source-id)"
        # Extract document_id from citation if available (not in citation dict directly)
        source_id = "source"  # Fallback ID
        annotated = f"{text} [{in_text}](#{source_id})"
        if include_full:
            annotated += f"\n\n[{source_id}]: {full}"
        start_pos = len(text) + 1  # After space
        end_pos = len(annotated) if not include_full else len(text) + 1 + len(f"[{in_text}](#{source_id})")

    else:
        # Default to inline if invalid style
        clean_text = text.rstrip()
        if clean_text.endswith('.'):
            clean_text = clean_text[:-1]
        annotated = f"{clean_text}. {in_text}"
        start_pos = len(clean_text) + 2
        end_pos = len(annotated)

    return annotated, start_pos, end_pos


@app.task(
    base=BaseFileIntelTask,
    name="fileintel.tasks.detect_plagiarism",
    max_retries=3,
    default_retry_delay=60,
    rate_limit="60/m",  # Limit to 60 per minute to avoid overwhelming system
    acks_late=True,
)
def detect_plagiarism_task(
    self,
    document_id: str,
    collection_id: str,
    min_similarity: float = 0.7,
    chunk_overlap_factor: float = 0.3,
    include_sources: bool = True,
    group_by_source: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Detect plagiarism by comparing document chunks against collection.

    Analyzes each chunk of the document for similarity against all chunks
    in the reference collection, identifies potential plagiarism, and
    calculates statistics.

    Args:
        document_id: Document to analyze (from any collection)
        collection_id: Reference collection to search against
        min_similarity: Minimum similarity to flag (0.0-1.0, default 0.7)
        chunk_overlap_factor: Min fraction of chunks matching to report source (0.0-1.0)
        include_sources: Include detailed match information
        group_by_source: Group results by source document
        **kwargs: Additional parameters

    Returns:
        Dict containing:
        - analyzed_document_id: Document analyzed
        - analyzed_filename: Filename
        - total_chunks: Total chunks in document
        - flagged_chunks_count: Chunks flagged as suspicious
        - suspicious_percentage: % of document flagged
        - overall_plagiarism_risk: "high"|"medium"|"low"|"none"
        - matches: List of source documents with matches
        - status: "completed" or "failed"
    """
    from datetime import datetime

    self.validate_input(
        ["document_id", "collection_id"],
        document_id=document_id,
        collection_id=collection_id
    )

    try:
        self.update_progress(0, 5, "Initializing plagiarism detection")

        from fileintel.celery_config import get_shared_storage

        storage = get_shared_storage()

        try:
            # Get document info
            document = storage.get_document(document_id)
            if not document:
                return {
                    "error": f"Document '{document_id}' not found",
                    "error_type": "document_not_found",
                    "status": "failed"
                }

            self.update_progress(1, 5, "Loading document chunks")

            # Get all chunks from document (CORRECTED method name)
            document_chunks = storage.get_all_chunks_for_document(document_id)

            if not document_chunks:
                return {
                    "error": f"Document '{document_id}' has no chunks",
                    "error_type": "no_chunks",
                    "status": "failed"
                }

            # Filter chunks with embeddings
            chunks_with_embeddings = [c for c in document_chunks if c.embedding is not None]

            if not chunks_with_embeddings:
                return {
                    "error": f"Document chunks have no embeddings",
                    "error_type": "no_embeddings",
                    "status": "failed"
                }

            total_chunks = len(chunks_with_embeddings)
            logger.info(f"Analyzing {total_chunks} chunks from document '{document.filename}'")

            self.update_progress(2, 5, f"Analyzing {total_chunks} chunks for similarity")

            # Find similar chunks for each document chunk
            matches_by_source = {}
            flagged_chunk_ids = set()

            for i, chunk in enumerate(chunks_with_embeddings):
                # Update progress every 10% of chunks
                if i % max(1, total_chunks // 10) == 0:
                    progress = 2 + (i / total_chunks) * 2  # Progress from 2 to 4
                    self.update_progress(
                        progress, 5,
                        f"Processed {i}/{total_chunks} chunks"
                    )

                # Search for similar chunks in reference collection
                similar_chunks = storage.find_relevant_chunks_in_collection(
                    collection_id=collection_id,
                    query_embedding=chunk.embedding,
                    limit=20,
                    similarity_threshold=min_similarity,
                    exclude_chunks=[chunk.id]  # Don't match against self
                )

                # Process matches
                for match in similar_chunks:
                    source_doc_id = match['document_id']

                    # Skip if matching against same document
                    if source_doc_id == document_id:
                        continue

                    # Mark this chunk as flagged
                    flagged_chunk_ids.add(chunk.id)

                    # Initialize source entry if first match
                    if source_doc_id not in matches_by_source:
                        matches_by_source[source_doc_id] = {
                            'source_document_id': source_doc_id,
                            'source_filename': match.get('filename', 'Unknown'),
                            'matched_chunks': [],
                            'similarities': []
                        }

                    # Add match details
                    match_info = {
                        'analyzed_chunk_text': chunk.chunk_text[:200] if include_sources else "",  # Truncate for size
                        'source_chunk_text': match['text'][:200] if include_sources else "",
                        'similarity': match['similarity'],
                        'source_page': match.get('metadata', {}).get('page_number')
                    }

                    matches_by_source[source_doc_id]['matched_chunks'].append(match_info)
                    matches_by_source[source_doc_id]['similarities'].append(match['similarity'])

            self.update_progress(4, 5, "Calculating statistics")

            # Calculate statistics
            flagged_chunks_count = len(flagged_chunk_ids)
            suspicious_percentage = (flagged_chunks_count / total_chunks) * 100 if total_chunks > 0 else 0.0

            # Determine risk level
            if suspicious_percentage >= 50:
                risk_level = "high"
            elif suspicious_percentage >= 20:
                risk_level = "medium"
            elif suspicious_percentage >= 5:
                risk_level = "low"
            else:
                risk_level = "none"

            # Build matches list
            matches = []
            for source_id, source_data in matches_by_source.items():
                # Calculate match percentage for this source
                num_matched_chunks = len(source_data['matched_chunks'])
                match_percentage = (num_matched_chunks / total_chunks) * 100

                # Filter by chunk_overlap_factor
                if match_percentage < (chunk_overlap_factor * 100):
                    continue  # Skip sources with too few matches

                # Calculate average similarity
                avg_similarity = sum(source_data['similarities']) / len(source_data['similarities'])

                # Build match entry
                match_entry = {
                    'source_document_id': source_data['source_document_id'],
                    'source_filename': source_data['source_filename'],
                    'match_percentage': round(match_percentage, 2),
                    'average_similarity': round(avg_similarity, 3),
                }

                # Include matched chunks if requested (limit to top 10 by similarity)
                if include_sources:
                    sorted_chunks = sorted(
                        source_data['matched_chunks'],
                        key=lambda x: x['similarity'],
                        reverse=True
                    )[:10]  # Top 10 matches
                    match_entry['matched_chunks'] = sorted_chunks
                else:
                    match_entry['matched_chunks'] = []

                matches.append(match_entry)

            # Sort matches by match percentage (highest first)
            matches.sort(key=lambda x: x['match_percentage'], reverse=True)

            self.update_progress(5, 5, "Plagiarism detection completed")

            result = {
                "analyzed_document_id": document_id,
                "analyzed_filename": document.filename,
                "total_chunks": total_chunks,
                "flagged_chunks_count": flagged_chunks_count,
                "suspicious_percentage": round(suspicious_percentage, 2),
                "overall_plagiarism_risk": risk_level,
                "matches": matches,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "status": "completed"
            }

            logger.info(
                f"Plagiarism detection complete: {flagged_chunks_count}/{total_chunks} chunks flagged "
                f"({suspicious_percentage:.1f}%), risk: {risk_level}, {len(matches)} source(s) identified"
            )

            return result

        finally:
            storage.close()

    except ValueError as e:
        # Input validation errors
        logger.error(f"Plagiarism detection validation error: {e}")
        return {
            "document_id": document_id,
            "collection_id": collection_id,
            "error": str(e),
            "error_type": "validation_error",
            "status": "failed",
        }

    except Exception as e:
        logger.error(f"Error detecting plagiarism: {e}", exc_info=True)

        # Retry for transient errors
        if "rate limit" in str(e).lower() or "timeout" in str(e).lower():
            raise self.retry(exc=e, countdown=60)

        return {
            "document_id": document_id,
            "collection_id": collection_id,
            "error": str(e),
            "error_type": "internal_error",
            "status": "failed",
        }
