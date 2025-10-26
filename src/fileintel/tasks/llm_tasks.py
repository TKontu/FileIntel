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
                return {
                    "chunk_id": chunk_id,
                    "embedding_dimensions": len(embedding),
                    "status": "completed",
                }
            else:
                raise ValueError(f"Failed to update chunk {chunk_id} in database")
        finally:
            storage.close()

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
