"""
LLM integration Celery tasks.

Converts LLM operations to distributed Celery tasks for parallel processing
with proper rate limiting, retry logic, and error handling.
"""

import logging
from typing import List, Dict, Any, Optional
from celery import group, chain

from fileintel.celery_config import app
from .base import BaseFileIntelTask
from fileintel.core.config import get_config

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
    Pure function to prepare text for embedding by truncating if needed.

    Args:
        text: Input text
        max_tokens: Maximum number of tokens allowed

    Returns:
        Truncated text if necessary
    """
    import tiktoken

    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    except Exception:
        tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")

    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text

    # Truncate and decode back to text
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens)


@app.task(
    base=BaseFileIntelTask,
    bind=True,
    queue="llm_processing",
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

        # Prepare text (truncate if needed)
        prepared_text = prepare_text_for_embedding(text)

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


@app.task(base=BaseFileIntelTask, bind=True, queue="llm_processing")
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
        from fileintel.storage.postgresql_storage import PostgreSQLStorage
        from fileintel.core.config import get_config

        config = get_config()

        # Generate embedding
        embedding_provider = OpenAIEmbeddingProvider(settings=config)
        embeddings = embedding_provider.get_embeddings([text])

        if not embeddings or len(embeddings) == 0:
            raise ValueError("No embedding generated")

        embedding = embeddings[0]  # Get the first (and only) embedding

        # Store embedding in database
        storage = PostgreSQLStorage(config)
        try:
            success = storage.update_chunk_embedding(chunk_id, embedding)

            if success:
                logger.info(f"Successfully stored embedding for chunk {chunk_id}")
                return {
                    "chunk_id": chunk_id,
                    "embedding_dimensions": len(embedding),
                    "status": "completed",
                }
            else:
                raise ValueError(f"Failed to update chunk {chunk_id} in database")
        finally:
            # Ensure proper cleanup of database connection
            storage.close()

    except Exception as e:
        logger.error(
            f"Error generating and storing embedding for chunk {chunk_id}: {e}"
        )
        return {"chunk_id": chunk_id, "error": str(e), "status": "failed"}


# REMOVED: generate_batch_embeddings - unnecessary wrapper function
# Use generate_text_embedding directly in groups for batch processing


@app.task(
    base=BaseFileIntelTask,
    bind=True,
    queue="llm_processing",
    rate_limit="10/m",
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
        from fileintel.storage.postgresql_storage import PostgreSQLStorage

        storage = PostgreSQLStorage(config)
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
