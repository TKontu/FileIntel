from abc import ABC, abstractmethod
from typing import List
import tiktoken
import logging
from openai import OpenAI
from fileintel.core.config import get_config, Settings
from fileintel.storage.postgresql_storage import PostgreSQLStorage
from tenacity import retry, stop_after_attempt, wait_exponential


class EmbeddingProvider(ABC):
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Converts a list of texts into a list of embeddings."""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, storage: PostgreSQLStorage = None, settings: Settings = None):
        if settings is None:
            settings = get_config()

        # Use separate embedding server if configured, otherwise use main LLM server
        embedding_base_url = (
            settings.llm.openai.embedding_base_url or settings.llm.openai.base_url
        )
        self.client = OpenAI(
            base_url=embedding_base_url,
            api_key=settings.llm.openai.api_key,
        )
        self.model = settings.rag.embedding_model

        # Initialize tokenizer for text truncation
        try:
            self.tokenizer = tiktoken.get_encoding(
                "cl100k_base"
            )  # Standard OpenAI tokenizer
        except Exception:
            self.tokenizer = tiktoken.encoding_for_model(
                "text-embedding-ada-002"
            )  # Fallback

        # Set token limit from configuration (with fallback)
        self.max_tokens = getattr(settings.rag, "embedding_max_tokens", 480)

        # Use even more conservative limit to prevent any edge cases
        self.max_tokens = min(
            self.max_tokens, 400
        )  # Hard cap at 400 tokens (safe margin for 512 limit)

        logger = logging.getLogger(__name__)
        logger.info(
            f"Embedding provider initialized - Model: {self.model}, Base URL: {embedding_base_url}, Max tokens: {self.max_tokens}"
        )

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the same tokenizer as truncation."""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            # Fallback to character estimation
            logger = logging.getLogger(__name__)
            logger.warning(f"Token counting failed, using character estimation: {e}")
            return len(text) // 4  # Conservative estimate

    def _truncate_text(self, text: str) -> str:
        """
        LAST RESORT: Truncate text to fit within token limit.
        This should rarely be needed if chunking is working properly.
        """
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= self.max_tokens:
            return text

        # This is a critical error - chunking should prevent this
        logger = logging.getLogger(__name__)
        logger.error(
            f"EMERGENCY TRUNCATION: Text has {len(tokens)} tokens, exceeds {self.max_tokens} limit. "
            f"This indicates a bug in the text chunking system. "
            f"Text preview: {text[:200]}..."
        )

        # Truncate tokens and decode back to text
        truncated_tokens = tokens[: self.max_tokens]
        truncated_text = self.tokenizer.decode(truncated_tokens)

        logger.error(
            f"Truncated from {len(tokens)} to {len(truncated_tokens)} tokens. "
            f"FIX THE CHUNKING SYSTEM TO PREVENT THIS!"
        )

        return truncated_text

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Gets embeddings from OpenAI with automatic text truncation."""
        logger = logging.getLogger(__name__)

        # Debug: Log input text stats
        token_counts = [self._count_tokens(text) for text in texts]
        max_tokens = max(token_counts) if token_counts else 0
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0

        logger.info(
            f"Embedding request: {len(texts)} texts, "
            f"token range: {min(token_counts) if token_counts else 0}-{max_tokens}, "
            f"avg: {avg_tokens:.1f}, limit: {self.max_tokens}"
        )

        # Critical check before processing
        oversized = [
            i for i, count in enumerate(token_counts) if count > self.max_tokens
        ]
        if oversized:
            logger.error(
                f"CRITICAL: {len(oversized)} texts exceed {self.max_tokens} token limit! "
                f"Indices: {oversized[:5]}... This will cause vLLM failures!"
            )
            for idx in oversized[:3]:  # Log first few oversized texts
                logger.error(
                    f"Oversized text {idx}: {token_counts[idx]} tokens: {texts[idx][:200]}..."
                )

        # Truncate all texts to fit within token limits
        truncated_texts = [self._truncate_text(text) for text in texts]

        return self._get_embeddings_internal(truncated_texts)

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
    )
    def _get_embeddings_internal(self, texts: List[str]) -> List[List[float]]:
        logger = logging.getLogger(__name__)

        # Log batch details before sending to vLLM
        total_chars = sum(len(text) for text in texts)
        individual_tokens = [self._count_tokens(text) for text in texts]
        max_individual = max(individual_tokens) if individual_tokens else 0

        logger.info(
            f"Sending to vLLM: {len(texts)} texts, "
            f"total chars: {total_chars}, "
            f"max individual tokens: {max_individual}, "
            f"first text preview: {texts[0][:100] if texts else 'N/A'}..."
        )

        # Try batch processing first, fall back to individual if needed
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            error_msg = str(e)
            if "tokens" in error_msg and "maximum" in error_msg:
                logger.warning(
                    f"Batch processing failed due to token limit, falling back to individual processing: {error_msg}"
                )
                # Process texts individually
                embeddings = []
                for i, text in enumerate(texts):
                    try:
                        individual_response = self.client.embeddings.create(
                            model=self.model,
                            input=[text],  # Single text as list
                        )
                        embeddings.append(individual_response.data[0].embedding)
                        logger.debug(
                            f"Successfully processed individual text {i+1}/{len(texts)}"
                        )
                    except Exception as individual_error:
                        logger.error(
                            f"Failed to process individual text {i+1}: {individual_error}"
                        )
                        # Try aggressive truncation as last resort
                        if "tokens" in str(individual_error) and "maximum" in str(
                            individual_error
                        ):
                            logger.warning(
                                f"Attempting emergency truncation for text {i+1}"
                            )
                            try:
                                # Extra aggressive truncation to 300 tokens
                                emergency_truncated = self.tokenizer.decode(
                                    self.tokenizer.encode(text)[:300]
                                )
                                emergency_response = self.client.embeddings.create(
                                    model=self.model,
                                    input=[emergency_truncated],
                                )
                                embeddings.append(emergency_response.data[0].embedding)
                                logger.warning(
                                    f"Emergency truncation succeeded for text {i+1}"
                                )
                                continue
                            except Exception as emergency_error:
                                logger.error(
                                    f"Emergency truncation failed for text {i+1}: {emergency_error}"
                                )

                        # Use zero vector as final fallback
                        embeddings.append([0.0] * 1024)  # Assuming 1024-dim embeddings
                return embeddings
            else:
                # Re-raise non-token-related errors
                raise e
