from abc import ABC, abstractmethod
from typing import List, Tuple
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

        # Initialize primary tokenizer (OpenAI)
        try:
            self.openai_tokenizer = tiktoken.get_encoding(
                "cl100k_base"
            )  # Standard OpenAI tokenizer
        except Exception:
            self.openai_tokenizer = tiktoken.encoding_for_model(
                "text-embedding-ada-002"
            )  # Fallback

        # Initialize BERT tokenizer for vLLM compatibility check
        self.bert_tokenizer = None
        try:
            from transformers import AutoTokenizer
            # Common BERT-style tokenizer used by many embedding models
            self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            logger = logging.getLogger(__name__)
            logger.debug("BERT tokenizer loaded for dual validation")
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not load BERT tokenizer: {e}. Using OpenAI tokenizer only.")

        # Maintain backward compatibility
        self.tokenizer = self.openai_tokenizer

        # Set token limit from configuration (with fallback)
        self.max_tokens = getattr(settings.rag, "embedding_max_tokens", 480)

        # Apply optimized safety margin for better token utilization
        # Preserve 90% of configured limit while accounting for tokenizer differences
        safety_margin = 440 if self.bert_tokenizer else 460
        self.max_tokens = min(self.max_tokens, safety_margin)

        logger = logging.getLogger(__name__)
        logger.debug(
            f"Embedding provider initialized - Model: {self.model}, Base URL: {embedding_base_url}, Max tokens: {self.max_tokens}"
        )

    def _count_tokens_dual(self, text: str) -> Tuple[int, int, str]:
        """
        Count tokens using both OpenAI and BERT tokenizers.

        Returns:
            Tuple of (openai_tokens, bert_tokens, analysis)
        """
        logger = logging.getLogger(__name__)

        # Count with OpenAI tokenizer
        try:
            openai_tokens = len(self.openai_tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"OpenAI token counting failed: {e}")
            openai_tokens = len(text) // 4

        # Count with BERT tokenizer if available
        bert_tokens = None
        if self.bert_tokenizer:
            try:
                bert_encoding = self.bert_tokenizer(text, return_tensors=None, add_special_tokens=True)
                bert_tokens = len(bert_encoding['input_ids'])
            except Exception as e:
                logger.warning(f"BERT token counting failed: {e}")
                bert_tokens = len(text) // 3  # BERT tends to use more tokens

        # Analysis and warning generation
        analysis = ""
        if bert_tokens is not None:
            ratio = bert_tokens / openai_tokens if openai_tokens > 0 else float('inf')
            if ratio > 2.0:
                analysis = f"HIGH_DIVERGENCE (BERT {ratio:.1f}x OpenAI)"
            elif ratio > 1.5:
                analysis = f"MEDIUM_DIVERGENCE (BERT {ratio:.1f}x OpenAI)"
            else:
                analysis = f"LOW_DIVERGENCE (BERT {ratio:.1f}x OpenAI)"

            # Use the higher count for safety
            max_tokens = max(openai_tokens, bert_tokens)
        else:
            analysis = "OPENAI_ONLY"
            max_tokens = openai_tokens
            bert_tokens = openai_tokens  # Fallback for return value

        return openai_tokens, bert_tokens, analysis

    def _count_tokens(self, text: str) -> int:
        """Count tokens using the most conservative approach between tokenizers."""
        openai_tokens, bert_tokens, _ = self._count_tokens_dual(text)
        # Return the higher count for safety
        return max(openai_tokens, bert_tokens) if self.bert_tokenizer else openai_tokens

    def _truncate_text(self, text: str, chunk_context: str = None) -> str:
        """
        LAST RESORT: Truncate text to fit within token limit.
        This should rarely be needed if chunking is working properly.

        Args:
            text: Text to truncate
            chunk_context: Optional context for logging (document_id, chunk_id, etc.)
        """
        logger = logging.getLogger(__name__)

        # CRITICAL: Use BERT tokenizer if available, as that's what vLLM uses
        # Using OpenAI tokenizer causes token count mismatches
        if self.bert_tokenizer:
            # BERT tokenizer (matches vLLM server)
            encoding = self.bert_tokenizer(text, return_tensors=None, add_special_tokens=True, truncation=False)
            tokens = encoding['input_ids']
            token_count = len(tokens)

            if token_count <= self.max_tokens:
                return text

            # Truncate using BERT tokenizer
            logger.debug(f"Truncating with BERT tokenizer: {token_count} → {self.max_tokens} tokens")
            truncated_encoding = self.bert_tokenizer(
                text,
                max_length=self.max_tokens,
                truncation=True,
                return_tensors=None,
                add_special_tokens=True
            )
            truncated_text = self.bert_tokenizer.decode(truncated_encoding['input_ids'], skip_special_tokens=True)
        else:
            # Fallback to OpenAI tokenizer
            tokens = self.tokenizer.encode(text)
            token_count = len(tokens)

            if token_count <= self.max_tokens:
                return text

            # Truncate using OpenAI tokenizer
            logger.debug(f"Truncating with OpenAI tokenizer: {token_count} → {self.max_tokens} tokens")
            truncated_tokens = tokens[: self.max_tokens]
            truncated_text = self.tokenizer.decode(truncated_tokens)

        # Log truncation
        text_preview = text[:500] + "..." if len(text) > 500 else text
        context_str = f" | {chunk_context}" if chunk_context else ""

        logger.debug(
            f"Auto-truncating text{context_str} | "
            f"Token count: {token_count}/{self.max_tokens} (exceeds by {token_count - self.max_tokens}) | "
            f"Text length: {len(text)} chars | "
            f"Preview:\n{text_preview}"
        )

        return truncated_text

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Gets embeddings from OpenAI with automatic text truncation."""
        logger = logging.getLogger(__name__)

        # Debug: Log input text stats with dual tokenizer analysis
        token_counts = [self._count_tokens(text) for text in texts]
        max_tokens = max(token_counts) if token_counts else 0
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0

        # Detailed analysis for first text if using dual tokenizers
        dual_analysis = ""
        if texts and self.bert_tokenizer:
            openai_count, bert_count, analysis = self._count_tokens_dual(texts[0])
            dual_analysis = f", first_text_analysis: {analysis} (OpenAI:{openai_count}/BERT:{bert_count})"

        # Log at DEBUG level for detailed token analysis
        logger.debug(
            f"Embedding request: {len(texts)} texts, "
            f"token range: {min(token_counts) if token_counts else 0}-{max_tokens}, "
            f"avg: {avg_tokens:.1f}, limit: {self.max_tokens}{dual_analysis}"
        )

        # Critical check before processing
        oversized = [
            i for i, count in enumerate(token_counts) if count > self.max_tokens
        ]
        if oversized:
            logger.warning(
                f"WARNING: {len(oversized)} texts exceed {self.max_tokens} token limit (will auto-truncate). "
                f"Indices: {oversized[:5]}..."
            )
            for idx in oversized[:3]:  # Log first few oversized texts
                text = texts[idx]
                # Show more text for debugging (500 chars instead of 100)
                text_preview = text[:500] + "..." if len(text) > 500 else text
                if self.bert_tokenizer:
                    openai_count, bert_count, analysis = self._count_tokens_dual(text)
                    logger.debug(
                        f"Oversized text | "
                        f"index={idx} | "
                        f"analysis={analysis} | "
                        f"OpenAI_tokens={openai_count} | "
                        f"BERT_tokens={bert_count} | "
                        f"chars={len(text)} | "
                        f"text:\n{text_preview}"
                    )
                else:
                    logger.debug(
                        f"Oversized text | "
                        f"index={idx} | "
                        f"tokens={token_counts[idx]}/{self.max_tokens} | "
                        f"chars={len(text)} | "
                        f"text:\n{text_preview}"
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

        # Log at DEBUG level for detailed API call tracking
        logger.debug(
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
                        # Show more text for debugging
                        text_preview = text[:500] + "..." if len(text) > 500 else text
                        token_count = self._count_tokens(text)
                        logger.error(
                            f"Failed to process individual text | "
                            f"index={i+1}/{len(texts)} | "
                            f"tokens={token_count} | "
                            f"chars={len(text)} | "
                            f"error={str(individual_error)} | "
                            f"text:\n{text_preview}"
                        )
                        # Try aggressive truncation as last resort
                        if "tokens" in str(individual_error) and "maximum" in str(
                            individual_error
                        ):
                            logger.warning(
                                f"Attempting emergency truncation for text {i+1}/{len(texts)}"
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
                                    f"Emergency truncation succeeded for text {i+1}/{len(texts)}"
                                )
                                continue
                            except Exception as emergency_error:
                                logger.error(
                                    f"Emergency truncation failed | "
                                    f"index={i+1}/{len(texts)} | "
                                    f"error={str(emergency_error)}"
                                )

                        # Use zero vector as final fallback
                        embeddings.append([0.0] * 1024)  # Assuming 1024-dim embeddings
                return embeddings
            else:
                # Re-raise non-token-related errors
                raise e
