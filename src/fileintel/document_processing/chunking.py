from typing import List, Tuple
import re
import tiktoken
import logging
from ..core.config import Settings

logger = logging.getLogger(__name__)

# Constants for token estimation and safety
CHARS_PER_TOKEN_ESTIMATE = 4  # Rough estimate for character-to-token ratio
TOKEN_WARNING_THRESHOLD = 0.9  # Warn when approaching 90% of token limit


class TextChunker:
    def __init__(self, config: Settings = None):
        from ..core.config import get_config

        if config is None:
            config = get_config()

        # Use unified chunking configuration
        chunking_config = config.rag.chunking
        self.target_sentences = chunking_config.target_sentences
        self.overlap_sentences = chunking_config.overlap_sentences
        self.max_chars = chunking_config.chunk_size

        # Token-based limits for embedding safety (configurable)
        self.vector_max_tokens = getattr(
            config.rag, "max_tokens", 450
        )  # Safe limit for 512 token embedding models
        self.graphrag_max_tokens = (
            config.rag.embedding_batch_max_tokens
        )  # Use configured GraphRAG token limit

        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.get_encoding(
                "cl100k_base"
            )  # Standard OpenAI tokenizer
        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to load cl100k_base tokenizer: {e}, using fallback")
            self.tokenizer = tiktoken.encoding_for_model(
                "text-embedding-ada-002"
            )  # Fallback

        # Simple sentence boundary pattern - matches periods, exclamation marks, and question marks
        self.sentence_pattern = re.compile(r"[.!?]+\s+")

        # Pattern for protecting abbreviations during sentence splitting
        self.abbreviation_pattern = re.compile(
            r"\b(Dr|Mr|Mrs|Ms|Prof|etc|vs|i\.e|e\.g)\."
        )

        logger.info(
            f"TextChunker initialized - Unified chunking: {self.max_chars} chars max, "
            f"Vector: {self.vector_max_tokens} tokens max, GraphRAG: {self.graphrag_max_tokens} tokens max, "
            f"Target sentences: {self.target_sentences}, Overlap: {self.overlap_sentences}"
        )

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple regex pattern."""
        # Protect common abbreviations that shouldn't break sentences
        text = self.abbreviation_pattern.sub(
            lambda m: m.group(0).replace(".", "<ABBREV>"), text
        )

        # Split on sentence boundaries
        sentences = self.sentence_pattern.split(text)

        # Clean up and restore abbreviations
        result = []
        for sentence in sentences:
            sentence = sentence.replace("<ABBREV>", ".").strip()
            if sentence:
                result.append(sentence)

        return result

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the tokenizer."""
        try:
            return len(self.tokenizer.encode(text))
        except (UnicodeDecodeError, UnicodeEncodeError, ValueError) as e:
            # Fallback to rough character-based estimation for encoding errors
            logger.warning(
                f"Token counting failed due to encoding error, using character estimation: {e}"
            )
            return len(text) // CHARS_PER_TOKEN_ESTIMATE

    def _check_token_safety(self, text: str, max_tokens: int, context: str) -> bool:
        """Check if text exceeds token limit and log warnings."""
        token_count = self._count_tokens(text)
        if token_count > max_tokens:
            logger.error(
                f"CRITICAL: {context} chunk has {token_count} tokens, exceeds {max_tokens} limit. "
                f"This will cause embedding failures. Text preview: {text[:100]}..."
            )
            return False
        elif token_count > max_tokens * TOKEN_WARNING_THRESHOLD:  # Warning threshold
            logger.warning(
                f"{context} chunk has {token_count} tokens, approaching {max_tokens} limit"
            )
        return True

    def _should_finalize_chunk(
        self,
        current_chunk: List[str],
        sentence: str,
        current_length: int,
        target_sentences: int,
        max_chars: int,
        max_tokens: int,
    ) -> bool:
        """Determine if current chunk should be finalized before adding sentence."""
        if not current_chunk:
            return False

        # Test potential chunk with this sentence added
        test_chunk_text = " ".join(current_chunk + [sentence])
        test_token_count = self._count_tokens(test_chunk_text)

        # Check if adding this sentence would exceed our targets
        would_exceed_sentences = len(current_chunk) >= target_sentences
        would_exceed_chars = current_length + len(sentence) > max_chars
        would_exceed_tokens = test_token_count > max_tokens

        # Decision logic: token limit is HARD limit, others are soft
        return would_exceed_tokens or (
            would_exceed_sentences
            and (would_exceed_chars or current_length > max_chars * 0.5)
        )

    def _create_chunk_with_overlap(
        self, current_chunk: List[str], sentence: str, overlap_sentences: int
    ) -> tuple[str, List[str], int]:
        """Create chunk and return new chunk with overlap."""
        # Create the completed chunk
        chunk_text = " ".join(current_chunk)

        # Calculate overlap: take last N sentences from current chunk as start of next
        overlap_start = max(0, len(current_chunk) - overlap_sentences)
        overlap_sentences_list = current_chunk[overlap_start:]

        # Start new chunk with overlap + current sentence
        new_chunk = overlap_sentences_list + [sentence]
        new_length = sum(len(s) for s in new_chunk) + len(new_chunk) - 1

        return chunk_text, new_chunk, new_length

    def _validate_chunks_against_token_limit(
        self, chunks: List[str], max_tokens: int
    ) -> List[str]:
        """Validate all chunks against token limits and filter out oversized ones."""
        validated_chunks = []
        for chunk in chunks:
            if self._check_token_safety(chunk, max_tokens, "Vector RAG"):
                validated_chunks.append(chunk)
            else:
                # This should rarely happen due to the logic above, but safety net
                logger.error(f"Dropping oversized chunk: {chunk[:100]}...")
        return validated_chunks

    def _chunk_by_sentences(
        self,
        text: str,
        target_sentences: int,
        max_chars: int,
        max_tokens: int,
        overlap_sentences: int = 1,
    ) -> List[str]:
        """
        Chunk text by complete sentences with intelligent overlap.
        Flexible with length to maintain semantic coherence.

        Args:
            text: Input text to chunk
            target_sentences: Target number of sentences per chunk
            max_chars: Soft character limit per chunk
            max_tokens: Hard token limit per chunk (for embedding safety)
            overlap_sentences: Number of sentences to overlap between chunks
        """
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if self._should_finalize_chunk(
                current_chunk,
                sentence,
                current_length,
                target_sentences,
                max_chars,
                max_tokens,
            ):
                # Finalize current chunk and start new one with overlap
                (
                    chunk_text,
                    current_chunk,
                    current_length,
                ) = self._create_chunk_with_overlap(
                    current_chunk, sentence, overlap_sentences
                )
                chunks.append(chunk_text)
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_length += sentence_length + 1  # +1 for space

        # Add final chunk if it has content
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Validate all chunks against token limits
        return self._validate_chunks_against_token_limit(chunks, max_tokens)

    def chunk_text(self, text: str) -> List[str]:
        """
        Chunks text into smaller pieces for Vector RAG.
        Prioritizes complete sentences over strict character limits.
        Enforces token limits for embedding safety.
        """
        logger.info(
            f"Chunking text: {len(text)} characters, token limit: {self.vector_max_tokens}"
        )

        chunks = self._chunk_by_sentences(
            text,
            self.target_sentences,
            self.max_chars,
            self.vector_max_tokens,  # Hard token limit
            self.overlap_sentences,
        )

        # Critical verification
        token_counts = [self._count_tokens(chunk) for chunk in chunks]
        max_tokens = max(token_counts) if token_counts else 0
        oversized = sum(1 for count in token_counts if count > self.vector_max_tokens)

        logger.info(
            f"Chunking complete: {len(chunks)} chunks, "
            f"token range: {min(token_counts) if token_counts else 0}-{max_tokens}, "
            f"limit: {self.vector_max_tokens}, oversized: {oversized}"
        )

        if oversized > 0:
            logger.error(
                f"CRITICAL: {oversized} chunks exceed token limit! This should not happen!"
            )
            for i, count in enumerate(token_counts):
                if count > self.vector_max_tokens:
                    logger.error(
                        f"Oversized chunk {i}: {count} tokens: {chunks[i][:200]}..."
                    )

        return chunks

    def chunk_text_for_graphrag(self, text: str) -> List[str]:
        """
        Chunks text into larger pieces for GraphRAG entity extraction.
        Prioritizes semantic coherence over strict limits.
        Enforces token limits for embedding safety.
        """
        chunks = self._chunk_by_sentences(
            text,
            self.target_sentences,
            self.max_chars,
            self.graphrag_max_tokens,  # Hard token limit
            self.overlap_sentences,
        )

        # Additional validation for GraphRAG chunks
        for chunk in chunks:
            self._check_token_safety(chunk, self.graphrag_max_tokens, "GraphRAG")

        return chunks

    def chunk_documents(self, documents: List[str]) -> List[str]:
        """Chunks a list of documents for Vector RAG."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_text(doc))
        return all_chunks

    def combine_vector_chunks_for_graphrag(
        self, vector_chunks: List[str], combine_count: int = 3
    ) -> List[str]:
        """
        Combines small vector chunks into larger chunks suitable for GraphRAG.
        This is a fallback method when we want to reuse existing vector chunks.
        """
        if not vector_chunks:
            return []

        combined_chunks = []
        for i in range(
            0, len(vector_chunks), max(1, combine_count - 1)
        ):  # Smart overlap
            chunk_group = vector_chunks[i : i + combine_count]
            combined_text = " ".join(chunk_group).strip()

            if len(combined_text) > 200:  # Minimum meaningful size
                combined_chunks.append(combined_text)

        return combined_chunks
