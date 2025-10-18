"""
Type-aware chunking strategies for different element types.

Applies different chunking strategies based on semantic type:
- TOC/LOF: Should be filtered out before reaching here
- Tables: Use caption or parse HTML
- Images: Use caption as single chunk
- Headers: Sentence-based chunking with context
- Prose: Standard sentence-based chunking
"""

from typing import List, Dict, Any
import logging

from .elements import TextElement

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """
    Rough token estimation (4 chars per token).

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    return len(text) // 4


def chunk_element_by_type(
    element: TextElement,
    max_tokens: int = 450,
    chunker = None
) -> List[Dict[str, Any]]:
    """
    Chunk a single element based on its semantic type.

    Args:
        element: TextElement to chunk
        max_tokens: Maximum tokens per chunk (default: 450 for embeddings)
        chunker: Optional TextChunker instance for sentence-based chunking

    Returns:
        List of chunk dicts with 'text' and 'metadata' keys
    """
    semantic_type = element.metadata.get('semantic_type', 'prose')
    layout_type = element.metadata.get('layout_type', 'text')

    chunks = []

    # Route to appropriate strategy based on type
    if layout_type == 'table':
        chunks = _chunk_table(element, max_tokens)
    elif layout_type == 'image':
        chunks = _chunk_image_caption(element, max_tokens)
    elif semantic_type in ['header', 'prose']:
        chunks = _chunk_text(element, max_tokens, chunker)
    elif semantic_type in ['toc', 'lof', 'lot']:
        # These should have been filtered out in Phase 2
        logger.warning(f"TOC/LOF element reached chunking (should be filtered): {semantic_type}")
        return []
    else:
        # Fallback: treat as prose
        chunks = _chunk_text(element, max_tokens, chunker)

    return chunks


def _chunk_table(element: TextElement, max_tokens: int) -> List[Dict[str, Any]]:
    """
    Chunk table elements using caption-based strategy.

    For now, uses table caption only. Future: parse table_body HTML.

    Args:
        element: Table element
        max_tokens: Maximum tokens per chunk

    Returns:
        List of chunks (usually just one for caption)
    """
    table_caption = element.metadata.get('table_caption', [])

    # Strategy 1: Use caption if available
    if table_caption:
        caption_text = ' '.join(table_caption)

        if estimate_tokens(caption_text) <= max_tokens:
            return [{
                'text': caption_text,
                'metadata': {
                    **element.metadata,
                    'chunk_strategy': 'table_caption',
                    'is_table': True
                }
            }]

    # Strategy 2: Use element text if caption is missing/too long
    if element.text:
        text_tokens = estimate_tokens(element.text)

        if text_tokens <= max_tokens:
            return [{
                'text': element.text,
                'metadata': {
                    **element.metadata,
                    'chunk_strategy': 'table_text',
                    'is_table': True
                }
            }]
        else:
            # Table text too long: use truncated caption
            logger.warning(f"Table text exceeds max_tokens ({text_tokens} > {max_tokens}), using truncated")
            return [{
                'text': element.text[:max_tokens * 4],  # Rough truncation
                'metadata': {
                    **element.metadata,
                    'chunk_strategy': 'table_truncated',
                    'is_table': True,
                    'truncated': True
                }
            }]

    # No usable text
    logger.warning("Table element has no caption or text")
    return []


def _chunk_image_caption(element: TextElement, max_tokens: int) -> List[Dict[str, Any]]:
    """
    Chunk image elements by using caption as single chunk.

    Args:
        element: Image element
        max_tokens: Maximum tokens per chunk

    Returns:
        List with single chunk (caption)
    """
    if not element.text:
        # No caption text (image_caption was empty)
        return []

    text_tokens = estimate_tokens(element.text)

    if text_tokens <= max_tokens:
        return [{
            'text': element.text,
            'metadata': {
                **element.metadata,
                'chunk_strategy': 'image_caption',
                'is_image_caption': True
            }
        }]
    else:
        # Caption too long (unusual): truncate
        logger.warning(f"Image caption exceeds max_tokens ({text_tokens} > {max_tokens}), truncating")
        return [{
            'text': element.text[:max_tokens * 4],
            'metadata': {
                **element.metadata,
                'chunk_strategy': 'image_caption_truncated',
                'is_image_caption': True,
                'truncated': True
            }
        }]


def _chunk_text(element: TextElement, max_tokens: int, chunker = None) -> List[Dict[str, Any]]:
    """
    Chunk prose and header elements using sentence-based chunking.

    Args:
        element: Text element (prose or header)
        max_tokens: Maximum tokens per chunk
        chunker: Optional TextChunker instance

    Returns:
        List of chunks
    """
    if not element.text:
        return []

    # For now, simple approach: check if fits in one chunk
    text_tokens = estimate_tokens(element.text)

    if text_tokens <= max_tokens:
        # Single chunk
        return [{
            'text': element.text,
            'metadata': {
                **element.metadata,
                'chunk_strategy': 'single_element'
            }
        }]
    else:
        # TODO: Use chunker for sentence-based splitting
        # For now, simple split by estimated size
        logger.debug(f"Element exceeds max_tokens ({text_tokens} > {max_tokens}), would split")

        # Simple split (placeholder until we integrate with TextChunker)
        max_chars = max_tokens * 4
        chunks = []
        text = element.text

        while text:
            chunk_text = text[:max_chars]
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    **element.metadata,
                    'chunk_strategy': 'simple_split'
                }
            })
            text = text[max_chars:]

        return chunks


def chunk_elements_by_type(
    elements: List[TextElement],
    max_tokens: int = 450,
    chunker = None
) -> List[Dict[str, Any]]:
    """
    Chunk multiple elements using type-aware strategies.

    Args:
        elements: List of TextElements to chunk
        max_tokens: Maximum tokens per chunk
        chunker: Optional TextChunker instance

    Returns:
        List of all chunks from all elements
    """
    all_chunks = []

    for element in elements:
        element_chunks = chunk_element_by_type(element, max_tokens, chunker)
        all_chunks.extend(element_chunks)

    logger.info(f"Chunked {len(elements)} elements into {len(all_chunks)} chunks using type-aware strategies")

    return all_chunks
