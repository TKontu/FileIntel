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
import re

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


# Phase 2: Statistical Heuristics for Content Classification

def analyze_text_statistics(text: str) -> Dict[str, float]:
    """
    Extract statistical features from text for classification.

    Uses format-agnostic metrics to identify content patterns without
    relying on hardcoded symbols or formatting.

    Args:
        text: Text to analyze

    Returns:
        Dictionary of statistical features
    """
    if not text or not text.strip():
        return {}

    lines = [l for l in text.split('\n') if l.strip()]
    if not lines:
        return {}

    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        sentences = [text]

    # Line-level statistics
    line_lengths = [len(l) for l in lines]
    avg_line_length = sum(line_lengths) / len(line_lengths)

    # Calculate standard deviation manually (avoid numpy dependency)
    variance = sum((l - avg_line_length) ** 2 for l in line_lengths) / len(line_lengths)
    line_length_std = variance ** 0.5

    # Sentence-level statistics
    sentence_lengths = [len(s) for s in sentences]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)

    return {
        # Line patterns (bullets/lists have short, varied lines)
        'line_count': len(lines),
        'avg_line_length': avg_line_length,
        'line_length_std': line_length_std,
        'short_lines_ratio': sum(1 for l in line_lengths if l < 80) / len(lines),

        # Sentence patterns (citations have very long sentences)
        'sentence_count': len(sentences),
        'avg_sentence_length': avg_sentence_length,
        'long_sentences_ratio': sum(1 for s in sentence_lengths if s > 200) / len(sentences),

        # Quote/citation indicators
        'quote_count': text.count('"') + text.count('"') + text.count('"'),
        'quote_density': (text.count('"') + text.count('"')) / max(len(text), 1),

        # Structure indicators
        'newline_density': text.count('\n') / max(len(text), 1),
        'has_section_numbers': bool(re.search(r'\d+\.\d+\.\d+', text)),
        'bullet_like_lines': sum(1 for l in lines if re.match(r'^\s*[•\-\*\d]+\.?\s', l))
    }


def classify_by_heuristics(text: str, stats: Dict = None) -> str:
    """
    Classify content type using statistical heuristics.

    This provides fallback classification when MinerU metadata is absent.
    Based on analysis of 13 oversized chunks from real data.

    Args:
        text: Text to classify
        stats: Pre-computed statistics (optional, will compute if None)

    Returns:
        Content type: 'bullet_list', 'citation_heavy', 'structured_sections', or 'prose'
    """
    if stats is None:
        stats = analyze_text_statistics(text)

    # Bullet list detection
    # Pattern from analysis: short lines, high variance, many bullet-like starters
    if (stats.get('short_lines_ratio', 0) > 0.6 and
        stats.get('line_length_std', 0) > 50 and
        stats.get('bullet_like_lines', 0) / max(stats.get('line_count', 1), 1) > 0.4):
        return 'bullet_list'

    # Citation-heavy prose detection
    # Pattern from analysis: high quote density, very long sentences
    if (stats.get('quote_density', 0) > 0.008 and
        stats.get('avg_sentence_length', 0) > 150):
        return 'citation_heavy'

    # Structured sections detection
    # Pattern from analysis: section numbering + clear breaks
    if (stats.get('has_section_numbers') and
        stats.get('newline_density', 0) > 0.03):
        return 'structured_sections'

    return 'prose'


def enrich_element_metadata(element: TextElement) -> TextElement:
    """
    Add statistical classification to element metadata if MinerU metadata is absent.

    Priority:
    1. Trust MinerU metadata (layout_type, semantic_type) if present
    2. Add statistical classification as fallback

    Args:
        element: TextElement to enrich

    Returns:
        Element with enriched metadata
    """
    metadata = element.metadata or {}

    # Skip if already has reliable MinerU metadata
    if metadata.get('layout_type') or metadata.get('semantic_type'):
        metadata['classification_source'] = 'mineru'
        element.metadata = metadata
        return element

    # Add statistical classification as fallback
    if element.text and element.text.strip():
        stats = analyze_text_statistics(element.text)
        content_type = classify_by_heuristics(element.text, stats)

        metadata.update({
            'classification_source': 'statistical',
            'heuristic_type': content_type,
            # Store key stats for debugging (only numeric/bool values)
            'stats_summary': {
                'line_count': stats.get('line_count', 0),
                'avg_line_length': round(stats.get('avg_line_length', 0), 1),
                'short_lines_ratio': round(stats.get('short_lines_ratio', 0), 2)
            }
        })
    else:
        # CRITICAL FIX: Always set metadata even for empty elements
        metadata.update({
            'classification_source': 'statistical',
            'heuristic_type': 'prose',  # Default for empty
            'empty_element': True
        })

    # CRITICAL FIX: Always set metadata back to element
    element.metadata = metadata

    return element


def chunk_element_by_type(
    element: TextElement,
    max_tokens: int = 450,
    chunker = None
) -> List[Dict[str, Any]]:
    """
    Chunk a single element based on its semantic type.

    Phase 2: Enriches metadata with statistical classification when MinerU data is absent.

    Args:
        element: TextElement to chunk
        max_tokens: Maximum tokens per chunk (default: 450 for embeddings)
        chunker: Optional TextChunker instance for sentence-based chunking

    Returns:
        List of chunk dicts with 'text' and 'metadata' keys
    """
    # Phase 2: Enrich metadata with statistical classification if needed
    element = enrich_element_metadata(element)

    semantic_type = element.metadata.get('semantic_type', 'prose')
    layout_type = element.metadata.get('layout_type', 'text')
    heuristic_type = element.metadata.get('heuristic_type')

    chunks = []

    # Phase 3: Check heuristic classification first (for elements without MinerU metadata)
    if heuristic_type and element.metadata.get('classification_source') == 'statistical':
        if heuristic_type == 'bullet_list':
            logger.debug(f"Using specialized bullet list chunker for heuristic classification")
            return _chunk_bullet_list(element, max_tokens, chunker)
        elif heuristic_type == 'citation_heavy':
            logger.debug(f"Using specialized citation prose chunker for heuristic classification")
            return _chunk_citation_prose(element, max_tokens, chunker)
        # Note: structured_sections falls through to default handling for now

    # Route to appropriate strategy based on MinerU type
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


# Phase 3: Specialized Chunkers for Different Content Types

def _chunk_bullet_list(element: TextElement, max_tokens: int, chunker) -> List[Dict[str, Any]]:
    """
    Split bullet lists at semantic boundaries (empty lines, headers).

    Strategy:
    - Group bullets by sections
    - Keep nested bullets with parents
    - Split when group exceeds token limit

    Args:
        element: TextElement with bullet list content
        max_tokens: Maximum tokens per chunk
        chunker: Optional TextChunker (unused, for signature compatibility)

    Returns:
        List of chunks, each containing a semantically coherent group of bullets
    """
    text = element.text
    if not text or not text.strip():
        return []

    lines = text.split('\n')

    # Group bullets by semantic boundaries
    groups = []
    current_group = []
    current_tokens = 0

    for line in lines:
        line_stripped = line.strip()

        # Empty line - potential group boundary
        if not line_stripped:
            # Finalize group if it's reasonably full
            if current_group and current_tokens > max_tokens * 0.8:
                groups.append('\n'.join(current_group))
                current_group = []
                current_tokens = 0
            continue

        line_tokens = estimate_tokens(line)

        # Start new group if adding this line significantly exceeds limit
        if current_tokens + line_tokens > max_tokens * 1.1 and current_group:
            groups.append('\n'.join(current_group))
            current_group = [line]
            current_tokens = line_tokens
        else:
            current_group.append(line)
            current_tokens += line_tokens

    # Add final group
    if current_group:
        groups.append('\n'.join(current_group))

    # Convert groups to chunks
    chunks = []
    for i, group_text in enumerate(groups):
        tokens = estimate_tokens(group_text)
        chunks.append({
            'text': group_text,
            'metadata': {
                **element.metadata,
                'chunk_strategy': 'bullet_group_split',
                'content_type': 'bullet_list',
                'group_index': i,
                'token_count': tokens,
                'within_limit': tokens <= max_tokens
            }
        })

    logger.debug(f"Split bullet list into {len(chunks)} groups")
    return chunks


def _chunk_citation_prose(element: TextElement, max_tokens: int, chunker) -> List[Dict[str, Any]]:
    """
    Split citation-heavy text using sentence-based chunking.

    Strategy:
    - Treat quoted passages as atomic units (sentences respect quotes)
    - Use existing sentence-based chunking logic
    - Tag chunks as citation-heavy for monitoring

    Args:
        element: TextElement with citation-heavy content
        max_tokens: Maximum tokens per chunk
        chunker: Optional TextChunker for sentence splitting

    Returns:
        List of chunks with citation_heavy metadata
    """
    # Use existing text chunker (it respects sentence boundaries)
    chunks = _chunk_text(element, max_tokens, chunker)

    # Enrich metadata to indicate citation-heavy content
    for chunk in chunks:
        chunk['metadata']['content_type'] = 'citation_heavy'
        if chunk['metadata'].get('chunk_strategy') != 'single_element':
            chunk['metadata']['chunk_strategy'] = 'citation_aware_sentence'

    logger.debug(f"Chunked citation-heavy prose into {len(chunks)} chunks")
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
    Chunk text using progressive fallback strategy.

    Phase 3 enhancement: Tries increasingly aggressive splits until chunks are under limit.

    Args:
        element: Text element (prose or header)
        max_tokens: Maximum tokens per chunk
        chunker: Optional TextChunker instance (for future sentence-based chunking)

    Returns:
        List of chunks
    """
    if not element.text:
        return []

    text_tokens = estimate_tokens(element.text)

    # If fits in one chunk, no splitting needed
    if text_tokens <= max_tokens:
        return [{
            'text': element.text,
            'metadata': {
                **element.metadata,
                'chunk_strategy': 'single_element',
                'token_count': text_tokens
            }
        }]

    # Progressive fallback splitting strategies
    # Try each delimiter in order, allowing increasing overage
    split_strategies = [
        ('\n\n', 'paragraph', 1.0),    # Paragraph breaks (no overage)
        ('\n', 'line', 1.1),            # Line breaks (10% overage)
        ('. ', 'sentence', 1.15),       # Sentences (15% overage)
        (', ', 'clause', 1.2),          # Clauses (20% overage)
    ]

    for delimiter, strategy_name, max_overage_factor in split_strategies:
        parts = element.text.split(delimiter)
        if len(parts) <= 1:
            continue  # Can't split with this delimiter

        chunks = []
        current_chunk = []
        current_tokens = 0

        for part in parts:
            part_tokens = estimate_tokens(part)

            # Start new chunk if adding this part exceeds limit
            if current_tokens + part_tokens > max_tokens and current_chunk:
                chunk_text = delimiter.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        **element.metadata,
                        'chunk_strategy': f'split_at_{strategy_name}',
                        'token_count': current_tokens
                    }
                })
                current_chunk = [part]
                current_tokens = part_tokens
            else:
                current_chunk.append(part)
                current_tokens += part_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = delimiter.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    **element.metadata,
                    'chunk_strategy': f'split_at_{strategy_name}',
                    'token_count': current_tokens
                }
            })

        # Check if this strategy worked (all chunks within overage allowance)
        max_chunk_tokens = max(c['metadata']['token_count'] for c in chunks)
        if max_chunk_tokens <= max_tokens * max_overage_factor:
            logger.debug(
                f"Split using {strategy_name} boundaries: "
                f"{len(chunks)} chunks, max={max_chunk_tokens} tokens"
            )
            return chunks

    # Last resort: hard truncate at character boundary
    logger.warning(f"No clean split found for {text_tokens} token element, truncating to {max_tokens} tokens")
    return [{
        'text': element.text[:max_tokens * 4],  # Rough char estimate
        'metadata': {
            **element.metadata,
            'chunk_strategy': 'truncated',
            'token_count': max_tokens,
            'truncated': True,
            'original_tokens': text_tokens
        }
    }]


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
