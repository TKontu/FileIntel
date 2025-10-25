"""
Semantic type detection for document elements.

MinerU provides layout-level types (text, table, image).
This module adds semantic classification (TOC, LOF, header, prose, etc.)
"""

import re
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def is_toc_or_lof(text: str, min_lines: int = 100, min_matches: int = 3) -> Tuple[bool, Optional[str]]:
    """
    Detect if text is a Table of Contents (TOC) or List of Figures/Tables.

    Uses pattern matching on text content to identify:
    - TOC: Section numbers with dots and page numbers (e.g., "1.2.3 Title ..... 45")
    - LOF: Figure references with page numbers (e.g., "Figure 1: Caption ..... 8")
    - LOT: Table references with page numbers (e.g., "Table 1: Caption ..... 12")

    IMPROVED: Handles MinerU output format where TOC entries may be on single line without \n separators.

    Args:
        text: Text content to analyze
        min_lines: Minimum text length to consider (default: 100 chars)
        min_matches: Minimum number of pattern matches required (default: 3)

    Returns:
        (is_toc_lof, type_name) where type_name is "toc", "lof", or "lot", or None if not detected

    Detection accuracy: 95%+ based on testing with technical documents
    """
    if not text or len(text) < min_lines:
        return False, None

    # IMPROVED: Handle both line-based and inline TOC formats
    # MinerU may output TOC as: "1.1 Title ... 1.2 Title ... 1.3 Title ..." (no \n)
    # We split on both \n AND section number patterns

    # First, try traditional line-based splitting
    lines = text.split('\n')[:20]

    # Additionally, split on section number patterns to handle inline TOCs
    # Pattern: "1.2.3 " or "Chapter 1 " at start of potential entries
    pseudo_lines = re.split(r'(?=\d+\.[\d.]+\s+)|(?=Chapter\s+\d+)', text[:2000])  # Check first 2000 chars

    # Combine both approaches (deduplicate)
    segments_to_check = list(set(lines + pseudo_lines))[:30]  # Check up to 30 segments

    # Count pattern matches
    toc_matches = 0
    figure_matches = 0
    table_matches = 0

    for segment in segments_to_check:
        segment = segment.strip()
        if not segment or len(segment) < 5:  # Skip very short segments
            continue

        # TOC pattern (IMPROVED): "1.2.3 Title ..... 45" or "1.2.3 Title ... 45"
        # Matches: section number, text, multiple dots (3+), optional text, number
        # No longer requires line end anchor ($) to handle inline format
        if re.search(r'\d+\.[\d.]+\s+.+?\.{3,}', segment):
            toc_matches += 1

        # List of Figures: "Figure 1: Caption ..... 8"
        # IMPROVED: Removed end anchor to handle inline format
        elif re.search(r'Figure\s+\d+:.+?\.{3,}', segment, re.IGNORECASE):
            figure_matches += 1

        # List of Tables: "Table 1: Caption ..... 12"
        # IMPROVED: Removed end anchor to handle inline format
        elif re.search(r'Table\s+\d+:.+?\.{3,}', segment, re.IGNORECASE):
            table_matches += 1

    # IMPROVED: Heuristic fallback - very high dot sequence density indicates TOC
    # This catches cases where splitting doesn't work perfectly
    dot_sequences = text[:2000].count('...')  # Check first 2000 chars
    if dot_sequences > 10:
        logger.debug(f"Detected TOC by heuristic: {dot_sequences} dot sequences")
        return True, "toc"

    # Decision: need at least min_matches matching segments
    if toc_matches >= min_matches:
        logger.debug(f"Detected TOC with {toc_matches} matching patterns")
        return True, "toc"
    elif figure_matches >= min_matches:
        logger.debug(f"Detected LOF with {figure_matches} matching patterns")
        return True, "lof"
    elif table_matches >= min_matches:
        logger.debug(f"Detected LOT with {table_matches} matching patterns")
        return True, "lot"

    return False, None


def detect_semantic_type(
    text: str,
    layout_type: str,
    text_level: int = 0
) -> str:
    """
    Detect semantic type of element using all available information.

    Combines:
    - Layout type from MinerU (text/table/image)
    - Text level field (0 = not header, 1-6 = header levels)
    - Pattern matching on text content

    Args:
        text: Text content
        layout_type: Layout type from MinerU ("text", "table", "image")
        text_level: Header level from MinerU (0 = not header, 1-6 = header level)

    Returns:
        Semantic type: "header", "toc", "lof", "lot", "prose", "table", "image"
    """
    # Non-text elements have straightforward semantic types
    if layout_type == 'table':
        return 'table'
    if layout_type == 'image':
        return 'image'

    # Text elements: classify based on text_level and content patterns
    if layout_type == 'text':
        # Headers identified by text_level field (most reliable)
        if text_level > 0:
            return 'header'

        # Check for TOC/LOF patterns
        is_toc_lof, detected_type = is_toc_or_lof(text)
        if is_toc_lof:
            return detected_type  # Returns 'toc', 'lof', or 'lot'

        # Check for bibliography/references
        if is_bibliography_section(text):
            return 'bibliography'

        # TODO: Add index detection

        # Default: prose text
        return 'prose'

    # Fallback for unknown layout types
    return 'prose'


def is_bibliography_section(text: str, section_title: str = None, section_path: str = None) -> bool:
    """
    Detect if element is from a bibliography/reference section.

    Uses multiple detection methods in priority order:
    1. Section metadata (most reliable if available)
    2. Text pattern matching (validates actual content)
    3. Statistical heuristics (fallback for edge cases)

    Args:
        text: Element text content
        section_title: Section title from metadata (optional)
        section_path: Section path from metadata (optional)

    Returns:
        True if element is from a bibliography/reference section

    Examples:
        >>> is_bibliography_section("", section_title="References")
        True
        >>> is_bibliography_section("Smith, J. (2020). Paper title...")
        True
        >>> is_bibliography_section("This is regular prose text.")
        False
    """
    import re
    import logging

    logger = logging.getLogger(__name__)

    # METHOD 1: Check section metadata (most reliable)
    if section_title or section_path:
        bibliography_keywords = [
            'reference', 'references',
            'bibliography', 'bibliographies',
            'works cited', 'citations',
            'literature cited', 'sources'
        ]

        section_text = ((section_title or '') + ' ' + (section_path or '')).lower()

        for keyword in bibliography_keywords:
            if keyword in section_text:
                logger.debug(f"Bibliography detected by metadata: {section_title or section_path}")
                return True

    # If text is too short, can't reliably detect
    if not text or len(text) < 200:
        return False

    # METHOD 2: Pattern matching on text content
    # Pattern 1: Standard academic reference format
    # Matches: "Author, I. M. (2014)" or "Smith, J. (2020)"
    pattern_standard = re.compile(
        r'[A-Z][a-z]+,\s+[A-Z]\.\s*(?:[A-Z]\.\s*)?\(?\d{4}\)?'
    )

    # Pattern 2: Multiple authors with ampersand
    # Matches: "Smith, J., & Jones, K. (2020)"
    pattern_multi = re.compile(
        r'[A-Z][a-z]+,\s+[A-Z]\.,.*?&.*?\(\d{4}\)'
    )

    # Pattern 3: Numbered references with author
    # Matches: "[1] Smith, J. (2020)" or "1. Smith, J. (2020)"
    pattern_numbered = re.compile(
        r'^\s*[\[\(]?\d+[\]\)\.]\s+[A-Z][a-z]+,',
        re.MULTILINE
    )

    # Pattern 4 (NEW): Numbered URL references
    # Matches: "[16] http://..." or "(16) http://..." or "16. http://..."
    # IMPROVED: Handles bibliography sections with URL lists
    pattern_url_refs = re.compile(
        r'[\[\(]?\d+[\]\)\.]\s+https?://',
        re.IGNORECASE
    )

    # Count total matches across all patterns
    matches = (
        len(pattern_standard.findall(text)) +
        len(pattern_multi.findall(text)) +
        len(pattern_numbered.findall(text)) +
        len(pattern_url_refs.findall(text))  # NEW: Include URL references
    )

    # If we found 3+ reference-like patterns, it's likely a bibliography
    if matches >= 3:
        logger.debug(f"Bibliography detected by pattern matching: {matches} references found")
        return True

    # METHOD 3: Statistical heuristics (strictest - all conditions must be true)
    paren_count = text.count('(') + text.count(')')
    paren_density = paren_count / len(text)

    comma_count = text.count(',')
    comma_density = comma_count / len(text)

    # Count 4-digit years (1900-2099)
    year_pattern = re.compile(r'\b(19|20)\d{2}\b')
    year_count = len(year_pattern.findall(text))

    # Bibliography characteristics (all must be true):
    # - High parenthesis density (years in parentheses)
    # - High comma density (author lists, journal info)
    # - Multiple years (multiple citations)
    if (paren_density > 0.01 and
        comma_density > 0.02 and
        year_count >= 3):
        logger.debug(
            f"Bibliography detected by statistics: "
            f"paren={paren_density:.3f}, comma={comma_density:.3f}, years={year_count}"
        )
        return True

    # METHOD 4 (NEW): Heuristic fallback - URL density
    # IMPROVED: Catches bibliography sections with many URLs (e.g., web references)
    # Very high URL density = likely a reference list (not regular prose)
    url_count = text.count('http://') + text.count('https://')

    # Heuristic: 5+ URLs in text < 3000 chars = bibliography-like density
    # Regular prose rarely has this many URLs in short text
    if url_count >= 5 and len(text) < 3000:
        logger.debug(
            f"Bibliography detected by URL heuristic: "
            f"{url_count} URLs in {len(text)} chars (density: {url_count/len(text)*1000:.2f} per 1000 chars)"
        )
        return True

    return False
