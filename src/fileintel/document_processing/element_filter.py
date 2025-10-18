"""
Element filtering for RAG (Retrieval-Augmented Generation).

Filters TextElements based on semantic type to include only content
relevant for embeddings while extracting structural information separately.
"""

from typing import List, Dict, Tuple, Optional
import logging
import re

from .elements import TextElement

logger = logging.getLogger(__name__)


def parse_toc_text(text: str) -> List[Dict]:
    """
    Parse TOC text into structured entries.

    Extracts section numbers, titles, and page numbers from TOC text.

    Args:
        text: Raw TOC text

    Returns:
        List of TOC entries [{"section": "1.1", "title": "...", "page": 5}, ...]
    """
    entries = []
    if not text:
        return entries

    try:
        lines = text.split('\n')
    except Exception as e:
        logger.error(f"Failed to split TOC text into lines: {e}")
        return entries

    for line in lines:
        try:
            line = line.strip()
            if not line:
                continue

            # Pattern 1: "1.2.3 Title ..... 45" (section number + title + dots + page)
            match = re.search(r'^([\d.]+)\s+(.+?)\.{3,}.*?(\d+)\s*$', line)
            if match:
                section, title, page = match.groups()
                entries.append({
                    'section': section.strip(),
                    'title': title.strip(),
                    'page': int(page)
                })
                continue

            # Pattern 2: "Chapter 1: Title ..... 5" (chapter + title + page)
            match = re.search(r'^(Chapter\s+\d+):?\s+(.+?)\.{3,}.*?(\d+)\s*$', line, re.IGNORECASE)
            if match:
                section, title, page = match.groups()
                entries.append({
                    'section': section.strip(),
                    'title': title.strip(),
                    'page': int(page)
                })
                continue

            # Pattern 3: Simple "Title ..... 12"
            match = re.search(r'^(.+?)\.{3,}.*?(\d+)\s*$', line)
            if match:
                title, page = match.groups()
                entries.append({
                    'section': '',
                    'title': title.strip(),
                    'page': int(page)
                })
        except (ValueError, AttributeError) as e:
            # Skip lines with invalid page numbers or other parsing errors
            logger.debug(f"Skipping TOC line due to parse error: {line[:50]}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error parsing TOC line: {e}")
            continue

    return entries


def parse_lof_text(text: str, list_type: str = 'figure') -> List[Dict]:
    """
    Parse List of Figures/Tables text into structured entries.

    Args:
        text: Raw LOF/LOT text
        list_type: 'figure' or 'table'

    Returns:
        List of entries [{"figure": "Figure 1", "title": "...", "page": 8}, ...]
    """
    entries = []
    if not text:
        return entries

    try:
        lines = text.split('\n')
    except Exception as e:
        logger.error(f"Failed to split {list_type} list text into lines: {e}")
        return entries

    # Determine prefix pattern based on type
    if list_type == 'figure':
        prefix_pattern = r'Figure\s+(\d+)'
        entry_key = 'figure'
    else:  # table
        prefix_pattern = r'Table\s+(\d+)'
        entry_key = 'table'

    for line in lines:
        try:
            line = line.strip()
            if not line:
                continue

            # Pattern: "Figure 1: Caption text ..... 8"
            match = re.search(rf'^{prefix_pattern}:?\s+(.+?)\.{{3,}}.*?(\d+)\s*$', line, re.IGNORECASE)
            if match:
                number, title, page = match.groups()
                entries.append({
                    entry_key: f"{list_type.capitalize()} {number}",
                    'title': title.strip(),
                    'page': int(page)
                })
                continue

            # Fallback: Extract any line with figure/table mention and page
            match = re.search(rf'{prefix_pattern}.*?(\d+)\s*$', line, re.IGNORECASE)
            if match:
                number, page = match.groups()
                entries.append({
                    entry_key: f"{list_type.capitalize()} {number}",
                    'title': line[:50].strip(),  # Use beginning of line as title
                    'page': int(page)
                })
        except (ValueError, AttributeError) as e:
            # Skip lines with invalid page numbers or other parsing errors
            logger.debug(f"Skipping {list_type} line due to parse error: {line[:50]}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error parsing {list_type} line: {e}")
            continue

    return entries


def filter_elements_for_rag(
    elements: List[TextElement],
    skip_semantic_types: List[str] = None,
    extract_structure_types: List[str] = None
) -> Tuple[List[TextElement], Dict]:
    """
    Filter elements based on RAG relevance using semantic classification.

    Args:
        elements: List of TextElements to filter
        skip_semantic_types: Semantic types to exclude from embeddings
                            Default: ['toc', 'lof', 'lot']
        extract_structure_types: Types to parse as structure (not embedded)
                                Default: ['toc', 'lof', 'lot']

    Returns:
        (filtered_elements, extracted_structure) tuple where:
        - filtered_elements: Elements to embed (TOC/LOF removed)
        - extracted_structure: Dict with parsed structure data
    """
    # Defaults
    if skip_semantic_types is None:
        skip_semantic_types = ['toc', 'lof', 'lot']
    if extract_structure_types is None:
        extract_structure_types = ['toc', 'lof', 'lot']

    filtered = []
    skipped_count = {}

    # Initialize structure with storage-compatible format
    extracted_structure = {
        'toc': {'entries': []},      # For DocumentStructure table (structure_type='toc')
        'lof': {'entries': []},      # For DocumentStructure table (structure_type='lof')
        'lot': {'entries': []},      # For DocumentStructure table (structure_type='lot')
        'headers': {'hierarchy': []}, # For DocumentStructure table (structure_type='headers')
        # Keep raw elements for backward compatibility
        'toc_elements': [],
        'lof_elements': [],
        'lot_elements': []
    }

    for elem in elements:
        semantic_type = elem.metadata.get('semantic_type', 'prose')
        layout_type = elem.metadata.get('layout_type', 'text')

        # Skip elements based on semantic type
        if semantic_type in skip_semantic_types:
            skipped_count[semantic_type] = skipped_count.get(semantic_type, 0) + 1

            # Extract structure if configured
            if semantic_type in extract_structure_types:
                try:
                    if semantic_type == 'toc':
                        # Parse TOC entries
                        parsed_entries = parse_toc_text(elem.text)
                        extracted_structure['toc']['entries'].extend(parsed_entries)

                        # Keep raw for backward compatibility
                        extracted_structure['toc_elements'].append({
                            'text': elem.text,
                            'page': elem.metadata.get('page_number'),
                            'length': len(elem.text)
                        })

                    elif semantic_type == 'lof':
                        # Parse LOF entries
                        parsed_entries = parse_lof_text(elem.text, list_type='figure')
                        extracted_structure['lof']['entries'].extend(parsed_entries)

                        # Keep raw for backward compatibility
                        extracted_structure['lof_elements'].append({
                            'text': elem.text,
                            'page': elem.metadata.get('page_number'),
                            'length': len(elem.text)
                        })

                    elif semantic_type == 'lot':
                        # Parse LOT entries
                        parsed_entries = parse_lof_text(elem.text, list_type='table')
                        extracted_structure['lot']['entries'].extend(parsed_entries)

                        # Keep raw for backward compatibility
                        extracted_structure['lot_elements'].append({
                            'text': elem.text,
                            'page': elem.metadata.get('page_number'),
                            'length': len(elem.text)
                        })

                except Exception as e:
                    logger.error(f"Failed to parse {semantic_type} structure: {e}")
                    logger.debug(f"Problematic text (first 200 chars): {elem.text[:200]}")
                    # Continue processing - don't let parsing failure crash pipeline

            continue  # Skip this element

        # Extract header information (but still include in filtered list)
        if semantic_type == 'header':
            header_level = elem.metadata.get('header_level', elem.metadata.get('text_level', 1))
            extracted_structure['headers']['hierarchy'].append({
                'text': elem.text,
                'page': elem.metadata.get('page_number'),
                'level': header_level
            })

        # Include element in filtered list
        filtered.append(elem)

    # Log filtering statistics
    if skipped_count:
        skipped_summary = ', '.join([f"{count} {stype}" for stype, count in skipped_count.items()])
        logger.info(f"Filtered out {sum(skipped_count.values())} elements: {skipped_summary}")

    # Log parsed structure counts
    toc_entries = len(extracted_structure['toc']['entries'])
    lof_entries = len(extracted_structure['lof']['entries'])
    lot_entries = len(extracted_structure['lot']['entries'])
    headers = len(extracted_structure['headers']['hierarchy'])

    logger.info(f"Filter results: {len(filtered)} elements to embed, "
                f"{toc_entries} TOC entries, "
                f"{lof_entries} LOF entries, "
                f"{lot_entries} LOT entries, "
                f"{headers} headers extracted")

    return filtered, extracted_structure


def get_filter_statistics(
    original_count: int,
    filtered_count: int,
    structure: Dict
) -> Dict:
    """
    Calculate filtering statistics for monitoring.

    Args:
        original_count: Number of elements before filtering
        filtered_count: Number of elements after filtering
        structure: Extracted structure dict from filter_elements_for_rag

    Returns:
        Dict with statistics
    """
    skipped = original_count - filtered_count

    return {
        'original_count': original_count,
        'filtered_count': filtered_count,
        'skipped_count': skipped,
        'skip_percentage': (skipped / original_count * 100) if original_count > 0 else 0,
        'toc_elements': len(structure.get('toc_elements', [])),
        'lof_elements': len(structure.get('lof_elements', [])),
        'lot_elements': len(structure.get('lot_elements', [])),
        'headers_extracted': len(structure.get('headers', []))
    }
