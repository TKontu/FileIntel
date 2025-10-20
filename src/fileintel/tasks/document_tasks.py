"""
Document processing Celery tasks.

Converts document processing workflows to distributed Celery tasks for multicore utilization.
Tasks are designed as pure functions with clear inputs/outputs and proper error handling.
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from celery import group

from fileintel.celery_config import app
from .base import BaseFileIntelTask
from fileintel.core.config import get_config
from fileintel.document_processing.chunking import TextChunker
from fileintel.document_processing.elements import TextElement

logger = logging.getLogger(__name__)

# Corruption detection thresholds (Phase 0: Prevent Data Loss)
CORRUPTION_THRESHOLDS = {
    'max_pdf_artifacts': 20,        # More than 20 (cid:X) indicates corrupt extraction
    'min_stat_table_tokens': 2000,  # Statistical tables are typically large appendix content
    'max_avg_word_length': 15,      # Normal English ~5-6 chars, >15 indicates no word spacing
    'max_table_lines': 500,         # Corrupt tables render one value per line
    'book_index_page_density': 0.3, # Index pages have 30%+ page number references
    'book_index_comma_density': 0.05, # Index pages use commas to separate page refs
    'extremely_large_element': 10000  # >10k tokens almost certainly corrupt
}


def _has_excessive_pdf_artifacts(text: str) -> bool:
    """Check for PDF extraction artifacts like (cid:X) placeholders."""
    return text.count('(cid:') > CORRUPTION_THRESHOLDS['max_pdf_artifacts']


def _is_statistical_reference_table(text: str) -> bool:
    """Detect statistical reference tables in appendices."""
    if not re.search(r'TABLE [IVX]+', text, re.IGNORECASE):
        return False

    stat_terms = ['Critical Values', 'Degrees of Freedom', 'Percentage Points', 'Distribution']
    term_count = sum(1 for term in stat_terms if term in text)

    from fileintel.document_processing.type_aware_chunking import estimate_tokens
    tokens = estimate_tokens(text)

    return term_count >= 2 and tokens > CORRUPTION_THRESHOLDS['min_stat_table_tokens']


def _has_missing_word_boundaries(text: str) -> bool:
    """Detect text extracted without word spacing."""
    words = text.split()
    if len(words) < 10:
        return False

    avg_length = sum(len(w) for w in words) / len(words)
    return avg_length > CORRUPTION_THRESHOLDS['max_avg_word_length']


def _is_corrupt_table_extraction(text: str) -> bool:
    """Detect tables rendered as one number per line."""
    lines = [l for l in text.split('\n') if l.strip()]
    if len(lines) <= CORRUPTION_THRESHOLDS['max_table_lines']:
        return False

    avg_line_length = sum(len(l) for l in lines) / len(lines)
    return avg_line_length < 15


def _is_book_index(text: str) -> bool:
    """Detect book index pages with page number references."""
    if len(text) < 1000:
        return False

    # Count page number references
    page_numbers = len(re.findall(r'\b\d{1,4}\b', text))
    words = text.split()
    if not words:
        return False

    page_density = page_numbers / len(words)
    comma_density = text.count(',') / len(text)

    if page_density <= CORRUPTION_THRESHOLDS['book_index_page_density']:
        return False
    if comma_density <= CORRUPTION_THRESHOLDS['book_index_comma_density']:
        return False

    # Additional check: short average line length
    lines = [l for l in text.split('\n') if l.strip()]
    if not lines:
        return False
    avg_line_len = sum(len(l) for l in lines) / len(lines)
    return avg_line_len < 80


def _should_filter_element(element: TextElement) -> Tuple[bool, Optional[str]]:
    """
    Determine if element should be filtered before chunking.

    Single responsibility: make filtering decision based on all checks.

    Returns: (should_filter, reason)
    """
    if not element or not element.text:
        return (True, 'empty_element')

    # Check 1: MinerU semantic type (trust MinerU if available)
    semantic_type = element.metadata.get('semantic_type') if element.metadata else None
    if semantic_type in ['toc', 'lof', 'lot']:
        return (True, f'semantic_type_{semantic_type}')

    # Check 2: Corruption patterns (early returns for clarity)
    text = element.text

    if _has_excessive_pdf_artifacts(text):
        return (True, 'excessive_pdf_artifacts')

    if _is_statistical_reference_table(text):
        return (True, 'statistical_reference_table')

    if _has_missing_word_boundaries(text):
        return (True, 'no_word_boundaries')

    if _is_corrupt_table_extraction(text):
        return (True, 'corrupt_table_extraction')

    if _is_book_index(text):
        return (True, 'book_index')

    # Check 3: Extremely large (likely corrupt)
    from fileintel.document_processing.type_aware_chunking import estimate_tokens
    if estimate_tokens(text) > CORRUPTION_THRESHOLDS['extremely_large_element']:
        return (True, 'extremely_large_element')

    return (False, None)


def _filter_elements(elements: List[TextElement]) -> Tuple[List[TextElement], List[Dict]]:
    """
    Filter corrupt/non-content elements.

    Pure function - no I/O, easily testable.

    Returns: (clean_elements, filtered_metadata)
    """
    clean = []
    filtered = []

    for idx, element in enumerate(elements):
        try:
            should_filter, reason = _should_filter_element(element)

            if should_filter:
                from fileintel.document_processing.type_aware_chunking import estimate_tokens

                logger.warning(
                    f"Filtering element {idx}: {reason} | "
                    f"{estimate_tokens(element.text)} tokens | "
                    f"preview: {element.text[:80]}..."
                )

                filtered.append({
                    'index': idx,
                    'reason': reason,
                    'token_count': estimate_tokens(element.text),
                    'char_count': len(element.text),
                    'preview': element.text[:500]
                })
            else:
                clean.append(element)

        except Exception as e:
            # HIGH FIX: Add full traceback and mark element as potentially problematic
            import traceback
            logger.error(
                f"Filter error on element {idx}: {e}\n"
                f"Traceback: {traceback.format_exc()}\n"
                f"Element preview: {element.text[:200] if element and element.text else 'No text'}"
            )
            # Fail open - keep element if filtering crashes (prevent data loss)
            # But mark it so we know filtering failed
            if hasattr(element, 'metadata') and element.metadata is not None:
                element.metadata['filtering_error'] = str(e)
            clean.append(element)

    return clean, filtered


def _store_filtering_results(
    document_id: str,
    total_elements: int,
    filtered_metadata: List[Dict],
    storage
) -> None:
    """
    Persist filtering results for transparency.

    Separate function for storage I/O - doesn't affect filtering logic.

    Stores filtered element metadata in document_structures table with
    structure_type='filtered_content'. This provides transparency into
    what content was filtered during Phase 0 corruption detection.
    """
    if not filtered_metadata:
        return

    try:
        storage.store_document_structure(
            document_id=document_id,
            structure_type='filtered_content',
            data={
                'filtered_count': len(filtered_metadata),
                'total_elements': total_elements,
                'items': filtered_metadata[:50]  # Limit storage to first 50 items
            }
        )
    except Exception as e:
        logger.error(f"Failed to store filtering metadata: {e}")
        # Don't fail the whole process if metadata storage fails


def read_document_with_elements(file_path: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any], List[TextElement]]:
    """
    Read document preserving structured elements for filtering and processing.

    Args:
        file_path: Path to the document file

    Returns:
        Tuple of (combined_text, page_mappings, metadata, elements)
        where:
        - combined_text: Combined text from all elements
        - page_mappings: Position and page info for each element
        - metadata: Processor metadata (may include 'document_structure')
        - elements: List of TextElement objects for filtering/processing

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file type is unsupported
    """
    from fileintel.document_processing.processors.traditional_pdf import (
        PDFProcessor as TraditionalPDFProcessor,
        validate_file_for_processing,
    )
    from fileintel.document_processing.processors.mineru_selfhosted import MinerUSelfHostedProcessor
    from fileintel.document_processing.processors.mineru_commercial import MinerUEnhancedProcessor
    from fileintel.document_processing.processors.epub_processor import (
        EPUBReader as EPUBProcessor,
    )
    from fileintel.document_processing.processors.mobi_processor import (
        MOBIReader as MOBIProcessor,
    )

    path = Path(file_path)

    # Basic existence check first
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()

    # Validate file before processing
    validate_file_for_processing(path, extension)

    # Get configuration for processor selection
    config = get_config()

    # Select PDF processor based on configuration
    def get_pdf_processor():
        if config.document_processing.primary_pdf_processor == "mineru":
            # Select MinerU processor based on API type - fail fast on invalid config
            mineru_api_type = config.document_processing.mineru.api_type
            if mineru_api_type == "selfhosted":
                return MinerUSelfHostedProcessor
            elif mineru_api_type == "commercial":
                return MinerUEnhancedProcessor
            else:
                # Fail fast on invalid configuration instead of defaulting
                valid_types = ["selfhosted", "commercial"]
                raise ValueError(
                    f"Invalid MinerU API type: '{mineru_api_type}'. "
                    f"Must be one of: {valid_types}. "
                    f"Check document_processing.mineru.api_type in configuration."
                )
        elif config.document_processing.primary_pdf_processor == "traditional":
            return TraditionalPDFProcessor
        else:
            # Also fail fast on invalid primary processor
            valid_processors = ["mineru", "traditional"]
            raise ValueError(
                f"Invalid primary PDF processor: '{config.document_processing.primary_pdf_processor}'. "
                f"Must be one of: {valid_processors}. "
                f"Check document_processing.primary_pdf_processor in configuration."
            )

    pdf_processor = get_pdf_processor()

    # Direct processor mapping - eliminates complex selection logic
    processors = {
        ".pdf": pdf_processor,
        ".epub": EPUBProcessor,
        ".mobi": MOBIProcessor,
    }

    processor_class = processors.get(extension)
    if not processor_class:
        supported_types = ", ".join(processors.keys())
        raise ValueError(
            f"Unsupported file type: {extension}. Supported types: {supported_types}"
        )

    # Process document and extract text with page mapping
    processor = processor_class()
    elements, metadata = processor.read(path)

    # Build text and page mapping
    text_parts = []
    page_mappings = []
    current_position = 0

    for elem in elements:
        if hasattr(elem, "text") and elem.text:
            text_content = elem.text
            text_parts.append(text_content)

            # Extract page information from element metadata
            elem_metadata = getattr(elem, "metadata", {})
            page_info = {
                "start_pos": current_position,
                "end_pos": current_position + len(text_content),
                "page_number": elem_metadata.get("page_number"),
                "chapter": elem_metadata.get("chapter"),
                "extraction_method": elem_metadata.get("extraction_method"),
                "section_title": elem_metadata.get("section_title"),
                "section_path": elem_metadata.get("section_path"),
                "markdown_headers": elem_metadata.get("markdown_headers")
            }
            page_mappings.append(page_info)
            current_position += len(text_content) + 1  # +1 for space separator

    combined_text = " ".join(text_parts)
    return combined_text, page_mappings, metadata, elements


def read_document_content(file_path: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    BACKWARDS COMPATIBLE: Read document without elements.

    For new code, use read_document_with_elements() to access structured elements.

    Args:
        file_path: Path to the document file

    Returns:
        Tuple of (raw_text_content, page_mappings, document_metadata)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file type is unsupported
    """
    text, mappings, meta, _ = read_document_with_elements(file_path)
    return text, mappings, meta


def clean_and_chunk_text(
    text: str, chunk_size: int = None, overlap: int = None, page_mappings: List[Dict[str, Any]] = None, return_full_result: bool = False
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    """
    Clean and chunk text using sentence-aware chunking for better semantic coherence.

    Args:
        text: Raw text content
        chunk_size: Size of each chunk (ignored - uses sentence-based chunking)
        overlap: Overlap between chunks (ignored - uses sentence-based overlap)
        page_mappings: List of page mapping information for text positions
        return_full_result: If True, returns tuple of (chunks, full_chunking_result)

    Returns:
        List of chunk dictionaries with text and metadata, or tuple with full result if return_full_result=True
    """
    # Clean text first
    text = text.encode("utf-8", "ignore").decode("utf-8")

    # Use adaptive chunker that supports both traditional and two-tier modes
    chunker = TextChunker()
    chunking_result = chunker.chunk_text_adaptive(text, page_mappings)

    # Handle two-tier chunking result
    if chunker.enable_two_tier and 'vector_chunks' in chunking_result:
        # Two-tier mode: use vector chunks for embedding generation
        vector_chunks = chunking_result['vector_chunks']

        # Convert vector chunks to expected format for backward compatibility
        if page_mappings is None:
            return [{"text": chunk['text'], "metadata": {"position": i, "chunk_type": "vector"}}
                   for i, chunk in enumerate(vector_chunks)]

        # Page mappings already integrated in two-tier mode
        chunk_list = []
        for i, chunk in enumerate(vector_chunks):
            page_info = chunk.get('page_info', {})
            metadata = {
                "position": i,
                "chunk_type": "vector",
                "pages": page_info.get('pages', []),
                "page_range": page_info.get('page_range'),
                "token_count": chunk.get('token_count', 0),
                "sentence_count": chunk.get('sentence_count', 0)
            }

            # Add enhanced metadata from page_info
            if page_info.get('extraction_methods'):
                metadata['extraction_methods'] = page_info['extraction_methods']

            if page_info.get('section_title'):
                metadata['section_title'] = page_info['section_title']

            if page_info.get('section_path'):
                metadata['section_path'] = page_info['section_path']

            if page_info.get('markdown_headers'):
                metadata['markdown_headers'] = page_info['markdown_headers']

            chunk_list.append({"text": chunk['text'], "metadata": metadata})

        return chunk_list

    # Traditional mode: extract chunks from result
    text_chunks = chunking_result.get('chunks', [])

    # If no page mappings provided, return simple chunks for backward compatibility
    if page_mappings is None:
        return [{"text": chunk, "metadata": {"position": i}} for i, chunk in enumerate(text_chunks)]

    # Map chunks to their page ranges using accumulative position tracking
    chunk_results = []
    current_text_pos = 0

    for i, chunk_text in enumerate(text_chunks):
        # Find the actual position of this chunk in the cleaned text
        # Start searching from current position to avoid finding duplicate text
        chunk_start = text.find(chunk_text, current_text_pos)
        if chunk_start == -1:
            # Fallback: estimate position based on previous chunks
            chunk_start = current_text_pos

        chunk_end = chunk_start + len(chunk_text)

        # Update current position for next chunk search
        current_text_pos = chunk_end

        # Find overlapping page mappings and collect metadata
        pages_involved = set()
        chapters_involved = set()
        extraction_methods = set()
        section_titles = []
        section_paths = []
        all_headers = []

        for page_info in page_mappings:
            # Check if this page_info overlaps with the chunk
            # Use more generous overlap detection
            page_start = page_info.get("start_pos", 0)
            page_end = page_info.get("end_pos", 0)

            # Check for any overlap between chunk and page ranges
            if not (chunk_end <= page_start or chunk_start >= page_end):
                # Collect basic metadata
                if page_info.get("page_number"):
                    pages_involved.add(page_info["page_number"])

                if page_info.get("chapter"):
                    chapters_involved.add(page_info["chapter"])

                if page_info.get("extraction_method"):
                    extraction_methods.add(page_info["extraction_method"])

                # Collect enhanced metadata
                if page_info.get("section_title"):
                    section_titles.append(page_info["section_title"])

                if page_info.get("section_path"):
                    section_paths.append(page_info["section_path"])

                if page_info.get("markdown_headers"):
                    all_headers.extend(page_info["markdown_headers"])

        # Build chunk metadata
        chunk_metadata = {
            "position": i,
            "char_start": chunk_start,
            "char_end": chunk_end
        }

        if pages_involved:
            chunk_metadata["pages"] = sorted(list(pages_involved))
            chunk_metadata["page_range"] = f"{min(pages_involved)}-{max(pages_involved)}" if len(pages_involved) > 1 else str(list(pages_involved)[0])

        if chapters_involved:
            chunk_metadata["chapters"] = list(chapters_involved)

        if extraction_methods:
            chunk_metadata["extraction_methods"] = list(extraction_methods)

        # Add enhanced metadata
        if section_titles:
            chunk_metadata["section_title"] = section_titles[0]  # Use first/primary

        if section_paths:
            chunk_metadata["section_path"] = section_paths[0]  # Use first/primary

        if all_headers:
            # Deduplicate headers by text
            seen_texts = set()
            unique_headers = []
            for header in all_headers:
                if header['text'] not in seen_texts:
                    seen_texts.add(header['text'])
                    unique_headers.append(header)
            chunk_metadata["markdown_headers"] = unique_headers

        chunk_results.append({
            "text": chunk_text,
            "metadata": chunk_metadata
        })

    # Return full result if requested (includes graph chunks for two-tier processing)
    if return_full_result:
        return chunk_results, chunking_result

    return chunk_results


def get_graph_chunks_for_collection(collection_id: str) -> List[Dict[str, Any]]:
    """Retrieve graph chunks for GraphRAG processing when two-tier chunking is enabled."""
    from fileintel.celery_config import get_shared_storage

    storage = get_shared_storage()
    try:
        # Get graph chunks directly using the new filtering method
        graph_chunks_raw = storage.get_chunks_by_type_for_collection(collection_id, 'graph')

        # Convert to expected format
        graph_chunks = []
        for chunk in graph_chunks_raw:
            graph_chunks.append({
                'id': chunk.id,
                'text': chunk.chunk_text,
                'metadata': chunk.chunk_metadata or {}
            })

        return graph_chunks
    finally:
        storage.close()


@app.task(base=BaseFileIntelTask, bind=True, queue="document_processing")
def validate_chunking_system(self, sample_text: str = None) -> Dict[str, Any]:
    """Validate the two-tier chunking system with comprehensive tests."""
    # Use sample text if none provided
    if not sample_text:
        sample_text = """
        Machine learning is a subset of artificial intelligence that focuses on algorithms
        that can learn and improve through experience. Deep learning is a specialized
        form of machine learning that uses neural networks with multiple layers.
        Natural language processing combines machine learning with linguistics to help
        computers understand human language. These technologies are transforming industries
        from healthcare to finance, enabling new applications and insights.
        """

    try:
        chunker = TextChunker()

        # Validate traditional chunking
        traditional_chunks = chunker.chunk_text(sample_text)
        traditional_metrics = {
            'count': len(traditional_chunks),
            'avg_tokens': sum(chunker._count_tokens(c) for c in traditional_chunks) / len(traditional_chunks) if traditional_chunks else 0
        }

        # Validate two-tier chunking if enabled
        two_tier_validation = None
        if chunker.enable_two_tier:
            two_tier_validation = chunker.validate_two_tier_chunking(sample_text)

        return {
            'success': True,
            'traditional_metrics': traditional_metrics,
            'two_tier_validation': two_tier_validation,
            'system_info': {
                'enable_two_tier': chunker.enable_two_tier,
                'vector_max_tokens': chunker.vector_max_tokens,
                'graphrag_max_tokens': chunker.graphrag_max_tokens,
                'has_bge_tokenizer': chunker.bge_tokenizer is not None
            }
        }

    except Exception as e:
        logger.error(f"Chunking validation failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }


@app.task(
    base=BaseFileIntelTask,
    bind=True,
    queue="document_processing",
    soft_time_limit=None,  # No soft limit - documents can take hours/days
    time_limit=None        # No hard limit - let them run as long as needed
)
def process_document(
    self, file_path: str, document_id: str = None, collection_id: str = None, **kwargs
) -> Dict[str, Any]:
    """
    Process a single document: extract content, clean, and chunk it.

    Args:
        file_path: Path to the document file
        document_id: Unique identifier for the document
        collection_id: Collection to which the document belongs
        **kwargs: Additional processing parameters

    Returns:
        Dict containing document processing results
    """
    self.validate_input(["file_path"], file_path=file_path)

    try:
        # Update progress
        self.update_progress(0, 3, "Reading document content")

        # Read document content with page mappings, metadata, and elements for filtering
        content, page_mappings, doc_metadata, elements = read_document_with_elements(file_path)
        logger.info(f"Extracted {len(content)} characters from {file_path} with {len(page_mappings)} page mappings")

        # Filter corrupt/non-content elements before chunking
        clean_elements, filtered_metadata = _filter_elements(elements)

        # CRITICAL: Validate we have elements remaining after filtering
        if not clean_elements:
            error_msg = (
                f"All {len(elements)} elements filtered as corrupt/non-content. "
                f"Document has no valid content to process. "
                f"Filtered reasons: {[f['reason'] for f in filtered_metadata[:5]]}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if filtered_metadata:
            logger.warning(f"Filtered {len(filtered_metadata)} corrupt/non-content elements from document")
            # Rebuild content from clean elements only
            clean_text_parts = []
            clean_page_mappings = []
            current_position = 0
            for elem in clean_elements:
                if hasattr(elem, "text") and elem.text:
                    text_content = elem.text
                    clean_text_parts.append(text_content)

                    elem_metadata = getattr(elem, "metadata", {})
                    page_info = {
                        "start_pos": current_position,
                        "end_pos": current_position + len(text_content),
                        "page_number": elem_metadata.get("page_number"),
                        "chapter": elem_metadata.get("chapter"),
                        "extraction_method": elem_metadata.get("extraction_method"),
                        "section_title": elem_metadata.get("section_title"),
                        "section_path": elem_metadata.get("section_path"),
                        "markdown_headers": elem_metadata.get("markdown_headers")
                    }
                    clean_page_mappings.append(page_info)
                    current_position += len(text_content) + 1

            content = " ".join(clean_text_parts)
            page_mappings = clean_page_mappings
            logger.info(f"After filtering: {len(content)} characters with {len(clean_page_mappings)} page mappings")

        # Update progress
        self.update_progress(1, 3, "Cleaning and chunking text")

        # Get configuration to determine chunking strategy
        config = get_config()

        # CRITICAL FIX: Use local variable to avoid race condition with concurrent tasks
        use_type_aware = config.document_processing.use_type_aware_chunking

        # Choose chunking strategy based on configuration
        if use_type_aware and clean_elements:
            # HIGH FIX: Add error boundary with fallback to traditional chunking
            try:
                # Phase 1: Type-aware chunking using element metadata
                from fileintel.document_processing.type_aware_chunking import chunk_elements_by_type
                logger.info(f"Using type-aware chunking for {len(clean_elements)} elements")

                chunker = TextChunker()
                chunks_list = chunk_elements_by_type(
                    clean_elements,
                    max_tokens=450,  # BGE embedding limit
                    chunker=chunker
                )
                # Convert to format expected by downstream code
                chunks = [chunk_dict for chunk_dict in chunks_list]
                full_chunking_result = None
                logger.info(f"Type-aware chunking created {len(chunks)} chunks")
            except Exception as e:
                # If type-aware chunking fails, fall back to traditional
                import traceback
                logger.error(
                    f"Type-aware chunking failed, falling back to traditional chunking: {e}\n"
                    f"Traceback: {traceback.format_exc()}"
                )
                # Fall through to traditional path below
                use_type_aware = False  # Only mutate local variable, not shared config

        if not (use_type_aware and clean_elements):
            # Traditional text-based chunking (backwards compatible)
            chunker = TextChunker()
            if chunker.enable_two_tier:
                # Two-tier mode: get both chunks and full result in one call
                chunks, full_chunking_result = clean_and_chunk_text(content, page_mappings=page_mappings, return_full_result=True)
            else:
                # Traditional mode: only need chunks
                chunks = clean_and_chunk_text(content, page_mappings=page_mappings)
                full_chunking_result = None
            logger.info(f"Traditional chunking created {len(chunks)} chunks")

        # Update progress
        self.update_progress(2, 3, "Storing chunks in database")

        # Store chunks in database
        if document_id and collection_id:
            from fileintel.celery_config import get_shared_storage
            import os
            import hashlib

            config = get_config()
            storage = get_shared_storage()
            try:
                # Check if document already exists (from upload-and-process workflow)
                existing_document = None
                if document_id.startswith(f"{collection_id}_doc_"):
                    # This is a generated document_id from workflow task, find existing document by file path
                    documents = storage.get_documents_by_collection(collection_id)
                    filename = os.path.basename(file_path)
                    for doc in documents:
                        # Check if document metadata contains this file path or has same filename
                        doc_metadata = doc.document_metadata or {}
                        if (doc_metadata.get("file_path") == file_path or
                            doc.filename == filename or
                            doc.original_filename == filename):
                            existing_document = doc
                            logger.info(f"Found existing document {doc.id} for {filename}")
                            break
                else:
                    # Try to get document by provided ID
                    existing_document = storage.get_document(document_id)

                if existing_document:
                    # Use existing document
                    actual_document_id = existing_document.id
                    logger.info(f"Using existing document record {actual_document_id}")
                else:
                    # Create new document record only if it doesn't exist
                    try:
                        # Get file information for document creation
                        file_size = os.path.getsize(file_path)
                        filename = os.path.basename(file_path)

                        # Generate content hash
                        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

                        # Determine MIME type based on file extension
                        mime_type = (
                            "application/pdf"
                            if file_path.lower().endswith(".pdf")
                            else "text/plain"
                        )

                        # Create the document record
                        document = storage.create_document(
                            filename=filename,
                            original_filename=filename,
                            content_hash=content_hash,
                            file_size=file_size,
                            mime_type=mime_type,
                            collection_id=collection_id,
                            metadata={"processed_by": "celery_task"},
                        )

                        # Update the document_id to use the one from the created document
                        actual_document_id = document.id
                        logger.info(
                            f"Created new document record {actual_document_id} for {filename}"
                        )

                    except Exception as e:
                        logger.warning(
                            f"Failed to create document record, using provided document_id {document_id}: {e}"
                        )
                        actual_document_id = document_id

                # HIGH FIX: Validate that chunking produced at least one chunk
                if not chunks:
                    error_msg = (
                        f"Chunking produced zero chunks for document {document_id}. "
                        f"Document has {len(clean_elements)} elements after filtering. "
                        f"This indicates a critical chunking failure."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                # Format chunks for storage
                # Chunks now come as dictionaries with text and metadata
                chunk_data = []
                for chunk_dict in chunks:
                    if isinstance(chunk_dict, dict) and "text" in chunk_dict:
                        # Use the metadata from the chunk processing and ensure chunk_type is set
                        metadata = chunk_dict.get("metadata", {})
                        if "chunk_type" not in metadata:
                            metadata["chunk_type"] = "vector"  # Default to vector chunks
                        chunk_data.append({
                            "text": chunk_dict["text"],
                            "metadata": metadata
                        })
                    else:
                        # Backward compatibility for plain text chunks
                        chunk_data.append({
                            "text": str(chunk_dict),
                            "metadata": {"position": len(chunk_data), "chunk_type": "vector"}
                        })

                storage.add_document_chunks(
                    actual_document_id, collection_id, chunk_data
                )

                # Store graph chunks separately if two-tier mode is enabled
                if full_chunking_result and 'graph_chunks' in full_chunking_result:
                    graph_chunk_data = []
                    for graph_chunk in full_chunking_result['graph_chunks']:
                        graph_chunk_data.append({
                            "text": graph_chunk['text'],
                            "metadata": {
                                **graph_chunk.get('page_info', {}),
                                'chunk_type': 'graph',
                                'token_count': graph_chunk['token_count'],
                                'sentence_count': graph_chunk['sentence_count'],
                                'vector_chunk_ids': graph_chunk['vector_chunk_ids'],
                                'deduplication_stats': graph_chunk['deduplication_stats']
                            }
                        })

                    # Store graph chunks with a different method or flag
                    storage.add_document_chunks(
                        actual_document_id, collection_id, graph_chunk_data
                    )
                logger.info(
                    f"Stored {len(chunks)} chunks in database for document {actual_document_id}"
                )

                # Store filtering results for transparency
                if filtered_metadata:
                    _store_filtering_results(
                        document_id=actual_document_id,
                        total_elements=len(elements),
                        filtered_metadata=filtered_metadata,
                        storage=storage
                    )

                # Store document structure if available (Phase 4: Structured Storage)
                if doc_metadata and 'document_structure' in doc_metadata:
                    structure_data = doc_metadata['document_structure']
                    structures_saved = 0

                    for struct_type in ['toc', 'lof', 'lot', 'headers']:
                        if struct_type in structure_data:
                            struct_entries = structure_data[struct_type]

                            # Check if we have actual entries to save
                            if struct_type == 'headers':
                                has_data = struct_entries.get('hierarchy')
                            else:
                                has_data = struct_entries.get('entries')

                            if has_data:
                                try:
                                    storage.store_document_structure(
                                        document_id=actual_document_id,
                                        structure_type=struct_type,
                                        data=struct_entries
                                    )
                                    structures_saved += 1
                                except Exception as struct_err:
                                    logger.error(f"Failed to store {struct_type} structure: {struct_err}")

                    if structures_saved > 0:
                        logger.info(f"Stored {structures_saved} document structures for {actual_document_id}")

            finally:
                storage.close()

        result = {
            "document_id": actual_document_id
            if "actual_document_id" in locals()
            else document_id,
            "collection_id": collection_id,
            "file_path": file_path,
            "content_length": len(content),
            "chunks_count": len(chunks),
            "chunks_stored": len(chunks) if document_id and collection_id else 0,
            "status": "completed",
        }

        self.update_progress(3, 3, "Document processing completed")
        return result

    except Exception as e:
        logger.error(f"Error processing document {file_path}: {e}")
        return {
            "document_id": document_id,
            "collection_id": collection_id,
            "file_path": file_path,
            "error": str(e),
            "status": "failed",
        }


@app.task(
    base=BaseFileIntelTask,
    bind=True,
    queue="document_processing",
    soft_time_limit=None,  # No soft limit - collections can take very long
    time_limit=None        # No hard limit - let them run as long as needed
)
def process_collection(
    self, collection_id: str, file_paths: List[str], **kwargs
) -> Dict[str, Any]:
    """
    Process multiple documents in a collection using parallel Celery tasks.

    Args:
        collection_id: Unique identifier for the collection
        file_paths: List of file paths to process
        **kwargs: Additional processing parameters

    Returns:
        Dict containing collection processing results
    """
    self.validate_input(
        ["collection_id", "file_paths"],
        collection_id=collection_id,
        file_paths=file_paths,
    )

    try:
        # Update collection status to processing
        from fileintel.celery_config import get_shared_storage

        storage = get_shared_storage()
        try:
            storage.update_collection_status(collection_id, "processing")

            self.update_progress(
                0,
                len(file_paths),
                f"Starting batch processing of {len(file_paths)} documents",
            )

            # Create a group of parallel document processing tasks
            job = group(
                process_document.s(
                    file_path=file_path,
                    document_id=f"{collection_id}_{i}",
                    collection_id=collection_id,
                    **kwargs,
                )
                for i, file_path in enumerate(file_paths)
            )

            # Execute the group without blocking
            result = job.apply_async()

            # Return task information instead of blocking for results
            return {
                "collection_id": collection_id,
                "total_files": len(file_paths),
                "processing_task_id": result.id,
                "status": "processing",
                "message": f"Started processing {len(file_paths)} documents",
            }
        finally:
            storage.close()

    except Exception as e:
        logger.error(f"Error processing collection {collection_id}: {e}")

        # Update collection status to failed
        try:
            storage = get_shared_storage()
            storage.update_collection_status(collection_id, "failed")
        except:
            pass  # Don't fail the task if status update fails

        return {"collection_id": collection_id, "error": str(e), "status": "failed"}


@app.task(
    base=BaseFileIntelTask,
    bind=True,
    queue="document_processing",
    soft_time_limit=None,  # No soft limit - metadata extraction can take long
    time_limit=None        # No hard limit - let it run as long as needed
)
def extract_document_metadata(
    self, file_path: str, content_chunks: List[str] = None, **kwargs
) -> Dict[str, Any]:
    """
    Extract structured metadata from document using sophisticated LLM analysis.

    Args:
        file_path: Path to the document file
        content_chunks: Pre-processed content chunks (optional)
        **kwargs: Additional extraction parameters

    Returns:
        Dict containing clean, structured metadata
    """
    self.validate_input(["file_path"], file_path=file_path)

    try:
        self.update_progress(0, 3, "Preparing metadata extraction")

        # Get chunks if not provided
        if content_chunks is None:
            content, page_mappings, _ = read_document_content(file_path)  # Ignore doc_metadata here
            chunk_dicts = clean_and_chunk_text(content, page_mappings=page_mappings)
            # Extract just the text for metadata extraction (preserving backward compatibility)
            content_chunks = [chunk_dict["text"] for chunk_dict in chunk_dicts]

        self.update_progress(1, 3, "Preparing metadata extraction")

        # Create sync LLM provider directly
        from fileintel.llm_integration.unified_provider import UnifiedLLMProvider
        from fileintel.celery_config import get_shared_storage

        config = get_config()
        storage = get_shared_storage()
        try:
            llm_provider = UnifiedLLMProvider(config, storage)
            prompts_dir = Path("prompts/templates")

            self.update_progress(2, 3, "Extracting metadata with MetadataExtractor")

            # Use MetadataExtractor with proper configuration
            from fileintel.document_processing.metadata_extractor import (
                MetadataExtractor,
            )

            extractor = MetadataExtractor(
                llm_provider=llm_provider,
                prompts_dir=prompts_dir,
                max_length=4000,
                max_chunks_for_extraction=3,
            )

            # Extract basic file metadata
            path = Path(file_path)
            file_metadata = {
                "file_name": path.name,
                "file_size": path.stat().st_size if path.exists() else 0,
                "file_path": str(path),
            }

            # Run extraction synchronously (no event loop needed)
            try:
                metadata = extractor.extract_metadata(content_chunks, file_metadata)
            except Exception as e:
                logger.warning(f"MetadataExtractor failed: {e}, using basic metadata")
                metadata = file_metadata
        finally:
            storage.close()

        self.update_progress(3, 3, "Metadata extraction completed")

        return {
            "metadata": metadata,
            "chunks_analyzed": min(len(content_chunks), 3),
            "status": "completed",
        }

    except Exception as e:
        logger.error(f"Error extracting metadata from {file_path}: {e}")
        return {"file_path": file_path, "error": str(e), "status": "failed"}


@app.task(base=BaseFileIntelTask, bind=True, queue="document_processing")
def chunk_existing_document(
    self, document_text: str, chunk_size: int = None, overlap: int = None, **kwargs
) -> Dict[str, Any]:
    """
    Re-chunk an already processed document with new parameters.

    Args:
        document_text: Full text content of the document
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        **kwargs: Additional chunking parameters

    Returns:
        Dict containing re-chunked document data
    """
    self.validate_input(["document_text"], document_text=document_text)

    try:
        self.update_progress(0, 1, "Re-chunking document")

        chunk_dicts = clean_and_chunk_text(document_text, chunk_size, overlap)

        # Extract text and metadata for backward compatibility
        chunks = [chunk_dict["text"] for chunk_dict in chunk_dicts]

        result = {
            "original_length": len(document_text),
            "chunks_count": len(chunks),
            "chunks": chunks,
            "chunk_metadata": [chunk_dict.get("metadata", {}) for chunk_dict in chunk_dicts],
            "chunk_size": chunk_size or get_config().rag.chunking.chunk_size,
            "overlap": overlap or get_config().rag.chunking.chunk_overlap,
            "status": "completed",
        }

        self.update_progress(1, 1, "Re-chunking completed")
        return result

    except Exception as e:
        logger.error(f"Error re-chunking document: {e}")
        return {"error": str(e), "status": "failed"}
