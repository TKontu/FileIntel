"""
Utility for tracing GraphRAG responses back to source document chunks.

Provides functions to traverse GraphRAG's knowledge graph structure
(context → communities → entities → text units) and map back to
FileIntel document chunks with page numbers for auditability.
"""

import os
from typing import Dict, Any, List, Set, Optional
import pandas as pd


def extract_source_chunks(
    graphrag_context: Dict[str, Any],
    workspace_path: str,
    storage,
) -> List[Dict[str, Any]]:
    """
    Trace GraphRAG context to source document chunks.

    Args:
        graphrag_context: The 'context' dict from GraphRAG response
        workspace_path: Path to GraphRAG workspace (parquet files location)
        storage: PostgreSQLStorage instance for chunk lookups

    Returns:
        List of source chunks with document, page_number, text_preview
    """
    # Step 1: Extract text unit IDs from GraphRAG context
    text_unit_ids = get_text_unit_ids_from_context(graphrag_context, workspace_path)

    if not text_unit_ids:
        return []

    # Step 2: Look up FileIntel chunks by text unit IDs
    sources = lookup_chunks(text_unit_ids, workspace_path, storage)

    # Step 3: Deduplicate by document+page
    return deduplicate_sources(sources)


def get_text_unit_ids_from_context(
    graphrag_context: Dict[str, Any], workspace_path: str
) -> Set[str]:
    """
    Extract text unit IDs from GraphRAG context by traversing knowledge graph.

    Traces: Reports → Communities → Entities → Text Unit IDs

    Args:
        graphrag_context: GraphRAG response context containing reports/entities
        workspace_path: Path to GraphRAG parquet files

    Returns:
        Set of text unit IDs (chunk UUIDs)
    """
    text_unit_ids = set()

    # Method 1: From community reports (most common in global search)
    if "reports" in graphrag_context:
        reports_df = graphrag_context["reports"]

        if isinstance(reports_df, pd.DataFrame) and not reports_df.empty:
            unit_ids = _trace_reports_to_text_units(reports_df, workspace_path)
            text_unit_ids.update(unit_ids)

    # Method 2: From entities directly (if provided in context)
    if "entities" in graphrag_context:
        entities_list = graphrag_context["entities"]

        if isinstance(entities_list, list):
            for entity in entities_list:
                unit_ids = entity.get("text_unit_ids", [])
                if isinstance(unit_ids, list):
                    text_unit_ids.update(unit_ids)

    # Method 3: From relationships (if provided)
    if "relationships" in graphrag_context:
        relationships_list = graphrag_context["relationships"]

        if isinstance(relationships_list, list):
            for relationship in relationships_list:
                unit_ids = relationship.get("text_unit_ids", [])
                if isinstance(unit_ids, list):
                    text_unit_ids.update(unit_ids)

    return text_unit_ids


def _trace_reports_to_text_units(
    reports_df: pd.DataFrame, workspace_path: str
) -> Set[str]:
    """
    Trace community reports to text units via communities and entities.

    Args:
        reports_df: DataFrame of community reports from GraphRAG
        workspace_path: Path to GraphRAG parquet files

    Returns:
        Set of text unit IDs
    """
    text_unit_ids = set()

    # Load GraphRAG parquet files
    entities_df = _load_parquet_safe(workspace_path, "entities.parquet")
    communities_df = _load_parquet_safe(workspace_path, "communities.parquet")

    if entities_df is None or communities_df is None:
        return text_unit_ids

    # Get community IDs from reports
    if "community" not in reports_df.columns:
        return text_unit_ids

    community_ids = reports_df["community"].tolist()

    # For each community, get its entities
    for comm_id in community_ids:
        comm_row = communities_df[communities_df["id"] == comm_id]

        if comm_row.empty:
            continue

        entity_ids = comm_row.iloc[0].get("entity_ids", [])

        if not isinstance(entity_ids, list):
            continue

        # For each entity, get its text unit IDs
        for entity_id in entity_ids:
            entity_row = entities_df[entities_df["id"] == entity_id]

            if entity_row.empty:
                continue

            unit_ids = entity_row.iloc[0].get("text_unit_ids", [])

            if isinstance(unit_ids, list):
                text_unit_ids.update(unit_ids)

    return text_unit_ids


def _format_page_numbers(pages: set) -> str:
    """Format page numbers according to citation style.

    Rules:
    - Single page: "p. 45"
    - Consecutive pages: "pp. 16-17"
    - Non-consecutive: "pp. 30, 35"

    Args:
        pages: Set of page numbers

    Returns:
        Formatted page string
    """
    if not pages:
        return None

    # Sort pages
    sorted_pages = sorted(pages)

    if len(sorted_pages) == 1:
        return f"p. {sorted_pages[0]}"

    # Check if pages are consecutive
    ranges = []
    start = sorted_pages[0]
    end = sorted_pages[0]

    for i in range(1, len(sorted_pages)):
        if sorted_pages[i] == end + 1:
            # Consecutive page
            end = sorted_pages[i]
        else:
            # Gap found, save current range
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = sorted_pages[i]
            end = sorted_pages[i]

    # Add final range
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")

    # Format output
    return f"pp. {', '.join(ranges)}"


def _load_parquet_safe(workspace_path: str, filename: str) -> Optional[pd.DataFrame]:
    """
    Safely load a parquet file, returning None if not found.

    Args:
        workspace_path: Base directory path
        filename: Parquet file name

    Returns:
        DataFrame or None if file doesn't exist
    """
    path = os.path.join(workspace_path, filename)

    if not os.path.exists(path):
        return None

    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def lookup_chunks(
    text_unit_ids: Set[str], workspace_path: str, storage
) -> List[Dict[str, Any]]:
    """
    Look up FileIntel chunks by text unit IDs.

    GraphRAG creates SHA512 hashes for text_unit IDs but preserves original
    chunk UUIDs in the document_ids field. We need to map text_unit IDs to
    chunk UUIDs before querying PostgreSQL.

    Args:
        text_unit_ids: Set of GraphRAG text_unit IDs (SHA512 hashes)
        workspace_path: Path to GraphRAG workspace (contains parquet files)
        storage: PostgreSQLStorage instance

    Returns:
        List of dicts with document, page_number, text_preview
    """
    sources = []

    # Load text_units.parquet to map text_unit IDs to chunk UUIDs
    text_units_df = _load_parquet_safe(workspace_path, "text_units.parquet")

    if text_units_df is None:
        return sources

    # Extract chunk UUIDs from text_units
    chunk_uuids = set()
    for unit_id in text_unit_ids:
        # Find text unit by its ID (SHA512 hash)
        text_unit_row = text_units_df[text_units_df["id"] == unit_id]

        if text_unit_row.empty:
            continue

        # Get document_ids field (contains original FileIntel chunk UUIDs)
        doc_ids = text_unit_row.iloc[0].get("document_ids", [])

        if doc_ids is not None and len(doc_ids) > 0:
            chunk_uuids.update(doc_ids)

    # Look up chunks in PostgreSQL using actual UUIDs
    for chunk_uuid in chunk_uuids:
        try:
            chunk = storage.get_chunk_by_id(str(chunk_uuid))

            if chunk:
                # Extract metadata
                page_number = None
                if chunk.chunk_metadata:
                    page_number = chunk.chunk_metadata.get("page_number")

                document_name = "Unknown"
                if chunk.document:
                    document_name = chunk.document.original_filename

                sources.append(
                    {
                        "chunk_id": str(chunk.id),
                        "document": document_name,
                        "document_id": str(chunk.document_id) if chunk.document_id else None,
                        "page_number": page_number,
                        "text_preview": chunk.chunk_text[:200] if chunk.chunk_text else "",
                    }
                )
        except Exception:
            # Skip chunks that can't be loaded
            continue

    return sources


def deduplicate_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate sources by document+page combination.

    Args:
        sources: List of source dicts

    Returns:
        Deduplicated list of sources
    """
    seen = set()
    unique_sources = []

    for source in sources:
        # Create key from document and page
        key = (source.get("document_id"), source.get("page_number"))

        if key not in seen:
            seen.add(key)
            unique_sources.append(source)

    return unique_sources


def trace_citations_to_sources(
    citation_ids: Dict[str, set],
    workspace_path: str,
    api_client
) -> List[Dict[str, Any]]:
    """
    Trace GraphRAG citations to source documents with page numbers.

    Flow:
    1. Report IDs → Community IDs → Entity IDs → Text Unit IDs
    2. Text Unit IDs (SHA512) → Chunk UUIDs (via text_units.parquet)
    3. Chunk UUIDs → Document metadata (via documents.parquet + API)

    Args:
        citation_ids: Dict with 'report_ids', 'entity_ids', 'relationship_ids'
        workspace_path: Path to GraphRAG workspace (parquet files location)
        api_client: API client for chunk metadata queries

    Returns:
        List of source dicts with document, page_number, chunk_count
    """
    import pandas as pd
    import os

    # Phase 2: Trace reports/entities to text_unit IDs
    text_unit_ids = set()

    # From report IDs
    if citation_ids.get("report_ids"):
        text_unit_ids.update(
            _trace_reports_to_text_units(citation_ids["report_ids"], workspace_path)
        )

    # From entity IDs directly
    if citation_ids.get("entity_ids"):
        text_unit_ids.update(
            _trace_entities_to_text_units(citation_ids["entity_ids"], workspace_path)
        )

    if not text_unit_ids:
        return []

    # Phase 3: Map text_unit IDs to chunk UUIDs
    chunk_uuids = _map_text_units_to_chunks(text_unit_ids, workspace_path)

    if not chunk_uuids:
        return []

    # Phase 4: Get source metadata (hybrid approach)
    sources = _get_source_metadata_hybrid(chunk_uuids, workspace_path, api_client)

    return sources


def _trace_reports_to_text_units(report_ids: set, workspace_path: str) -> set:
    """Trace report IDs to text_unit IDs via communities and entities.

    Note: GraphRAG inline citations like [Data: Reports (5)] use community IDs directly,
    not report UUIDs. The numbers in citations ARE the community IDs.
    """
    communities_df = _load_parquet_safe(workspace_path, "communities.parquet")
    entities_df = _load_parquet_safe(workspace_path, "entities.parquet")

    if communities_df is None:
        print(f"DEBUG: communities.parquet not found at {workspace_path}")
        return set()
    if entities_df is None:
        print(f"DEBUG: entities.parquet not found at {workspace_path}")
        return set()

    text_unit_ids = set()

    # The citation numbers ARE community IDs, use them directly
    community_ids = report_ids

    # Step 2: Get entity IDs from communities
    # Note: Look up by 'community' field (integer), not 'id' field (UUID)
    entity_ids = set()
    for community_id in community_ids:
        comm_row = communities_df[communities_df["community"] == community_id]
        if not comm_row.empty:
            ent_ids = comm_row.iloc[0].get("entity_ids", [])
            if ent_ids is not None and len(ent_ids) > 0:
                entity_ids.update(ent_ids)

    # Step 3: Get text_unit IDs from entities
    for entity_id in entity_ids:
        entity_row = entities_df[entities_df["id"] == entity_id]
        if not entity_row.empty:
            tu_ids = entity_row.iloc[0].get("text_unit_ids", [])
            if tu_ids is not None and len(tu_ids) > 0:
                text_unit_ids.update(tu_ids)

    return text_unit_ids


def _trace_entities_to_text_units(entity_ids: set, workspace_path: str) -> set:
    """Trace entity IDs directly to text_unit IDs."""
    entities_df = _load_parquet_safe(workspace_path, "entities.parquet")

    if entities_df is None:
        return set()

    text_unit_ids = set()

    for entity_id in entity_ids:
        entity_row = entities_df[entities_df["id"] == entity_id]
        if not entity_row.empty:
            tu_ids = entity_row.iloc[0].get("text_unit_ids", [])
            if tu_ids is not None and len(tu_ids) > 0:
                text_unit_ids.update(tu_ids)

    return text_unit_ids


def _map_text_units_to_chunks(text_unit_ids: set, workspace_path: str) -> set:
    """Map text_unit IDs (SHA512) to chunk UUIDs."""
    text_units_df = _load_parquet_safe(workspace_path, "text_units.parquet")

    if text_units_df is None:
        return set()

    chunk_uuids = set()

    for unit_id in text_unit_ids:
        tu_row = text_units_df[text_units_df["id"] == unit_id]
        if not tu_row.empty:
            doc_ids = tu_row.iloc[0].get("document_ids", [])
            if doc_ids is not None and len(doc_ids) > 0:
                chunk_uuids.update(doc_ids)

    return chunk_uuids


def _get_source_metadata_hybrid(
    chunk_uuids: set, workspace_path: str, api_client
) -> List[Dict[str, Any]]:
    """
    Get source metadata using hybrid approach:
    1. Group chunks by document (from documents.parquet)
    2. Get page numbers for one chunk per document (from API)
    """
    import os

    docs_df = _load_parquet_safe(workspace_path, "documents.parquet")

    if docs_df is None:
        return []

    # Group chunks by document
    doc_chunks = {}

    for chunk_uuid in chunk_uuids:
        doc_row = docs_df[docs_df["id"] == str(chunk_uuid)]
        if not doc_row.empty:
            doc_title = doc_row.iloc[0].get("title", "Unknown")
            if doc_title not in doc_chunks:
                doc_chunks[doc_title] = []
            doc_chunks[doc_title].append(str(chunk_uuid))

    # Get document metadata and ALL page numbers from API
    sources = []

    for doc_title, chunk_list in doc_chunks.items():
        import requests

        pages = set()
        doc_metadata = None
        document_id = None

        # Query ALL chunks to get all page numbers referenced
        for chunk_uuid in chunk_list:
            try:
                # Direct request without error printing
                url = f"{api_client.base_url_v2}/chunks/{chunk_uuid}"
                response = requests.get(url, timeout=(30, 300))
                if response.status_code == 200:
                    chunk_info = response.json().get("data", response.json())
                    page = chunk_info.get("chunk_metadata", {}).get("page_number")
                    if page is not None:
                        pages.add(page)
                    if not document_id:
                        document_id = chunk_info.get("document_id")
            except Exception:
                # Continue to next chunk
                continue

        # Get full document metadata if we found a valid chunk
        if document_id:
            try:
                url = f"{api_client.base_url_v2}/documents/{document_id}"
                response = requests.get(url, timeout=(30, 300))
                if response.status_code == 200:
                    doc_info = response.json().get("data", response.json())
                    doc_metadata = doc_info.get("metadata", {})
            except Exception:
                pass

        # Format page numbers: single, consecutive, or non-consecutive
        page_str = _format_page_numbers(pages) if pages else None

        sources.append({
            "document": doc_title,
            "page_numbers": page_str,
            "chunk_count": len(chunk_list),
            "metadata": doc_metadata or {},
        })

    # Sort sources by chunk count (descending) - most referenced sources first
    sources.sort(key=lambda x: x["chunk_count"], reverse=True)

    return sources
