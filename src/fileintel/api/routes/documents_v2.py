"""
Documents API v2 - Document operations and exports.

Provides endpoints for document inspection, chunk exports, and metadata retrieval.
"""

import logging
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel
from datetime import datetime
from io import StringIO

from fileintel.api.dependencies import get_storage, get_api_key
from fileintel.api.error_handlers import api_error_handler, create_success_response
from fileintel.storage.postgresql_storage import PostgreSQLStorage
from fileintel.storage.models import DocumentChunk

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(get_api_key)])


class ChunkMetadata(BaseModel):
    """Chunk metadata response model."""
    position: Optional[int] = None
    chunk_type: Optional[str] = None
    page_number: Optional[int] = None
    token_count: Optional[int] = None
    heuristic_type: Optional[str] = None
    classification_source: Optional[str] = None
    vector_chunk_ids: Optional[List[str]] = None
    sentence_count: Optional[int] = None
    filtering_error: Optional[str] = None
    truncated: Optional[bool] = None


class ChunkResponse(BaseModel):
    """Single chunk response model."""
    id: str
    text: str
    position: int
    metadata: dict


class DocumentChunksResponse(BaseModel):
    """Document chunks list response."""
    document_id: str
    filename: str
    collection_ids: List[str]
    total_chunks: int
    chunks: List[ChunkResponse]


@router.get(
    "/documents/{document_id}/chunks",
    response_model=DocumentChunksResponse,
    summary="Get document chunks as JSON"
)
@api_error_handler("get document chunks")
async def get_document_chunks(
    document_id: str,
    chunk_type: Optional[str] = Query(None, description="Filter by chunk type (vector, graph)"),
    storage: PostgreSQLStorage = Depends(get_storage)
):
    """
    Retrieve all chunks for a document as JSON.

    **Parameters:**
    - **document_id**: Document UUID
    - **chunk_type**: Optional filter (vector, graph, or omit for all)

    **Returns:**
    - Document metadata
    - List of all chunks in order with text and metadata

    **Example:**
    ```
    GET /api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863/chunks
    GET /api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863/chunks?chunk_type=vector
    ```
    """
    import asyncio

    # Log request
    logger.info(f"Document chunks requested (JSON): document_id={document_id} chunk_type={chunk_type}")

    # Get document info (wrap blocking storage call)
    doc = await asyncio.to_thread(storage.get_document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    # Get chunks (wrap blocking query)
    def _get_chunks():
        query = storage.db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id
        )

        # Filter by chunk type if specified
        if chunk_type:
            query = query.filter(
                DocumentChunk.chunk_metadata['chunk_type'].astext == chunk_type
            )

        # Order by position column to maintain document order
        return query.order_by(DocumentChunk.position).all()

    chunks = await asyncio.to_thread(_get_chunks)

    # Format response - already ordered by position from query
    chunk_list = []
    for chunk in chunks:
        metadata = chunk.chunk_metadata or {}

        chunk_list.append(ChunkResponse(
            id=chunk.id,
            text=chunk.chunk_text,
            position=chunk.position,  # Use actual position from database
            metadata=metadata
        ))

    return DocumentChunksResponse(
        document_id=doc.id,
        filename=doc.filename or doc.original_filename or "Unknown",
        collection_ids=[c.id for c in doc.collections],
        total_chunks=len(chunk_list),
        chunks=chunk_list
    )


@router.get(
    "/documents/{document_id}/export",
    response_class=PlainTextResponse,
    summary="Export document chunks as markdown"
)
async def export_document_chunks_markdown(
    document_id: str,
    chunk_type: Optional[str] = Query(None, description="Filter by chunk type (vector, graph)"),
    include_metadata: bool = Query(False, description="Include chunk metadata in export"),
    storage: PostgreSQLStorage = Depends(get_storage)
):
    """
    Export document chunks as a downloadable markdown file.

    **Parameters:**
    - **document_id**: Document UUID
    - **chunk_type**: Optional filter (vector, graph, or omit for all)
    - **include_metadata**: Include detailed chunk metadata (default: false)

    **Returns:**
    - Markdown file with all chunks in order
    - Content-Disposition header for download

    **Example:**
    ```
    GET /api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863/export
    GET /api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863/export?include_metadata=true
    GET /api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863/export?chunk_type=graph
    ```
    """
    import asyncio

    # Log request
    logger.info(f"Document export requested (markdown): document_id={document_id} chunk_type={chunk_type} include_metadata={include_metadata}")

    # Get document info (wrap blocking storage call)
    doc = await asyncio.to_thread(storage.get_document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    # Get chunks (wrap blocking query)
    def _get_chunks():
        query = storage.db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id
        )

        # Filter by chunk type if specified
        if chunk_type:
            query = query.filter(
                DocumentChunk.chunk_metadata['chunk_type'].astext == chunk_type
            )

        # Order by position column to maintain document order
        return query.order_by(DocumentChunk.position).all()

    chunks = await asyncio.to_thread(_get_chunks)

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail=f"No chunks found for document {document_id}" +
                   (f" with chunk_type={chunk_type}" if chunk_type else "")
        )

    # Format chunks - already ordered by position from query
    chunk_list = []
    for chunk in chunks:
        metadata = chunk.chunk_metadata or {}
        chunk_list.append({
            'id': chunk.id,
            'text': chunk.chunk_text,
            'position': chunk.position,  # Use actual position from database
            'metadata': metadata
        })

    # Generate markdown
    output = StringIO()

    # Header
    filename = doc.filename or doc.original_filename or "Unknown"
    output.write(f"# Document Export: {filename}\n\n")
    output.write(f"**Document ID**: `{doc.id}`\n")
    collection_names = [f"{c.name} ({c.id})" for c in doc.collections]
    output.write(f"**Collections**: {', '.join(collection_names) if collection_names else 'None'}\n")
    output.write(f"**Created**: {doc.created_at}\n")
    output.write(f"**Total Chunks**: {len(chunk_list)}\n")
    output.write(f"**Exported**: {datetime.now().isoformat()}\n\n")

    # Document metadata
    if doc.document_metadata:
        output.write("## Document Metadata\n\n")
        for key, value in doc.document_metadata.items():
            output.write(f"- **{key}**: {value}\n")
        output.write("\n")

    output.write("---\n\n")
    output.write("## Document Content\n\n")

    # Chunks
    total_tokens = 0
    for idx, chunk in enumerate(chunk_list, 1):
        output.write(f"### Chunk {idx}\n\n")

        # Metadata if requested
        if include_metadata:
            output.write("<details>\n")
            output.write("<summary>Chunk Metadata</summary>\n\n")

            metadata = chunk['metadata']
            if 'position' in metadata:
                output.write(f"**Position**: {metadata['position']}\n")
            if 'chunk_type' in metadata:
                output.write(f"**Type**: {metadata['chunk_type']}\n")
            if 'page_number' in metadata:
                output.write(f"**Page**: {metadata['page_number']}\n")
            if 'token_count' in metadata:
                output.write(f"**Tokens**: {metadata['token_count']}\n")
                total_tokens += metadata['token_count']
            if 'heuristic_type' in metadata:
                output.write(f"**Content Type**: {metadata['heuristic_type']}\n")
            if 'classification_source' in metadata:
                output.write(f"**Classification**: {metadata['classification_source']}\n")
            if 'vector_chunk_ids' in metadata:
                output.write(f"**Vector Chunks**: {len(metadata['vector_chunk_ids'])} chunks\n")
            if 'sentence_count' in metadata:
                output.write(f"**Sentences**: {metadata['sentence_count']}\n")
            if 'filtering_error' in metadata:
                output.write(f"**⚠ Filtering Error**: {metadata['filtering_error']}\n")
            if metadata.get('truncated'):
                output.write("**⚠ Truncated**: Content was truncated to fit limits\n")

            output.write("\n</details>\n\n")

        # Chunk text
        output.write(chunk['text'])
        output.write("\n\n")
        output.write("---\n\n")

    # Summary
    output.write("## Export Summary\n\n")
    output.write(f"- **Total Chunks**: {len(chunk_list)}\n")
    output.write(f"- **Total Tokens**: {total_tokens:,}\n")
    output.write(f"- **Average Tokens/Chunk**: {total_tokens // len(chunk_list) if chunk_list else 0}\n")

    # Get the markdown content
    markdown_content = output.getvalue()
    output.close()

    # Return as downloadable file
    safe_filename = filename.replace(' ', '_').replace('/', '_')
    headers = {
        'Content-Disposition': f'attachment; filename="{safe_filename}_chunks.md"'
    }

    return PlainTextResponse(
        content=markdown_content,
        headers=headers,
        media_type='text/markdown'
    )


@router.get(
    "/documents/{document_id}",
    summary="Get document information"
)
@api_error_handler("get document info")
async def get_document_info(
    document_id: str,
    storage: PostgreSQLStorage = Depends(get_storage)
):
    """
    Get document metadata and statistics.

    **Parameters:**
    - **document_id**: Document UUID

    **Returns:**
    - Document metadata
    - Chunk count and statistics

    **Example:**
    ```
    GET /api/v2/documents/3b9e6ac7-2152-4133-bd87-2cd0ffc09863
    ```
    """
    import asyncio

    # Log request
    logger.info(f"Document info requested: document_id={document_id}")

    # Wrap blocking storage call
    doc = await asyncio.to_thread(storage.get_document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    # Get chunk count (wrap blocking query)
    def _get_chunk_count():
        return storage.db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id
        ).count()

    chunk_count = await asyncio.to_thread(_get_chunk_count)

    # Get chunk types breakdown (wrap blocking queries)
    def _get_vector_count():
        return storage.db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id,
            DocumentChunk.chunk_metadata['chunk_type'].astext == 'vector'
        ).count()

    vector_count = await asyncio.to_thread(_get_vector_count)

    def _get_graph_count():
        return storage.db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id,
            DocumentChunk.chunk_metadata['chunk_type'].astext == 'graph'
        ).count()

    graph_count = await asyncio.to_thread(_get_graph_count)

    return create_success_response({
        'document_id': doc.id,
        'filename': doc.filename or doc.original_filename,
        'collection_ids': [c.id for c in doc.collections],
        'collections': [{'id': c.id, 'name': c.name} for c in doc.collections],
        'created_at': doc.created_at.isoformat() if doc.created_at else None,
        'metadata': doc.document_metadata or {},
        'statistics': {
            'total_chunks': chunk_count,
            'vector_chunks': vector_count,
            'graph_chunks': graph_count
        }
    })


@router.get("/chunks/{chunk_id}")
@api_error_handler("get chunk by id")
async def get_chunk_by_id(
    chunk_id: str,
    storage: PostgreSQLStorage = Depends(get_storage)
):
    """Get a single chunk by its UUID for GraphRAG source tracing."""
    # Log request
    logger.info(f"Chunk requested: chunk_id={chunk_id}")

    chunk = storage.get_chunk_by_id(chunk_id)

    if not chunk:
        raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")

    return create_success_response({
        'chunk_id': str(chunk.id),
        'document_id': str(chunk.document_id),
        'chunk_text': chunk.chunk_text,
        'chunk_metadata': chunk.chunk_metadata or {},
        'position': chunk.position,
        'has_embedding': chunk.embedding is not None
    })
