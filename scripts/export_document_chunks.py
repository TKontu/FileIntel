#!/usr/bin/env python3
"""
Export Document Chunks to Markdown

Helper script to retrieve all chunks from a document and merge them into
a single inspectable markdown file in the correct order.

Usage:
    poetry run python scripts/export_document_chunks.py <document_id> [options]

Examples:
    # Export to default output file
    poetry run python scripts/export_document_chunks.py 3b9e6ac7-2152-4133-bd87-2cd0ffc09863

    # Export to specific file
    poetry run python scripts/export_document_chunks.py 3b9e6ac7-2152-4133-bd87-2cd0ffc09863 -o output.md

    # Include metadata in output
    poetry run python scripts/export_document_chunks.py 3b9e6ac7-2152-4133-bd87-2cd0ffc09863 --include-metadata

    # Export only graph chunks (if two-tier chunking enabled)
    poetry run python scripts/export_document_chunks.py 3b9e6ac7-2152-4133-bd87-2cd0ffc09863 --chunk-type graph

    # Verbose output
    poetry run python scripts/export_document_chunks.py 3b9e6ac7-2152-4133-bd87-2cd0ffc09863 -v
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fileintel.core.config import get_config
from src.fileintel.storage.postgresql_storage import PostgreSQLStorage


def get_document_info(storage: PostgreSQLStorage, document_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve document metadata."""
    try:
        doc = storage.get_document(document_id)
        if not doc:
            return None

        return {
            'id': doc.id,
            'filename': doc.filename or doc.original_filename or 'Unknown',
            'collection_id': doc.collection_id,
            'created_at': doc.created_at,
            'metadata': doc.document_metadata or {}
        }
    except Exception as e:
        print(f"Error retrieving document info: {e}", file=sys.stderr)
        return None


def get_document_chunks(
    storage: PostgreSQLStorage,
    document_id: str,
    chunk_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve all chunks for a document in order.

    Args:
        storage: Storage instance
        document_id: Document UUID
        chunk_type: Filter by chunk type ('vector', 'graph', or None for all)

    Returns:
        List of chunks with text and metadata, ordered by position
    """
    try:
        # Query chunks ordered by position (which should reflect document order)
        from src.fileintel.storage.models import DocumentChunk

        query = storage.db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id
        )

        # Filter by chunk type if specified
        if chunk_type:
            query = query.filter(
                DocumentChunk.chunk_metadata['chunk_type'].astext == chunk_type
            )

        # Order by position column to maintain document order
        chunks = query.order_by(DocumentChunk.position).all()

        # Convert to dict format - already ordered by position from query
        chunk_list = []
        for chunk in chunks:
            metadata = chunk.chunk_metadata or {}

            chunk_list.append({
                'id': chunk.id,
                'text': chunk.chunk_text,
                'position': chunk.position,  # Use actual position from database
                'metadata': metadata
            })

        return chunk_list

    except Exception as e:
        print(f"Error retrieving chunks: {e}", file=sys.stderr)
        return []


def format_chunk_metadata(metadata: Dict[str, Any]) -> str:
    """Format chunk metadata as markdown."""
    lines = []

    # Key metadata fields
    if 'position' in metadata:
        lines.append(f"**Position**: {metadata['position']}")

    if 'chunk_type' in metadata:
        lines.append(f"**Type**: {metadata['chunk_type']}")

    if 'page_number' in metadata:
        lines.append(f"**Page**: {metadata['page_number']}")

    if 'token_count' in metadata:
        lines.append(f"**Tokens**: {metadata['token_count']}")

    # Type-aware chunking metadata
    if 'heuristic_type' in metadata:
        lines.append(f"**Content Type**: {metadata['heuristic_type']}")

    if 'classification_source' in metadata:
        lines.append(f"**Classification**: {metadata['classification_source']}")

    # Two-tier chunking metadata
    if 'vector_chunk_ids' in metadata:
        chunk_ids = metadata['vector_chunk_ids']
        lines.append(f"**Vector Chunks**: {len(chunk_ids)} chunks")

    if 'sentence_count' in metadata:
        lines.append(f"**Sentences**: {metadata['sentence_count']}")

    # Filtering metadata
    if 'filtering_error' in metadata:
        lines.append(f"**⚠ Filtering Error**: {metadata['filtering_error']}")

    if 'truncated' in metadata and metadata['truncated']:
        lines.append(f"**⚠ Truncated**: Content was truncated to fit limits")

    return "\n".join(lines)


def export_to_markdown(
    chunks: List[Dict[str, Any]],
    document_info: Dict[str, Any],
    output_path: Path,
    include_metadata: bool = False,
    verbose: bool = False
) -> None:
    """
    Export chunks to a markdown file.

    Args:
        chunks: List of chunk dictionaries
        document_info: Document metadata
        output_path: Path to output markdown file
        include_metadata: Whether to include chunk metadata
        verbose: Enable verbose output
    """
    if verbose:
        print(f"Exporting {len(chunks)} chunks to {output_path}...")

    with open(output_path, 'w', encoding='utf-8') as f:
        # Document header
        f.write(f"# Document Export: {document_info['filename']}\n\n")
        f.write(f"**Document ID**: `{document_info['id']}`\n")
        f.write(f"**Collection ID**: `{document_info['collection_id']}`\n")
        f.write(f"**Created**: {document_info['created_at']}\n")
        f.write(f"**Total Chunks**: {len(chunks)}\n")
        f.write(f"**Exported**: {datetime.now().isoformat()}\n\n")

        # Document metadata if available
        if document_info['metadata']:
            f.write("## Document Metadata\n\n")
            for key, value in document_info['metadata'].items():
                f.write(f"- **{key}**: {value}\n")
            f.write("\n")

        f.write("---\n\n")

        # Chunks
        f.write("## Document Content\n\n")

        total_tokens = 0
        for idx, chunk in enumerate(chunks, 1):
            # Chunk header
            f.write(f"### Chunk {idx}\n\n")

            # Metadata if requested
            if include_metadata:
                f.write("<details>\n")
                f.write("<summary>Chunk Metadata</summary>\n\n")
                f.write(format_chunk_metadata(chunk['metadata']))
                f.write("\n\n</details>\n\n")

            # Chunk text
            f.write(chunk['text'])
            f.write("\n\n")
            f.write("---\n\n")

            # Track tokens
            token_count = chunk['metadata'].get('token_count', 0)
            total_tokens += token_count

            if verbose and idx % 100 == 0:
                print(f"  Processed {idx}/{len(chunks)} chunks...")

        # Summary
        f.write("## Export Summary\n\n")
        f.write(f"- **Total Chunks**: {len(chunks)}\n")
        f.write(f"- **Total Tokens**: {total_tokens:,}\n")
        f.write(f"- **Average Tokens/Chunk**: {total_tokens // len(chunks) if chunks else 0}\n")

    if verbose:
        print(f"✓ Export complete: {output_path}")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  File size: {output_path.stat().st_size:,} bytes")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export document chunks to a readable markdown file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        'document_id',
        help='Document UUID to export'
    )

    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output markdown file path (default: <document_id>.md)'
    )

    parser.add_argument(
        '-t', '--chunk-type',
        choices=['vector', 'graph', 'all'],
        default='all',
        help='Chunk type to export (default: all)'
    )

    parser.add_argument(
        '-m', '--include-metadata',
        action='store_true',
        help='Include chunk metadata in output'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = Path(f"{args.document_id}.md")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(f"Connecting to database...")

    # Initialize storage
    config = get_config()
    storage = PostgreSQLStorage(config)

    try:
        # Get document info
        if args.verbose:
            print(f"Retrieving document info for {args.document_id}...")

        doc_info = get_document_info(storage, args.document_id)
        if not doc_info:
            print(f"Error: Document {args.document_id} not found", file=sys.stderr)
            return 1

        if args.verbose:
            print(f"  Document: {doc_info['filename']}")
            print(f"  Collection: {doc_info['collection_id']}")

        # Get chunks
        if args.verbose:
            print(f"Retrieving chunks...")

        chunk_type_filter = None if args.chunk_type == 'all' else args.chunk_type
        chunks = get_document_chunks(storage, args.document_id, chunk_type_filter)

        if not chunks:
            print(f"Warning: No chunks found for document {args.document_id}", file=sys.stderr)
            print(f"  Chunk type filter: {args.chunk_type}", file=sys.stderr)
            return 1

        if args.verbose:
            print(f"  Found {len(chunks)} chunks")

        # Export to markdown
        export_to_markdown(
            chunks,
            doc_info,
            output_path,
            include_metadata=args.include_metadata,
            verbose=args.verbose
        )

        print(f"✓ Exported to: {output_path}")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 1

    finally:
        storage.close()


if __name__ == '__main__':
    sys.exit(main())
