"""
Formatter utilities for CLI output (CSV, Markdown, JSON).

Provides functions to format collection metadata exports in various formats.
"""

import csv
import json
from io import StringIO
from typing import Any, Dict, List, Optional


def format_as_csv(data: Dict[str, Any], selected_fields: Optional[List[str]] = None) -> str:
    """
    Format collection metadata as CSV with pipe delimiter.

    Uses pipe (|) as the column delimiter instead of comma, since commas
    are very common in metadata content (author lists, titles, etc.).

    Supported field types:
    - Strings: exported as-is
    - Numbers: converted to strings (e.g., 2023 → "2023")
    - Lists: joined with semicolons (e.g., ["a", "b"] → "a; b")
    - Dicts: EXCLUDED (don't round-trip properly through CSV)

    Args:
        data: Export data from API containing documents with metadata
        selected_fields: Optional list of metadata fields to include (includes all if None)

    Returns:
        Pipe-delimited CSV string with only simple metadata types
    """
    documents = data.get("documents", [])
    if not documents:
        return ""

    # Standard columns
    standard_cols = ["filename", "document_id", "has_extracted_metadata"]

    # Determine metadata columns
    if selected_fields:
        metadata_cols = selected_fields
    else:
        # Get all unique metadata keys across all documents
        all_keys = set()
        for doc in documents:
            metadata = doc.get("metadata", {})
            # Only include simple types (strings, numbers, lists)
            # Exclude dicts as they don't round-trip properly
            for key, value in metadata.items():
                if not isinstance(value, dict):
                    all_keys.add(key)
        # Filter out internal keys and sort
        metadata_cols = sorted([k for k in all_keys if not k.startswith("_")])

    # Create CSV with pipe delimiter (commas are too common in metadata content)
    output = StringIO()
    writer = csv.writer(output, delimiter='|')

    # Write header
    writer.writerow(standard_cols + metadata_cols)

    # Write data rows
    for doc in documents:
        metadata = doc.get("metadata", {})
        row = [
            doc.get("filename", ""),
            doc.get("document_id", ""),
            "Yes" if doc.get("has_extracted_metadata", False) else "No"
        ]

        # Add metadata fields
        for field in metadata_cols:
            value = metadata.get(field, "")
            row.append(_serialize_value_for_csv(value))

        writer.writerow(row)

    return output.getvalue()


def format_as_markdown(data: Dict[str, Any], selected_fields: Optional[List[str]] = None) -> str:
    """
    Format collection metadata as Markdown table.

    Args:
        data: Export data from API containing documents with metadata
        selected_fields: Optional list of metadata fields to include (includes all if None)

    Returns:
        Markdown table-formatted string
    """
    documents = data.get("documents", [])
    if not documents:
        return "No documents found."

    # Standard column headers
    standard_headers = ["Filename", "Document ID", "Extracted"]

    # Determine metadata columns
    if selected_fields:
        metadata_keys = selected_fields
        metadata_headers = [_format_header(field) for field in selected_fields]
    else:
        # Get all unique metadata keys across all documents
        all_keys = set()
        for doc in documents:
            metadata = doc.get("metadata", {})
            # Only include simple types (strings, numbers, lists)
            # Exclude dicts as they don't display/round-trip well
            for key, value in metadata.items():
                if not isinstance(value, dict):
                    all_keys.add(key)
        # Filter out internal keys and sort
        metadata_keys = sorted([k for k in all_keys if not k.startswith("_")])
        metadata_headers = [_format_header(key) for key in metadata_keys]

    # Build table
    lines = []
    all_headers = standard_headers + metadata_headers

    # Header row
    lines.append("| " + " | ".join(all_headers) + " |")

    # Separator row
    lines.append("| " + " | ".join(["---"] * len(all_headers)) + " |")

    # Data rows
    for doc in documents:
        metadata = doc.get("metadata", {})
        cells = [
            _truncate(doc.get("filename", ""), 30),
            doc.get("document_id", "")[:8] + "...",
            "✓" if doc.get("has_extracted_metadata", False) else "✗"
        ]

        # Add metadata fields
        for key in metadata_keys:
            value = metadata.get(key, "")
            formatted_value = _format_markdown_value(value)
            cells.append(_truncate(formatted_value, 40))

        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def format_as_json(data: Dict[str, Any], selected_fields: Optional[List[str]] = None) -> str:
    """
    Format collection metadata as JSON.

    Args:
        data: Export data from API containing documents with metadata
        selected_fields: Optional list of metadata fields to include (includes all if None)

    Returns:
        JSON-formatted string
    """
    if selected_fields:
        # Filter metadata to selected fields only
        filtered_data = data.copy()
        filtered_docs = []

        for doc in data.get("documents", []):
            metadata = doc.get("metadata", {})
            filtered_metadata = {
                k: v for k, v in metadata.items()
                if k in selected_fields
            }
            filtered_doc = doc.copy()
            filtered_doc["metadata"] = filtered_metadata
            filtered_docs.append(filtered_doc)

        filtered_data["documents"] = filtered_docs
        return json.dumps(filtered_data, indent=2, default=str)
    else:
        return json.dumps(data, indent=2, default=str)


# Helper functions

def _serialize_value_for_csv(value: Any) -> str:
    """
    Serialize a value for CSV output.

    Handles lists and simple types. Dicts are excluded from export
    as they don't round-trip properly through CSV.
    """
    if value is None:
        return ""
    elif isinstance(value, dict):
        # Skip dicts - they don't round-trip properly
        return ""
    elif isinstance(value, list):
        # Join list items with semicolon
        return "; ".join(str(v) for v in value)
    else:
        return str(value)


def _format_markdown_value(value: Any) -> str:
    """
    Format a value for Markdown table cell.

    Handles lists, dicts, and other complex types with truncation.
    """
    if value is None or value == "":
        return ""
    elif isinstance(value, list):
        if len(value) == 0:
            return ""
        elif len(value) <= 3:
            return ", ".join(str(v) for v in value)
        else:
            # Show first 3 items with ellipsis
            return ", ".join(str(v) for v in value[:3]) + ", ..."
    elif isinstance(value, dict):
        return "{...}"
    else:
        return str(value)


def _format_header(field_name: str) -> str:
    """
    Format a metadata field name as a table header.

    Converts snake_case to Title Case.
    """
    return field_name.replace("_", " ").title()


def _truncate(text: str, max_len: int) -> str:
    """
    Truncate text to maximum length with ellipsis.
    """
    text = str(text)
    if len(text) <= max_len:
        return text
    else:
        return text[:max_len - 3] + "..."


def parse_csv_import(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse a pipe-delimited CSV file for metadata import.

    Expected format:
    - First row: headers (must include 'document_id')
    - Subsequent rows: metadata values
    - Delimiter: | (pipe)
    - Standard columns: filename, document_id, has_extracted_metadata (ignored)
    - Other columns: treated as metadata fields

    Args:
        file_path: Path to the CSV file

    Returns:
        List of document updates with document_id and metadata dict

    Raises:
        ValueError: If file format is invalid or document_id column is missing
    """
    import csv

    updates = []
    standard_cols = {"filename", "document_id", "has_extracted_metadata", "created_at", "updated_at"}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter='|')

            # Validate headers
            if not reader.fieldnames:
                raise ValueError("CSV file is empty or has no headers")

            if "document_id" not in reader.fieldnames:
                raise ValueError("CSV file must have a 'document_id' column")

            # Determine metadata columns (non-standard columns)
            metadata_cols = [col for col in reader.fieldnames if col not in standard_cols]

            # Parse rows
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                document_id = row.get("document_id", "").strip()

                if not document_id:
                    raise ValueError(f"Row {row_num}: document_id is required")

                # Build metadata dict from non-standard columns
                metadata = {}
                for col in metadata_cols:
                    value = row.get(col, "").strip()

                    if value:
                        # Parse value (handle lists separated by semicolons)
                        metadata[col] = _parse_csv_value(value)

                updates.append({
                    "document_id": document_id,
                    "metadata": metadata
                })

        return updates

    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file: {e}")


def _parse_csv_value(value: str) -> Any:
    """
    Parse a CSV cell value, handling lists and other types.

    Lists are indicated by semicolons (;).
    """
    if not value:
        return ""

    # Check if it's a list (contains semicolons)
    if ";" in value:
        # Split by semicolon and clean up
        items = [item.strip() for item in value.split(";")]
        return [item for item in items if item]  # Remove empty items
    else:
        return value
