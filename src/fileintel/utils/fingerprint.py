"""
Content-based fingerprinting utilities for deterministic file identification.

Generates UUID v5 fingerprints from file content, enabling:
- Idempotent processing (same file → same ID)
- Global deduplication across collections
- MinerU output caching and reuse
- Storage optimization
"""

import uuid
import hashlib
from pathlib import Path
from typing import Union


# FileIntel namespace UUID (generated once, persists forever)
# This UUID was generated specifically for FileIntel and should never change
FILEINTEL_NAMESPACE = uuid.UUID('f11e1a1e-1111-5555-8888-1dea11111111')


def generate_content_fingerprint(content: Union[bytes, Path]) -> str:
    """
    Generate deterministic UUID v5 fingerprint from file content.

    Same content always produces same fingerprint, regardless of:
    - File name
    - Upload time
    - Collection
    - System/environment

    Args:
        content: Either file bytes or Path to file

    Returns:
        UUID v5 string (e.g., "8f3d2c1b-4a5e-5678-9abc-def123456789")

    Examples:
        >>> # From bytes
        >>> content = Path("report.pdf").read_bytes()
        >>> fp1 = generate_content_fingerprint(content)

        >>> # From path
        >>> fp2 = generate_content_fingerprint(Path("report.pdf"))

        >>> # Same content → same fingerprint
        >>> assert fp1 == fp2

        >>> # Rename file, upload again
        >>> Path("report.pdf").rename("summary.pdf")
        >>> fp3 = generate_content_fingerprint(Path("summary.pdf"))
        >>> assert fp1 == fp3  # Still same fingerprint!
    """
    # Handle Path input
    if isinstance(content, Path):
        content = content.read_bytes()

    # Calculate SHA256 hash of content
    content_hash = hashlib.sha256(content).hexdigest()

    # Generate deterministic UUID v5
    # Same content_hash always produces same UUID
    fingerprint = uuid.uuid5(FILEINTEL_NAMESPACE, content_hash)

    return str(fingerprint)


def generate_fingerprint_from_hash(content_hash: str) -> str:
    """
    Generate fingerprint from existing SHA256 hash.

    Use this when you already have the SHA256 hash computed
    (e.g., from document.content_hash field).

    Args:
        content_hash: SHA256 hexadecimal string

    Returns:
        UUID v5 string

    Examples:
        >>> content_hash = "abc123def456..."
        >>> fp = generate_fingerprint_from_hash(content_hash)
    """
    return str(uuid.uuid5(FILEINTEL_NAMESPACE, content_hash))


def verify_fingerprint(content: Union[bytes, Path], expected_fingerprint: str) -> bool:
    """
    Verify that content matches expected fingerprint.

    Useful for:
    - Integrity checks
    - Verifying cached content
    - Deduplication validation

    Args:
        content: File bytes or Path
        expected_fingerprint: UUID string to verify against

    Returns:
        True if content matches fingerprint, False otherwise

    Examples:
        >>> content = Path("report.pdf").read_bytes()
        >>> fp = generate_content_fingerprint(content)
        >>> verify_fingerprint(content, fp)
        True
        >>> verify_fingerprint(b"different content", fp)
        False
    """
    actual_fingerprint = generate_content_fingerprint(content)
    return actual_fingerprint == expected_fingerprint
