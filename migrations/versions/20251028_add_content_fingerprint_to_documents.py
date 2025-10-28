"""Add content_fingerprint to documents

Revision ID: 20251028_content_fingerprint
Revises: 20251019_filtered_content
Create Date: 2025-10-28

This migration adds content-based fingerprinting to documents table.

The content_fingerprint field stores a deterministic UUID v5 generated from
file content (SHA256 hash), enabling:
- Idempotent processing: same content → same UUID
- Global deduplication across collections
- MinerU output caching and reuse
- Storage optimization

Key features:
- Nullable for backward compatibility with existing documents
- Indexed for fast fingerprint lookups
- Generated from file content (not filename)
- Same file uploaded multiple times gets same fingerprint
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20251028_content_fingerprint"
down_revision = "20251019_filtered_content"
branch_labels = None
depends_on = None


def upgrade():
    """Add content_fingerprint column to documents table."""
    # Add content_fingerprint column
    # VARCHAR(36) to store UUID strings (format: "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
    # Nullable to support existing documents without fingerprints
    op.add_column(
        "documents",
        sa.Column("content_fingerprint", sa.String(36), nullable=True)
    )

    # Create index for fast fingerprint lookups
    # Used when checking for duplicate uploads
    op.create_index(
        "ix_documents_content_fingerprint",
        "documents",
        ["content_fingerprint"],
        unique=False  # Allow NULL values, non-unique for safety
    )

    # Optional: Add unique constraint to enforce global deduplication
    # Uncomment the following lines to prevent same content in multiple collections:
    #
    # op.create_unique_constraint(
    #     "uq_documents_content_fingerprint",
    #     "documents",
    #     ["content_fingerprint"]
    # )
    #
    # Note: This would require handling cases where same file needs to be
    # in multiple collections (e.g., via document-collection linking table)


def downgrade():
    """Remove content_fingerprint column from documents table."""
    # Drop index first
    op.drop_index("ix_documents_content_fingerprint", table_name="documents")

    # Drop column
    op.drop_column("documents", "content_fingerprint")
