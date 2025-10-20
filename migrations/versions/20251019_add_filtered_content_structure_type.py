"""Add filtered_content structure type support

Revision ID: 20251019_filtered_content
Revises: 20251019_create_celery_task_registry
Create Date: 2025-10-19

This migration adds support for 'filtered_content' structure_type to the
document_structures table. This enables storing metadata about elements
filtered during Phase 0 corruption detection in the chunking pipeline.

The filtered_content structure stores:
- filtered_count: Number of elements filtered
- total_elements: Total elements before filtering
- items: Sample of filtered elements with reasons

This provides transparency into what content was filtered and why,
without affecting the primary document processing flow.

NOTE: Since structure_type is a String column (not an enum), this migration
is documentation-only. The actual support is added by updating the
validation logic in structure_storage.py.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20251019_filtered_content"
down_revision = "20251019_create_celery_task_registry"
branch_labels = None
depends_on = None


def upgrade():
    """
    Add support for 'filtered_content' structure_type.

    This is a documentation-only migration since structure_type is a String
    column, not an enum. The actual validation change happens in the
    application code (structure_storage.py).

    Valid structure_types after this migration:
    - 'toc': Table of Contents
    - 'lof': List of Figures
    - 'lot': List of Tables
    - 'headers': Document headers/sections
    - 'filtered_content': Corruption-filtered elements (NEW)
    """
    # No database changes needed - structure_type is a String column
    # The validation in structure_storage.py will be updated to allow 'filtered_content'
    pass


def downgrade():
    """
    Remove support for 'filtered_content' structure_type.

    This removes any existing filtered_content records from the database
    to maintain consistency with the pre-migration schema.
    """
    # Remove any filtered_content records that may have been created
    op.execute("""
        DELETE FROM document_structures
        WHERE structure_type = 'filtered_content'
    """)
