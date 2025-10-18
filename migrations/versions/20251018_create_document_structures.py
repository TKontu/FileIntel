"""Create document_structures table

Revision ID: 20251018_document_structures
Revises: 20251013_task_tracking
Create Date: 2025-10-18

This migration creates the document_structures table to store extracted
document structure data (TOC, LOF, LOT, headers) from MinerU processing.
This enables structure-based navigation and querying without embedding
TOC/LOF in the vector store.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = "20251018_document_structures"
down_revision = "20251013_task_tracking"
branch_labels = None
depends_on = None


def upgrade():
    """Create document_structures table."""
    op.create_table(
        'document_structures',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('document_id', sa.String(), nullable=False),
        sa.Column('structure_type', sa.String(), nullable=False),
        sa.Column('data', JSONB, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True),
                 server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'],
                               ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes for faster lookups
    op.create_index(
        'ix_document_structures_document_id',
        'document_structures',
        ['document_id'],
        unique=False
    )

    op.create_index(
        'ix_document_structures_structure_type',
        'document_structures',
        ['structure_type'],
        unique=False
    )


def downgrade():
    """Drop document_structures table."""
    # Drop indexes first
    op.drop_index('ix_document_structures_structure_type',
                 table_name='document_structures')
    op.drop_index('ix_document_structures_document_id',
                 table_name='document_structures')

    # Drop table
    op.drop_table('document_structures')
