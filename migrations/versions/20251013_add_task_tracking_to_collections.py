"""Add task tracking fields to collections

Revision ID: 20251013_task_tracking
Revises: 61ee6f04df66
Create Date: 2025-10-13 22:14:28

This migration adds task tracking fields to the collections table to enable
correlation between collections and their processing tasks, preventing stuck
collections and enabling better status tracking.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = "20251013_task_tracking"
down_revision = "61ee6f04df66"
branch_labels = None
depends_on = None


def upgrade():
    """Add task tracking fields to collections table."""
    # Add current_task_id column - tracks the active task for this collection
    op.add_column(
        "collections",
        sa.Column("current_task_id", sa.String(), nullable=True)
    )

    # Add task_history column - stores history of all tasks for this collection
    op.add_column(
        "collections",
        sa.Column("task_history", JSONB, nullable=True)
    )

    # Add status_updated_at column - tracks when status was last changed
    op.add_column(
        "collections",
        sa.Column("status_updated_at", sa.DateTime(timezone=True), nullable=True)
    )

    # Create index on current_task_id for faster lookups
    op.create_index(
        "ix_collections_current_task_id",
        "collections",
        ["current_task_id"],
        unique=False
    )


def downgrade():
    """Remove task tracking fields from collections table."""
    # Drop index first
    op.drop_index("ix_collections_current_task_id", table_name="collections")

    # Drop columns
    op.drop_column("collections", "status_updated_at")
    op.drop_column("collections", "task_history")
    op.drop_column("collections", "current_task_id")
