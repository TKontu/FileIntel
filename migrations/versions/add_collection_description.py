"""add collection description

Revision ID: add_collection_description
Revises: drop_job_infrastructure
Create Date: 2025-09-20 20:45:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "add_collection_description"
down_revision = "drop_job_infrastructure"
branch_labels = None
depends_on = None


def upgrade():
    # Add description column to collections table
    op.add_column("collections", sa.Column("description", sa.String(), nullable=True))


def downgrade():
    # Remove description column from collections table
    op.drop_column("collections", "description")
