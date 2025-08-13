"""remove_default_from_job_type

Revision ID: 1
Revises: 
Create Date: 2025-08-13 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '1'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.alter_column('jobs', 'job_type', server_default=None)


def downgrade():
    op.alter_column('jobs', 'job_type', server_default='single_file')
