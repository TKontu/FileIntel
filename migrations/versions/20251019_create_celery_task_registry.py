"""Create celery_task_registry table for stale task detection

Revision ID: 20251019_celery_task_registry
Revises: 20251018_document_structures
Create Date: 2025-10-19

This migration creates the celery_task_registry table to track active
Celery tasks and enable detection of stale tasks from dead workers.

When workers die unexpectedly (docker-compose down, crashes), tasks can be
left in STARTED state. This table enables automatic cleanup on worker restart
by tracking which worker is processing which task, with heartbeat timestamps
for long-running tasks.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = "20251019_celery_task_registry"
down_revision = "20251018_document_structures"
branch_labels = None
depends_on = None


def upgrade():
    """Create celery_task_registry table."""
    op.create_table(
        'celery_task_registry',
        sa.Column('task_id', sa.String(), nullable=False),
        sa.Column('task_name', sa.String(), nullable=False),
        sa.Column('worker_id', sa.String(), nullable=False),
        sa.Column('worker_pid', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('queued_at', sa.DateTime(timezone=True),
                 server_default=sa.text('now()'), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('args', JSONB, nullable=True),
        sa.Column('kwargs', JSONB, nullable=True),
        sa.Column('result', JSONB, nullable=True),
        sa.Column('last_heartbeat', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True),
                 server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('task_id')
    )

    # Create indexes for faster lookups
    op.create_index(
        'ix_celery_task_registry_task_name',
        'celery_task_registry',
        ['task_name'],
        unique=False
    )

    op.create_index(
        'ix_celery_task_registry_worker_id',
        'celery_task_registry',
        ['worker_id'],
        unique=False
    )

    op.create_index(
        'ix_celery_task_registry_status',
        'celery_task_registry',
        ['status'],
        unique=False
    )


def downgrade():
    """Drop celery_task_registry table."""
    # Drop indexes first
    op.drop_index('ix_celery_task_registry_status',
                 table_name='celery_task_registry')
    op.drop_index('ix_celery_task_registry_worker_id',
                 table_name='celery_task_registry')
    op.drop_index('ix_celery_task_registry_task_name',
                 table_name='celery_task_registry')

    # Drop table
    op.drop_table('celery_task_registry')
