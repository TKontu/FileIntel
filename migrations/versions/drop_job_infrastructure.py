"""Drop job infrastructure tables

Revision ID: 2025_01_19_drop_jobs
Revises: 61ee6f04df66
Create Date: 2025-01-19 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "2025_01_19_drop_jobs"
down_revision = "61ee6f04df66"
branch_labels = None
depends_on = None


def upgrade():
    """Drop job infrastructure tables and foreign key constraints."""

    # Drop foreign key constraints first to avoid dependency issues
    op.drop_constraint("jobs_collection_id_fkey", "jobs", type_="foreignkey")
    op.drop_constraint(
        "dead_letter_jobs_collection_id_fkey", "dead_letter_jobs", type_="foreignkey"
    )

    # Drop job-related tables
    op.drop_table("circuit_breaker_states")
    op.drop_table("dead_letter_jobs")
    op.drop_table("workers")
    op.drop_table("jobs")

    # Drop job-related indexes that might still exist
    try:
        op.drop_index("ix_jobs_collection_id", table_name="jobs")
    except Exception:
        pass  # Index might not exist

    try:
        op.drop_index("ix_jobs_status", table_name="jobs")
    except Exception:
        pass  # Index might not exist

    try:
        op.drop_index("ix_jobs_worker_id", table_name="jobs")
    except Exception:
        pass  # Index might not exist

    try:
        op.drop_index(
            "ix_dead_letter_jobs_collection_id", table_name="dead_letter_jobs"
        )
    except Exception:
        pass  # Index might not exist


def downgrade():
    """Recreate job infrastructure tables if needed for rollback."""

    # Note: This is a destructive migration. Downgrade recreates empty tables
    # but does not restore data that was dropped during upgrade.

    # Recreate jobs table
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("collection_id", sa.String(), nullable=False),
        sa.Column("job_type", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("worker_id", sa.String(), nullable=True),
        sa.Column("parameters", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_retries", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("priority", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("timeout_seconds", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # Recreate workers table
    op.create_table(
        "workers",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column(
            "capabilities", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("last_heartbeat", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "shutdown_requested", sa.Boolean(), nullable=False, server_default="false"
        ),
        sa.Column("current_job_id", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # Recreate dead_letter_jobs table
    op.create_table(
        "dead_letter_jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("collection_id", sa.String(), nullable=False),
        sa.Column(
            "original_job_data", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("failure_reason", sa.Text(), nullable=False),
        sa.Column(
            "failed_at", sa.DateTime(timezone=True), server_default=sa.text("now()")
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Recreate circuit_breaker_states table
    op.create_table(
        "circuit_breaker_states",
        sa.Column("operation_type", sa.String(), nullable=False),
        sa.Column("state", sa.String(), nullable=False),
        sa.Column("failure_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_failure_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_success_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_attempt_time", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("operation_type"),
    )

    # Recreate indexes
    op.create_index("ix_jobs_collection_id", "jobs", ["collection_id"])
    op.create_index("ix_jobs_status", "jobs", ["status"])
    op.create_index("ix_jobs_worker_id", "jobs", ["worker_id"])
    op.create_index(
        "ix_dead_letter_jobs_collection_id", "dead_letter_jobs", ["collection_id"]
    )

    # Recreate foreign key constraints
    op.create_foreign_key(
        "jobs_collection_id_fkey", "jobs", "collections", ["collection_id"], ["id"]
    )
    op.create_foreign_key(
        "dead_letter_jobs_collection_id_fkey",
        "dead_letter_jobs",
        "collections",
        ["collection_id"],
        ["id"],
    )
