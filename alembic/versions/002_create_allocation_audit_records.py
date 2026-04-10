"""create allocation_audit_records table

Revision ID: 002
Revises: 001
Create Date: 2026-04-10
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers
revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create allocation_audit_records table with composite index."""
    op.create_table(
        "allocation_audit_records",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("task_id", sa.String(255), nullable=False),
        sa.Column("task_type", sa.String(64), nullable=False),
        sa.Column("assigned_agent", sa.String(255), nullable=True),
        sa.Column("escalated", sa.Boolean, nullable=False),
        sa.Column("all_bids", sa.JSON, nullable=False),
        sa.Column("health_snapshot", sa.JSON, nullable=False),
        sa.Column("rationale", sa.String(1024), nullable=False),
        sa.Column("threshold", sa.Float, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_aar_task_ts",
        "allocation_audit_records",
        ["task_id", "timestamp"],
    )


def downgrade() -> None:
    """Drop allocation_audit_records table."""
    op.drop_index("ix_aar_task_ts", table_name="allocation_audit_records")
    op.drop_table("allocation_audit_records")
