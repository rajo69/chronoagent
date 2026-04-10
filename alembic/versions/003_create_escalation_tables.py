"""create audit_events and escalation_records tables

Revision ID: 003
Revises: 002
Create Date: 2026-04-10
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers
revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create audit_events and escalation_records tables with indexes."""
    op.create_table(
        "audit_events",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("event_type", sa.String(32), nullable=False),
        sa.Column("agent_id", sa.String(255), nullable=True),
        sa.Column("payload", sa.JSON, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_ae_event_ts", "audit_events", ["event_type", "timestamp"])
    op.create_index("ix_ae_agent_ts", "audit_events", ["agent_id", "timestamp"])

    op.create_table(
        "escalation_records",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("agent_id", sa.String(255), nullable=False),
        sa.Column("trigger", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("context", sa.JSON, nullable=False),
        sa.Column("resolution_notes", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_er_status_created", "escalation_records", ["status", "created_at"])
    op.create_index("ix_er_agent_created", "escalation_records", ["agent_id", "created_at"])


def downgrade() -> None:
    """Drop escalation_records and audit_events tables."""
    op.drop_index("ix_er_agent_created", table_name="escalation_records")
    op.drop_index("ix_er_status_created", table_name="escalation_records")
    op.drop_table("escalation_records")
    op.drop_index("ix_ae_agent_ts", table_name="audit_events")
    op.drop_index("ix_ae_event_ts", table_name="audit_events")
    op.drop_table("audit_events")
