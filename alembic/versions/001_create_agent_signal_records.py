"""create agent_signal_records table

Revision ID: 001
Revises:
Create Date: 2026-04-10
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers
revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create agent_signal_records table with composite index."""
    op.create_table(
        "agent_signal_records",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("agent_id", sa.String(255), nullable=False),
        sa.Column("task_id", sa.String(255), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("total_latency_ms", sa.Float, nullable=False),
        sa.Column("retrieval_count", sa.Integer, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False),
        sa.Column("kl_divergence", sa.Float, nullable=False),
        sa.Column("tool_calls", sa.Integer, nullable=False),
        sa.Column("memory_query_entropy", sa.Float, nullable=False),
    )
    op.create_index(
        "ix_asr_agent_ts",
        "agent_signal_records",
        ["agent_id", "timestamp"],
    )


def downgrade() -> None:
    """Drop agent_signal_records table."""
    op.drop_index("ix_asr_agent_ts", table_name="agent_signal_records")
    op.drop_table("agent_signal_records")
