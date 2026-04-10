"""SQLAlchemy ORM models for ChronoAgent.

All tables are declared here.  Import :class:`Base` to access the shared
metadata for Alembic migrations and ``create_all`` calls.
"""

from __future__ import annotations

import datetime

from sqlalchemy import DateTime, Float, Index, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared declarative base for all ChronoAgent ORM models."""


class AgentSignalRecord(Base):
    """Persisted behavioral signal record — one row per pipeline step.

    Written after every completed pipeline step via
    :meth:`~chronoagent.monitor.collector.BehavioralCollector.persist_step`.

    Column ordering of the six signal fields matches
    :data:`~chronoagent.monitor.collector.SIGNAL_LABELS`.

    Attributes:
        id: Auto-increment primary key.
        agent_id: Identifier for the agent (e.g. ``"security_reviewer"``).
        task_id: Optional identifier for the work unit (e.g. PR id).
        timestamp: UTC timestamp of the step completion.
        total_latency_ms: End-to-end wall-clock time for the step in ms.
        retrieval_count: Total ChromaDB documents retrieved.
        token_count: Approximate LLM input token count.
        kl_divergence: KL divergence from clean retrieval baseline.
        tool_calls: Number of discrete tool / retrieval calls.
        memory_query_entropy: Normalised Shannon entropy of similarity scores.
    """

    __tablename__ = "agent_signal_records"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Identity columns
    agent_id: Mapped[str] = mapped_column(String(255), nullable=False)
    task_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Timestamp — always stored as UTC
    timestamp: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    # 6 behavioral signals (order matches SIGNAL_LABELS)
    total_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    retrieval_count: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    kl_divergence: Mapped[float] = mapped_column(Float, nullable=False)
    tool_calls: Mapped[int] = mapped_column(Integer, nullable=False)
    memory_query_entropy: Mapped[float] = mapped_column(Float, nullable=False)

    # Composite index — common query: all signals for an agent in a time window
    __table_args__ = (Index("ix_asr_agent_ts", "agent_id", "timestamp"),)

    def __repr__(self) -> str:
        return (
            f"AgentSignalRecord(id={self.id!r}, agent_id={self.agent_id!r}, "
            f"task_id={self.task_id!r}, timestamp={self.timestamp!r})"
        )
