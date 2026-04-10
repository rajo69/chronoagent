"""SQLAlchemy ORM models for ChronoAgent.

All tables are declared here.  Import :class:`Base` to access the shared
metadata for Alembic migrations and ``create_all`` calls.
"""

from __future__ import annotations

import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, Index, Integer, String
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


class AllocationAuditRecord(Base):
    """Persisted audit trail for one contract-net allocation decision.

    Phase 5 task 5.5.  Written by the task allocator (or pipeline) after
    every :func:`~chronoagent.allocator.negotiation.run_contract_net`
    round so the full rationale, the bid ledger, and the health snapshot
    that produced the decision can be reconstructed offline.

    The two structured columns (``all_bids``, ``health_snapshot``) use
    SQLAlchemy's generic :class:`sqlalchemy.JSON` type, which maps to
    SQLite's ``JSON`` (text) and PostgreSQL's ``JSONB``.  Bids are
    stored as a list of ``{agent_id, capability, health, score}`` dicts
    in :data:`~chronoagent.allocator.capability_weights.AGENT_IDS`
    order so the ledger is replayable byte-for-byte.

    Attributes:
        id: Auto-increment primary key.
        task_id: Opaque task identifier from the
            :class:`~chronoagent.allocator.negotiation.NegotiationResult`
            (e.g. ``"pr-42::security_review"``).
        task_type: One of
            :data:`~chronoagent.allocator.capability_weights.TASK_TYPES`.
        assigned_agent: Winning agent ID, or ``None`` when escalated.
        escalated: True when the round ended in human escalation
            because every bid fell below threshold.
        all_bids: JSON list of every bid in
            :data:`~chronoagent.allocator.capability_weights.AGENT_IDS`
            order.  Each entry is
            ``{"agent_id", "capability", "health", "score"}``.
        health_snapshot: JSON object ``{agent_id: health}`` mirroring
            the snapshot the allocator used at decision time.
        rationale: Human-readable explanation copied from
            :attr:`NegotiationResult.rationale`.
        threshold: Minimum-bid threshold in force for this round.
        timestamp: UTC timestamp of the decision.
    """

    __tablename__ = "allocation_audit_records"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Identity columns
    task_id: Mapped[str] = mapped_column(String(255), nullable=False)
    task_type: Mapped[str] = mapped_column(String(64), nullable=False)

    # Outcome
    assigned_agent: Mapped[str | None] = mapped_column(String(255), nullable=True)
    escalated: Mapped[bool] = mapped_column(Boolean, nullable=False)

    # Decision rationale + structured ledger
    all_bids: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    health_snapshot: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    rationale: Mapped[str] = mapped_column(String(1024), nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)

    # Timestamp -- always stored as UTC
    timestamp: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    # Composite index -- common query: full audit trail for one task
    __table_args__ = (Index("ix_aar_task_ts", "task_id", "timestamp"),)

    def __repr__(self) -> str:
        return (
            f"AllocationAuditRecord(id={self.id!r}, task_id={self.task_id!r}, "
            f"task_type={self.task_type!r}, assigned_agent={self.assigned_agent!r}, "
            f"escalated={self.escalated!r})"
        )
