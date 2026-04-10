"""Static capability matrix for the four ChronoAgent review agents.

Phase 5, task 5.1 of the project plan.

Overview
--------
The :class:`CapabilityMatrix` stores a fixed proficiency value in ``[0, 1]``
for every (agent, task type) pair.  ``1.0`` means the agent is the
designated specialist for the task; values below ``1.0`` quantify how
well a non-specialist can stand in if the specialist is degraded or
unavailable.  Phase 5's negotiation protocol multiplies these weights by
the live :class:`~chronoagent.scorer.health_scorer.HealthUpdate` to
compute bids:

    bid(agent, task) = capability_weight(agent, task) * health(agent)

Design choices
--------------
* **Diagonal = 1.0.** Each agent is fully proficient at its primary task.
  This anchors the matrix and makes the "happy path" deterministic: a
  healthy specialist always wins its own task.
* **Off-diagonal in (0, 1).** No off-diagonal entry is exactly zero, so
  Phase 5's contract-net protocol can always redistribute work when a
  specialist is degraded (the exit criterion in PLAN.md).  When *all*
  bids fall below the negotiation threshold the allocator escalates to a
  human; the matrix never silently drops a task.
* **Soft overlap reflects task affinity.** Security and style reviewers
  share a "code review" affinity, so their off-diagonal weights for each
  other are higher than for ``plan`` or ``summarize``.  The summarizer
  is a strong fallback for ``plan`` because both are language-heavy
  composition tasks.
* **Immutable, deterministic.** The matrix is exposed as a frozen
  dataclass and the module-level :data:`DEFAULT_CAPABILITY_MATRIX`
  instance is hashed by identity.  No runtime mutation; lookups are
  pure functions of agent_id and task_type.

The agent IDs match those used by the agents themselves
(:mod:`chronoagent.agents`).  The task types match
:data:`chronoagent.agents.registry._CAPABILITY_MAP`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

# ---------------------------------------------------------------------------
# Canonical agent IDs and task types
# ---------------------------------------------------------------------------

#: Canonical agent IDs in deterministic order.  Order is used for
#: tie-breaking in Phase 5.2 and as the row order of the matrix.
AGENT_IDS: tuple[str, ...] = (
    "planner",
    "security_reviewer",
    "style_reviewer",
    "summarizer",
)

#: Canonical task types, in the natural order they appear in the
#: pipeline (plan -> reviews in parallel -> summarise).
TASK_TYPES: tuple[str, ...] = (
    "plan",
    "security_review",
    "style_review",
    "summarize",
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class UnknownAgentError(KeyError):
    """Raised when a lookup uses an agent_id outside :data:`AGENT_IDS`."""

    def __init__(self, agent_id: str) -> None:
        super().__init__(f"Unknown agent_id={agent_id!r}. Known agents: {list(AGENT_IDS)}")
        self.agent_id = agent_id


class UnknownTaskTypeError(KeyError):
    """Raised when a lookup uses a task_type outside :data:`TASK_TYPES`."""

    def __init__(self, task_type: str) -> None:
        super().__init__(f"Unknown task_type={task_type!r}. Known task types: {list(TASK_TYPES)}")
        self.task_type = task_type


# ---------------------------------------------------------------------------
# CapabilityMatrix
# ---------------------------------------------------------------------------


def _build_default_weights() -> dict[str, dict[str, float]]:
    """Construct the default 4 agents x 4 task types proficiency table.

    See module docstring for the rationale behind each off-diagonal value.
    Specialists score 1.0 on their own task.  All other entries are
    strictly inside ``(0, 1)`` to keep every agent eligible (with reduced
    bid) when its specialist peer is degraded.
    """

    return {
        "planner": {
            "plan": 1.00,
            "security_review": 0.30,
            "style_review": 0.30,
            "summarize": 0.60,
        },
        "security_reviewer": {
            "plan": 0.25,
            "security_review": 1.00,
            "style_review": 0.55,
            "summarize": 0.40,
        },
        "style_reviewer": {
            "plan": 0.25,
            "security_review": 0.55,
            "style_review": 1.00,
            "summarize": 0.40,
        },
        "summarizer": {
            "plan": 0.50,
            "security_review": 0.35,
            "style_review": 0.35,
            "summarize": 1.00,
        },
    }


@dataclass(frozen=True)
class CapabilityMatrix:
    """Immutable proficiency table for agent / task-type pairs.

    Parameters
    ----------
    weights:
        Nested mapping ``{agent_id: {task_type: proficiency}}``.  Every
        agent in :data:`AGENT_IDS` must appear with a complete row over
        :data:`TASK_TYPES`, and every value must lie in ``[0, 1]``.

    Notes
    -----
    The dataclass is ``frozen=True``; the inner mapping is wrapped in a
    :class:`types.MappingProxyType` so callers cannot mutate it.  Use
    :meth:`as_dict` to obtain a deep, mutable copy.
    """

    weights: Mapping[str, Mapping[str, float]] = field(
        default_factory=lambda: MappingProxyType(
            {agent: MappingProxyType(row) for agent, row in _build_default_weights().items()}
        )
    )

    def __post_init__(self) -> None:
        self._validate(self.weights)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(weights: Mapping[str, Mapping[str, float]]) -> None:
        """Reject malformed matrices early so lookups are total.

        Checks:
            1. The set of agent rows equals :data:`AGENT_IDS`.
            2. Each row's keys equal :data:`TASK_TYPES`.
            3. Every value is a finite float in ``[0, 1]``.
        """
        expected_agents = set(AGENT_IDS)
        actual_agents = set(weights.keys())
        if actual_agents != expected_agents:
            missing = expected_agents - actual_agents
            extra = actual_agents - expected_agents
            raise ValueError(
                "CapabilityMatrix rows must match AGENT_IDS exactly. "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )

        expected_tasks = set(TASK_TYPES)
        for agent_id in AGENT_IDS:
            row = weights[agent_id]
            actual_tasks = set(row.keys())
            if actual_tasks != expected_tasks:
                missing = expected_tasks - actual_tasks
                extra = actual_tasks - expected_tasks
                raise ValueError(
                    f"CapabilityMatrix row {agent_id!r} must cover all "
                    f"TASK_TYPES. missing={sorted(missing)}, "
                    f"extra={sorted(extra)}"
                )
            for task_type in TASK_TYPES:
                value = row[task_type]
                if not isinstance(value, (int, float)):
                    raise TypeError(
                        f"CapabilityMatrix[{agent_id!r}][{task_type!r}] "
                        f"must be a number, got {type(value).__name__}"
                    )
                fvalue = float(value)
                if fvalue != fvalue:  # NaN check (NaN != NaN)
                    raise ValueError(f"CapabilityMatrix[{agent_id!r}][{task_type!r}] is NaN")
                if not (0.0 <= fvalue <= 1.0):
                    raise ValueError(
                        f"CapabilityMatrix[{agent_id!r}][{task_type!r}]={fvalue} must be in [0, 1]"
                    )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def proficiency(self, agent_id: str, task_type: str) -> float:
        """Return the proficiency of *agent_id* on *task_type*.

        Args:
            agent_id: One of :data:`AGENT_IDS`.
            task_type: One of :data:`TASK_TYPES`.

        Returns:
            Proficiency in ``[0, 1]``.

        Raises:
            UnknownAgentError: If *agent_id* is not in :data:`AGENT_IDS`.
            UnknownTaskTypeError: If *task_type* is not in :data:`TASK_TYPES`.
        """
        if agent_id not in self.weights:
            raise UnknownAgentError(agent_id)
        row = self.weights[agent_id]
        if task_type not in row:
            raise UnknownTaskTypeError(task_type)
        return float(row[task_type])

    def row(self, agent_id: str) -> dict[str, float]:
        """Return a copy of *agent_id*'s proficiency row.

        Args:
            agent_id: One of :data:`AGENT_IDS`.

        Returns:
            Mutable ``{task_type: proficiency}`` copy.

        Raises:
            UnknownAgentError: If *agent_id* is not registered.
        """
        if agent_id not in self.weights:
            raise UnknownAgentError(agent_id)
        return {task: float(value) for task, value in self.weights[agent_id].items()}

    def column(self, task_type: str) -> dict[str, float]:
        """Return all agents' proficiencies for *task_type*.

        Useful when the negotiation protocol asks "who can do this task
        and how well?" before scaling by health.

        Args:
            task_type: One of :data:`TASK_TYPES`.

        Returns:
            Mutable ``{agent_id: proficiency}`` copy ordered by
            :data:`AGENT_IDS`.

        Raises:
            UnknownTaskTypeError: If *task_type* is not registered.
        """
        if task_type not in TASK_TYPES:
            raise UnknownTaskTypeError(task_type)
        return {agent_id: float(self.weights[agent_id][task_type]) for agent_id in AGENT_IDS}

    def primary_agent(self, task_type: str) -> str:
        """Return the highest-proficiency agent for *task_type*.

        For the default matrix this is the diagonal specialist (the
        agent whose name encodes the task).  Ties are broken by the
        order of :data:`AGENT_IDS` so the choice is deterministic even
        if a future matrix has equal entries.

        Args:
            task_type: One of :data:`TASK_TYPES`.

        Returns:
            The agent_id with the highest proficiency for the task.

        Raises:
            UnknownTaskTypeError: If *task_type* is not registered.
        """
        column = self.column(task_type)
        best_agent = AGENT_IDS[0]
        best_score = column[best_agent]
        for agent_id in AGENT_IDS[1:]:
            score = column[agent_id]
            if score > best_score:
                best_agent = agent_id
                best_score = score
        return best_agent

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def as_dict(self) -> dict[str, dict[str, float]]:
        """Return a deep, mutable copy of the underlying weights.

        Useful for serialisation (audit logs, JSON responses) without
        risking mutation of the canonical instance.
        """
        return {
            agent_id: {task: float(value) for task, value in row.items()}
            for agent_id, row in self.weights.items()
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

#: The canonical capability matrix used by Phase 5 negotiation.  Built
#: once at import time so every importer shares the same identity.
DEFAULT_CAPABILITY_MATRIX: CapabilityMatrix = CapabilityMatrix()
