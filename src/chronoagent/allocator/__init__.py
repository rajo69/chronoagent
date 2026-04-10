"""Decentralized task allocator package (Phase 5).

This package implements health-aware, capability-weighted task routing
between ChronoAgent's review agents.  Phase 5 layers on top of the
Temporal Health Scorer (Phase 4) and the Behavioral Monitor (Phase 3) so
that allocation decisions can react to per-agent health in real time.

Modules
-------
``capability_weights``
    Static :class:`CapabilityMatrix`: a 4 agents x 4 task types proficiency
    table in [0, 1].  Provides lookup, listing, and snapshot helpers.
"""

from __future__ import annotations

from chronoagent.allocator.capability_weights import (
    AGENT_IDS,
    DEFAULT_CAPABILITY_MATRIX,
    TASK_TYPES,
    CapabilityMatrix,
    UnknownAgentError,
    UnknownTaskTypeError,
)

__all__ = [
    "AGENT_IDS",
    "DEFAULT_CAPABILITY_MATRIX",
    "TASK_TYPES",
    "CapabilityMatrix",
    "UnknownAgentError",
    "UnknownTaskTypeError",
]
