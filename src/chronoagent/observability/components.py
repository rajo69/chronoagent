"""Component status reporting for graceful degradation (Phase 9 task 9.3).

Each external dependency (Redis, Postgres, ChromaDB, Chronos forecaster) is
initialised in :func:`chronoagent.main.lifespan` with a primary attempt and a
local fallback.  The outcome of that attempt is captured as a
:class:`ComponentStatus` dataclass and recorded on ``app.state.component_status``
so the forthcoming ``/api/v1/health`` endpoint (task 9.4) can label every
subsystem without re-probing the underlying connection.

``ComponentStatus`` instances are intentionally small, hashable, and
secret-free: the ``detail`` field carries short operator-facing text (backend
identifier plus, for fallbacks, the underlying error type) and never contains
credentials or full stack traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ComponentMode = Literal["primary", "fallback", "unavailable"]


@dataclass(frozen=True)
class ComponentStatus:
    """Result of attempting to initialise an external component.

    Attributes:
        name: Logical component name (``"bus"``, ``"database"``,
            ``"chromadb"``, ``"forecaster"``, ...).
        mode: ``"primary"`` if the preferred backend is active,
            ``"fallback"`` if a local alternative is in use, or
            ``"unavailable"`` if the component cannot be initialised
            at all (no fallback possible).
        detail: Short human-readable description that includes the
            backend identifier and, for fallbacks, the underlying
            error type.  Never contains secrets or full stack traces.
    """

    name: str
    mode: ComponentMode
    detail: str
