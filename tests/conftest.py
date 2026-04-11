"""Shared pytest fixtures for the ChronoAgent test suite.

Phase 9 task 9.5 migrates every ``src/`` module to the structlog-backed
``chronoagent.observability.logging.get_logger`` wrapper.  Structlog's
``get_logger`` returns a lazy proxy that only binds to its concrete
backend on first use; without an explicit :func:`configure_logging`
call, that backend defaults to ``structlog.PrintLogger`` (stdout), so
records never reach pytest's ``caplog`` fixture.

This conftest installs an autouse session-scoped fixture that calls
:func:`configure_logging` once before any test runs.  The call wires
structlog through the ``structlog.stdlib.LoggerFactory`` so every
bound-logger emission creates a real ``logging.LogRecord`` on the
stdlib logger named after the module, which ``caplog`` then captures
transparently.
"""

from __future__ import annotations

import pytest

from chronoagent.observability.logging import configure_logging


@pytest.fixture(autouse=True, scope="session")
def _configure_structlog_for_tests() -> None:
    """Route structlog through stdlib logging so ``caplog`` works everywhere.

    Called once per session via the ``autouse=True`` flag; individual
    tests are not aware of the fixture.  We pick the ``test`` env so the
    renderer matches what CI assertions expect.
    """
    configure_logging("test")
