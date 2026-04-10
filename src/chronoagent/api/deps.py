"""FastAPI dependency functions shared across routers."""

from __future__ import annotations

from collections.abc import Generator

from fastapi import Request
from sqlalchemy.orm import Session


def get_db(request: Request) -> Generator[Session, None, None]:
    """Yield an open SQLAlchemy session, closing it on exit.

    Reads ``request.app.state.session_factory`` which is set during
    application lifespan startup.  Override this dependency in tests via
    ``app.dependency_overrides[get_db]`` to inject an in-memory session.

    Args:
        request: FastAPI request (injected by the framework).

    Yields:
        An open :class:`~sqlalchemy.orm.Session`.
    """
    factory = request.app.state.session_factory
    with factory() as session:
        yield session
