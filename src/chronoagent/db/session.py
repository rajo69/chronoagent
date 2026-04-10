"""SQLAlchemy engine and session factory for ChronoAgent.

Usage in application code::

    from chronoagent.db.session import make_session_factory

    SessionLocal = make_session_factory()

    with SessionLocal() as session:
        session.add(record)
        session.commit()

Usage in tests with an in-memory SQLite database::

    from chronoagent.config import Settings
    from chronoagent.db.models import Base
    from chronoagent.db.session import make_engine, make_session_factory

    settings = Settings(database_url="sqlite:///:memory:")
    engine = make_engine(settings)
    Base.metadata.create_all(engine)
    SessionLocal = make_session_factory(settings)
"""

from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from chronoagent.config import Settings, load_settings


def make_engine(settings: Settings | None = None) -> Engine:
    """Create a SQLAlchemy :class:`~sqlalchemy.Engine` from settings.

    Adds ``check_same_thread=False`` for SQLite engines (required for
    multi-threaded FastAPI usage).

    Args:
        settings: Optional pre-built :class:`~chronoagent.config.Settings`.
            Defaults to :func:`~chronoagent.config.load_settings`.

    Returns:
        Configured :class:`~sqlalchemy.Engine`.
    """
    resolved = settings or load_settings()
    kwargs: dict[str, object] = {}
    if resolved.database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(resolved.database_url, **kwargs)


def make_session_factory(settings: Settings | None = None) -> sessionmaker[Session]:
    """Return a :class:`~sqlalchemy.orm.sessionmaker` bound to a new engine.

    Args:
        settings: Optional :class:`~chronoagent.config.Settings`.

    Returns:
        :class:`~sqlalchemy.orm.sessionmaker` configured with
        ``autocommit=False`` and ``autoflush=False``.
    """
    engine = make_engine(settings)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def make_session_factory_from_engine(engine: Engine) -> sessionmaker[Session]:
    """Return a :class:`~sqlalchemy.orm.sessionmaker` bound to an existing engine.

    Use this when you already hold a reference to the engine (e.g. after calling
    :func:`make_engine` + ``Base.metadata.create_all``) so that the session
    factory shares the same connection pool.  This is especially important for
    in-memory SQLite databases where each engine has its own isolated database.

    Args:
        engine: A pre-built :class:`~sqlalchemy.Engine`.

    Returns:
        :class:`~sqlalchemy.orm.sessionmaker` configured with
        ``autocommit=False`` and ``autoflush=False``.
    """
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)
