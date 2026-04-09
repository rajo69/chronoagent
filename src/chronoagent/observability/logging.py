"""Structured logging configuration using structlog.

JSON format in production; colourised console output in dev/test.
Every log record carries consistent fields: ``agent_id``, ``task_id``, ``phase``.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(env: str = "dev") -> None:
    """Configure structlog for the given runtime environment.

    Args:
        env: One of ``"dev"``, ``"test"``, or ``"prod"``.
             Dev/test use a human-friendly colourised renderer;
             prod uses JSON for log-aggregation pipelines.
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if env == "prod":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG if env != "prod" else logging.INFO)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger with the given name.

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        A :class:`structlog.stdlib.BoundLogger` instance.
    """
    return structlog.get_logger(name)  # type: ignore[return-value]
