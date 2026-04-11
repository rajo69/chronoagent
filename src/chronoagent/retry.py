"""Central tenacity retry policies for every external ChronoAgent call.

Every call that crosses a process boundary (LLM HTTP, Redis, ChromaDB,
Postgres/SQLAlchemy) is wrapped with a named policy exported from this
module so the "3 attempts, exponential backoff" knob lives in one file.

Design notes
------------
* Every policy uses the same ``stop_after_attempt(3)`` and an exponential
  backoff of 0.25s..2.0s.  Callers that need a different cadence should
  pass an explicit ``retry`` kwarg instead of inventing a new policy here.
* Each policy restricts ``retry_if_exception_type`` to a narrow base class
  that genuinely means "transient external failure" (network errors,
  Redis connection loss, Chroma I/O flakes, SQLAlchemy operational
  errors).  Exceptions that indicate a bug in our own code
  (``ValueError``, ``TypeError``, SQLAlchemy ``IntegrityError`` from a
  duplicate primary key, ...) propagate on the first raise and are
  never retried.
* ``before_sleep`` emits a structured ``logging.WARNING`` entry naming
  the policy and the exception class so operators can spot retry storms
  in production logs.

Usage::

    from chronoagent.retry import llm_retry, db_retry, redis_retry, chroma_retry

    class TogetherAIBackend:
        @llm_retry
        def generate(self, prompt: str) -> str:
            ...
"""

from __future__ import annotations

import logging

import chromadb.errors
import httpx
import redis.exceptions
import sqlalchemy.exc
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Shared knobs.  ``stop_after_attempt(3)`` = at most two retries after the
# initial attempt, matching the PLAN's "3 attempts" wording.
_MAX_ATTEMPTS = 3
_WAIT = wait_exponential(multiplier=0.25, min=0.25, max=2.0)

# Single logger for every retry warning so log filters can target one name.
_logger = logging.getLogger("chronoagent.retry")
_before_sleep = before_sleep_log(_logger, logging.WARNING)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

# LLM HTTP calls (Together.ai, Ollama).  ``httpx.HTTPError`` is the common
# base of ``RequestError`` (network, timeout, connection) and
# ``HTTPStatusError`` (raised by ``raise_for_status``), so transient 5xx
# responses and flaky sockets are both covered.
llm_retry = retry(
    stop=stop_after_attempt(_MAX_ATTEMPTS),
    wait=_WAIT,
    retry=retry_if_exception_type(httpx.HTTPError),
    before_sleep=_before_sleep,
    reraise=True,
)

# Redis pub/sub + commands.  ``redis.exceptions.RedisError`` is the base
# for connection errors, timeouts, and response errors.
redis_retry = retry(
    stop=stop_after_attempt(_MAX_ATTEMPTS),
    wait=_WAIT,
    retry=retry_if_exception_type(redis.exceptions.RedisError),
    before_sleep=_before_sleep,
    reraise=True,
)

# ChromaDB collection operations.  ``chromadb.errors.ChromaError`` is the
# base of every Chroma-raised error; concrete client-side mistakes
# (``InvalidDimensionException``, duplicate-id errors) are subclasses and
# will also retry twice before surfacing, which is acceptable because
# those failures still bubble up on attempt three.
chroma_retry = retry(
    stop=stop_after_attempt(_MAX_ATTEMPTS),
    wait=_WAIT,
    retry=retry_if_exception_type(chromadb.errors.ChromaError),
    before_sleep=_before_sleep,
    reraise=True,
)

# SQLAlchemy / Postgres.  ``OperationalError`` covers "connection closed",
# "could not connect", server restarts, and transient lock failures.
# ``IntegrityError`` (duplicate key, FK violation) indicates a bug in the
# caller and is deliberately NOT retried.
db_retry = retry(
    stop=stop_after_attempt(_MAX_ATTEMPTS),
    wait=_WAIT,
    retry=retry_if_exception_type(sqlalchemy.exc.OperationalError),
    before_sleep=_before_sleep,
    reraise=True,
)


__all__ = [
    "chroma_retry",
    "db_retry",
    "llm_retry",
    "redis_retry",
]
