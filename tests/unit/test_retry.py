"""Unit tests for Phase 9 task 9.2 tenacity retry policies.

Covers each named policy exported from :mod:`chronoagent.retry`:

- ``llm_retry``: retries ``httpx.HTTPError``, propagates ``ValueError``.
- ``redis_retry``: retries ``redis.exceptions.RedisError``.
- ``chroma_retry``: retries ``chromadb.errors.ChromaError``.
- ``db_retry``: retries ``sqlalchemy.exc.OperationalError``, propagates
  ``sqlalchemy.exc.IntegrityError`` (caller bug, never retried).

Each policy is exercised with a fake that raises ``N-1`` times then
returns a sentinel value, so we can assert the decorator makes exactly
three attempts before surfacing success, that non-retryable exceptions
propagate on the first raise, and that a structured-logging WARNING is
emitted on every intermediate retry.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import chromadb.errors
import httpx
import pytest
import redis.exceptions
import sqlalchemy.exc

from chronoagent.retry import chroma_retry, db_retry, llm_retry, redis_retry

# ---------------------------------------------------------------------------
# Flaky callable helper
# ---------------------------------------------------------------------------


class _Flaky:
    """Callable that raises *exc* ``fail_times`` times, then returns *result*.

    Tracks call count so tests can assert the total attempts made.
    """

    def __init__(
        self,
        *,
        fail_times: int,
        exc: BaseException,
        result: Any = "success",
    ) -> None:
        self.fail_times = fail_times
        self.exc = exc
        self.result = result
        self.calls = 0

    def __call__(self, *args: object, **kwargs: object) -> Any:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        return self.result


def _make_operational_error() -> sqlalchemy.exc.OperationalError:
    """Build a genuine ``OperationalError`` without hitting a real engine."""
    return sqlalchemy.exc.OperationalError(
        "SELECT 1",
        params={},
        orig=Exception("connection refused"),
    )


def _make_integrity_error() -> sqlalchemy.exc.IntegrityError:
    """Build a genuine ``IntegrityError`` without hitting a real engine."""
    return sqlalchemy.exc.IntegrityError(
        "INSERT INTO foo VALUES (1)",
        params={},
        orig=Exception("duplicate key"),
    )


# ---------------------------------------------------------------------------
# Shared cross-policy assertions
# ---------------------------------------------------------------------------


_POLICIES: list[tuple[str, Callable[..., Any], BaseException]] = [
    ("llm_retry", llm_retry, httpx.ConnectError("boom")),
    ("redis_retry", redis_retry, redis.exceptions.ConnectionError("boom")),
    (
        "chroma_retry",
        chroma_retry,
        chromadb.errors.ChromaError("boom"),
    ),
    ("db_retry", db_retry, _make_operational_error()),
]


@pytest.mark.parametrize(
    ("_policy_name", "policy", "retryable_exc"),
    _POLICIES,
    ids=[name for name, _, _ in _POLICIES],
)
class TestPolicyRetryBehaviour:
    """Every policy must retry exactly 3 attempts total on its own exception."""

    def test_succeeds_on_third_attempt_exactly(
        self,
        _policy_name: str,
        policy: Callable[..., Any],
        retryable_exc: BaseException,
    ) -> None:
        flaky = _Flaky(fail_times=2, exc=retryable_exc, result="ok")
        wrapped = policy(flaky)

        assert wrapped() == "ok"
        assert flaky.calls == 3

    def test_single_failure_still_retries_and_returns(
        self,
        _policy_name: str,
        policy: Callable[..., Any],
        retryable_exc: BaseException,
    ) -> None:
        flaky = _Flaky(fail_times=1, exc=retryable_exc, result="ok")
        wrapped = policy(flaky)

        assert wrapped() == "ok"
        assert flaky.calls == 2

    def test_exhausts_at_three_attempts_and_reraises(
        self,
        _policy_name: str,
        policy: Callable[..., Any],
        retryable_exc: BaseException,
    ) -> None:
        flaky = _Flaky(fail_times=10, exc=retryable_exc)
        wrapped = policy(flaky)

        with pytest.raises(type(retryable_exc)):
            wrapped()
        assert flaky.calls == 3

    def test_non_retryable_propagates_on_first_raise(
        self,
        _policy_name: str,
        policy: Callable[..., Any],
        retryable_exc: BaseException,
    ) -> None:
        # ValueError indicates a bug in the caller; all policies must
        # refuse to retry it and surface on the first raise.
        flaky = _Flaky(fail_times=10, exc=ValueError("bug"))
        wrapped = policy(flaky)

        with pytest.raises(ValueError, match="bug"):
            wrapped()
        assert flaky.calls == 1

    def test_warning_logged_before_each_retry(
        self,
        _policy_name: str,
        policy: Callable[..., Any],
        retryable_exc: BaseException,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        flaky = _Flaky(fail_times=2, exc=retryable_exc, result="ok")
        wrapped = policy(flaky)

        with caplog.at_level(logging.WARNING, logger="chronoagent.retry"):
            assert wrapped() == "ok"

        warn_records = [
            r
            for r in caplog.records
            if r.name == "chronoagent.retry" and r.levelno == logging.WARNING
        ]
        # 3 attempts total -> 2 ``before_sleep`` warnings (one before
        # each retry; the successful third attempt is not logged).
        assert len(warn_records) == 2


# ---------------------------------------------------------------------------
# Policy-specific narrow-exception checks
# ---------------------------------------------------------------------------


class TestDbRetryDoesNotRetryIntegrityError:
    """``IntegrityError`` is a caller-bug (duplicate key, FK violation).

    Retrying it would hide the bug behind a distributed-systems veneer,
    so ``db_retry`` must propagate it on the first raise.
    """

    def test_integrity_error_propagates_on_first_raise(self) -> None:
        flaky = _Flaky(fail_times=5, exc=_make_integrity_error())
        wrapped = db_retry(flaky)

        with pytest.raises(sqlalchemy.exc.IntegrityError):
            wrapped()
        assert flaky.calls == 1


class TestLlmRetryCoversBothSubclasses:
    """``llm_retry`` must retry both transport and HTTP-status errors.

    Both are concrete ``httpx.HTTPError`` subclasses; in production they
    correspond to network failures (``ConnectError``, ``ReadTimeout``)
    and 5xx-style ``HTTPStatusError`` respectively.
    """

    def test_read_timeout_retries(self) -> None:
        flaky = _Flaky(fail_times=2, exc=httpx.ReadTimeout("slow"), result="ok")
        wrapped = llm_retry(flaky)

        assert wrapped() == "ok"
        assert flaky.calls == 3

    def test_http_status_error_retries(self) -> None:
        request = httpx.Request("POST", "https://example.invalid/x")
        response = httpx.Response(503, request=request)
        status_exc = httpx.HTTPStatusError(
            "server down",
            request=request,
            response=response,
        )
        flaky = _Flaky(fail_times=2, exc=status_exc, result="ok")
        wrapped = llm_retry(flaky)

        assert wrapped() == "ok"
        assert flaky.calls == 3


class TestRedisRetryCoversSubclasses:
    """``redis_retry`` must retry both connection and timeout errors."""

    def test_timeout_error_retries(self) -> None:
        flaky = _Flaky(
            fail_times=2,
            exc=redis.exceptions.TimeoutError("timed out"),
            result="ok",
        )
        wrapped = redis_retry(flaky)

        assert wrapped() == "ok"
        assert flaky.calls == 3


# ---------------------------------------------------------------------------
# Method decoration preserves self binding
# ---------------------------------------------------------------------------


class TestPolicyDecoratorOnInstanceMethod:
    """The policies must work when used as method decorators.

    All real call sites (Together/Ollama/MemoryStore/QuarantineStore/
    RedisBus/AuditTrailLogger/EscalationHandler) apply the decorator to
    instance methods, so we pin the self-binding behaviour here too.
    """

    def test_llm_retry_on_method_binds_self(self) -> None:
        class _Svc:
            def __init__(self) -> None:
                self.calls = 0

            @llm_retry
            def hit(self, arg: str) -> str:
                self.calls += 1
                if self.calls < 3:
                    raise httpx.ConnectError("boom")
                return f"ok:{arg}"

        svc = _Svc()
        assert svc.hit("x") == "ok:x"
        assert svc.calls == 3

    def test_db_retry_on_method_binds_self(self) -> None:
        class _Repo:
            def __init__(self) -> None:
                self.calls = 0

            @db_retry
            def query(self) -> int:
                self.calls += 1
                if self.calls < 3:
                    raise _make_operational_error()
                return 42

        repo = _Repo()
        assert repo.query() == 42
        assert repo.calls == 3
