"""Unit tests for Phase 9 task 9.1 rate-limit middleware.

Covers:
- ``RateLimitConfig``: default values match the PLAN (POST 10/min,
  GET 60/min, WS 5 concurrent) and include ``/health``/``/metrics``
  on the exempt list.
- ``RateLimitMiddleware`` HTTP path: per-method budgets,
  exempt-path bypass, fixed-window rollover via an injected clock,
  ``X-RateLimit-*`` response headers on allowed requests, 429 body
  shape and ``Retry-After`` header on rejected requests, non-(GET|POST)
  methods pass through.
- ``RateLimitMiddleware`` per-client isolation: two distinct client
  IPs draw from independent budgets, driven via crafted ASGI scopes.
- ``RateLimitMiddleware`` WebSocket path: concurrent cap enforced,
  slot released on disconnect, exempt paths bypass the cap.
- ``create_app`` wiring: the middleware is installed by default and
  a custom clock can be injected at construction time.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import ExitStack
from typing import Any

import pytest
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.testclient import TestClient

from chronoagent.api.middleware import (
    RateLimitConfig,
    RateLimitMiddleware,
    _client_ip,
    _seconds_until_next_bucket,
)
from chronoagent.config import Settings
from chronoagent.main import create_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Clock:
    """Mutable monotonic stub the middleware can read on every call.

    Tests advance ``self.t`` to force fixed-window bucket transitions
    without relying on real wall-clock progress.
    """

    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, delta: float) -> None:
        self.t += delta


async def _null_app(scope: Any, receive: Any, send: Any) -> None:
    """Empty ASGI app used when the middleware's own state is under test."""
    return None


def _build_mini_app(
    config: RateLimitConfig | None = None,
    *,
    clock: _Clock | None = None,
) -> FastAPI:
    """Build a minimal FastAPI app wrapped in :class:`RateLimitMiddleware`.

    The stock ``chronoagent`` app pulls in pipelines, ChromaDB, and
    the real lifespan.  For middleware-focused tests we want none of
    that: a few toy routes and a toy WebSocket give us full coverage
    of the middleware's branches without any cross-module coupling.
    """
    app = FastAPI()
    app.add_middleware(
        RateLimitMiddleware,
        config=config,
        clock=clock,
    )

    @app.get("/ping")
    def ping() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/submit")
    def submit() -> dict[str, bool]:
        return {"ok": True}

    @app.put("/put")
    def put_handler() -> dict[str, bool]:
        return {"ok": True}

    @app.delete("/delete")
    def delete_handler() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/metrics")
    def metrics() -> dict[str, bool]:
        return {"ok": True}

    @app.websocket("/ws")
    async def ws_echo(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                msg = await websocket.receive_text()
                await websocket.send_text(msg)
        except WebSocketDisconnect:
            return

    @app.websocket("/metrics/ws")
    async def ws_exempt(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                msg = await websocket.receive_text()
                await websocket.send_text(msg)
        except WebSocketDisconnect:
            return

    return app


# ---------------------------------------------------------------------------
# RateLimitConfig defaults
# ---------------------------------------------------------------------------


class TestRateLimitConfigDefaults:
    """The PLAN pins explicit numbers we do not want to drift."""

    def test_post_default_is_ten_per_minute(self) -> None:
        assert RateLimitConfig().post_per_minute == 10

    def test_get_default_is_sixty_per_minute(self) -> None:
        assert RateLimitConfig().get_per_minute == 60

    def test_ws_default_is_five_concurrent(self) -> None:
        assert RateLimitConfig().ws_max_concurrent == 5

    def test_exempt_paths_include_health_and_metrics(self) -> None:
        exempt = RateLimitConfig().exempt_paths
        assert "/health" in exempt
        assert "/metrics" in exempt


# ---------------------------------------------------------------------------
# HTTP budgets
# ---------------------------------------------------------------------------


class TestHttpGetBudget:
    """Per-client GET budget behaviour."""

    def test_get_within_limit_all_succeed(self) -> None:
        config = RateLimitConfig(post_per_minute=10, get_per_minute=5)
        app = _build_mini_app(config=config)
        with TestClient(app) as client:
            for _ in range(5):
                assert client.get("/ping").status_code == 200

    def test_get_over_limit_is_429(self) -> None:
        config = RateLimitConfig(post_per_minute=10, get_per_minute=3)
        app = _build_mini_app(config=config)
        with TestClient(app) as client:
            for _ in range(3):
                assert client.get("/ping").status_code == 200
            over = client.get("/ping")
        assert over.status_code == 429
        body = over.json()
        assert body["detail"] == "rate_limit_exceeded"
        assert body["limit"] == 3
        assert body["retry_after_seconds"] >= 1

    def test_429_response_has_retry_after_header(self) -> None:
        config = RateLimitConfig(post_per_minute=10, get_per_minute=1)
        app = _build_mini_app(config=config)
        with TestClient(app) as client:
            client.get("/ping")
            over = client.get("/ping")
        assert int(over.headers["Retry-After"]) >= 1
        assert over.headers["X-RateLimit-Limit"] == "1"
        assert over.headers["X-RateLimit-Remaining"] == "0"

    def test_allowed_response_has_ratelimit_headers(self) -> None:
        config = RateLimitConfig(post_per_minute=10, get_per_minute=5)
        app = _build_mini_app(config=config)
        with TestClient(app) as client:
            response = client.get("/ping")
        assert response.status_code == 200
        assert response.headers["x-ratelimit-limit"] == "5"
        assert response.headers["x-ratelimit-remaining"] == "4"
        assert int(response.headers["x-ratelimit-reset"]) >= 1


class TestHttpPostBudget:
    """Per-client POST budget behaviour."""

    def test_post_within_limit_all_succeed(self) -> None:
        config = RateLimitConfig(post_per_minute=3, get_per_minute=60)
        app = _build_mini_app(config=config)
        with TestClient(app) as client:
            for _ in range(3):
                assert client.post("/submit").status_code == 200

    def test_post_over_limit_is_429(self) -> None:
        config = RateLimitConfig(post_per_minute=2, get_per_minute=60)
        app = _build_mini_app(config=config)
        with TestClient(app) as client:
            assert client.post("/submit").status_code == 200
            assert client.post("/submit").status_code == 200
            over = client.post("/submit")
        assert over.status_code == 429
        assert over.json()["limit"] == 2


class TestPostAndGetBucketsAreIndependent:
    """GET and POST budgets must not cross-contaminate."""

    def test_get_not_affected_by_post_exhaustion(self) -> None:
        config = RateLimitConfig(post_per_minute=2, get_per_minute=2)
        app = _build_mini_app(config=config)
        with TestClient(app) as client:
            client.post("/submit")
            client.post("/submit")
            assert client.post("/submit").status_code == 429
            assert client.get("/ping").status_code == 200
            assert client.get("/ping").status_code == 200
            assert client.get("/ping").status_code == 429


class TestExemptPaths:
    """``/health`` and ``/metrics`` must never be throttled."""

    def test_health_bypasses_limit(self) -> None:
        config = RateLimitConfig(post_per_minute=1, get_per_minute=1)
        app = _build_mini_app(config=config)
        with TestClient(app) as client:
            assert client.get("/ping").status_code == 200
            assert client.get("/ping").status_code == 429
            for _ in range(10):
                assert client.get("/health").status_code == 200

    def test_metrics_bypasses_limit(self) -> None:
        config = RateLimitConfig(post_per_minute=1, get_per_minute=1)
        app = _build_mini_app(config=config)
        with TestClient(app) as client:
            assert client.get("/ping").status_code == 200
            assert client.get("/ping").status_code == 429
            for _ in range(10):
                assert client.get("/metrics").status_code == 200

    def test_exempt_subpath_also_bypasses(self) -> None:
        """``/metrics/foo`` is exempt because ``/metrics`` is."""
        config = RateLimitConfig(post_per_minute=1, get_per_minute=1)
        app = _build_mini_app(config=config)

        # Add a toy subpath under /metrics to prove the prefix match.
        @app.get("/metrics/sub")
        def sub() -> dict[str, bool]:
            return {"ok": True}

        with TestClient(app) as client:
            for _ in range(10):
                assert client.get("/metrics/sub").status_code == 200

    def test_exempt_prefix_does_not_match_arbitrary_substring(self) -> None:
        """``/healthcheck`` is NOT exempt just because ``/health`` is."""
        config = RateLimitConfig(post_per_minute=10, get_per_minute=1)
        app = _build_mini_app(config=config)

        @app.get("/healthcheck")
        def not_exempt() -> dict[str, bool]:
            return {"ok": True}

        with TestClient(app) as client:
            assert client.get("/healthcheck").status_code == 200
            assert client.get("/healthcheck").status_code == 429


class TestBucketRollover:
    """Fixed-window rollover must drop old counters exactly on the minute."""

    def test_rollover_allows_new_requests(self) -> None:
        clock = _Clock(start=0.0)
        config = RateLimitConfig(post_per_minute=10, get_per_minute=1)
        app = _build_mini_app(config=config, clock=clock)
        with TestClient(app) as client:
            assert client.get("/ping").status_code == 200
            assert client.get("/ping").status_code == 429
            clock.advance(61.0)
            assert client.get("/ping").status_code == 200

    def test_rollover_prunes_stale_counters(self) -> None:
        clock = _Clock(start=0.0)
        config = RateLimitConfig(post_per_minute=10, get_per_minute=2)
        middleware = RateLimitMiddleware(_null_app, config=config, clock=clock)
        # Seed counters for bucket 0 and bucket 5; after a prune at bucket 10
        # both should be gone (both are < 10 - 1 = 9).
        middleware._http_counts[("1.2.3.4", "GET", 0)] = 2
        middleware._http_counts[("5.6.7.8", "GET", 5)] = 1
        middleware._prune_http(current_bucket=10)
        assert middleware._http_counts == {}

    def test_rollover_keeps_previous_bucket(self) -> None:
        """Adjacent-bucket counters survive prune so in-flight requests
        near a minute boundary are not double-counted or lost."""
        clock = _Clock(start=0.0)
        config = RateLimitConfig()
        middleware = RateLimitMiddleware(_null_app, config=config, clock=clock)
        middleware._http_counts[("1.2.3.4", "GET", 9)] = 3  # previous
        middleware._http_counts[("1.2.3.4", "GET", 10)] = 1  # current
        middleware._prune_http(current_bucket=10)
        assert ("1.2.3.4", "GET", 9) in middleware._http_counts
        assert ("1.2.3.4", "GET", 10) in middleware._http_counts


class TestNonGetPostMethodsPassThrough:
    """PUT/DELETE/PATCH/OPTIONS are out of scope for Phase 9.1."""

    def test_put_not_counted(self) -> None:
        config = RateLimitConfig(post_per_minute=1, get_per_minute=1)
        app = _build_mini_app(config=config)
        with TestClient(app) as client:
            for _ in range(5):
                assert client.put("/put").status_code == 200

    def test_delete_not_counted(self) -> None:
        config = RateLimitConfig(post_per_minute=1, get_per_minute=1)
        app = _build_mini_app(config=config)
        with TestClient(app) as client:
            for _ in range(5):
                assert client.delete("/delete").status_code == 200


# ---------------------------------------------------------------------------
# Per-client isolation (crafted ASGI scopes)
# ---------------------------------------------------------------------------


class TestPerClientIsolation:
    """Two distinct remote IPs must get independent HTTP budgets."""

    async def test_distinct_clients_have_independent_buckets(self) -> None:
        clock = _Clock(start=0.0)
        config = RateLimitConfig(post_per_minute=2, get_per_minute=60)

        async def downstream(scope: Any, receive: Any, send: Any) -> None:
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [],
                }
            )
            await send({"type": "http.response.body", "body": b"", "more_body": False})

        middleware = RateLimitMiddleware(downstream, config=config, clock=clock)

        async def call(ip: str) -> int:
            scope: dict[str, Any] = {
                "type": "http",
                "method": "POST",
                "path": "/submit",
                "client": (ip, 50000),
                "headers": [],
            }
            sent_messages: list[dict[str, Any]] = []

            async def receive() -> dict[str, Any]:
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(message: dict[str, Any]) -> None:
                sent_messages.append(message)

            await middleware(scope, receive, send)
            start = next(m for m in sent_messages if m["type"] == "http.response.start")
            status: int = start["status"]
            return status

        # Exhaust client A.
        assert await call("10.0.0.1") == 200
        assert await call("10.0.0.1") == 200
        assert await call("10.0.0.1") == 429
        # Client B has a fresh budget.
        assert await call("10.0.0.2") == 200
        assert await call("10.0.0.2") == 200
        assert await call("10.0.0.2") == 429


# ---------------------------------------------------------------------------
# WebSocket concurrency
# ---------------------------------------------------------------------------


class TestWebSocketConcurrency:
    """Global concurrency cap behaviour."""

    def test_ws_under_limit_all_accepted(self) -> None:
        config = RateLimitConfig(ws_max_concurrent=3)
        app = _build_mini_app(config=config)
        with TestClient(app) as client, ExitStack() as stack:
            sockets = [stack.enter_context(client.websocket_connect("/ws")) for _ in range(3)]
            for ws in sockets:
                ws.send_text("hi")
                assert ws.receive_text() == "hi"

    def test_ws_over_limit_rejected_with_1008(self) -> None:
        config = RateLimitConfig(ws_max_concurrent=2)
        app = _build_mini_app(config=config)
        with TestClient(app) as client, ExitStack() as stack:
            for _ in range(2):
                stack.enter_context(client.websocket_connect("/ws"))
            with (
                pytest.raises(WebSocketDisconnect) as exc_info,
                client.websocket_connect("/ws") as ws_extra,
            ):
                # The middleware sends websocket.close before accept;
                # receive_text raises WebSocketDisconnect on the client.
                ws_extra.receive_text()
            assert exc_info.value.code == 1008

    def test_ws_slot_freed_after_disconnect(self) -> None:
        config = RateLimitConfig(ws_max_concurrent=2)
        app = _build_mini_app(config=config)
        with TestClient(app) as client:
            with (
                client.websocket_connect("/ws") as ws1,
                client.websocket_connect("/ws") as ws2,
            ):
                ws1.send_text("a")
                assert ws1.receive_text() == "a"
                ws2.send_text("b")
                assert ws2.receive_text() == "b"
            # Both closed; a fresh connection should claim a freed slot.
            with client.websocket_connect("/ws") as ws3:
                ws3.send_text("c")
                assert ws3.receive_text() == "c"

    def test_ws_exempt_path_bypasses_limit(self) -> None:
        """A WebSocket under an exempt prefix does not count against the cap."""
        config = RateLimitConfig(ws_max_concurrent=1)
        app = _build_mini_app(config=config)
        with (
            TestClient(app) as client,
            # Exhaust the single slot on /ws.
            client.websocket_connect("/ws") as blocker,
        ):
            blocker.send_text("ping")
            assert blocker.receive_text() == "ping"
            # Exempt ws still opens.
            with client.websocket_connect("/metrics/ws") as exempt_ws:
                exempt_ws.send_text("hi")
                assert exempt_ws.receive_text() == "hi"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class TestClientIpExtraction:
    """``_client_ip`` must degrade gracefully for odd ASGI scopes."""

    def test_tuple_client(self) -> None:
        assert _client_ip({"client": ("1.2.3.4", 12345)}) == "1.2.3.4"

    def test_list_client(self) -> None:
        assert _client_ip({"client": ["5.6.7.8", 50000]}) == "5.6.7.8"

    def test_missing_client(self) -> None:
        assert _client_ip({}) == "unknown"

    def test_empty_tuple_client(self) -> None:
        assert _client_ip({"client": ()}) == "unknown"

    def test_non_string_host(self) -> None:
        assert _client_ip({"client": (None, 50000)}) == "unknown"


class TestSecondsUntilNextBucket:
    """Retry-After seconds must always be >=1 and strictly bounded."""

    def test_midway_through_bucket(self) -> None:
        assert _seconds_until_next_bucket(30.0) == 30

    def test_at_bucket_boundary(self) -> None:
        # now % 60 == 0 → remainder == 60.0 → int(60.0) == 60 (floor).
        assert _seconds_until_next_bucket(0.0) == 60

    def test_late_in_bucket_clamped_to_one(self) -> None:
        # now = 59.5 → remainder = 0.5 → clamped to 1.
        assert _seconds_until_next_bucket(59.5) == 1


# ---------------------------------------------------------------------------
# create_app() wiring
# ---------------------------------------------------------------------------


@pytest.fixture()
def chrono_client() -> Generator[TestClient, None, None]:
    """Real ChronoAgent app with a small POST budget for targeted tests.

    Using a very small budget keeps the test fast: we only need to
    prove that the middleware is wired into the production app factory.
    """
    settings = Settings(env="test", llm_backend="mock")
    config = RateLimitConfig(
        post_per_minute=2,
        get_per_minute=60,
        ws_max_concurrent=5,
    )
    app = create_app(settings=settings, rate_limit_config=config)
    with TestClient(app) as client:
        yield client


class TestCreateAppWiring:
    """Minimum proof that ``create_app`` actually installs the middleware."""

    def test_post_review_subject_to_rate_limit(self, chrono_client: TestClient) -> None:
        payload = {
            "pr_id": "rate_limit_smoke",
            "title": "Smoke test",
            "description": "Tests that the real app enforces POST limits.",
            "diff": "+ noop\n",
            "files_changed": ["noop.py"],
        }
        # Budget is 2.  The third POST (with a different pr_id so the
        # handler does not reject duplicates) should land on 429.
        payload_1 = dict(payload, pr_id="rate_limit_smoke_1")
        payload_2 = dict(payload, pr_id="rate_limit_smoke_2")
        payload_3 = dict(payload, pr_id="rate_limit_smoke_3")
        assert chrono_client.post("/api/v1/review", json=payload_1).status_code == 201
        assert chrono_client.post("/api/v1/review", json=payload_2).status_code == 201
        over = chrono_client.post("/api/v1/review", json=payload_3)
        assert over.status_code == 429
        assert over.json()["detail"] == "rate_limit_exceeded"

    def test_health_endpoint_still_works_under_load(self, chrono_client: TestClient) -> None:
        # /health is on the default exempt list, so a flood of probes
        # must all return 200 regardless of the global budget.
        for _ in range(30):
            assert chrono_client.get("/health").status_code == 200

    def test_custom_clock_injected_through_create_app(self) -> None:
        clock = _Clock(start=0.0)
        settings = Settings(env="test", llm_backend="mock")
        config = RateLimitConfig(post_per_minute=10, get_per_minute=1)
        app = create_app(
            settings=settings,
            rate_limit_config=config,
            rate_limit_clock=clock,
        )
        with TestClient(app) as client:
            # Lifespan already ran; two GETs exhaust the GET=1 budget.
            assert client.get("/health").status_code == 200  # exempt, does not count
            first = client.get("/api/v1/agents/health")
            assert first.status_code == 200
            over = client.get("/api/v1/agents/health")
            assert over.status_code == 429
            # Rollover → fresh budget.
            clock.advance(61.0)
            after = client.get("/api/v1/agents/health")
            assert after.status_code == 200
