"""HTTP + WebSocket rate limiting middleware (Phase 9 task 9.1).

This module ships :class:`RateLimitMiddleware`, a pure ASGI middleware
that enforces two complementary throttling policies:

1. **Per-client HTTP rate limits.** Each remote client IP gets a fixed
   budget of POST requests and GET requests per minute, evaluated on a
   fixed (wall-clock) one-minute window.  Any request beyond the budget
   is rejected with a ``429 Too Many Requests`` JSON response carrying a
   ``Retry-After`` header.
2. **Global WebSocket concurrency cap.** The number of simultaneously
   live WebSocket connections to the application is capped at
   ``config.ws_max_concurrent``.  New connections beyond the cap are
   rejected with ``websocket.close`` code ``1008`` (policy violation)
   *before* the downstream handler can accept them, so they never consume
   any application resources.

Design decisions
----------------

* **Pure ASGI, not ``BaseHTTPMiddleware``.** We need to intercept both
  HTTP and WebSocket scopes, which ``BaseHTTPMiddleware`` does not
  support.  A bare ``async __call__(scope, receive, send)`` middleware
  also avoids the request/response re-materialisation overhead that
  ``BaseHTTPMiddleware`` imposes on every HTTP call.
* **Fixed-window counters, not token buckets.** The PLAN specifies
  "N per minute", which maps cleanly onto ``int(now // 60)`` buckets.
  Fixed windows are trivially testable with an injectable clock and
  have no warm-up/refill semantics to get wrong.
* **``threading.Lock``, not ``asyncio.Lock``.** Starlette's
  ``TestClient`` spawns a fresh event loop per concurrent WebSocket
  portal, so an ``asyncio.Lock`` created lazily on one loop cannot be
  awaited from another.  The critical sections here are
  counter-increment only (never any I/O), so a plain threading lock
  is fast enough and correct across both single-loop production
  deployments and multi-loop test harnesses.
* **Injectable clock.** Tests drive the window rollover deterministically
  by passing a mutable clock callable.  The middleware uses
  :func:`time.monotonic` by default because wall-clock jumps (NTP step,
  timezone change) would otherwise cause spurious bucket transitions in
  production.
* **Exempt paths.** Ops probes (``/health``) and the Prometheus scrape
  endpoint (``/metrics``) bypass all limits unconditionally so a traffic
  flood cannot starve observability.

The middleware is wired into :func:`chronoagent.main.create_app` with
the defaults from :class:`RateLimitConfig`; both the config and the
clock can be overridden at app-construction time for tests.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from chronoagent.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RateLimitConfig:
    """Immutable configuration for :class:`RateLimitMiddleware`.

    Attributes:
        post_per_minute: Maximum allowed ``POST`` requests from a single
            client IP inside any fixed one-minute wall-clock window.
            Defaults to the PLAN's 10.
        get_per_minute: Maximum allowed ``GET`` requests from a single
            client IP inside any fixed one-minute wall-clock window.
            Defaults to the PLAN's 60.
        ws_max_concurrent: Maximum number of simultaneously live
            WebSocket connections across the entire application.
            Defaults to the PLAN's 5.
        exempt_paths: Tuple of path prefixes that bypass all limits.  A
            request matches when its path equals one of these or is a
            subpath (``path.startswith(prefix + "/")``).  Defaults to
            ``("/health", "/metrics")`` so ops cannot be starved.
    """

    post_per_minute: int = 10
    get_per_minute: int = 60
    ws_max_concurrent: int = 5
    exempt_paths: tuple[str, ...] = ("/health", "/metrics")


class RateLimitMiddleware:
    """Pure ASGI middleware enforcing HTTP and WebSocket rate limits.

    Instantiated once per application; Starlette's middleware stack
    calls ``__call__`` for every incoming connection.  HTTP scopes are
    routed through per-client fixed-window counters and WebSocket
    scopes through a global concurrency counter.  Anything else
    (``lifespan`` events, custom scope types) passes through untouched.

    Args:
        app: The downstream ASGI application.
        config: Rate-limit configuration.  Defaults to
            :class:`RateLimitConfig`\\ 's defaults when ``None``.
        clock: Callable returning the current time in seconds as a
            float.  Defaults to :func:`time.monotonic`.  Tests inject a
            mutable clock to drive bucket rollover deterministically.
    """

    def __init__(
        self,
        app: ASGIApp,
        config: RateLimitConfig | None = None,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.app = app
        self.config = config or RateLimitConfig()
        self._clock: Callable[[], float] = clock or time.monotonic
        self._http_counts: dict[tuple[str, str, int], int] = {}
        self._ws_active: int = 0
        self._lock = threading.Lock()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Dispatch by scope type.

        HTTP scopes go through :meth:`_handle_http`, WebSocket scopes
        through :meth:`_handle_ws`, and every other scope type
        (``lifespan``, custom) passes through to the wrapped app.
        """
        scope_type = scope.get("type")
        if scope_type == "http":
            await self._handle_http(scope, receive, send)
        elif scope_type == "websocket":
            await self._handle_ws(scope, receive, send)
        else:
            await self.app(scope, receive, send)

    # ------------------------------------------------------------------
    # HTTP path
    # ------------------------------------------------------------------

    async def _handle_http(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Enforce per-client per-method HTTP limits.

        Exempt paths and non-(GET|POST) methods pass through untouched.
        A request over its budget is answered directly with a 429 JSON
        payload and never reaches the downstream app.  Allowed requests
        have ``X-RateLimit-*`` headers injected into their response.
        """
        path = _path_of(scope)
        if self._is_exempt(path):
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method", "GET")).upper()
        if method == "POST":
            limit = self.config.post_per_minute
        elif method == "GET":
            limit = self.config.get_per_minute
        else:
            # HEAD/OPTIONS/PUT/DELETE/PATCH are out of scope for Phase 9.1.
            await self.app(scope, receive, send)
            return

        client_ip = _client_ip(scope)
        now = self._clock()
        bucket = int(now // 60)
        key = (client_ip, method, bucket)

        with self._lock:
            self._prune_http(bucket)
            count = self._http_counts.get(key, 0)
            if count >= limit:
                retry_after = _seconds_until_next_bucket(now)
                logger.warning(
                    "rate_limit_exceeded",
                    client=client_ip,
                    method=method,
                    path=path,
                    limit=limit,
                    retry_after=retry_after,
                )
                response = JSONResponse(
                    status_code=429,
                    content={
                        "detail": "rate_limit_exceeded",
                        "limit": limit,
                        "retry_after_seconds": retry_after,
                    },
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(retry_after),
                    },
                )
                await response(scope, receive, send)
                return
            self._http_counts[key] = count + 1
            remaining = limit - (count + 1)
            reset = _seconds_until_next_bucket(now)

        headers_to_inject: list[tuple[bytes, bytes]] = [
            (b"x-ratelimit-limit", str(limit).encode("latin-1")),
            (b"x-ratelimit-remaining", str(remaining).encode("latin-1")),
            (b"x-ratelimit-reset", str(reset).encode("latin-1")),
        ]

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                message = dict(message)
                existing = list(message.get("headers", []))
                existing.extend(headers_to_inject)
                message["headers"] = existing
            await send(message)

        await self.app(scope, receive, send_with_headers)

    def _prune_http(self, current_bucket: int) -> None:
        """Drop counter entries older than the previous minute.

        Called under ``self._lock`` from :meth:`_handle_http` before
        each counter check.  Keeping the previous bucket alongside the
        current one is cheap and lets the pruner stay strictly less
        than ``current_bucket - 1`` for simplicity.
        """
        stale = [k for k in self._http_counts if k[2] < current_bucket - 1]
        for k in stale:
            del self._http_counts[k]

    def _is_exempt(self, path: str) -> bool:
        """Return whether ``path`` is on the exempt list.

        A path is exempt if it exactly matches one of
        ``config.exempt_paths`` *or* is a subpath of one (e.g.
        ``/metrics/foo`` is exempt because ``/metrics`` is).
        """
        for prefix in self.config.exempt_paths:
            if path == prefix or path.startswith(prefix + "/"):
                return True
        return False

    # ------------------------------------------------------------------
    # WebSocket path
    # ------------------------------------------------------------------

    async def _handle_ws(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Enforce the global WebSocket concurrency cap.

        Exempt paths bypass the cap.  When the cap is reached the
        middleware drains the initial ``websocket.connect`` message and
        responds with ``websocket.close`` code ``1008``; the downstream
        app is never invoked, so the connection never consumes handler
        resources.  Otherwise the middleware increments the counter,
        delegates to the app, and decrements in a ``finally`` block so
        the slot is released on any exit path (normal close, client
        disconnect, or handler exception).
        """
        path = _path_of(scope)
        if self._is_exempt(path):
            await self.app(scope, receive, send)
            return

        with self._lock:
            if self._ws_active >= self.config.ws_max_concurrent:
                over_limit = True
            else:
                self._ws_active += 1
                over_limit = False

        if over_limit:
            logger.warning(
                "ws_rate_limit_exceeded",
                path=path,
                active=self.config.ws_max_concurrent,
                limit=self.config.ws_max_concurrent,
            )
            await self._reject_ws(receive, send)
            return

        try:
            await self.app(scope, receive, send)
        finally:
            with self._lock:
                self._ws_active -= 1

    async def _reject_ws(self, receive: Receive, send: Send) -> None:
        """Politely refuse an incoming WebSocket before the handshake.

        The ASGI WebSocket protocol requires the server to first read a
        ``websocket.connect`` message from the client; only then can it
        respond with either ``websocket.accept`` or ``websocket.close``.
        We read the connect message and immediately emit
        ``websocket.close`` with code ``1008`` (policy violation) and a
        human-readable reason.
        """
        try:
            message = await receive()
        except Exception:  # pragma: no cover - defensive; Starlette always sends it
            return
        if message.get("type") != "websocket.connect":  # pragma: no cover - defensive
            return
        await send(
            {
                "type": "websocket.close",
                "code": 1008,
                "reason": "rate_limit_exceeded",
            }
        )


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


def _path_of(scope: Scope) -> str:
    """Extract a request path from an ASGI scope as a plain string."""
    raw = scope.get("path", "")
    return raw if isinstance(raw, str) else ""


def _client_ip(scope: Scope) -> str:
    """Return the client IP from an ASGI scope, or ``"unknown"``.

    ASGI servers populate ``scope["client"]`` as an ``(host, port)``
    tuple when the connection has a remote peer.  A missing or
    malformed entry collapses to a single shared bucket named
    ``"unknown"``, which is a safe conservative default: anonymous
    sources share a single budget rather than each getting an
    unlimited one.
    """
    client = scope.get("client")
    if isinstance(client, (tuple, list)) and client:
        host = client[0]
        if isinstance(host, str) and host:
            return host
    return "unknown"


def _seconds_until_next_bucket(now: float) -> int:
    """Return the integer seconds remaining in the current minute bucket.

    Always at least ``1`` so clients honouring ``Retry-After`` wait
    long enough for the window to roll over, even when the middleware
    rejects the very last millisecond of a bucket.
    """
    remainder = 60.0 - (now % 60.0)
    return max(1, int(remainder)) if remainder > 0 else 60
