"""Redis pub/sub message bus for production use.

Messages are JSON-serialised before publishing and deserialised on receipt.
Subscription listeners run in a background thread managed by redis-py's
``PubSub.run_in_thread``.

Usage::

    bus = RedisBus(url="redis://localhost:6379/0")
    bus.subscribe("health_updates", my_handler)
    bus.publish("health_updates", {"agent_id": "planner", "health": 0.92})
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
from collections import defaultdict
from typing import Any

import redis

from chronoagent.messaging.bus import MessageBus, MessageHandler

logger = logging.getLogger(__name__)


class RedisBus(MessageBus):
    """Production Redis-backed pub/sub message bus.

    Parameters
    ----------
    url:
        Redis connection URL (e.g. ``"redis://localhost:6379/0"``).
    """

    def __init__(self, url: str = "redis://localhost:6379/0") -> None:
        self._client: redis.Redis[bytes] = redis.Redis.from_url(url)
        self._pubsub: redis.client.PubSub = self._client.pubsub(ignore_subscribe_messages=True)
        self._handlers: dict[str, list[MessageHandler]] = defaultdict(list)
        self._lock = threading.Lock()
        self._listener_thread: Any | None = None

    # ------------------------------------------------------------------
    # MessageBus interface
    # ------------------------------------------------------------------

    def publish(self, channel: str, message: object) -> None:
        """Serialise ``message`` as JSON and publish to ``channel``."""
        payload = json.dumps(message)
        self._client.publish(channel, payload)

    def subscribe(self, channel: str, handler: MessageHandler) -> None:
        """Register ``handler`` and ensure the Redis subscription is active."""
        with self._lock:
            first = channel not in self._handlers or not self._handlers[channel]
            self._handlers[channel].append(handler)

        if first:
            self._pubsub.subscribe(**{channel: self._dispatch})
            self._ensure_listener()

    def unsubscribe(self, channel: str, handler: MessageHandler) -> None:
        """Remove ``handler`` from ``channel``.  Unsubscribes Redis if empty."""
        with self._lock:
            handlers = self._handlers.get(channel, [])
            with contextlib.suppress(ValueError):
                handlers.remove(handler)
            if not handlers:
                self._pubsub.unsubscribe(channel)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _dispatch(self, raw: dict[str, Any]) -> None:
        """Deserialise a raw Redis message and fan out to local handlers."""
        channel_bytes = raw.get("channel", b"")
        channel = channel_bytes.decode() if isinstance(channel_bytes, bytes) else channel_bytes
        data_bytes = raw.get("data", b"")
        try:
            data: Any = json.loads(
                data_bytes.decode() if isinstance(data_bytes, bytes) else data_bytes
            )
        except (json.JSONDecodeError, AttributeError):
            logger.warning("redis_bus: could not decode message on %s", channel)
            return

        with self._lock:
            handlers = list(self._handlers.get(channel, []))
        for h in handlers:
            try:
                h(channel, data)
            except Exception:  # noqa: BLE001
                logger.exception("redis_bus: handler error on channel %s", channel)

    def _ensure_listener(self) -> None:
        """Start the background listener thread if it is not running."""
        if self._listener_thread is None or not self._listener_thread.is_alive():
            self._listener_thread = self._pubsub.run_in_thread(sleep_time=0.01, daemon=True)
