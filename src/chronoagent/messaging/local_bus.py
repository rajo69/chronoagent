"""In-memory synchronous message bus for development and testing.

Handlers are called synchronously in the order they were registered.
Thread-safe via a ``threading.Lock``.
"""
from __future__ import annotations

import contextlib
import threading
from collections import defaultdict

from chronoagent.messaging.bus import MessageBus, MessageHandler


class LocalBus(MessageBus):
    """Simple in-process pub/sub message bus.

    Suitable for unit tests and local ``make dev`` runs where Redis is
    not available.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[MessageHandler]] = defaultdict(list)
        self._lock = threading.Lock()

    def publish(self, channel: str, message: object) -> None:
        """Deliver ``message`` to all handlers subscribed to ``channel``."""
        with self._lock:
            handlers = list(self._handlers.get(channel, []))
        for handler in handlers:
            handler(channel, message)

    def subscribe(self, channel: str, handler: MessageHandler) -> None:
        """Register ``handler`` to receive messages on ``channel``."""
        with self._lock:
            self._handlers[channel].append(handler)

    def unsubscribe(self, channel: str, handler: MessageHandler) -> None:
        """Remove ``handler`` from ``channel``.  No-op if not registered."""
        with self._lock:
            handlers = self._handlers.get(channel, [])
            with contextlib.suppress(ValueError):
                handlers.remove(handler)
