"""Abstract MessageBus interface for pub/sub messaging.

Concrete implementations:
- ``LocalBus``  — in-memory, sync, suitable for dev and tests
- ``RedisBus``  — Redis pub/sub, suitable for production
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

# Type alias for message handler callbacks.
MessageHandler = Callable[[str, Any], None]


class MessageBus(ABC):
    """Abstract pub/sub message bus.

    All implementations must be thread-safe for subscribe/publish.
    """

    @abstractmethod
    def publish(self, channel: str, message: Any) -> None:
        """Publish a message to a channel.

        Parameters
        ----------
        channel:
            Channel name (e.g. ``"health_updates"``).
        message:
            Serialisable payload.
        """

    @abstractmethod
    def subscribe(self, channel: str, handler: MessageHandler) -> None:
        """Register a handler for messages on a channel.

        Parameters
        ----------
        channel:
            Channel name to subscribe to.
        handler:
            Callable invoked as ``handler(channel, message)`` on each message.
        """

    @abstractmethod
    def unsubscribe(self, channel: str, handler: MessageHandler) -> None:
        """Remove a previously registered handler.

        Parameters
        ----------
        channel:
            Channel name.
        handler:
            The exact handler instance that was passed to ``subscribe``.
        """
