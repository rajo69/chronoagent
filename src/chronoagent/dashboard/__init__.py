"""ChronoAgent observability dashboard package.

Holds the static HTML/JS assets served at ``GET /dashboard/`` by
:mod:`chronoagent.api.routers.dashboard`.  The dashboard is a single
self-contained HTML file that calls the ``/dashboard/api/*`` REST
endpoints and subscribes to the ``/dashboard/ws/live`` WebSocket.

No build step; Chart.js is loaded from a CDN.
"""

from __future__ import annotations

from pathlib import Path

STATIC_DIR: Path = Path(__file__).resolve().parent / "static"
"""Absolute path to the dashboard static asset directory."""

INDEX_HTML: Path = STATIC_DIR / "index.html"
"""Absolute path to the dashboard single-page HTML entrypoint."""
