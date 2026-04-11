"""Repository-wide logging audit tests (Phase 9 task 9.5).

Task 9.5 is a code audit, not a behaviour change, so the tests lock in
three invariants that keep the project's structured logging consistent:

* Every ``src/`` module that owns a module-level ``logger`` must route
  through :func:`chronoagent.observability.logging.get_logger` rather
  than call ``logging.getLogger`` or ``structlog.get_logger`` directly.
  One exception is whitelisted: :mod:`chronoagent.retry` keeps a stdlib
  logger because tenacity's ``before_sleep_log`` hook requires a real
  :class:`logging.Logger` instance.  :mod:`chronoagent.observability.logging`
  is also exempt because it *defines* the wrapper.
* ``configure_logging`` returns cleanly for every declared runtime env
  (``dev``, ``test``, ``prod``) and installs exactly one handler on the
  root logger each time.
* The ``get_logger`` wrapper returns a structlog bound logger that
  accepts kwargs (``logger.info("event_name", key=value)``) and renders
  through the project's shared ``ProcessorFormatter`` without raising.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
import re
from pathlib import Path

import pytest

import chronoagent
from chronoagent.observability.logging import configure_logging, get_logger

# ── Directory discovery ─────────────────────────────────────────────────────

SRC_ROOT: Path = Path(chronoagent.__file__).parent

# Modules that are allowed to call ``logging.getLogger`` / ``structlog.get_logger``
# directly (the audit below skips these).  Paths are ``"/"``-joined relative to
# ``SRC_ROOT`` so the suite runs on both POSIX and Windows.
_ALLOWED_STDLIB_CALLERS: frozenset[str] = frozenset(
    {
        # Defines the ``get_logger`` wrapper and configures the stdlib root.
        "observability/logging.py",
        # Tenacity's ``before_sleep_log`` needs a real ``logging.Logger``.
        "retry.py",
    }
)


def _iter_py_files() -> list[Path]:
    """Return every ``.py`` file under ``src/chronoagent`` sorted for stability."""
    return sorted(p for p in SRC_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _relative(path: Path) -> str:
    """Return the path relative to ``SRC_ROOT`` with forward slashes."""
    return path.relative_to(SRC_ROOT).as_posix()


# ── No stray stdlib / bare-structlog loggers ────────────────────────────────


class TestNoStrayStdlibLoggers:
    """Every src/ module must acquire loggers via ``get_logger``."""

    _STDLIB_PATTERN = re.compile(r"logging\.getLogger\s*\(")
    _STRUCTLOG_PATTERN = re.compile(r"structlog\.get_logger\s*\(")

    def test_no_stdlib_get_logger_outside_whitelist(self) -> None:
        offenders: list[str] = []
        for path in _iter_py_files():
            rel = _relative(path)
            if rel in _ALLOWED_STDLIB_CALLERS:
                continue
            text = path.read_text(encoding="utf-8")
            if self._STDLIB_PATTERN.search(text):
                offenders.append(rel)
        assert not offenders, (
            "These modules still call logging.getLogger directly; route them "
            "through chronoagent.observability.logging.get_logger instead: "
            f"{offenders}"
        )

    def test_no_structlog_get_logger_calls(self) -> None:
        """Only ``observability/logging.py`` may call ``structlog.get_logger``."""
        offenders: list[str] = []
        for path in _iter_py_files():
            rel = _relative(path)
            if rel == "observability/logging.py":
                continue
            text = path.read_text(encoding="utf-8")
            if self._STRUCTLOG_PATTERN.search(text):
                offenders.append(rel)
        assert not offenders, (
            "These modules call structlog.get_logger directly; use "
            "chronoagent.observability.logging.get_logger instead: "
            f"{offenders}"
        )

    def test_retry_module_stays_on_stdlib(self) -> None:
        """Lock in the 9.2 / 9.5 decision: retry.py keeps ``logging.getLogger``."""
        retry_path = SRC_ROOT / "retry.py"
        text = retry_path.read_text(encoding="utf-8")
        assert self._STDLIB_PATTERN.search(text), (
            "retry.py must continue to use logging.getLogger for tenacity's before_sleep_log hook."
        )


# ── Every module imports through get_logger ─────────────────────────────────


class TestGetLoggerImport:
    """Every module that owns a ``logger`` attribute imports ``get_logger``."""

    def _modules_with_logger(self) -> list[str]:
        """Return dotted module names that define a module-level ``logger``."""
        names: list[str] = []
        for modinfo in pkgutil.walk_packages(chronoagent.__path__, prefix="chronoagent."):
            # Skip submodules that fail to import in the test env; the broader
            # test suite already catches import errors, we only care about
            # modules that load cleanly.
            try:
                module = importlib.import_module(modinfo.name)
            except Exception:  # noqa: BLE001 — best-effort discovery
                continue
            if hasattr(module, "logger"):
                names.append(modinfo.name)
        return names

    def test_all_loggers_are_structlog_backed(self) -> None:
        """Every module-level ``logger`` is a structlog bound logger.

        ``structlog.get_logger`` returns a ``BoundLoggerLazyProxy`` that
        delegates to a real bound logger on first use, so we accept any
        type coming out of :mod:`structlog` as long as it is not a plain
        :class:`logging.Logger`.  The whitelisted stdlib caller
        (:mod:`chronoagent.retry`) exposes its logger as the private
        name ``_logger``, which is skipped by
        :meth:`_modules_with_logger`, so it does not show up here.
        """
        mismatched: list[str] = []
        for name in self._modules_with_logger():
            module = importlib.import_module(name)
            logger_type = type(module.logger)
            module_path = getattr(logger_type, "__module__", "")
            if isinstance(module.logger, logging.Logger):
                mismatched.append(f"{name} -> stdlib {logger_type.__name__}")
            elif not module_path.startswith("structlog"):
                mismatched.append(f"{name} -> {module_path}.{logger_type.__name__}")
        assert not mismatched, (
            f"These modules have a non-structlog ``logger`` attribute: {mismatched}"
        )


# ── configure_logging across environments ───────────────────────────────────


class TestConfigureLoggingAllEnvs:
    """``configure_logging`` installs exactly one handler for every env."""

    @pytest.mark.parametrize("env", ["dev", "test", "prod"])
    def test_installs_single_handler(self, env: str) -> None:
        configure_logging(env)
        root = logging.getLogger()
        assert len(root.handlers) == 1

    @pytest.mark.parametrize("env", ["dev", "test", "prod"])
    def test_handler_has_processor_formatter(self, env: str) -> None:
        """The installed handler uses ``structlog.stdlib.ProcessorFormatter``."""
        import structlog

        configure_logging(env)
        root = logging.getLogger()
        formatter = root.handlers[0].formatter
        assert isinstance(formatter, structlog.stdlib.ProcessorFormatter)

    def test_idempotent_across_envs(self) -> None:
        """Switching envs does not stack handlers."""
        for env in ("dev", "test", "prod", "dev"):
            configure_logging(env)
        assert len(logging.getLogger().handlers) == 1


# ── get_logger returns a usable bound logger ────────────────────────────────


class TestGetLoggerRoundTrip:
    """``get_logger`` returns a structlog bound logger that accepts kwargs."""

    def test_returns_non_none(self) -> None:
        assert get_logger(__name__) is not None

    def test_accepts_kwargs_without_raising(self) -> None:
        """The idiomatic ``logger.info("event", key=value)`` call path works."""
        configure_logging("test")
        logger = get_logger("chronoagent.tests.logging_audit_probe")
        # Each level should accept structured kwargs.
        logger.debug("audit_probe_debug", component="audit", level="debug")
        logger.info("audit_probe_info", component="audit", level="info")
        logger.warning("audit_probe_warning", component="audit", level="warning")
        logger.error("audit_probe_error", component="audit", level="error")

    def test_exception_method_accepts_kwargs(self) -> None:
        """``.exception`` inside an except block renders with kwargs too."""
        configure_logging("test")
        logger = get_logger("chronoagent.tests.logging_audit_probe")
        try:
            raise RuntimeError("simulated")
        except RuntimeError:
            logger.exception("audit_probe_exception", component="audit")


# ── Field-name convention spot checks ───────────────────────────────────────


class TestStructuredFieldConvention:
    """A handful of migrated call sites still use kwargs, not %s interpolation."""

    _PERCENT_S_PATTERN = re.compile(r"logger\.\w+\([^)]*%[sdrf]")

    @pytest.mark.parametrize(
        "module_path",
        [
            "allocator/task_allocator.py",
            "scorer/health_scorer.py",
            "scorer/chronos_forecaster.py",
            "messaging/redis_bus.py",
        ],
    )
    def test_migrated_modules_have_no_percent_s_logger_calls(
        self,
        module_path: str,
    ) -> None:
        """Migrated modules must not contain ``logger.foo("... %s", x)`` calls."""
        text = (SRC_ROOT / module_path).read_text(encoding="utf-8")
        match = self._PERCENT_S_PATTERN.search(text)
        assert match is None, (
            f"{module_path} still contains a %-style logger call: "
            f"{match.group(0) if match else 'n/a'}"
        )
