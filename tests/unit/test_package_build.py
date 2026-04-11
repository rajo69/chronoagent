"""Packaging regression tests (Phase 9 task 9.6).

These tests encode three invariants that 9.6 locks in so a future change
cannot silently regress the wheel / sdist:

* The bundled dashboard HTML asset is reachable at runtime via the
  ``chronoagent.dashboard`` path constants.  This works for both dev
  checkouts (where the file lives inside the source tree) and installed
  wheels (where hatchling has bundled it under
  ``chronoagent/dashboard/static/index.html``).
* ``pyproject.toml`` declares the ``chronoagent`` console-script entry
  point at ``chronoagent.cli:app``.
* The ``[tool.hatch.build.targets.sdist]`` section excludes the local-only
  artefacts that default-mode hatchling would otherwise pick up
  (``.claude/``, ``.hypothesis/``, build / cache directories).  The sdist
  include list is a whitelist, so excluding is technically redundant for
  the listed patterns, but asserting it keeps the intent explicit and
  gives a clear error message if someone switches back to the default
  behaviour.

The tests deliberately do NOT shell out to ``py -m build``.  A real build
is slow (30+ s) and needs an isolated environment, so we rely on
``tests/unit/test_dashboard_router.py`` + ``test_health_endpoint.py`` for
the end-to-end "the handler serves the file" coverage and use these
tests purely for the packaging-side invariants.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import chronoagent
from chronoagent.dashboard import INDEX_HTML, STATIC_DIR

# Repo root is three levels up from this file: tests/unit/test_package_build.py
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
PYPROJECT: Path = REPO_ROOT / "pyproject.toml"


# ── Bundled dashboard asset ──────────────────────────────────────────────────


class TestBundledDashboardAsset:
    """The dashboard HTML resolves via package-relative paths.

    Works for both ``pip install -e .`` and a built wheel because
    ``chronoagent/dashboard/__init__.py`` uses ``Path(__file__).parent``
    to locate the ``static/`` directory.
    """

    def test_static_dir_resolves(self) -> None:
        assert STATIC_DIR.exists(), f"STATIC_DIR does not exist: {STATIC_DIR}"
        assert STATIC_DIR.is_dir()

    def test_index_html_resolves(self) -> None:
        assert INDEX_HTML.exists(), f"INDEX_HTML does not exist: {INDEX_HTML}"
        assert INDEX_HTML.is_file()

    def test_index_html_has_dashboard_landmark(self) -> None:
        """Smoke-check that the bundled file is the dashboard page, not a stub."""
        text = INDEX_HTML.read_text(encoding="utf-8")
        assert "ChronoAgent" in text
        # ``test_dashboard_router.py::TestDashboardIndex`` already locks in the
        # full landmark set; this one assertion is enough to catch "file was
        # truncated during packaging" regressions.
        assert len(text) > 1000

    def test_index_html_lives_inside_chronoagent_package(self) -> None:
        """``INDEX_HTML`` must sit under the ``chronoagent`` package root.

        If someone refactors the asset out of ``src/chronoagent/dashboard/``
        (e.g. moves it to a top-level ``static/`` folder) the wheel will no
        longer bundle it under hatchling's default ``packages`` rule.  This
        test fails loudly before the wheel ships broken.
        """
        package_root = Path(chronoagent.__file__).resolve().parent
        assert package_root in INDEX_HTML.resolve().parents


# ── pyproject.toml invariants ────────────────────────────────────────────────


class TestPyprojectPackagingInvariants:
    """Lock in the 9.6 hatchling configuration."""

    @pytest.fixture()
    def pyproject_data(self) -> dict[str, object]:
        with PYPROJECT.open("rb") as fh:
            return tomllib.load(fh)

    def test_hatchling_is_pinned_with_minimum(self, pyproject_data: dict[str, object]) -> None:
        """``build-system.requires`` pins a hatchling minimum version."""
        build_system = pyproject_data["build-system"]
        assert isinstance(build_system, dict)
        requires = build_system["requires"]
        assert any(isinstance(r, str) and r.startswith("hatchling>=") for r in requires), (
            f"Expected pinned hatchling>=X.Y.Z, got {requires!r}"
        )

    def test_console_script_entry_point(self, pyproject_data: dict[str, object]) -> None:
        """``chronoagent`` console script points at ``chronoagent.cli:app``."""
        project = pyproject_data["project"]
        assert isinstance(project, dict)
        scripts = project["scripts"]
        assert isinstance(scripts, dict)
        assert scripts.get("chronoagent") == "chronoagent.cli:app"

    def test_wheel_packages_listed(self, pyproject_data: dict[str, object]) -> None:
        """``[tool.hatch.build.targets.wheel]`` lists ``src/chronoagent``."""
        tool = pyproject_data["tool"]
        assert isinstance(tool, dict)
        wheel_cfg = tool["hatch"]["build"]["targets"]["wheel"]  # type: ignore[index]
        assert "src/chronoagent" in wheel_cfg["packages"]  # type: ignore[index,operator]

    def test_sdist_include_list_present(self, pyproject_data: dict[str, object]) -> None:
        """The sdist target has an explicit whitelist, not the default glob."""
        tool = pyproject_data["tool"]
        sdist_cfg = tool["hatch"]["build"]["targets"]["sdist"]  # type: ignore[index]
        include = sdist_cfg["include"]  # type: ignore[index]
        assert isinstance(include, list)
        # Every include entry must be anchored to the repo root so hatchling
        # does not walk into stray subdirectories that share the name.
        for entry in include:
            assert isinstance(entry, str)
            assert entry.startswith("/"), f"sdist include entry not anchored: {entry!r}"
        # Core source and tests must be in the whitelist.
        assert "/src" in include
        assert "/tests" in include
        assert "/pyproject.toml" in include

    def test_sdist_exclude_list_covers_local_artefacts(
        self, pyproject_data: dict[str, object]
    ) -> None:
        """Local-only directories are explicitly excluded from the sdist."""
        tool = pyproject_data["tool"]
        sdist_cfg = tool["hatch"]["build"]["targets"]["sdist"]  # type: ignore[index]
        exclude = sdist_cfg["exclude"]  # type: ignore[index]
        assert isinstance(exclude, list)
        required = {
            "**/__pycache__",
            "**/*.py[cod]",
            "**/.pytest_cache",
            "**/.mypy_cache",
            "**/.ruff_cache",
            "**/.hypothesis",
        }
        missing = required - set(exclude)
        assert not missing, f"sdist exclude list missing patterns: {missing}"


# ── Version string ───────────────────────────────────────────────────────────


class TestVersionStringMatches:
    """``chronoagent.__version__`` matches ``pyproject.toml::project.version``."""

    def test_versions_agree(self) -> None:
        with PYPROJECT.open("rb") as fh:
            data = tomllib.load(fh)
        project = data["project"]
        assert isinstance(project, dict)
        pyproject_version = project["version"]
        assert pyproject_version == chronoagent.__version__, (
            f"pyproject.toml says {pyproject_version!r} but "
            f"chronoagent.__version__ is {chronoagent.__version__!r}"
        )
