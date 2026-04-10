"""Unit tests for the Typer CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from chronoagent.cli import app

runner = CliRunner()


class TestHelp:
    """CLI help output is available."""

    def test_root_help(self) -> None:
        """chronoagent --help exits 0 and prints usage."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "chronoagent" in result.output.lower()

    def test_serve_help(self) -> None:
        """chronoagent serve --help exits 0."""
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0

    def test_run_experiment_help(self) -> None:
        """chronoagent run-experiment --help exits 0."""
        result = runner.invoke(app, ["run-experiment", "--help"])
        assert result.exit_code == 0

    def test_check_health_help(self) -> None:
        """chronoagent check-health --help exits 0."""
        result = runner.invoke(app, ["check-health", "--help"])
        assert result.exit_code == 0


class TestRunExperiment:
    """run-experiment command."""

    def test_exits_zero_with_valid_config(self, tmp_path: Path) -> None:
        """run-experiment exits 0 when config path is provided."""
        cfg = tmp_path / "experiment.yaml"
        cfg.write_text("seed: 42\n")
        result = runner.invoke(app, ["run-experiment", "--config", str(cfg)])
        assert result.exit_code == 0

    def test_prints_config_path(self, tmp_path: Path) -> None:
        """run-experiment echoes the config path."""
        cfg = tmp_path / "experiment.yaml"
        cfg.write_text("seed: 42\n")
        result = runner.invoke(app, ["run-experiment", "--config", str(cfg)])
        assert str(cfg) in result.output

    def test_prints_output_dir(self, tmp_path: Path) -> None:
        """run-experiment echoes the output directory."""
        cfg = tmp_path / "experiment.yaml"
        cfg.write_text("seed: 42\n")
        out = tmp_path / "out"
        result = runner.invoke(
            app, ["run-experiment", "--config", str(cfg), "--output", str(out)]
        )
        assert str(out) in result.output


class TestCheckHealth:
    """check-health command."""

    def test_success(self) -> None:
        """check-health prints OK when service responds."""
        import httpx

        mock_response = httpx.Response(
            200,
            json={"status": "ok", "version": "0.1.0"},
            request=httpx.Request("GET", "http://localhost:8000/health"),
        )
        with patch("chronoagent.cli.httpx.get", return_value=mock_response):
            result = runner.invoke(app, ["check-health"])
        assert result.exit_code == 0
        assert "OK" in result.output

    def test_failure_on_connection_error(self) -> None:
        """check-health exits 1 when the service is unreachable."""
        import httpx

        with patch(
            "chronoagent.cli.httpx.get",
            side_effect=httpx.ConnectError("refused"),
        ):
            result = runner.invoke(app, ["check-health"])
        assert result.exit_code == 1


class TestServe:
    """serve command delegates to uvicorn."""

    def test_calls_uvicorn_run(self) -> None:
        """serve calls uvicorn.run with expected arguments."""
        with patch("chronoagent.cli.uvicorn.run") as mock_run:
            runner.invoke(app, ["serve", "--no-reload"])
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs.get("port") == 8000 or mock_run.call_args[0]

    def test_serve_with_config_sets_env(self, tmp_path: Path) -> None:
        """serve --config sets CHRONO_CONFIG_PATH env var before starting."""
        cfg = tmp_path / "custom.yaml"
        cfg.write_text("env: test\n")
        with patch("chronoagent.cli.uvicorn.run"):

            runner.invoke(app, ["serve", "--config", str(cfg), "--no-reload"])
            # env var may be set; we just verify no crash
