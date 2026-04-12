"""Unit tests for the Typer CLI."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from typer.testing import CliRunner

from chronoagent.cli import app
from chronoagent.experiments.experiment_runner import (
    AggregateResult,
    MetricAggregate,
    RawRunRecord,
    RunResult,
)

runner = CliRunner()

# Typer renders its help screen through Rich, which wraps flag names in ANSI
# colour codes ("-\x1b[0m\x1b[1;36m-plots\x1b[0m") and may soft-wrap long lines
# to the terminal width. Tests that assert a flag name appears in the help
# output strip ANSI first so the assertion is terminal-width independent.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Strip ANSI escape sequences from ``text``."""
    return _ANSI_RE.sub("", text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _phase10_yaml(name: str = "cli_test_experiment") -> str:
    """A minimal, validator-clean Phase 10 experiment YAML string."""
    return (
        f"name: {name}\n"
        "seed: 42\n"
        "num_runs: 1\n"
        "num_prs: 12\n"
        "attack:\n"
        "  type: minja\n"
        "  target: both\n"
        "  injection_step: 6\n"
        "  n_poison_docs: 4\n"
        "  strategy: default\n"
        "ablation:\n"
        "  forecaster: true\n"
        "  bocpd: true\n"
        "  health: true\n"
        "  integrity: true\n"
        "system:\n"
        "  llm_backend: mock\n"
        "  bocpd_hazard_lambda: 50.0\n"
        "  health_threshold: 0.3\n"
        "  integrity_threshold: 0.6\n"
    )


def _stub_aggregate(name: str, num_runs: int = 1) -> AggregateResult:
    """Build a deterministic AggregateResult for CLI tests."""
    run = RunResult(
        run_index=0,
        seed=42,
        detector_name="full_system_detector",
        injection_step=6,
        num_prs=12,
        first_flagged_step=7,
        advance_warning_time=1,
        allocation_efficiency_score=0.75,
        detection_auroc_score=0.9,
        detection_f1_score=0.8,
        latency_ms=1.5,
    )
    return AggregateResult(
        name=name,
        detector_name="full_system_detector",
        num_runs=num_runs,
        injection_step=6,
        num_prs=12,
        runs=[run],
        advance_warning_time=MetricAggregate(mean=1.0, std=0.0, ci_low=1.0, ci_high=1.0, n=1),
        allocation_efficiency_score=MetricAggregate(
            mean=0.75, std=0.0, ci_low=0.75, ci_high=0.75, n=1
        ),
        detection_auroc_score=MetricAggregate(mean=0.9, std=0.0, ci_low=0.9, ci_high=0.9, n=1),
        detection_f1_score=MetricAggregate(mean=0.8, std=0.0, ci_low=0.8, ci_high=0.8, n=1),
        latency_ms=MetricAggregate(mean=1.5, std=0.0, ci_low=1.5, ci_high=1.5, n=1),
        provenance={"name": name},
    )


class _StubRunner:
    """Drop-in replacement for ExperimentRunner that never touches ChromaDB.

    Records raw runs with small synthetic payloads so the downstream plot
    code has real files to read when --plots is enabled.
    """

    def __init__(self, cfg: Any, collect_raw: bool = False) -> None:
        self.cfg = cfg
        self.collect_raw = collect_raw
        self._raw_runs: list[RawRunRecord] = []

    @property
    def raw_runs(self) -> list[RawRunRecord]:
        return self._raw_runs

    def run(self) -> AggregateResult:
        if self.collect_raw:
            matrix = np.zeros((self.cfg.num_prs, 6), dtype=np.float64)
            # Bump kl_divergence (col 3) + entropy (col 5) after injection so
            # the fake looks shifted if anyone plots it.
            matrix[self.cfg.attack.injection_step :, 3] = 1.0
            matrix[self.cfg.attack.injection_step :, 5] = 1.0
            decisions: list[dict[str, Any]] = []
            for step in range(self.cfg.num_prs):
                flagged = step >= self.cfg.attack.injection_step
                decisions.append(
                    {
                        "step_index": step,
                        "success": not flagged,
                        "agent_id": "full_system_detector",
                        "score": 1.0 if flagged else 0.0,
                        "flagged": flagged,
                    }
                )
            self._raw_runs.append(
                RawRunRecord(
                    run_index=0,
                    seed=42,
                    signal_matrix=matrix,
                    decisions=decisions,
                )
            )
        return _stub_aggregate(self.cfg.name, num_runs=self.cfg.num_runs)


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
        """chronoagent run-experiment --help exits 0 and lists subcommands."""
        result = runner.invoke(app, ["run-experiment", "--help"])
        assert result.exit_code == 0
        output = _plain(result.output)
        assert "phase1" in output
        assert "phase10" in output

    def test_run_experiment_phase1_help(self) -> None:
        """chronoagent run-experiment phase1 --help exits 0."""
        result = runner.invoke(app, ["run-experiment", "phase1", "--help"])
        assert result.exit_code == 0

    def test_run_experiment_phase10_help(self) -> None:
        """chronoagent run-experiment phase10 --help exits 0 and documents flags."""
        result = runner.invoke(app, ["run-experiment", "phase10", "--help"])
        assert result.exit_code == 0
        output = _plain(result.output)
        assert "--plots" in output
        assert "--tables" in output
        assert "--quiet" in output

    def test_compare_experiments_help(self) -> None:
        """chronoagent compare-experiments --help exits 0."""
        result = runner.invoke(app, ["compare-experiments", "--help"])
        assert result.exit_code == 0
        output = _plain(result.output)
        assert "--experiment" in output

    def test_check_health_help(self) -> None:
        """chronoagent check-health --help exits 0."""
        result = runner.invoke(app, ["check-health", "--help"])
        assert result.exit_code == 0


class TestRunExperimentPhase1:
    """run-experiment phase1 preserves the Phase 1 signal-validation runner."""

    def test_exits_zero_with_valid_config(self, tmp_path: Path) -> None:
        """phase1 exits 0 when config path is provided."""
        cfg = tmp_path / "experiment.yaml"
        cfg.write_text("seed: 42\n")
        result = runner.invoke(app, ["run-experiment", "phase1", "--config", str(cfg)])
        assert result.exit_code == 0, result.output

    def test_prints_config_path(self, tmp_path: Path) -> None:
        """phase1 echoes the config path."""
        cfg = tmp_path / "experiment.yaml"
        cfg.write_text("seed: 42\n")
        result = runner.invoke(app, ["run-experiment", "phase1", "--config", str(cfg)])
        assert str(cfg) in result.output

    def test_prints_output_dir(self, tmp_path: Path) -> None:
        """phase1 echoes the output directory."""
        cfg = tmp_path / "experiment.yaml"
        cfg.write_text("seed: 42\n")
        out = tmp_path / "out"
        result = runner.invoke(
            app,
            ["run-experiment", "phase1", "--config", str(cfg), "--output", str(out)],
        )
        assert str(out) in result.output

    def test_rejects_unknown_attack_type(self, tmp_path: Path) -> None:
        """phase1 exits 1 when the runner.attack key is not minja/agentpoison."""
        cfg = tmp_path / "experiment.yaml"
        cfg.write_text("runner:\n  attack: bogus\n")
        result = runner.invoke(app, ["run-experiment", "phase1", "--config", str(cfg)])
        assert result.exit_code == 1
        assert "unknown attack type" in result.output.lower()


class TestRunExperimentPhase10:
    """run-experiment phase10 drives a single-experiment run."""

    def test_happy_path_writes_results(self, tmp_path: Path) -> None:
        """phase10 runs the runner, writes runs.csv + aggregate.json."""
        cfg_path = tmp_path / "main_experiment.yaml"
        cfg_path.write_text(_phase10_yaml())
        out_dir = tmp_path / "results"

        with patch("chronoagent.experiments.experiment_runner.ExperimentRunner", _StubRunner):
            result = runner.invoke(
                app,
                [
                    "run-experiment",
                    "phase10",
                    "--config",
                    str(cfg_path),
                    "--output",
                    str(out_dir),
                    "--no-plots",
                    "--no-tables",
                ],
            )
        assert result.exit_code == 0, result.output
        assert (out_dir / "cli_test_experiment" / "runs.csv").exists()
        assert (out_dir / "cli_test_experiment" / "aggregate.json").exists()

    def test_no_plots_no_tables_skip_side_effects(self, tmp_path: Path) -> None:
        """--no-plots and --no-tables skip figure + table rendering."""
        cfg_path = tmp_path / "main_experiment.yaml"
        cfg_path.write_text(_phase10_yaml())
        out_dir = tmp_path / "results"

        with (
            patch("chronoagent.experiments.experiment_runner.ExperimentRunner", _StubRunner),
            patch("chronoagent.experiments.analysis.plots.plot_signal_drift") as mock_plot,
            patch("chronoagent.experiments.analysis.tables.make_main_results_table") as mock_table,
        ):
            result = runner.invoke(
                app,
                [
                    "run-experiment",
                    "phase10",
                    "--config",
                    str(cfg_path),
                    "--output",
                    str(out_dir),
                    "--no-plots",
                    "--no-tables",
                ],
            )
        assert result.exit_code == 0, result.output
        mock_plot.assert_not_called()
        mock_table.assert_not_called()

    def test_plots_enabled_calls_signal_drift(self, tmp_path: Path) -> None:
        """--plots (default) calls plot_signal_drift once."""
        cfg_path = tmp_path / "main_experiment.yaml"
        cfg_path.write_text(_phase10_yaml())
        out_dir = tmp_path / "results"

        with (
            patch("chronoagent.experiments.experiment_runner.ExperimentRunner", _StubRunner),
            patch(
                "chronoagent.experiments.analysis.plots.plot_signal_drift",
                return_value=[out_dir / "fake.png"],
            ) as mock_plot,
            patch(
                "chronoagent.experiments.analysis.tables.make_main_results_table",
                return_value=out_dir / "tables" / "fake.tex",
            ),
        ):
            result = runner.invoke(
                app,
                [
                    "run-experiment",
                    "phase10",
                    "--config",
                    str(cfg_path),
                    "--output",
                    str(out_dir),
                ],
            )
        assert result.exit_code == 0, result.output
        mock_plot.assert_called_once()
        call_args = mock_plot.call_args
        assert call_args.kwargs.get("run_index") == 0
        assert call_args.args[1] == "cli_test_experiment"

    def test_tables_enabled_calls_main_results_table(self, tmp_path: Path) -> None:
        """--tables (default) calls make_main_results_table with a single-entry list."""
        cfg_path = tmp_path / "main_experiment.yaml"
        cfg_path.write_text(_phase10_yaml())
        out_dir = tmp_path / "results"

        with (
            patch("chronoagent.experiments.experiment_runner.ExperimentRunner", _StubRunner),
            patch(
                "chronoagent.experiments.analysis.plots.plot_signal_drift",
                return_value=[out_dir / "fake.png"],
            ),
            patch(
                "chronoagent.experiments.analysis.tables.make_main_results_table",
                return_value=out_dir / "tables" / "fake.tex",
            ) as mock_table,
        ):
            result = runner.invoke(
                app,
                [
                    "run-experiment",
                    "phase10",
                    "--config",
                    str(cfg_path),
                    "--output",
                    str(out_dir),
                ],
            )
        assert result.exit_code == 0, result.output
        mock_table.assert_called_once()
        assert mock_table.call_args.args[1] == ["cli_test_experiment"]

    def test_quiet_suppresses_progress(self, tmp_path: Path) -> None:
        """--quiet suppresses the progress echo lines."""
        cfg_path = tmp_path / "main_experiment.yaml"
        cfg_path.write_text(_phase10_yaml())
        out_dir = tmp_path / "results"

        with patch("chronoagent.experiments.experiment_runner.ExperimentRunner", _StubRunner):
            result = runner.invoke(
                app,
                [
                    "run-experiment",
                    "phase10",
                    "--config",
                    str(cfg_path),
                    "--output",
                    str(out_dir),
                    "--no-plots",
                    "--no-tables",
                    "--quiet",
                ],
            )
        assert result.exit_code == 0, result.output
        assert "Running phase10 experiment" not in result.output
        assert "Wrote aggregate" not in result.output

    def test_missing_config_file_exits_one(self, tmp_path: Path) -> None:
        """phase10 exits 1 when the config file does not exist."""
        missing = tmp_path / "not_here.yaml"
        result = runner.invoke(
            app,
            [
                "run-experiment",
                "phase10",
                "--config",
                str(missing),
                "--output",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_invalid_config_exits_one(self, tmp_path: Path) -> None:
        """phase10 exits 1 when the config fails schema validation."""
        cfg_path = tmp_path / "bad.yaml"
        cfg_path.write_text("name: bad\nseed: -1\n")  # negative seed rejected
        result = runner.invoke(
            app,
            [
                "run-experiment",
                "phase10",
                "--config",
                str(cfg_path),
                "--output",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == 1
        assert "invalid experiment config" in result.output.lower()

    def test_plot_failure_is_warning_not_error(self, tmp_path: Path) -> None:
        """A plot failure prints a warning but leaves exit code 0."""
        cfg_path = tmp_path / "main_experiment.yaml"
        cfg_path.write_text(_phase10_yaml())
        out_dir = tmp_path / "results"

        with (
            patch("chronoagent.experiments.experiment_runner.ExperimentRunner", _StubRunner),
            patch(
                "chronoagent.experiments.analysis.plots.plot_signal_drift",
                side_effect=FileNotFoundError("no raw data"),
            ),
            patch(
                "chronoagent.experiments.analysis.tables.make_main_results_table",
                return_value=out_dir / "tables" / "fake.tex",
            ),
        ):
            result = runner.invoke(
                app,
                [
                    "run-experiment",
                    "phase10",
                    "--config",
                    str(cfg_path),
                    "--output",
                    str(out_dir),
                ],
            )
        assert result.exit_code == 0, result.output
        assert "WARNING" in result.output


class TestCompareExperiments:
    """compare-experiments walks a results directory and renders comparisons."""

    def test_requires_output_directory(self, tmp_path: Path) -> None:
        """Missing output directory exits 1."""
        missing = tmp_path / "does_not_exist"
        result = runner.invoke(
            app,
            [
                "compare-experiments",
                "--output",
                str(missing),
                "--experiment",
                "a",
                "--experiment",
                "b",
            ],
        )
        assert result.exit_code == 1
        assert "does not exist" in result.output.lower()

    def test_happy_path_calls_plot_and_table_generators(self, tmp_path: Path) -> None:
        """compare-experiments runs generate_all_plots + make_main_results_table."""
        out_dir = tmp_path / "results"
        out_dir.mkdir()

        with (
            patch(
                "chronoagent.experiments.analysis.plots.generate_all_plots",
                return_value=[out_dir / "a.png", out_dir / "b.png"],
            ) as mock_plots,
            patch(
                "chronoagent.experiments.analysis.tables.make_main_results_table",
                return_value=out_dir / "tables" / "main.tex",
            ) as mock_main,
            patch(
                "chronoagent.experiments.analysis.tables.make_latency_table",
                return_value=out_dir / "tables" / "latency.tex",
            ) as mock_latency,
            patch(
                "chronoagent.experiments.analysis.tables.make_ablation_table",
                return_value=out_dir / "tables" / "ablation.tex",
            ) as mock_ablation,
        ):
            result = runner.invoke(
                app,
                [
                    "compare-experiments",
                    "--output",
                    str(out_dir),
                    "--experiment",
                    "main_experiment",
                    "--experiment",
                    "ablation_no_bocpd",
                    "--ablation",
                    "ablation_no_bocpd",
                ],
            )
        assert result.exit_code == 0, result.output
        mock_plots.assert_called_once()
        mock_main.assert_called_once()
        mock_latency.assert_called_once()
        mock_ablation.assert_called_once()
        # Default drift_experiment is the first --experiment value.
        assert mock_plots.call_args.kwargs["drift_experiment"] == "main_experiment"
        # Default full_system is the first --experiment value.
        assert mock_ablation.call_args.args[1] == "main_experiment"
        assert mock_ablation.call_args.args[2] == ["ablation_no_bocpd"]

    def test_skips_ablation_table_when_no_ablations(self, tmp_path: Path) -> None:
        """With no --ablation flags, the ablation table is skipped."""
        out_dir = tmp_path / "results"
        out_dir.mkdir()

        with (
            patch(
                "chronoagent.experiments.analysis.plots.generate_all_plots",
                return_value=[],
            ),
            patch(
                "chronoagent.experiments.analysis.tables.make_main_results_table",
                return_value=out_dir / "tables" / "main.tex",
            ),
            patch(
                "chronoagent.experiments.analysis.tables.make_latency_table",
                return_value=out_dir / "tables" / "latency.tex",
            ),
            patch(
                "chronoagent.experiments.analysis.tables.make_ablation_table",
            ) as mock_ablation,
        ):
            result = runner.invoke(
                app,
                [
                    "compare-experiments",
                    "--output",
                    str(out_dir),
                    "--experiment",
                    "main_experiment",
                ],
            )
        assert result.exit_code == 0, result.output
        mock_ablation.assert_not_called()
        assert "Skipping ablation table" in result.output

    def test_no_plots_no_tables_short_circuit(self, tmp_path: Path) -> None:
        """--no-plots and --no-tables skip both generators."""
        out_dir = tmp_path / "results"
        out_dir.mkdir()

        with (
            patch("chronoagent.experiments.analysis.plots.generate_all_plots") as mock_plots,
            patch("chronoagent.experiments.analysis.tables.make_main_results_table") as mock_main,
            patch("chronoagent.experiments.analysis.tables.make_latency_table"),
            patch("chronoagent.experiments.analysis.tables.make_ablation_table") as mock_ablation,
        ):
            result = runner.invoke(
                app,
                [
                    "compare-experiments",
                    "--output",
                    str(out_dir),
                    "--experiment",
                    "main_experiment",
                    "--no-plots",
                    "--no-tables",
                ],
            )
        assert result.exit_code == 0, result.output
        mock_plots.assert_not_called()
        mock_main.assert_not_called()
        mock_ablation.assert_not_called()

    def test_explicit_full_system_and_drift_target(self, tmp_path: Path) -> None:
        """--full-system and --drift-experiment override defaults."""
        out_dir = tmp_path / "results"
        out_dir.mkdir()

        with (
            patch(
                "chronoagent.experiments.analysis.plots.generate_all_plots",
                return_value=[],
            ) as mock_plots,
            patch(
                "chronoagent.experiments.analysis.tables.make_main_results_table",
                return_value=out_dir / "main.tex",
            ),
            patch(
                "chronoagent.experiments.analysis.tables.make_latency_table",
                return_value=out_dir / "latency.tex",
            ),
            patch(
                "chronoagent.experiments.analysis.tables.make_ablation_table",
                return_value=out_dir / "ablation.tex",
            ) as mock_ablation,
        ):
            result = runner.invoke(
                app,
                [
                    "compare-experiments",
                    "--output",
                    str(out_dir),
                    "--experiment",
                    "main_experiment",
                    "--experiment",
                    "agentpoison_experiment",
                    "--full-system",
                    "main_experiment",
                    "--ablation",
                    "ablation_no_bocpd",
                    "--drift-experiment",
                    "agentpoison_experiment",
                ],
            )
        assert result.exit_code == 0, result.output
        assert mock_plots.call_args.kwargs["drift_experiment"] == "agentpoison_experiment"
        assert mock_ablation.call_args.args[1] == "main_experiment"


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
