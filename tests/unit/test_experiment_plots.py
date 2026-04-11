"""Unit tests for the Phase 10 task 10.7 plot module.

These tests cover three layers:

1. The 10.6 runner's raw-data collection + persistence path. The
   ``collect_raw=True`` constructor flag, the
   :class:`~chronoagent.experiments.experiment_runner.RawRunRecord`
   shape, and the new ``raw_runs`` parameter on
   :func:`~chronoagent.experiments.experiment_runner.write_experiment_results`
   must all round-trip cleanly through disk.
2. The :class:`~chronoagent.experiments.analysis.plots.ExperimentArtefacts`
   loader, which parses ``aggregate.json`` + ``runs.csv`` + (optionally)
   the per-run ``raw/`` directory.
3. Each of the six plot functions in
   :mod:`chronoagent.experiments.analysis.plots`. Every test writes
   minimal artefacts under ``tmp_path``, calls the function, and
   asserts a PNG + SVG with non-trivial size landed in the right
   directory. Matplotlib state is reset via the autouse
   ``_close_matplotlib`` fixture so a failure in one figure does not
   leak open figures into the next test.

The tests deliberately do not invoke the real
``SignalValidationRunner``: every signal matrix is a deterministic
seeded gaussian shifted on the KL + entropy columns after the
injection row, identical to the helper used in the 10.6 runner test
file. This keeps the inner-loop for plot tests well under 10 seconds.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pytest
from numpy.typing import NDArray

# Lock the headless backend before pytest collects any matplotlib
# figures. matplotlib.use is a no-op once any pyplot module has been
# imported, so this must run at module load time.
matplotlib.use("Agg")

from chronoagent.experiments.analysis.plots import (  # noqa: E402
    ABLATION_BAR_FIG_STEM,
    ALLOC_EFF_OVER_TIME_FIG_STEM,
    AWT_BOX_FIG_STEM,
    HEALTH_COMPARISON_FIG_STEM,
    ROC_CURVE_FIG_STEM,
    SIGNAL_DRIFT_FIG_STEM,
    ExperimentArtefacts,
    _empirical_roc,
    generate_all_plots,
    load_artefacts,
    plot_ablation_bar_chart,
    plot_allocation_efficiency_over_time,
    plot_awt_box,
    plot_health_score_comparison,
    plot_roc_curve,
    plot_signal_drift,
)
from chronoagent.experiments.config_schema import (  # noqa: E402
    AblationConfig,
    AttackConfig,
    ExperimentConfig,
    SystemConfig,
)
from chronoagent.experiments.experiment_runner import (  # noqa: E402
    ExperimentRunner,
    RawRunRecord,
    SignalMatrixFactory,
    write_experiment_results,
)
from chronoagent.monitor.collector import NUM_SIGNALS  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "experiments"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _close_matplotlib() -> Iterator[None]:
    """Make sure no test leaks an open matplotlib figure."""
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


def _shifted_factory(shift_kl: float = 5.0, shift_entropy: float = 3.0) -> SignalMatrixFactory:
    def factory(
        *,
        attack_type: str,
        seed: int,
        injection_step: int,
        num_prs: int,
        n_poison_docs: int,
    ) -> NDArray[np.float64]:
        rng = np.random.default_rng(seed)
        mat = rng.normal(loc=0.0, scale=0.1, size=(num_prs, NUM_SIGNALS)).astype(np.float64)
        mat[injection_step:, 3] += shift_kl  # KL column
        mat[injection_step:, 5] += shift_entropy  # entropy column
        return mat

    return factory  # type: ignore[return-value]


def _make_cfg(
    *,
    name: str = "main_experiment",
    seed: int = 7,
    num_runs: int = 3,
    num_prs: int = 16,
    injection_step: int = 8,
    attack_type: str = "minja",
    forecaster: bool = True,
    bocpd: bool = True,
    health: bool = True,
    integrity: bool = True,
) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        seed=seed,
        num_runs=num_runs,
        num_prs=num_prs,
        attack=AttackConfig(
            type=attack_type,  # type: ignore[arg-type]
            injection_step=injection_step,
            n_poison_docs=5,
        ),
        ablation=AblationConfig(
            forecaster=forecaster,
            bocpd=bocpd,
            health=health,
            integrity=integrity,
        ),
        system=SystemConfig(),
    )


def _seed_experiment(
    tmp_path: Path,
    *,
    name: str,
    forecaster: bool = True,
    bocpd: bool = True,
    health: bool = True,
    integrity: bool = True,
    num_runs: int = 3,
    num_prs: int = 16,
    injection_step: int = 8,
) -> ExperimentRunner:
    """Persist one experiment under tmp_path with raw data on disk."""
    cfg = _make_cfg(
        name=name,
        num_runs=num_runs,
        num_prs=num_prs,
        injection_step=injection_step,
        forecaster=forecaster,
        bocpd=bocpd,
        health=health,
        integrity=integrity,
    )
    runner = ExperimentRunner(cfg, signal_matrix_factory=_shifted_factory(), collect_raw=True)
    aggregate = runner.run()
    write_experiment_results(aggregate, tmp_path, raw_runs=runner.raw_runs)
    return runner


def _seed_three_experiments(tmp_path: Path) -> list[str]:
    """Persist three experiments (full + one ablation + sentinel)."""
    names = ["main_experiment", "ablation_no_forecaster", "baseline_sentinel"]
    _seed_experiment(tmp_path, name="main_experiment")
    _seed_experiment(tmp_path, name="ablation_no_forecaster", forecaster=False)
    _seed_experiment(tmp_path, name="baseline_sentinel")
    return names


def _read_aggregate(tmp_path: Path, name: str) -> dict[str, Any]:
    return json.loads((tmp_path / name / "aggregate.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Runner raw data collection (extension landed alongside 10.7)
# ---------------------------------------------------------------------------


class TestRunnerRawData:
    def test_collect_raw_default_is_off(self) -> None:
        cfg = _make_cfg()
        runner = ExperimentRunner(cfg, signal_matrix_factory=_shifted_factory())
        runner.run()
        assert runner.raw_runs == []

    def test_collect_raw_populates_raw_runs(self) -> None:
        cfg = _make_cfg(num_runs=2)
        runner = ExperimentRunner(cfg, signal_matrix_factory=_shifted_factory(), collect_raw=True)
        runner.run()
        assert len(runner.raw_runs) == 2
        for record in runner.raw_runs:
            assert isinstance(record, RawRunRecord)
            assert record.signal_matrix.shape == (cfg.num_prs, NUM_SIGNALS)
            assert len(record.decisions) == cfg.num_prs
            for dec in record.decisions:
                assert {"step_index", "score", "flagged", "success", "agent_id"} <= dec.keys()

    def test_collect_raw_preserves_run_index_and_seed(self) -> None:
        cfg = _make_cfg(seed=11, num_runs=3)
        runner = ExperimentRunner(cfg, signal_matrix_factory=_shifted_factory(), collect_raw=True)
        runner.run()
        for i, record in enumerate(runner.raw_runs):
            assert record.run_index == i
            assert record.seed == 11 + i

    def test_full_system_raw_decisions_carry_subscores(self) -> None:
        cfg = _make_cfg()  # full system, all flags True
        runner = ExperimentRunner(cfg, signal_matrix_factory=_shifted_factory(), collect_raw=True)
        runner.run()
        for record in runner.raw_runs:
            for dec in record.decisions:
                assert "bocpd_score" in dec
                assert "forecaster_score" in dec
                assert "integrity_score" in dec

    def test_no_monitoring_raw_decisions_have_no_subscores(self) -> None:
        cfg = _make_cfg(name="no_mon_test", health=False)
        runner = ExperimentRunner(cfg, signal_matrix_factory=_shifted_factory(), collect_raw=True)
        runner.run()
        for record in runner.raw_runs:
            for dec in record.decisions:
                assert "bocpd_score" not in dec
                assert "forecaster_score" not in dec
                # Score and flagged default to 0.0 / False on the projection
                assert dec["score"] == 0.0
                assert dec["flagged"] is False
                assert dec["success"] is True

    def test_collect_raw_resets_on_second_run(self) -> None:
        cfg = _make_cfg(num_runs=2)
        runner = ExperimentRunner(cfg, signal_matrix_factory=_shifted_factory(), collect_raw=True)
        runner.run()
        assert len(runner.raw_runs) == 2
        runner.run()
        assert len(runner.raw_runs) == 2  # not 4

    def test_write_experiment_results_skips_raw_when_omitted(self, tmp_path: Path) -> None:
        cfg = _make_cfg()
        runner = ExperimentRunner(cfg, signal_matrix_factory=_shifted_factory(), collect_raw=True)
        agg = runner.run()
        write_experiment_results(agg, tmp_path)  # no raw_runs
        assert not (tmp_path / cfg.name / "raw").exists()

    def test_write_experiment_results_persists_raw(self, tmp_path: Path) -> None:
        cfg = _make_cfg(num_runs=2)
        runner = ExperimentRunner(cfg, signal_matrix_factory=_shifted_factory(), collect_raw=True)
        agg = runner.run()
        write_experiment_results(agg, tmp_path, raw_runs=runner.raw_runs)
        raw_dir = tmp_path / cfg.name / "raw"
        assert raw_dir.is_dir()
        assert (raw_dir / "run_000_signals.npy").is_file()
        assert (raw_dir / "run_000_decisions.json").is_file()
        assert (raw_dir / "run_001_signals.npy").is_file()
        assert (raw_dir / "run_001_decisions.json").is_file()

    def test_persisted_signals_npy_round_trip(self, tmp_path: Path) -> None:
        cfg = _make_cfg(num_runs=1)
        runner = ExperimentRunner(cfg, signal_matrix_factory=_shifted_factory(), collect_raw=True)
        agg = runner.run()
        write_experiment_results(agg, tmp_path, raw_runs=runner.raw_runs)
        loaded = np.load(tmp_path / cfg.name / "raw" / "run_000_signals.npy")
        np.testing.assert_array_equal(loaded, runner.raw_runs[0].signal_matrix)

    def test_persisted_decisions_json_round_trip(self, tmp_path: Path) -> None:
        cfg = _make_cfg(num_runs=1)
        runner = ExperimentRunner(cfg, signal_matrix_factory=_shifted_factory(), collect_raw=True)
        agg = runner.run()
        write_experiment_results(agg, tmp_path, raw_runs=runner.raw_runs)
        payload = json.loads(
            (tmp_path / cfg.name / "raw" / "run_000_decisions.json").read_text(encoding="utf-8")
        )
        assert payload["run_index"] == 0
        assert payload["seed"] == cfg.per_run_seed(0)
        assert len(payload["decisions"]) == cfg.num_prs


# ---------------------------------------------------------------------------
# load_artefacts
# ---------------------------------------------------------------------------


class TestLoadArtefacts:
    def test_loader_returns_dataclass(self, tmp_path: Path) -> None:
        _seed_experiment(tmp_path, name="main_experiment")
        a = load_artefacts(tmp_path, "main_experiment")
        assert isinstance(a, ExperimentArtefacts)
        assert a.name == "main_experiment"
        assert a.aggregate["num_runs"] == 3
        assert len(a.runs) == 3

    def test_loader_reads_raw_data_when_present(self, tmp_path: Path) -> None:
        _seed_experiment(tmp_path, name="main_experiment", num_runs=2)
        a = load_artefacts(tmp_path, "main_experiment")
        assert len(a.signal_matrices) == 2
        assert len(a.decision_streams) == 2
        for sig in a.signal_matrices:
            assert sig.shape == (16, NUM_SIGNALS)
        for stream in a.decision_streams:
            assert len(stream) == 16

    def test_loader_handles_missing_raw_dir(self, tmp_path: Path) -> None:
        cfg = _make_cfg()
        runner = ExperimentRunner(cfg, signal_matrix_factory=_shifted_factory())
        agg = runner.run()
        write_experiment_results(agg, tmp_path)
        a = load_artefacts(tmp_path, "main_experiment")
        assert a.signal_matrices == []
        assert a.decision_streams == []

    def test_loader_raises_on_missing_aggregate(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="aggregate.json"):
            load_artefacts(tmp_path, "missing")


# ---------------------------------------------------------------------------
# Empirical ROC helper
# ---------------------------------------------------------------------------


class TestEmpiricalRoc:
    def test_empty_returns_diagonal(self) -> None:
        fpr, tpr = _empirical_roc(np.array([], dtype=bool), np.array([], dtype=np.float64))
        np.testing.assert_array_equal(fpr, np.array([0.0, 1.0]))
        np.testing.assert_array_equal(tpr, np.array([0.0, 1.0]))

    def test_single_class_returns_diagonal(self) -> None:
        fpr, tpr = _empirical_roc(
            np.array([True, True, True], dtype=bool),
            np.array([0.1, 0.5, 0.9], dtype=np.float64),
        )
        np.testing.assert_array_equal(fpr, np.array([0.0, 1.0]))
        np.testing.assert_array_equal(tpr, np.array([0.0, 1.0]))

    def test_perfect_separation_corner_at_zero_one(self) -> None:
        fpr, tpr = _empirical_roc(
            np.array([False, False, True, True], dtype=bool),
            np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64),
        )
        # Sorted by descending score: 0.9 (T) -> 0.8 (T) -> 0.2 (F) -> 0.1 (F)
        # tps = [1, 2, 2, 2]; fps = [0, 0, 1, 2]; pos = 2; neg = 2
        np.testing.assert_array_almost_equal(tpr, np.array([0.0, 0.5, 1.0, 1.0, 1.0]))
        np.testing.assert_array_almost_equal(fpr, np.array([0.0, 0.0, 0.0, 0.5, 1.0]))


# ---------------------------------------------------------------------------
# Figure 1: signal drift
# ---------------------------------------------------------------------------


class TestPlotSignalDrift:
    def test_creates_png_and_svg(self, tmp_path: Path) -> None:
        _seed_experiment(tmp_path, name="main_experiment")
        paths = plot_signal_drift(tmp_path, "main_experiment", run_index=0)
        assert len(paths) == 2
        for path in paths:
            assert path.is_file()
            assert path.stat().st_size > 1000
        names = sorted(p.name for p in paths)
        assert names == [f"{SIGNAL_DRIFT_FIG_STEM}.png", f"{SIGNAL_DRIFT_FIG_STEM}.svg"]

    def test_writes_into_per_experiment_figures_dir(self, tmp_path: Path) -> None:
        _seed_experiment(tmp_path, name="main_experiment")
        paths = plot_signal_drift(tmp_path, "main_experiment")
        for path in paths:
            assert path.parent == tmp_path / "main_experiment" / "figures"

    def test_raises_on_missing_raw(self, tmp_path: Path) -> None:
        cfg = _make_cfg()
        runner = ExperimentRunner(cfg, signal_matrix_factory=_shifted_factory())
        agg = runner.run()
        write_experiment_results(agg, tmp_path)
        with pytest.raises(FileNotFoundError, match="no raw signal data"):
            plot_signal_drift(tmp_path, "main_experiment")

    def test_run_index_out_of_range_raises(self, tmp_path: Path) -> None:
        _seed_experiment(tmp_path, name="main_experiment", num_runs=2)
        with pytest.raises(IndexError, match="outside"):
            plot_signal_drift(tmp_path, "main_experiment", run_index=5)


# ---------------------------------------------------------------------------
# Figure 2: health score comparison
# ---------------------------------------------------------------------------


class TestPlotHealthScoreComparison:
    def test_creates_png_and_svg(self, tmp_path: Path) -> None:
        names = _seed_three_experiments(tmp_path)
        paths = plot_health_score_comparison(tmp_path, names)
        assert len(paths) == 2
        for path in paths:
            assert path.is_file()
            assert path.stat().st_size > 1000

    def test_writes_to_root_figures_dir(self, tmp_path: Path) -> None:
        names = _seed_three_experiments(tmp_path)
        paths = plot_health_score_comparison(tmp_path, names)
        for path in paths:
            assert path.parent == tmp_path / "figures"

    def test_empty_experiment_list_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            plot_health_score_comparison(tmp_path, [])

    def test_missing_raw_data_raises(self, tmp_path: Path) -> None:
        cfg = _make_cfg()
        runner = ExperimentRunner(cfg, signal_matrix_factory=_shifted_factory())
        agg = runner.run()
        write_experiment_results(agg, tmp_path)
        with pytest.raises(FileNotFoundError, match="no raw decision data"):
            plot_health_score_comparison(tmp_path, ["main_experiment"])


# ---------------------------------------------------------------------------
# Figure 3: AWT box plot
# ---------------------------------------------------------------------------


class TestPlotAwtBox:
    def test_creates_png_and_svg(self, tmp_path: Path) -> None:
        names = _seed_three_experiments(tmp_path)
        paths = plot_awt_box(tmp_path, names)
        assert len(paths) == 2
        for path in paths:
            assert path.stat().st_size > 1000

    def test_handles_no_monitoring_with_empty_awt(self, tmp_path: Path) -> None:
        # Health=False -> NoMonitoring -> never flags -> empty AWT column.
        _seed_experiment(tmp_path, name="ablation_no_health_scores", health=False)
        _seed_experiment(tmp_path, name="main_experiment")
        paths = plot_awt_box(
            tmp_path,
            ["main_experiment", "ablation_no_health_scores"],
        )
        assert len(paths) == 2
        # The function must NOT crash on the empty box; it draws a hatched
        # placeholder instead. Confirm the file is non-trivial.
        for path in paths:
            assert path.stat().st_size > 1000

    def test_uses_runs_csv_only_no_raw_required(self, tmp_path: Path) -> None:
        cfg = _make_cfg()
        runner = ExperimentRunner(cfg, signal_matrix_factory=_shifted_factory())
        agg = runner.run()
        write_experiment_results(agg, tmp_path)  # no raw
        paths = plot_awt_box(tmp_path, ["main_experiment"])
        assert all(p.is_file() for p in paths)

    def test_empty_experiment_list_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            plot_awt_box(tmp_path, [])


# ---------------------------------------------------------------------------
# Figure 4: allocation efficiency over time
# ---------------------------------------------------------------------------


class TestPlotAllocationEfficiencyOverTime:
    def test_creates_png_and_svg(self, tmp_path: Path) -> None:
        names = _seed_three_experiments(tmp_path)
        paths = plot_allocation_efficiency_over_time(tmp_path, names)
        assert len(paths) == 2
        for path in paths:
            assert path.stat().st_size > 1000

    def test_missing_raw_data_raises(self, tmp_path: Path) -> None:
        cfg = _make_cfg()
        runner = ExperimentRunner(cfg, signal_matrix_factory=_shifted_factory())
        agg = runner.run()
        write_experiment_results(agg, tmp_path)
        with pytest.raises(FileNotFoundError, match="no raw decision data"):
            plot_allocation_efficiency_over_time(tmp_path, ["main_experiment"])

    def test_writes_to_root_figures_dir(self, tmp_path: Path) -> None:
        names = _seed_three_experiments(tmp_path)
        paths = plot_allocation_efficiency_over_time(tmp_path, names)
        for path in paths:
            assert path.parent == tmp_path / "figures"


# ---------------------------------------------------------------------------
# Figure 5: ROC curve
# ---------------------------------------------------------------------------


class TestPlotRocCurve:
    def test_creates_png_and_svg(self, tmp_path: Path) -> None:
        names = _seed_three_experiments(tmp_path)
        paths = plot_roc_curve(tmp_path, names)
        assert len(paths) == 2
        for path in paths:
            assert path.stat().st_size > 1000

    def test_missing_raw_data_raises(self, tmp_path: Path) -> None:
        cfg = _make_cfg()
        runner = ExperimentRunner(cfg, signal_matrix_factory=_shifted_factory())
        agg = runner.run()
        write_experiment_results(agg, tmp_path)
        with pytest.raises(FileNotFoundError, match="no raw decision data"):
            plot_roc_curve(tmp_path, ["main_experiment"])


# ---------------------------------------------------------------------------
# Figure 6: ablation bar chart
# ---------------------------------------------------------------------------


class TestPlotAblationBarChart:
    def test_creates_png_and_svg(self, tmp_path: Path) -> None:
        names = _seed_three_experiments(tmp_path)
        paths = plot_ablation_bar_chart(tmp_path, names)
        assert len(paths) == 2
        for path in paths:
            assert path.stat().st_size > 1000

    def test_no_raw_data_required(self, tmp_path: Path) -> None:
        cfg = _make_cfg()
        runner = ExperimentRunner(cfg, signal_matrix_factory=_shifted_factory())
        agg = runner.run()
        write_experiment_results(agg, tmp_path)
        paths = plot_ablation_bar_chart(tmp_path, ["main_experiment"])
        for path in paths:
            assert path.is_file()

    def test_handles_nan_metrics(self, tmp_path: Path) -> None:
        # NoMonitoring never fires => AWT mean is NaN => null in JSON.
        _seed_experiment(tmp_path, name="ablation_no_health_scores", health=False)
        _seed_experiment(tmp_path, name="main_experiment")
        agg = _read_aggregate(tmp_path, "ablation_no_health_scores")
        assert agg["metrics"]["advance_warning_time"]["mean"] is None
        # Plot should still draw (zero-height bar) without crashing.
        paths = plot_ablation_bar_chart(
            tmp_path,
            ["main_experiment", "ablation_no_health_scores"],
        )
        for path in paths:
            assert path.is_file()
            assert path.stat().st_size > 1000

    def test_empty_experiment_list_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            plot_ablation_bar_chart(tmp_path, [])


# ---------------------------------------------------------------------------
# generate_all_plots
# ---------------------------------------------------------------------------


class TestGenerateAllPlots:
    def test_runs_every_figure_function(self, tmp_path: Path) -> None:
        names = _seed_three_experiments(tmp_path)
        paths = generate_all_plots(tmp_path, names)
        assert len(paths) == 12  # 6 figures x (PNG + SVG)
        stems = {p.stem for p in paths}
        assert SIGNAL_DRIFT_FIG_STEM in stems
        assert HEALTH_COMPARISON_FIG_STEM in stems
        assert AWT_BOX_FIG_STEM in stems
        assert ALLOC_EFF_OVER_TIME_FIG_STEM in stems
        assert ROC_CURVE_FIG_STEM in stems
        assert ABLATION_BAR_FIG_STEM in stems

    def test_drift_experiment_override(self, tmp_path: Path) -> None:
        names = _seed_three_experiments(tmp_path)
        paths = generate_all_plots(
            tmp_path,
            names,
            drift_experiment="ablation_no_forecaster",
        )
        # Signal drift figure should land under the drift_experiment dir
        drift_paths = [p for p in paths if SIGNAL_DRIFT_FIG_STEM in p.name]
        assert len(drift_paths) == 2
        for p in drift_paths:
            assert "ablation_no_forecaster" in p.parts

    def test_empty_experiment_list_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            generate_all_plots(tmp_path, [])
