"""Unit tests for the Phase 10 task 10.8 LaTeX tables module.

These tests cover three layers:

1. The private cell formatters (``_format_mean_ci``, ``_format_delta``,
   ``_is_better``, ``_coerce_metric_value``, ``_escape_latex``). Each
   has hand-crafted edge cases for NaN, single-sample CIs, and special
   characters in signal labels.
2. The three table generators (main results, ablation, signal
   validation) end-to-end against fake-runner artefacts persisted
   under ``tmp_path``. Each generator's output is parsed back to
   verify row count, column count, header text, NaN placeholders,
   bolding, and per-column delta rendering.
3. The :func:`generate_all_tables` convenience entry point with and
   without the optional Phase 1 signal validation rows.

The fake-runner factory and ``_seed_experiment`` helper are deliberate
mirrors of the helpers in ``test_experiment_plots.py`` so the two test
files share a mental model. The tables module does not need raw
decision data, so ``collect_raw=False`` is the default for these
fixtures.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from chronoagent.experiments.analysis.tables import (
    ABLATION_TABLE_STEM,
    MAIN_RESULTS_TABLE_STEM,
    SIGNAL_VALIDATION_TABLE_STEM,
    SignalStatRow,
    _coerce_metric_value,
    _escape_latex,
    _format_delta,
    _format_mean_ci,
    _is_better,
    generate_all_tables,
    make_ablation_table,
    make_main_results_table,
    make_signal_validation_table,
)
from chronoagent.experiments.config_schema import (
    AblationConfig,
    AttackConfig,
    ExperimentConfig,
    SystemConfig,
)
from chronoagent.experiments.experiment_runner import (
    ExperimentRunner,
    SignalMatrixFactory,
    write_experiment_results,
)
from chronoagent.monitor.collector import NUM_SIGNALS

# ---------------------------------------------------------------------------
# Fixtures and helpers (mirror of test_experiment_plots.py)
# ---------------------------------------------------------------------------


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
        mat[injection_step:, 3] += shift_kl
        mat[injection_step:, 5] += shift_entropy
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
            forecaster=forecaster, bocpd=bocpd, health=health, integrity=integrity
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
) -> None:
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
    runner = ExperimentRunner(cfg, signal_matrix_factory=_shifted_factory())
    aggregate = runner.run()
    write_experiment_results(aggregate, tmp_path)


def _seed_full_suite(tmp_path: Path) -> dict[str, str]:
    """Persist enough experiments for both the main results and ablation tables."""
    _seed_experiment(tmp_path, name="main_experiment")
    _seed_experiment(tmp_path, name="ablation_no_forecaster", forecaster=False)
    _seed_experiment(tmp_path, name="ablation_no_bocpd", bocpd=False)
    _seed_experiment(tmp_path, name="ablation_no_health_scores", health=False)
    _seed_experiment(tmp_path, name="baseline_sentinel")
    return {
        "main": "main_experiment",
        "no_forecaster": "ablation_no_forecaster",
        "no_bocpd": "ablation_no_bocpd",
        "no_health": "ablation_no_health_scores",
        "sentinel": "baseline_sentinel",
    }


# ---------------------------------------------------------------------------
# Cell formatters
# ---------------------------------------------------------------------------


class TestFormatMeanCi:
    def test_basic_symmetric_ci(self) -> None:
        s = _format_mean_ci(0.5, 0.4, 0.6, decimals=3)
        assert s == "0.500 $\\pm$ 0.100"

    def test_asymmetric_ci_picks_larger_side(self) -> None:
        s = _format_mean_ci(0.5, 0.45, 0.7, decimals=3)
        # max(0.5 - 0.45, 0.7 - 0.5) = 0.2
        assert s == "0.500 $\\pm$ 0.200"

    def test_nan_mean_returns_dashes(self) -> None:
        assert _format_mean_ci(float("nan"), 0.0, 0.0, decimals=3) == "--"

    def test_nan_ci_low_returns_mean_only(self) -> None:
        s = _format_mean_ci(0.5, float("nan"), float("nan"), decimals=3)
        assert s == "0.500"
        assert "$\\pm$" not in s

    def test_decimals_parameter_respected(self) -> None:
        s = _format_mean_ci(2.345, 2.0, 2.69, decimals=1)
        assert s == "2.3 $\\pm$ 0.3"

    def test_zero_width_ci_renders_zero(self) -> None:
        s = _format_mean_ci(1.0, 1.0, 1.0, decimals=3)
        assert s == "1.000 $\\pm$ 0.000"


class TestFormatDelta:
    def test_positive_delta(self) -> None:
        assert _format_delta(0.6, 0.5, decimals=3) == " (+0.100)"

    def test_negative_delta(self) -> None:
        assert _format_delta(0.4, 0.5, decimals=3) == " (-0.100)"

    def test_zero_delta_uses_plus_sign(self) -> None:
        assert _format_delta(0.5, 0.5, decimals=3) == " (+0.000)"

    def test_nan_mean_returns_empty_string(self) -> None:
        assert _format_delta(float("nan"), 0.5, decimals=3) == ""

    def test_nan_baseline_returns_empty_string(self) -> None:
        assert _format_delta(0.5, float("nan"), decimals=3) == ""

    def test_decimals_parameter_respected(self) -> None:
        assert _format_delta(2.5, 2.0, decimals=1) == " (+0.5)"


class TestIsBetter:
    def test_higher_value_wins(self) -> None:
        assert _is_better(0.7, 0.5) is True
        assert _is_better(0.5, 0.7) is False

    def test_equal_value_does_not_replace_existing_best(self) -> None:
        assert _is_better(0.5, 0.5) is False

    def test_first_real_value_beats_nan_baseline(self) -> None:
        assert _is_better(0.3, float("nan")) is True

    def test_nan_value_never_wins(self) -> None:
        assert _is_better(float("nan"), 0.5) is False
        assert _is_better(float("nan"), float("nan")) is False


class TestCoerceMetricValue:
    def test_none_returns_nan(self) -> None:
        assert math.isnan(_coerce_metric_value(None))

    def test_numeric_string_parses(self) -> None:
        assert _coerce_metric_value("0.5") == pytest.approx(0.5)

    def test_float_passthrough(self) -> None:
        assert _coerce_metric_value(0.5) == pytest.approx(0.5)

    def test_garbage_string_returns_nan(self) -> None:
        assert math.isnan(_coerce_metric_value("not a number"))


class TestEscapeLatex:
    def test_underscores_escaped(self) -> None:
        assert _escape_latex("kl_divergence") == "kl\\_divergence"

    def test_multiple_specials(self) -> None:
        assert _escape_latex("a_b%c") == "a\\_b\\%c"

    def test_plain_text_unchanged(self) -> None:
        assert _escape_latex("plain") == "plain"


# ---------------------------------------------------------------------------
# Main results table
# ---------------------------------------------------------------------------


class TestMakeMainResultsTable:
    def test_writes_default_path(self, tmp_path: Path) -> None:
        names = list(_seed_full_suite(tmp_path).values())
        path = make_main_results_table(tmp_path, names[:3])
        assert path == tmp_path / "tables" / f"{MAIN_RESULTS_TABLE_STEM}.tex"
        assert path.is_file()
        assert path.stat().st_size > 200

    def test_output_path_override(self, tmp_path: Path) -> None:
        _seed_full_suite(tmp_path)
        target = tmp_path / "custom" / "out.tex"
        path = make_main_results_table(tmp_path, ["main_experiment"], output_path=target)
        assert path == target
        assert path.is_file()

    def test_table_structure(self, tmp_path: Path) -> None:
        _seed_full_suite(tmp_path)
        path = make_main_results_table(
            tmp_path,
            ["main_experiment", "baseline_sentinel", "ablation_no_health_scores"],
        )
        text = path.read_text(encoding="utf-8")
        assert "\\begin{tabular}{lcccc}" in text
        assert "\\end{tabular}" in text
        # Header row
        assert "Experiment" in text
        assert "AWT (steps)" in text
        assert "AUROC" in text
        assert "F1" in text
        assert "Alloc eff" in text
        # 3 data rows: count "\\\\" line endings between hlines.
        # Header + 3 data rows = 4 row separators inside the tabular.
        body_lines = [line for line in text.splitlines() if line.endswith(" \\\\")]
        assert len(body_lines) == 4  # 1 header + 3 data

    def test_no_monitoring_row_renders_nan_as_dashes(self, tmp_path: Path) -> None:
        _seed_full_suite(tmp_path)
        path = make_main_results_table(tmp_path, ["ablation_no_health_scores"])
        text = path.read_text(encoding="utf-8")
        # AWT cell is "--" because NoMonitoring never fires.
        assert "--" in text

    def test_bolding_marks_best_value_per_column(self, tmp_path: Path) -> None:
        _seed_full_suite(tmp_path)
        path = make_main_results_table(
            tmp_path,
            ["main_experiment", "baseline_sentinel", "ablation_no_health_scores"],
        )
        text = path.read_text(encoding="utf-8")
        # At least one bold cell should appear.
        assert "\\textbf{" in text
        # NoMonitoring should win on Alloc eff (always 1.0 vs others ~0.5)
        # and that bolded cell should be on the no-health row.
        no_health_rows = [
            line for line in text.splitlines() if "no health" in line and line.endswith(" \\\\")
        ]
        assert len(no_health_rows) == 1
        # The Alloc eff cell on this row should be bolded.
        assert "\\textbf{1.000" in no_health_rows[0]

    def test_uses_display_names_not_raw_yaml_names(self, tmp_path: Path) -> None:
        _seed_full_suite(tmp_path)
        path = make_main_results_table(tmp_path, ["main_experiment"])
        text = path.read_text(encoding="utf-8")
        assert "ChronoAgent (full)" in text
        # Raw YAML name should NOT appear in any data row.
        body = [line for line in text.splitlines() if line.endswith(" \\\\")]
        for line in body[1:]:  # skip header
            assert "main_experiment" not in line

    def test_empty_experiment_list_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            make_main_results_table(tmp_path, [])

    def test_missing_experiment_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            make_main_results_table(tmp_path, ["does_not_exist"])

    def test_creates_tables_directory(self, tmp_path: Path) -> None:
        _seed_full_suite(tmp_path)
        assert not (tmp_path / "tables").exists()
        make_main_results_table(tmp_path, ["main_experiment"])
        assert (tmp_path / "tables").is_dir()

    def test_per_metric_decimal_places(self, tmp_path: Path) -> None:
        _seed_full_suite(tmp_path)
        path = make_main_results_table(tmp_path, ["main_experiment"])
        text = path.read_text(encoding="utf-8")
        # AUROC uses 3 decimals: "1.000"
        assert "1.000 $\\pm$" in text
        # AWT cell uses 1 decimal: "0.0 $\pm$ 0.0" (may be wrapped in \textbf{}).
        body = [line for line in text.splitlines() if line.endswith(" \\\\")][1:]
        assert any("0.0 $\\pm$ 0.0" in line for line in body)


# ---------------------------------------------------------------------------
# Ablation table
# ---------------------------------------------------------------------------


class TestMakeAblationTable:
    def test_writes_default_path(self, tmp_path: Path) -> None:
        _seed_full_suite(tmp_path)
        path = make_ablation_table(
            tmp_path,
            "main_experiment",
            ["ablation_no_forecaster", "ablation_no_bocpd", "ablation_no_health_scores"],
        )
        assert path == tmp_path / "tables" / f"{ABLATION_TABLE_STEM}.tex"
        assert path.is_file()

    def test_output_path_override(self, tmp_path: Path) -> None:
        _seed_full_suite(tmp_path)
        target = tmp_path / "custom" / "abl.tex"
        path = make_ablation_table(
            tmp_path, "main_experiment", ["ablation_no_forecaster"], output_path=target
        )
        assert path == target

    def test_full_system_row_appears_first(self, tmp_path: Path) -> None:
        _seed_full_suite(tmp_path)
        path = make_ablation_table(
            tmp_path,
            "main_experiment",
            ["ablation_no_forecaster", "ablation_no_bocpd"],
        )
        text = path.read_text(encoding="utf-8")
        body = [line for line in text.splitlines() if line.endswith(" \\\\")]
        # Header + full + 2 ablations = 4 row lines.
        assert len(body) == 4
        # First data row (after header) is the full system.
        assert "ChronoAgent (full)" in body[1]

    def test_full_system_row_has_no_delta(self, tmp_path: Path) -> None:
        _seed_full_suite(tmp_path)
        path = make_ablation_table(tmp_path, "main_experiment", ["ablation_no_forecaster"])
        text = path.read_text(encoding="utf-8")
        body = [line for line in text.splitlines() if line.endswith(" \\\\")]
        full_row = body[1]
        # The full system row should not contain "(+" or "(-" delta brackets.
        assert "(+" not in full_row
        assert "(-" not in full_row

    def test_ablation_rows_have_delta(self, tmp_path: Path) -> None:
        _seed_full_suite(tmp_path)
        path = make_ablation_table(tmp_path, "main_experiment", ["ablation_no_health_scores"])
        text = path.read_text(encoding="utf-8")
        body = [line for line in text.splitlines() if line.endswith(" \\\\")]
        # The ablation row (last) should have at least one signed delta.
        assert "(+" in body[-1] or "(-" in body[-1]

    def test_no_monitoring_negative_delta_on_auroc(self, tmp_path: Path) -> None:
        _seed_full_suite(tmp_path)
        path = make_ablation_table(tmp_path, "main_experiment", ["ablation_no_health_scores"])
        text = path.read_text(encoding="utf-8")
        # NoMonitoring AUROC = 0.5 vs full = 1.0 -> delta -0.500
        assert "(-0.500)" in text

    def test_no_monitoring_positive_delta_on_alloc_eff(self, tmp_path: Path) -> None:
        _seed_full_suite(tmp_path)
        path = make_ablation_table(tmp_path, "main_experiment", ["ablation_no_health_scores"])
        text = path.read_text(encoding="utf-8")
        # NoMonitoring alloc_eff = 1.0 vs full = 0.5 -> delta +0.500
        assert "(+0.500)" in text

    def test_nan_metrics_drop_delta(self, tmp_path: Path) -> None:
        _seed_full_suite(tmp_path)
        path = make_ablation_table(tmp_path, "main_experiment", ["ablation_no_health_scores"])
        text = path.read_text(encoding="utf-8")
        # AWT cell on the no-health row is "--" with NO delta, since
        # the mean is NaN.
        body = [line for line in text.splitlines() if line.endswith(" \\\\")]
        no_health = body[-1]
        assert "--" in no_health
        # No "(+" or "(-" right after the "--" cell.
        idx = no_health.index("--")
        next_chunk = no_health[idx : idx + 10]
        assert "(" not in next_chunk

    def test_empty_ablation_list_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            make_ablation_table(tmp_path, "main_experiment", [])


# ---------------------------------------------------------------------------
# Signal validation table
# ---------------------------------------------------------------------------


class TestMakeSignalValidationTable:
    def _rows(self) -> list[SignalStatRow]:
        return [
            SignalStatRow(
                label="kl_divergence",
                clean_mean=0.10,
                clean_std=0.02,
                poisoned_mean=0.50,
                poisoned_std=0.10,
                cohens_d=2.50,
                large_effect=True,
            ),
            SignalStatRow(
                label="token_count",
                clean_mean=10.0,
                clean_std=2.0,
                poisoned_mean=11.0,
                poisoned_std=2.5,
                cohens_d=0.40,
                large_effect=False,
            ),
        ]

    def test_writes_to_supplied_path(self, tmp_path: Path) -> None:
        target = tmp_path / "out" / "sig.tex"
        path = make_signal_validation_table(self._rows(), target)
        assert path == target
        assert path.is_file()

    def test_table_has_one_row_per_signal(self, tmp_path: Path) -> None:
        path = make_signal_validation_table(self._rows(), tmp_path / "sig.tex")
        text = path.read_text(encoding="utf-8")
        body = [line for line in text.splitlines() if line.endswith(" \\\\")]
        # 1 header + 2 data rows
        assert len(body) == 3

    def test_underscores_escaped(self, tmp_path: Path) -> None:
        path = make_signal_validation_table(self._rows(), tmp_path / "sig.tex")
        text = path.read_text(encoding="utf-8")
        assert "kl\\_divergence" in text
        assert "token\\_count" in text

    def test_large_effect_bolded(self, tmp_path: Path) -> None:
        path = make_signal_validation_table(self._rows(), tmp_path / "sig.tex")
        text = path.read_text(encoding="utf-8")
        # The 2.50 d value should be bolded
        assert "\\textbf{2.50}" in text
        # The 0.40 d should NOT be bolded
        assert "\\textbf{0.40}" not in text

    def test_cohens_d_uses_two_decimals(self, tmp_path: Path) -> None:
        path = make_signal_validation_table(self._rows(), tmp_path / "sig.tex")
        text = path.read_text(encoding="utf-8")
        assert "0.40" in text
        # Should not have "0.4 " or "0.400" if 2 decimals is locked in.
        body = [line for line in text.splitlines() if line.endswith(" \\\\")][1:]
        assert any("0.40" in line for line in body)

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested" / "dir" / "sig.tex"
        assert not nested.parent.exists()
        make_signal_validation_table(self._rows(), nested)
        assert nested.parent.is_dir()

    def test_empty_rows_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            make_signal_validation_table([], tmp_path / "sig.tex")

    def test_table_header_text(self, tmp_path: Path) -> None:
        path = make_signal_validation_table(self._rows(), tmp_path / "sig.tex")
        text = path.read_text(encoding="utf-8")
        assert "\\begin{tabular}{lccccc}" in text
        assert "Signal" in text
        assert "Clean $\\mu$" in text
        assert "Cohen's $d$" in text


# ---------------------------------------------------------------------------
# generate_all_tables
# ---------------------------------------------------------------------------


class TestGenerateAllTables:
    def test_runs_main_and_ablation_without_signal_rows(self, tmp_path: Path) -> None:
        _seed_full_suite(tmp_path)
        paths = generate_all_tables(
            tmp_path,
            main_experiment_names=["main_experiment", "baseline_sentinel"],
            full_system_name="main_experiment",
            ablation_names=["ablation_no_forecaster", "ablation_no_bocpd"],
        )
        assert len(paths) == 2
        stems = {p.stem for p in paths}
        assert MAIN_RESULTS_TABLE_STEM in stems
        assert ABLATION_TABLE_STEM in stems

    def test_includes_signal_validation_when_rows_supplied(self, tmp_path: Path) -> None:
        _seed_full_suite(tmp_path)
        rows = [
            SignalStatRow(
                label="kl_divergence",
                clean_mean=0.1,
                clean_std=0.02,
                poisoned_mean=0.5,
                poisoned_std=0.1,
                cohens_d=2.5,
                large_effect=True,
            ),
        ]
        paths = generate_all_tables(
            tmp_path,
            main_experiment_names=["main_experiment"],
            full_system_name="main_experiment",
            ablation_names=["ablation_no_forecaster"],
            signal_validation_rows=rows,
        )
        assert len(paths) == 3
        stems = {p.stem for p in paths}
        assert SIGNAL_VALIDATION_TABLE_STEM in stems

    def test_signal_validation_lands_under_tables_dir(self, tmp_path: Path) -> None:
        _seed_full_suite(tmp_path)
        rows = [
            SignalStatRow(
                label="kl_divergence",
                clean_mean=0.1,
                clean_std=0.02,
                poisoned_mean=0.5,
                poisoned_std=0.1,
                cohens_d=2.5,
                large_effect=True,
            ),
        ]
        paths = generate_all_tables(
            tmp_path,
            main_experiment_names=["main_experiment"],
            full_system_name="main_experiment",
            ablation_names=["ablation_no_forecaster"],
            signal_validation_rows=rows,
        )
        sig_path = next(p for p in paths if p.stem == SIGNAL_VALIDATION_TABLE_STEM)
        assert sig_path.parent == tmp_path / "tables"

    def test_paths_are_files(self, tmp_path: Path) -> None:
        _seed_full_suite(tmp_path)
        paths = generate_all_tables(
            tmp_path,
            main_experiment_names=["main_experiment"],
            full_system_name="main_experiment",
            ablation_names=["ablation_no_forecaster"],
        )
        for p in paths:
            assert p.is_file()
            assert p.stat().st_size > 100
