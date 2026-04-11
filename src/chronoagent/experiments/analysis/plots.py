"""Phase 10 task 10.7 paper figures.

This module ships the six paper figures the Phase 10 research suite
needs. Every function reads the artefacts the 10.6
:class:`~chronoagent.experiments.experiment_runner.ExperimentRunner`
persists under ``<results_dir>/<experiment_name>/`` and writes a
matplotlib figure to ``<results_dir>/<experiment_name>/figures/`` (or
to a multi-experiment ``figures/`` root for the comparison plots) in
both PNG and SVG formats.

Six figures, mapping one-to-one onto Section 5 of the paper:

#. :func:`plot_signal_drift` (Section 5.1) - time-series of every
   signal column for one representative run, with the injection step
   and the first-flagged step marked. Reads
   ``raw/run_<i>_signals.npy`` plus ``aggregate.json`` (for the
   injection step) and ``runs.csv`` (for ``first_flagged_step``).
#. :func:`plot_health_score_comparison` (Section 5.2) - mean per-step
   detector score across runs for the full system vs the three
   ablations on the same axes. Reads ``raw/run_<i>_decisions.json``
   from each named experiment.
#. :func:`plot_awt_box` (Section 5.3) - box plot of per-run advance
   warning time across baselines + the full system. Reads
   ``runs.csv`` from each experiment.
#. :func:`plot_allocation_efficiency_over_time` (Section 5.4) -
   cumulative success rate per step, averaged across runs, for each
   experiment. Reads ``raw/run_<i>_decisions.json``.
#. :func:`plot_roc_curve` (Section 5.5) - per-experiment ROC curve
   built from the per-step scores in ``raw/run_<i>_decisions.json``
   pooled across runs.
#. :func:`plot_ablation_bar_chart` (Section 5.6) - one bar per (
   experiment, metric) cell with 95% CI error bars. Reads
   ``aggregate.json`` only.

The convenience function :func:`generate_all_plots` runs every figure
function for a list of experiment names and returns the list of files
written; it is the entry point the 10.9 CLI will hook into.

Implementation notes:

* The module imports :mod:`matplotlib` lazily inside each function so
  unit tests for the runner / aggregator can import this file's
  helpers without pulling in matplotlib's heavy backend stack.
* All functions explicitly call ``matplotlib.use("Agg")`` before
  importing ``matplotlib.pyplot`` so figures render headless on CI
  and on operator workstations without an X display.
* Every figure is closed via ``plt.close(fig)`` after saving so a
  test that calls multiple plot functions in one process does not leak
  matplotlib state across calls.
* Saved files use the convention
  ``<figure_name>.png`` + ``<figure_name>.svg`` so PNG is good enough
  for the README screenshots and SVG is the canonical input for the
  LaTeX paper.
"""

from __future__ import annotations

import contextlib
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from chronoagent.experiments.metrics import detection_auroc
from chronoagent.monitor.collector import NUM_SIGNALS, SIGNAL_LABELS

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Output filename stems for each figure. Pinned as module constants so
# downstream code (and tests) can import the names without re-deriving
# them. Each function writes ``<stem>.png`` + ``<stem>.svg``.
SIGNAL_DRIFT_FIG_STEM: str = "fig1_signal_drift"
HEALTH_COMPARISON_FIG_STEM: str = "fig2_health_score_comparison"
AWT_BOX_FIG_STEM: str = "fig3_awt_box_plot"
ALLOC_EFF_OVER_TIME_FIG_STEM: str = "fig4_allocation_efficiency_over_time"
ROC_CURVE_FIG_STEM: str = "fig5_roc_curve"
ABLATION_BAR_FIG_STEM: str = "fig6_ablation_bar_chart"

# Default DPI for the saved PNGs. SVG is resolution-independent.
_FIG_DPI: int = 150

# Stable colour palette for multi-experiment plots. Pinned so figures
# generated in different sessions are visually consistent across
# regenerations.
_EXPERIMENT_COLOURS: dict[str, str] = {
    "main_experiment": "#1f77b4",
    "agentpoison_experiment": "#9467bd",
    "ablation_no_forecaster": "#ff7f0e",
    "ablation_no_bocpd": "#2ca02c",
    "ablation_no_health_scores": "#d62728",
    "baseline_sentinel": "#7f7f7f",
    "baseline_no_monitoring": "#17becf",
}

# Display labels (paper-friendly) keyed by experiment name. Falls back
# to the raw name if the experiment is not in the table.
_EXPERIMENT_DISPLAY_NAMES: dict[str, str] = {
    "main_experiment": "ChronoAgent (full)",
    "agentpoison_experiment": "ChronoAgent (full, AgentPoison)",
    "ablation_no_forecaster": "Ablation: no forecaster",
    "ablation_no_bocpd": "Ablation: no BOCPD",
    "ablation_no_health_scores": "Ablation: no health (= no monitoring)",
    "baseline_sentinel": "Baseline: Sentinel",
    "baseline_no_monitoring": "Baseline: no monitoring",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentArtefacts:
    """All persisted artefacts for one experiment.

    Each instance carries the parsed ``aggregate.json``, the per-run
    rows from ``runs.csv``, and (when raw data was persisted) the
    per-run signal matrices and decision streams. The plot functions
    take this dataclass directly so the loader logic can be unit-tested
    in isolation.
    """

    name: str
    base_dir: Path
    aggregate: dict[str, Any]
    runs: list[dict[str, Any]]
    signal_matrices: list[NDArray[np.float64]]
    decision_streams: list[list[dict[str, Any]]]


def load_artefacts(results_dir: Path | str, experiment_name: str) -> ExperimentArtefacts:
    """Load every persisted artefact for ``experiment_name``.

    Args:
        results_dir: Filesystem root the runner wrote into.
        experiment_name: Name of the experiment subdirectory.

    Returns:
        :class:`ExperimentArtefacts` with empty
        ``signal_matrices`` / ``decision_streams`` lists when raw data
        was not persisted (so the consumer can decide what to plot
        based on what is available).

    Raises:
        FileNotFoundError: If ``aggregate.json`` or ``runs.csv`` is
            missing under the experiment directory.
    """
    base = Path(results_dir) / experiment_name
    agg_path = base / "aggregate.json"
    runs_path = base / "runs.csv"
    if not agg_path.is_file():
        raise FileNotFoundError(f"missing aggregate.json: {agg_path}")
    if not runs_path.is_file():
        raise FileNotFoundError(f"missing runs.csv: {runs_path}")

    aggregate = json.loads(agg_path.read_text(encoding="utf-8"))
    with runs_path.open(encoding="utf-8") as fh:
        runs = list(csv.DictReader(fh))

    signal_matrices: list[NDArray[np.float64]] = []
    decision_streams: list[list[dict[str, Any]]] = []
    raw_dir = base / "raw"
    if raw_dir.is_dir():
        sig_files = sorted(raw_dir.glob("run_*_signals.npy"))
        for sig_file in sig_files:
            run_id = sig_file.stem.split("_")[1]
            dec_file = raw_dir / f"run_{run_id}_decisions.json"
            signal_matrices.append(np.load(sig_file))
            if dec_file.is_file():
                payload = json.loads(dec_file.read_text(encoding="utf-8"))
                decision_streams.append(payload["decisions"])
            else:
                decision_streams.append([])

    return ExperimentArtefacts(
        name=experiment_name,
        base_dir=base,
        aggregate=aggregate,
        runs=runs,
        signal_matrices=signal_matrices,
        decision_streams=decision_streams,
    )


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def _save_figure(fig: Figure, output_dir: Path, stem: str) -> list[Path]:
    """Save ``fig`` as PNG + SVG under ``output_dir`` and close it."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext in ("png", "svg"):
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=_FIG_DPI, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _setup_matplotlib() -> Any:
    """Import matplotlib with the headless Agg backend.

    Centralised so each plot function shares one import path. Returns
    the ``pyplot`` module so the caller can build figures.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _display_name(experiment_name: str) -> str:
    return _EXPERIMENT_DISPLAY_NAMES.get(experiment_name, experiment_name)


def _experiment_colour(experiment_name: str, index: int) -> str:
    if experiment_name in _EXPERIMENT_COLOURS:
        return _EXPERIMENT_COLOURS[experiment_name]
    fallback = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    return fallback[index % len(fallback)]


def _coerce_float_or_nan(value: Any) -> float:
    """Parse CSV values that may be empty strings or numeric strings."""
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _figures_dir(results_dir: Path | str, experiment_name: str | None = None) -> Path:
    """Return the figures output directory.

    For per-experiment figures, lives under
    ``<results_dir>/<experiment_name>/figures/``. For multi-experiment
    figures (comparisons), lives at ``<results_dir>/figures/``.
    """
    if experiment_name is None:
        return Path(results_dir) / "figures"
    return Path(results_dir) / experiment_name / "figures"


# ---------------------------------------------------------------------------
# Figure 1: signal drift viz
# ---------------------------------------------------------------------------


def plot_signal_drift(
    results_dir: Path | str,
    experiment_name: str,
    run_index: int = 0,
) -> list[Path]:
    """Time-series of every signal column for one representative run.

    The injection row is drawn as a vertical dashed line; if the
    detector flagged any row in the same run, the first-flagged row is
    drawn as a vertical solid line so the AWT is visible at a glance.

    Args:
        results_dir: Runner output root.
        experiment_name: Experiment subdirectory name.
        run_index: Which run to draw signals from. Defaults to the
            first run, which has seed ``cfg.seed`` and is the most
            stable choice for paper-quality figures.

    Returns:
        List of created file paths (PNG + SVG).

    Raises:
        FileNotFoundError: If raw signal data was not persisted for
            this experiment.
        IndexError: If ``run_index`` is outside the persisted range.
    """
    plt = _setup_matplotlib()
    artefacts = load_artefacts(results_dir, experiment_name)
    if not artefacts.signal_matrices:
        raise FileNotFoundError(
            f"no raw signal data for experiment {experiment_name!r}; re-run with collect_raw=True"
        )
    if run_index < 0 or run_index >= len(artefacts.signal_matrices):
        raise IndexError(f"run_index {run_index} outside [0, {len(artefacts.signal_matrices)})")

    signal_matrix = artefacts.signal_matrices[run_index]
    if signal_matrix.shape[1] != NUM_SIGNALS:
        raise ValueError(
            f"signal matrix has {signal_matrix.shape[1]} columns, expected {NUM_SIGNALS}"
        )

    injection_step = int(artefacts.aggregate["injection_step"])
    first_flagged: int | None = None
    if run_index < len(artefacts.runs):
        raw_value = artefacts.runs[run_index].get("first_flagged_step", "")
        if raw_value not in ("", None):
            try:
                first_flagged = int(raw_value)
            except (TypeError, ValueError):
                first_flagged = None

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    if run_index < len(artefacts.runs):
        seed_label = artefacts.runs[run_index].get("seed", "?")
    else:
        seed_label = "?"
    fig.suptitle(
        f"Signal drift - {_display_name(experiment_name)} (run {run_index}, seed {seed_label})",
        fontsize=14,
    )
    n_steps = signal_matrix.shape[0]
    x = np.arange(n_steps)
    for col_idx, label in enumerate(SIGNAL_LABELS):
        ax = axes[col_idx // 3, col_idx % 3]
        ax.plot(x, signal_matrix[:, col_idx], color="#1f77b4", linewidth=1.4)
        ax.axvline(
            injection_step,
            color="#d62728",
            linestyle="--",
            linewidth=1.2,
            label=f"injection (step {injection_step})",
        )
        if first_flagged is not None:
            ax.axvline(
                first_flagged,
                color="#2ca02c",
                linestyle="-",
                linewidth=1.2,
                label=f"first flagged (step {first_flagged})",
            )
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("step")
        ax.set_ylabel("value")
        if col_idx == 0:
            ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    return _save_figure(fig, _figures_dir(results_dir, experiment_name), SIGNAL_DRIFT_FIG_STEM)


# ---------------------------------------------------------------------------
# Figure 2: health score comparison
# ---------------------------------------------------------------------------


def plot_health_score_comparison(
    results_dir: Path | str,
    experiment_names: list[str],
) -> list[Path]:
    """Per-step detector score, mean across runs, for each experiment.

    The figure has one line per experiment with a shaded
    mean +/- std band. The y-axis is the combined detection score in
    ``[0, 1]``; the x-axis is step index. The injection step (taken
    from the first experiment's aggregate.json) is marked with a
    vertical dashed line.

    Note: under the 10.6 design there is no single scalar "health
    score" output, so the per-step combined score from the detector
    decision stream is used as the health-score stand-in. The figure
    title carries this caveat.

    Args:
        results_dir: Runner output root.
        experiment_names: Experiments to overlay. Order determines the
            legend ordering.

    Returns:
        List of created file paths.

    Raises:
        FileNotFoundError: If any named experiment is missing raw data.
        ValueError: If ``experiment_names`` is empty.
    """
    if not experiment_names:
        raise ValueError("experiment_names must not be empty")
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))

    injection_step: int | None = None
    for idx, name in enumerate(experiment_names):
        artefacts = load_artefacts(results_dir, name)
        if injection_step is None:
            injection_step = int(artefacts.aggregate["injection_step"])
        if not artefacts.decision_streams:
            raise FileNotFoundError(
                f"no raw decision data for experiment {name!r}; re-run with collect_raw=True"
            )
        n_runs = len(artefacts.decision_streams)
        n_steps = max((len(d) for d in artefacts.decision_streams), default=0)
        score_matrix = np.full((n_runs, n_steps), np.nan, dtype=np.float64)
        for r_idx, decisions in enumerate(artefacts.decision_streams):
            for step_idx, dec in enumerate(decisions):
                score_matrix[r_idx, step_idx] = float(dec.get("score", 0.0))
        mean = np.nanmean(score_matrix, axis=0)
        std = np.nanstd(score_matrix, axis=0)
        x = np.arange(n_steps)
        colour = _experiment_colour(name, idx)
        ax.plot(x, mean, label=_display_name(name), color=colour, linewidth=1.6)
        ax.fill_between(x, mean - std, mean + std, color=colour, alpha=0.2)

    if injection_step is not None:
        ax.axvline(
            injection_step,
            color="#d62728",
            linestyle="--",
            linewidth=1.0,
            label=f"injection (step {injection_step})",
        )
    ax.set_xlabel("step")
    ax.set_ylabel("detector combined score (per-step health stand-in)")
    ax.set_title("Health score comparison across experiments")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    return _save_figure(fig, _figures_dir(results_dir), HEALTH_COMPARISON_FIG_STEM)


# ---------------------------------------------------------------------------
# Figure 3: AWT box plot
# ---------------------------------------------------------------------------


def plot_awt_box(
    results_dir: Path | str,
    experiment_names: list[str],
) -> list[Path]:
    """Box plot of per-run advance warning time across experiments.

    Pulls AWT values from each experiment's ``runs.csv``. Runs where
    the detector never fired (empty AWT cell) are dropped from the
    box plot but counted in the per-experiment label below the box.

    Args:
        results_dir: Runner output root.
        experiment_names: Experiments to compare.

    Returns:
        List of created file paths.
    """
    if not experiment_names:
        raise ValueError("experiment_names must not be empty")
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(experiment_names)), 6))

    box_data: list[list[float]] = []
    labels: list[str] = []
    for idx, name in enumerate(experiment_names):
        artefacts = load_artefacts(results_dir, name)
        values: list[float] = []
        for row in artefacts.runs:
            raw = row.get("advance_warning_time", "")
            if raw not in ("", None):
                with contextlib.suppress(TypeError, ValueError):
                    values.append(float(raw))
        box_data.append(values)
        n_total = len(artefacts.runs)
        labels.append(f"{_display_name(name)}\n(n={len(values)}/{n_total})")
        # Suppress unused-loop-var lint by referencing idx for colour map
        _ = _experiment_colour(name, idx)

    # Use empty placeholder for fully-empty boxes so matplotlib does
    # not crash on the box draw step.
    drawable = [d if d else [0.0] for d in box_data]
    parts = ax.boxplot(drawable, tick_labels=labels, patch_artist=True)
    for idx, patch in enumerate(parts["boxes"]):
        patch.set_facecolor(_experiment_colour(experiment_names[idx], idx))
        patch.set_alpha(0.4)
        if not box_data[idx]:
            patch.set_facecolor("#cccccc")
            patch.set_hatch("///")

    ax.axhline(0.0, color="#888888", linestyle=":", linewidth=1.0)
    ax.set_ylabel("Advance Warning Time (steps)")
    ax.set_title("AWT distribution across experiments")
    fig.autofmt_xdate(rotation=20, ha="right")
    fig.tight_layout()

    return _save_figure(fig, _figures_dir(results_dir), AWT_BOX_FIG_STEM)


# ---------------------------------------------------------------------------
# Figure 4: allocation efficiency over time
# ---------------------------------------------------------------------------


def plot_allocation_efficiency_over_time(
    results_dir: Path | str,
    experiment_names: list[str],
) -> list[Path]:
    """Cumulative success rate per step, mean across runs.

    Reads ``raw/run_<i>_decisions.json`` for each named experiment,
    computes the per-step cumulative ``success`` fraction, then
    averages across runs. Plots one line per experiment with a shaded
    +/- 1 std band.
    """
    if not experiment_names:
        raise ValueError("experiment_names must not be empty")
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))

    injection_step: int | None = None
    for idx, name in enumerate(experiment_names):
        artefacts = load_artefacts(results_dir, name)
        if injection_step is None:
            injection_step = int(artefacts.aggregate["injection_step"])
        if not artefacts.decision_streams:
            raise FileNotFoundError(
                f"no raw decision data for experiment {name!r}; re-run with collect_raw=True"
            )
        n_runs = len(artefacts.decision_streams)
        n_steps = max((len(d) for d in artefacts.decision_streams), default=0)
        cum = np.full((n_runs, n_steps), np.nan, dtype=np.float64)
        for r_idx, decisions in enumerate(artefacts.decision_streams):
            running = 0
            for step_idx, dec in enumerate(decisions):
                running += 1 if bool(dec.get("success", False)) else 0
                cum[r_idx, step_idx] = running / (step_idx + 1)
        mean = np.nanmean(cum, axis=0)
        std = np.nanstd(cum, axis=0)
        x = np.arange(n_steps)
        colour = _experiment_colour(name, idx)
        ax.plot(x, mean, label=_display_name(name), color=colour, linewidth=1.6)
        ax.fill_between(x, mean - std, mean + std, color=colour, alpha=0.2)

    if injection_step is not None:
        ax.axvline(
            injection_step,
            color="#d62728",
            linestyle="--",
            linewidth=1.0,
            label=f"injection (step {injection_step})",
        )
    ax.set_xlabel("step")
    ax.set_ylabel("cumulative allocation efficiency")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Allocation efficiency over time")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    return _save_figure(fig, _figures_dir(results_dir), ALLOC_EFF_OVER_TIME_FIG_STEM)


# ---------------------------------------------------------------------------
# Figure 5: ROC curve
# ---------------------------------------------------------------------------


def plot_roc_curve(
    results_dir: Path | str,
    experiment_names: list[str],
) -> list[Path]:
    """Pooled ROC curve per experiment, plus per-experiment AUROC.

    Pools the per-step ``score`` field from every run in each
    experiment against the ground-truth labels (``False`` for the
    clean phase, ``True`` for the post-injection phase) and computes
    the empirical ROC. The legend reports the per-experiment pooled
    AUROC computed via the 10.2
    :func:`~chronoagent.experiments.metrics.detection_auroc` helper so
    the figure label and the aggregate.json metric agree.
    """
    if not experiment_names:
        raise ValueError("experiment_names must not be empty")
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 7))

    for idx, name in enumerate(experiment_names):
        artefacts = load_artefacts(results_dir, name)
        if not artefacts.decision_streams:
            raise FileNotFoundError(
                f"no raw decision data for experiment {name!r}; re-run with collect_raw=True"
            )
        injection_step = int(artefacts.aggregate["injection_step"])
        scores: list[float] = []
        labels: list[bool] = []
        for decisions in artefacts.decision_streams:
            for step_idx, dec in enumerate(decisions):
                scores.append(float(dec.get("score", 0.0)))
                labels.append(step_idx >= injection_step)
        score_arr = np.asarray(scores, dtype=np.float64)
        label_arr = np.asarray(labels, dtype=bool)
        fpr, tpr = _empirical_roc(label_arr, score_arr)
        auc = detection_auroc(label_arr, score_arr)
        colour = _experiment_colour(name, idx)
        ax.plot(
            fpr,
            tpr,
            label=f"{_display_name(name)} (AUC={auc:.3f})",
            color=colour,
            linewidth=1.6,
        )

    ax.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1.0, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("ROC curves (per-step scores pooled across runs)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    return _save_figure(fig, _figures_dir(results_dir), ROC_CURVE_FIG_STEM)


def _empirical_roc(
    y_true: NDArray[np.bool_],
    y_scores: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the empirical ROC curve.

    Returns ``(fpr, tpr)`` arrays sorted by ascending FPR. Falls back
    to a degenerate ``[0, 1]`` step when one class is missing or the
    score array is empty so the figure can still draw a legend entry.
    """
    if y_true.size == 0 or y_scores.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    pos = int(np.sum(y_true))
    neg = int(y_true.size - pos)
    if pos == 0 or neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    order = np.argsort(-y_scores)
    sorted_labels = y_true[order]
    tps = np.cumsum(sorted_labels.astype(np.int64))
    fps = np.cumsum((~sorted_labels).astype(np.int64))
    tpr = np.concatenate([[0.0], tps / pos])
    fpr = np.concatenate([[0.0], fps / neg])
    return fpr, tpr


# ---------------------------------------------------------------------------
# Figure 6: ablation bar chart
# ---------------------------------------------------------------------------


_BAR_METRICS: tuple[tuple[str, str], ...] = (
    ("advance_warning_time", "Advance Warning Time (steps)"),
    ("allocation_efficiency_score", "Allocation efficiency"),
    ("detection_auroc_score", "Detection AUROC"),
    ("detection_f1_score", "Detection F1"),
)


def plot_ablation_bar_chart(
    results_dir: Path | str,
    experiment_names: list[str],
) -> list[Path]:
    """Four metrics across all experiments as a grouped bar chart.

    Reads only ``aggregate.json`` for each named experiment, so this
    figure works without raw data persistence.

    Each subplot is one metric (AWT, allocation efficiency, AUROC, F1)
    with one bar per experiment and 95% CI error bars where the
    aggregate's ``n`` is at least 2 (so a one-sample bound does not
    misleadingly imply a confidence interval).
    """
    if not experiment_names:
        raise ValueError("experiment_names must not be empty")
    plt = _setup_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for ax, (metric_key, metric_label) in zip(axes.flat, _BAR_METRICS, strict=True):
        means: list[float] = []
        ci_low_errors: list[float] = []
        ci_high_errors: list[float] = []
        x_labels: list[str] = []
        colours: list[str] = []
        for idx, name in enumerate(experiment_names):
            artefacts = load_artefacts(results_dir, name)
            metric = artefacts.aggregate["metrics"][metric_key]
            mean = _coerce_float_or_nan(metric.get("mean"))
            ci_low = _coerce_float_or_nan(metric.get("ci_low"))
            ci_high = _coerce_float_or_nan(metric.get("ci_high"))
            means.append(0.0 if np.isnan(mean) else mean)
            n = int(metric.get("n", 0))
            if n >= 2 and not (np.isnan(ci_low) or np.isnan(ci_high)):
                ci_low_errors.append(max(0.0, mean - ci_low))
                ci_high_errors.append(max(0.0, ci_high - mean))
            else:
                ci_low_errors.append(0.0)
                ci_high_errors.append(0.0)
            x_labels.append(_display_name(name))
            colours.append(_experiment_colour(name, idx))

        x = np.arange(len(experiment_names))
        bars = ax.bar(x, means, color=colours, alpha=0.8)
        ax.errorbar(
            x,
            means,
            yerr=[ci_low_errors, ci_high_errors],
            fmt="none",
            ecolor="#222222",
            capsize=3,
            linewidth=1.0,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.grid(axis="y", linestyle=":", linewidth=0.5)
        # Suppress unused-bars lint by touching the return value.
        _ = bars

    fig.suptitle("Ablation results across experiments", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    return _save_figure(fig, _figures_dir(results_dir), ABLATION_BAR_FIG_STEM)


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


def generate_all_plots(
    results_dir: Path | str,
    experiment_names: list[str],
    *,
    drift_experiment: str | None = None,
    drift_run_index: int = 0,
) -> list[Path]:
    """Run every figure function for ``experiment_names``.

    Args:
        results_dir: Runner output root.
        experiment_names: Experiments included in every multi-experiment
            comparison figure (figs 2, 3, 4, 5, 6).
        drift_experiment: Which experiment's signal matrix to draw the
            single-run signal-drift figure from. Defaults to the first
            entry of ``experiment_names``.
        drift_run_index: Which run to draw signals from. Defaults to
            run 0.

    Returns:
        Concatenated list of every file path created.
    """
    if not experiment_names:
        raise ValueError("experiment_names must not be empty")
    drift_target = drift_experiment if drift_experiment is not None else experiment_names[0]
    paths: list[Path] = []
    paths.extend(plot_signal_drift(results_dir, drift_target, run_index=drift_run_index))
    paths.extend(plot_health_score_comparison(results_dir, experiment_names))
    paths.extend(plot_awt_box(results_dir, experiment_names))
    paths.extend(plot_allocation_efficiency_over_time(results_dir, experiment_names))
    paths.extend(plot_roc_curve(results_dir, experiment_names))
    paths.extend(plot_ablation_bar_chart(results_dir, experiment_names))
    return paths
