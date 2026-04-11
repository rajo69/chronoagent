"""Phase 10 task 10.8 paper LaTeX tables.

This module ships three LaTeX ``tabular`` generators consumed by the
Phase 10 paper. Every function reads the artefacts the 10.6
:class:`~chronoagent.experiments.experiment_runner.ExperimentRunner`
persists under ``<results_dir>/<experiment_name>/aggregate.json`` (no
raw data needed) and writes a ``.tex`` file under
``<results_dir>/tables/`` ready to ``\\input{...}`` from the paper
source.

Three tables, mapping one-to-one onto the paper's Section 5 numbers:

#. :func:`make_main_results_table` (Section 5 headline) - one row per
   experiment in the supplied list with columns AWT, AUROC, F1, and
   allocation efficiency. Each cell is rendered as ``mean \\pm
   half_ci`` from the corresponding ``aggregate.json`` ``MetricAggregate``,
   or as ``--`` when the mean is NaN (e.g. NoMonitoring AWT). The
   highest mean per column wraps in ``\\textbf{...}``.
#. :func:`make_ablation_table` - full system + the three named
   ablations. Each ablation cell additionally carries the delta vs the
   full-system row, formatted as ``0.732 (-0.151)`` so the paper's
   ablation table has a clean per-knob contribution column.
#. :func:`make_signal_validation_table` - per-signal Cohen's d from
   Phase 1, taking a sequence of :class:`SignalStatRow` so the 10.9
   CLI (or a hand-crafted test fixture) can drive the row data
   independently of where the Phase 1 SignalValidationRunner is run.

The convenience function :func:`generate_all_tables` runs every table
function and returns the list of files written; it is the entry point
the 10.9 CLI will hook into.

Implementation notes:

* Output is plain LaTeX ``tabular`` with ``\\hline`` separators - no
  booktabs dep required, so the only LaTeX package needed in the
  paper preamble is the standard ``\\usepackage{}`` set.
* All cell formatters live in private ``_format_*`` helpers so the
  test file can pin per-metric decimal rules in isolation without
  re-rendering a whole table.
* Experiment display names reuse the ``_EXPERIMENT_DISPLAY_NAMES``
  table from :mod:`chronoagent.experiments.analysis.plots` so the
  paper's figures and tables agree on row labels.
* The module deliberately does NOT import matplotlib or any plotting
  code: importing ``tables.py`` is cheap, which lets the runner CLI
  emit tables in an environment without a graphics stack.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chronoagent.experiments.analysis.plots import (
    _EXPERIMENT_DISPLAY_NAMES,
    load_artefacts,
)

# Output filename stems. Pinned as module constants so downstream
# code (and tests) can import the names without re-deriving them.
MAIN_RESULTS_TABLE_STEM: str = "table1_main_results"
ABLATION_TABLE_STEM: str = "table2_ablation"
SIGNAL_VALIDATION_TABLE_STEM: str = "table3_signal_validation"

# Decimal places per metric. Locked here so every table function and
# every test asserts against the same numbers.
_AWT_DECIMALS: int = 1
_AUROC_DECIMALS: int = 3
_F1_DECIMALS: int = 3
_ALLOC_EFF_DECIMALS: int = 3
_COHENS_D_DECIMALS: int = 2

# (column_key, header_label, decimals) tuples used by the main results
# and ablation tables. Both tables share the same metric column set so
# the per-row data is identical and only the row composition + delta
# rendering differ.
_METRIC_COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("advance_warning_time", "AWT (steps)", _AWT_DECIMALS),
    ("detection_auroc_score", "AUROC", _AUROC_DECIMALS),
    ("detection_f1_score", "F1", _F1_DECIMALS),
    ("allocation_efficiency_score", "Alloc eff", _ALLOC_EFF_DECIMALS),
)

# String emitted in place of NaN means / undefined CI half-widths.
_NAN_PLACEHOLDER: str = "--"


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalStatRow:
    """One row of the Phase 1 signal validation table.

    Mirrors :class:`chronoagent.experiments.analysis.phase1.SignalStats`
    intentionally so the 10.9 CLI (or a future runner that re-runs
    Phase 1) can convert one to the other with a one-line constructor.
    Kept independent of the Phase 1 dataclass so this module does not
    pull the Phase 1 runner's heavy ChromaDB / MockBackend imports
    transitively.

    Attributes:
        label: Signal column name (e.g. ``"kl_divergence"``).
        clean_mean: Per-signal mean across the clean phase.
        clean_std: Per-signal std (regularised) across the clean phase.
        poisoned_mean: Per-signal mean across the poisoned phase.
        poisoned_std: Per-signal std across the poisoned phase.
        cohens_d: Pooled Cohen's d effect size, always non-negative.
        large_effect: ``True`` iff ``cohens_d > 0.8``.
    """

    label: str
    clean_mean: float
    clean_std: float
    poisoned_mean: float
    poisoned_std: float
    cohens_d: float
    large_effect: bool


# ---------------------------------------------------------------------------
# Cell formatters
# ---------------------------------------------------------------------------


def _format_mean_ci(
    mean: float,
    ci_low: float,
    ci_high: float,
    *,
    decimals: int,
) -> str:
    """Render ``mean \\pm half_ci`` or the NaN placeholder.

    The half-width uses ``max(mean - ci_low, ci_high - mean)`` to
    conservatively report the larger side of an asymmetric CI; for the
    symmetric Student-t CI the 10.6 aggregator emits, both sides are
    equal so this is the natural width.

    NaN handling:

    * NaN ``mean`` -> :data:`_NAN_PLACEHOLDER`.
    * NaN ``ci_low`` / ``ci_high`` (e.g. single-sample aggregate) ->
      print mean alone, no ``\\pm`` block.
    """
    if math.isnan(mean):
        return _NAN_PLACEHOLDER
    if math.isnan(ci_low) or math.isnan(ci_high):
        return f"{mean:.{decimals}f}"
    half = max(mean - ci_low, ci_high - mean)
    if half < 0:
        # Shouldn't happen with the 10.6 aggregator, but defensive.
        half = 0.0
    return f"{mean:.{decimals}f} $\\pm$ {half:.{decimals}f}"


def _format_delta(
    mean: float,
    baseline: float,
    *,
    decimals: int,
) -> str:
    """Format an ablation delta as ``(+0.012)`` or ``(-0.151)``.

    Returns the empty string when either value is NaN so the cell is
    just the bare ``mean +/- ci``. The leading sign is always present
    so a positive delta is unambiguous.
    """
    if math.isnan(mean) or math.isnan(baseline):
        return ""
    delta = mean - baseline
    sign = "+" if delta >= 0 else "-"
    return f" ({sign}{abs(delta):.{decimals}f})"


def _is_better(value: float, current_best: float) -> bool:
    """Higher is better for every metric in the main results table.

    Both AWT (more advance warning) and the three quality metrics
    (AUROC, F1, allocation efficiency) read upward as a positive
    signal, so a single comparison rule works across the four
    columns. NaN values are never "better" than a real number.
    """
    if math.isnan(value):
        return False
    if math.isnan(current_best):
        return True
    return value > current_best


def _coerce_metric_value(value: Any) -> float:
    """Parse JSON metric values that may be ``None`` (NaN) or numeric."""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _display_name(experiment_name: str) -> str:
    return _EXPERIMENT_DISPLAY_NAMES.get(experiment_name, experiment_name)


# ---------------------------------------------------------------------------
# Main results table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MetricCell:
    """Internal scratch struct used to build a per-cell rendering."""

    mean: float
    ci_low: float
    ci_high: float
    n: int
    decimals: int


def _load_metric_row(
    results_dir: Path | str,
    experiment_name: str,
) -> dict[str, _MetricCell]:
    """Pull every column's :class:`MetricAggregate` for one experiment."""
    artefacts = load_artefacts(results_dir, experiment_name)
    metrics_block = artefacts.aggregate["metrics"]
    row: dict[str, _MetricCell] = {}
    for key, _label, decimals in _METRIC_COLUMNS:
        block = metrics_block[key]
        row[key] = _MetricCell(
            mean=_coerce_metric_value(block.get("mean")),
            ci_low=_coerce_metric_value(block.get("ci_low")),
            ci_high=_coerce_metric_value(block.get("ci_high")),
            n=int(block.get("n", 0)),
            decimals=decimals,
        )
    return row


def make_main_results_table(
    results_dir: Path | str,
    experiment_names: list[str],
    output_path: Path | str | None = None,
) -> Path:
    """Generate the headline LaTeX results table.

    Args:
        results_dir: Runner output root.
        experiment_names: Experiments to include as rows. Order is
            preserved in the output.
        output_path: Optional override for the output ``.tex`` file
            location. Defaults to
            ``<results_dir>/tables/<MAIN_RESULTS_TABLE_STEM>.tex``.

    Returns:
        Path to the written ``.tex`` file.

    Raises:
        ValueError: If ``experiment_names`` is empty.
        FileNotFoundError: If any named experiment is missing
            ``aggregate.json`` (propagated from the loader).
    """
    if not experiment_names:
        raise ValueError("experiment_names must not be empty")

    rows: list[tuple[str, dict[str, _MetricCell]]] = []
    for name in experiment_names:
        rows.append((name, _load_metric_row(results_dir, name)))

    # Find the per-column best mean so we can bold it.
    best_per_column: dict[str, float] = {key: float("nan") for key, _, _ in _METRIC_COLUMNS}
    for _, cells in rows:
        for key in best_per_column:
            if _is_better(cells[key].mean, best_per_column[key]):
                best_per_column[key] = cells[key].mean

    lines: list[str] = []
    lines.append("% Auto-generated by chronoagent.experiments.analysis.tables")
    lines.append("% Phase 10 task 10.8 - main results table")
    lines.append("\\begin{tabular}{l" + "c" * len(_METRIC_COLUMNS) + "}")
    lines.append("\\hline")
    header_cells = ["Experiment"] + [label for _, label, _ in _METRIC_COLUMNS]
    lines.append(" & ".join(header_cells) + " \\\\")
    lines.append("\\hline")
    for name, cells in rows:
        cell_strings: list[str] = [_display_name(name)]
        for key, _label, decimals in _METRIC_COLUMNS:
            cell = cells[key]
            rendered = _format_mean_ci(cell.mean, cell.ci_low, cell.ci_high, decimals=decimals)
            if (
                not math.isnan(cell.mean)
                and not math.isnan(best_per_column[key])
                and cell.mean == best_per_column[key]
            ):
                rendered = f"\\textbf{{{rendered}}}"
            cell_strings.append(rendered)
        lines.append(" & ".join(cell_strings) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")

    target = _resolve_table_path(results_dir, output_path, MAIN_RESULTS_TABLE_STEM)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Ablation table
# ---------------------------------------------------------------------------


def make_ablation_table(
    results_dir: Path | str,
    full_system_name: str,
    ablation_names: list[str],
    output_path: Path | str | None = None,
) -> Path:
    """Generate the per-knob ablation LaTeX table.

    The first row is the full system; subsequent rows are the named
    ablations, each with a parenthesised delta vs the full system on
    every metric column.

    Args:
        results_dir: Runner output root.
        full_system_name: Experiment name of the full system row
            (typically ``"main_experiment"``); used as the delta
            baseline.
        ablation_names: Experiment names of the ablation rows. Order
            preserved.
        output_path: Optional override for the output ``.tex`` file.
            Defaults to
            ``<results_dir>/tables/<ABLATION_TABLE_STEM>.tex``.

    Returns:
        Path to the written ``.tex`` file.

    Raises:
        ValueError: If ``ablation_names`` is empty.
    """
    if not ablation_names:
        raise ValueError("ablation_names must not be empty")

    full_row = _load_metric_row(results_dir, full_system_name)
    ablation_rows: list[tuple[str, dict[str, _MetricCell]]] = []
    for name in ablation_names:
        ablation_rows.append((name, _load_metric_row(results_dir, name)))

    lines: list[str] = []
    lines.append("% Auto-generated by chronoagent.experiments.analysis.tables")
    lines.append("% Phase 10 task 10.8 - ablation table")
    lines.append("\\begin{tabular}{l" + "c" * len(_METRIC_COLUMNS) + "}")
    lines.append("\\hline")
    header_cells = ["Condition"] + [label for _, label, _ in _METRIC_COLUMNS]
    lines.append(" & ".join(header_cells) + " \\\\")
    lines.append("\\hline")

    # Full-system row, no delta column.
    full_cells: list[str] = [_display_name(full_system_name)]
    for key, _label, decimals in _METRIC_COLUMNS:
        cell = full_row[key]
        full_cells.append(_format_mean_ci(cell.mean, cell.ci_low, cell.ci_high, decimals=decimals))
    lines.append(" & ".join(full_cells) + " \\\\")
    lines.append("\\hline")

    for name, cells in ablation_rows:
        ablation_cells: list[str] = [_display_name(name)]
        for key, _label, decimals in _METRIC_COLUMNS:
            cell = cells[key]
            base_str = _format_mean_ci(cell.mean, cell.ci_low, cell.ci_high, decimals=decimals)
            delta_str = _format_delta(cell.mean, full_row[key].mean, decimals=decimals)
            ablation_cells.append(base_str + delta_str)
        lines.append(" & ".join(ablation_cells) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")

    target = _resolve_table_path(results_dir, output_path, ABLATION_TABLE_STEM)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Signal validation table
# ---------------------------------------------------------------------------


def make_signal_validation_table(
    rows: Sequence[SignalStatRow],
    output_path: Path | str,
) -> Path:
    """Generate the Phase 1 per-signal Cohen's d LaTeX table.

    Args:
        rows: One :class:`SignalStatRow` per signal column. The 10.9
            CLI builds these from a fresh
            :class:`chronoagent.experiments.runner.SignalValidationRunner`
            run, but tests can hand-craft them inline.
        output_path: Where to write the ``.tex`` file. Required
            (unlike the multi-experiment tables) because Phase 1
            results do not live under a 10.6 results directory.

    Returns:
        Path to the written ``.tex`` file.

    Raises:
        ValueError: If ``rows`` is empty.
    """
    if not rows:
        raise ValueError("rows must not be empty")

    lines: list[str] = []
    lines.append("% Auto-generated by chronoagent.experiments.analysis.tables")
    lines.append("% Phase 10 task 10.8 - Phase 1 signal validation table")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\hline")
    lines.append(
        "Signal & Clean $\\mu$ & Clean $\\sigma$ & Poison $\\mu$ "
        "& Poison $\\sigma$ & Cohen's $d$ \\\\"
    )
    lines.append("\\hline")
    for row in rows:
        d_cell = f"{row.cohens_d:.{_COHENS_D_DECIMALS}f}"
        if row.large_effect:
            d_cell = f"\\textbf{{{d_cell}}}"
        cells = [
            _escape_latex(row.label),
            f"{row.clean_mean:.4f}",
            f"{row.clean_std:.4f}",
            f"{row.poisoned_mean:.4f}",
            f"{row.poisoned_std:.4f}",
            d_cell,
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


def generate_all_tables(
    results_dir: Path | str,
    main_experiment_names: list[str],
    full_system_name: str,
    ablation_names: list[str],
    *,
    signal_validation_rows: Sequence[SignalStatRow] | None = None,
) -> list[Path]:
    """Run every table generator for the supplied configuration.

    Args:
        results_dir: Runner output root.
        main_experiment_names: Experiments to include in the main
            results table (full + generalisation + baselines).
        full_system_name: Experiment used as the ablation baseline.
        ablation_names: Ablation experiments to include after the
            full-system row in the ablation table.
        signal_validation_rows: Optional Phase 1 rows. When omitted,
            the signal validation table is skipped (since Phase 1
            data is not derivable from 10.6 artefacts alone).

    Returns:
        List of every ``.tex`` file written. Order:
        main results, ablation, then signal validation if supplied.
    """
    paths: list[Path] = []
    paths.append(make_main_results_table(results_dir, main_experiment_names))
    paths.append(make_ablation_table(results_dir, full_system_name, ablation_names))
    if signal_validation_rows is not None:
        sig_path = Path(results_dir) / "tables" / f"{SIGNAL_VALIDATION_TABLE_STEM}.tex"
        paths.append(make_signal_validation_table(signal_validation_rows, sig_path))
    return paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_table_path(
    results_dir: Path | str,
    output_path: Path | str | None,
    default_stem: str,
) -> Path:
    if output_path is not None:
        return Path(output_path)
    return Path(results_dir) / "tables" / f"{default_stem}.tex"


def _escape_latex(text: str) -> str:
    """Escape the LaTeX-special characters that show up in signal labels."""
    replacements = {
        "_": "\\_",
        "%": "\\%",
        "#": "\\#",
        "&": "\\&",
        "$": "\\$",
    }
    out = text
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out
