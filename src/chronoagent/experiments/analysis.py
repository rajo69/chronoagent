"""Analysis script for Phase 1 signal validation — task 1.8.

Produces per-signal time-series plots, PELT changepoint detection,
Advance Warning Time (AWT) estimation, and a decision matrix CSV/table.

Usage::

    from pathlib import Path
    from chronoagent.experiments.analysis import AnalysisConfig, SignalAnalyzer

    config = AnalysisConfig.from_yaml_section(yaml_cfg["analysis"])
    analyzer = SignalAnalyzer(result, config)
    analyzer.run(output_dir=Path("results/"))
    print(analyzer.decision_table())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import ruptures as rpt
from numpy.typing import NDArray

from chronoagent.experiments.runner import ExperimentResult
from chronoagent.monitor.collector import SIGNAL_LABELS

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class AnalysisConfig:
    """Analysis parameters consumed by :class:`SignalAnalyzer`.

    All fields map directly to the ``analysis:`` section of the experiment
    YAML produced in task 1.7.

    Attributes:
        large_effect_threshold: Cohen's d threshold for "large effect" label.
        go_threshold: Minimum large-effect signals needed for a GO ruling.
        changepoint_model: ruptures cost model (``'rbf'``, ``'l2'``, ``'l1'``).
        changepoint_min_size: Minimum segment length in steps.
        changepoint_jump: Sub-sampling step (1 = exact search).
        changepoint_penalty: Penalty passed to ``Pelt.predict(pen=...)``.
        awt_detection_multiplier: σ multiplier for threshold crossing.
        awt_min_awt: Floor for AWT (0 = concurrent detection counts).
        save_csv: Write decision matrix as ``decision_matrix.csv``.
        save_plots: Write per-signal plots.
        plot_format: Image format — ``'png'`` or ``'svg'``.
        plot_dpi: DPI for raster plots (ignored for SVG).
    """

    large_effect_threshold: float = 0.8
    go_threshold: int = 2
    changepoint_model: str = "rbf"
    changepoint_min_size: int = 3
    changepoint_jump: int = 1
    changepoint_penalty: float = 1.0
    awt_detection_multiplier: float = 2.0
    awt_min_awt: int = 0
    save_csv: bool = True
    save_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 150

    @classmethod
    def from_yaml_section(cls, section: dict[str, Any]) -> AnalysisConfig:
        """Build an :class:`AnalysisConfig` from the raw YAML ``analysis:`` dict.

        Args:
            section: Parsed YAML ``analysis:`` mapping.

        Returns:
            Populated :class:`AnalysisConfig`.
        """
        cp = section.get("changepoint", {})
        awt = section.get("awt", {})
        out = section.get("output", {})
        return cls(
            large_effect_threshold=float(section.get("large_effect_threshold", 0.8)),
            go_threshold=int(section.get("go_threshold", 2)),
            changepoint_model=str(cp.get("model", "rbf")),
            changepoint_min_size=int(cp.get("min_size", 3)),
            changepoint_jump=int(cp.get("jump", 1)),
            changepoint_penalty=float(cp.get("penalty", 1.0)),
            awt_detection_multiplier=float(awt.get("detection_multiplier", 2.0)),
            awt_min_awt=int(awt.get("min_awt", 0)),
            save_csv=bool(out.get("save_csv", True)),
            save_plots=bool(out.get("save_plots", True)),
            plot_format=str(out.get("plot_format", "png")),
            plot_dpi=int(out.get("plot_dpi", 150)),
        )


# ---------------------------------------------------------------------------
# Per-signal analysis result
# ---------------------------------------------------------------------------


@dataclass
class SignalAnalysisResult:
    """PELT changepoint detection and AWT for a single signal.

    Attributes:
        label: Signal name.
        pelt_breakpoint: Detected changepoint index in the full
            (clean ‖ poisoned) series.  ``-1`` if PELT found no breakpoint.
        awt: Advance Warning Time in steps.  Positive = early warning;
            ``0`` = concurrent; negative = late detection;
            ``-1`` when no breakpoint was found.
        threshold: Threshold used for AWT crossing detection
            (``clean_mean + multiplier × clean_std``).
        first_crossing: Step index where the signal first exceeded *threshold*
            in the full series.  ``-1`` if no crossing occurred.
    """

    label: str
    pelt_breakpoint: int
    awt: int
    threshold: float
    first_crossing: int


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


@dataclass
class SignalAnalyzer:
    """Runs PELT changepoint detection, AWT estimation, and output generation.

    Args:
        result: :class:`~chronoagent.experiments.runner.ExperimentResult` from
            :class:`~chronoagent.experiments.runner.SignalValidationRunner`.
        config: Analysis configuration.
    """

    result: ExperimentResult
    config: AnalysisConfig
    signal_analyses: list[SignalAnalysisResult] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.signal_analyses = self._analyse_all_signals()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, output_dir: Path) -> None:
        """Generate all output artefacts in *output_dir*.

        Writes:
        - ``decision_matrix.csv`` (if :attr:`AnalysisConfig.save_csv`)
        - ``signals_<attack>.png`` or ``.svg`` (if :attr:`AnalysisConfig.save_plots`)

        Args:
            output_dir: Directory to write artefacts into (created if absent).
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.save_csv:
            self._save_csv(output_dir)

        if self.config.save_plots:
            self._save_plots(output_dir)

    def decision_table(self) -> str:
        """Return the full decision matrix as a formatted string.

        Columns: signal, Cohen's d, large?, PELT CP, AWT, threshold crossing.

        Returns:
            Multi-line formatted table string.
        """
        header = (
            f"{'Signal':<26} {'Cohen d':>9} {'Large?':>7} "
            f"{'PELT CP':>8} {'AWT':>5} {'1st cross':>10}"
        )
        sep = "-" * 70
        lines: list[str] = [
            f"=== Decision Matrix ({self.result.attack_type}) ===",
            header,
            sep,
        ]

        stat_map = {s.label: s for s in self.result.signal_stats}

        for sa in self.signal_analyses:
            stat = stat_map[sa.label]
            cp_str = str(sa.pelt_breakpoint) if sa.pelt_breakpoint >= 0 else "—"
            awt_str = str(sa.awt) if sa.pelt_breakpoint >= 0 else "—"
            cross_str = str(sa.first_crossing) if sa.first_crossing >= 0 else "—"
            lines.append(
                f"{sa.label:<26} {stat.cohens_d:>9.3f} "
                f"{'YES' if stat.large_effect else 'no':>7} "
                f"{cp_str:>8} {awt_str:>5} {cross_str:>10}"
            )

        n_large = self.result.n_large_effects
        lines += [
            sep,
            f"Signals with large effect (d>{self.config.large_effect_threshold}): {n_large}/6",
            f"GO/NO-GO decision: {self.result.go_no_go}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Signal-level analysis
    # ------------------------------------------------------------------

    def _analyse_all_signals(self) -> list[SignalAnalysisResult]:
        """Run PELT + AWT analysis for every signal column.

        Returns:
            List of :class:`SignalAnalysisResult`, one per signal.
        """
        results: list[SignalAnalysisResult] = []
        for i, label in enumerate(SIGNAL_LABELS):
            clean_col = self.result.clean_matrix[:, i]
            poison_col = self.result.poisoned_matrix[:, i]
            full_series = np.concatenate([clean_col, poison_col])

            cp_idx = self._run_pelt(full_series)
            threshold = float(np.mean(clean_col)) + self.config.awt_detection_multiplier * float(
                np.std(clean_col, ddof=1) if len(clean_col) > 1 else 0.0
            )
            first_cross = self._first_threshold_crossing(full_series, threshold)
            awt = self._compute_awt(cp_idx, first_cross)

            results.append(
                SignalAnalysisResult(
                    label=label,
                    pelt_breakpoint=cp_idx,
                    awt=awt,
                    threshold=threshold,
                    first_crossing=first_cross,
                )
            )
        return results

    def _run_pelt(self, series: NDArray[np.float64]) -> int:
        """Detect a single changepoint in *series* using PELT.

        Reshapes to ``(T, 1)`` as required by ruptures when ``model='rbf'``.

        Args:
            series: 1-D signal array of length T.

        Returns:
            Index of the first detected breakpoint (0-based), or ``-1`` if
            PELT found no breakpoint (i.e., only the terminal index is
            returned, meaning the whole series is one segment).
        """
        data = series.reshape(-1, 1)
        algo = rpt.Pelt(
            model=self.config.changepoint_model,
            min_size=self.config.changepoint_min_size,
            jump=self.config.changepoint_jump,
        ).fit(data)
        breakpoints: list[int] = algo.predict(pen=self.config.changepoint_penalty)
        # ruptures always appends len(series) as the final sentinel.
        # If only the sentinel is present, no real changepoint was found.
        interior = [b for b in breakpoints if b < len(series)]
        return interior[0] if interior else -1

    def _first_threshold_crossing(
        self,
        series: NDArray[np.float64],
        threshold: float,
    ) -> int:
        """Find the first index where *series* exceeds *threshold*.

        Args:
            series: Full (clean ‖ poisoned) 1-D signal array.
            threshold: Detection threshold value.

        Returns:
            0-based index of first crossing, or ``-1`` if none.
        """
        crossings = np.where(series > threshold)[0]
        return int(crossings[0]) if len(crossings) > 0 else -1

    def _compute_awt(self, pelt_cp: int, first_cross: int) -> int:
        """Compute Advance Warning Time given a PELT changepoint and first crossing.

        AWT = pelt_cp - first_cross (steps of advance warning before confirmed CP).
        Returns ``0`` if first crossing is at or after the changepoint.
        Returns ``-1`` if no PELT changepoint was detected.

        Args:
            pelt_cp: PELT-detected changepoint index (``-1`` = not found).
            first_cross: First threshold-crossing index (``-1`` = not found).

        Returns:
            AWT in steps.  Negative means late detection; ``-1`` means missed.
        """
        if pelt_cp < 0:
            return -1
        if first_cross < 0:
            return 0
        return max(self.config.awt_min_awt, pelt_cp - first_cross)

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _save_csv(self, output_dir: Path) -> None:
        """Write decision matrix to ``decision_matrix.csv``.

        Args:
            output_dir: Directory to write into.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for CSV output. "
                "Install with: pip install 'chronoagent[experiments]'"
            ) from exc

        stat_map = {s.label: s for s in self.result.signal_stats}
        rows = []
        for sa in self.signal_analyses:
            stat = stat_map[sa.label]
            rows.append(
                {
                    "signal": sa.label,
                    "clean_mean": round(stat.clean_mean, 6),
                    "clean_std": round(stat.clean_std, 6),
                    "poisoned_mean": round(stat.poisoned_mean, 6),
                    "poisoned_std": round(stat.poisoned_std, 6),
                    "cohens_d": round(stat.cohens_d, 6),
                    "large_effect": stat.large_effect,
                    "pelt_breakpoint": sa.pelt_breakpoint,
                    "awt": sa.awt,
                    "threshold": round(sa.threshold, 6),
                    "first_crossing": sa.first_crossing,
                }
            )

        df = pd.DataFrame(rows)
        csv_path = output_dir / "decision_matrix.csv"
        df.to_csv(csv_path, index=False)

    def _save_plots(self, output_dir: Path) -> None:
        """Write per-signal time-series plots to *output_dir*.

        Creates a single figure with a 2×3 subplot grid — one panel per signal.
        Each panel shows the full (clean ‖ poisoned) time series with:
        - Shaded clean/poisoned regions.
        - Vertical dashed line at ground-truth phase boundary.
        - Vertical solid line at PELT-detected changepoint (if found).
        - Horizontal dotted line at the AWT detection threshold.
        - Marker at the first threshold crossing (if any).

        Args:
            output_dir: Directory to write the plot file into.
        """
        try:
            import matplotlib
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plot output. "
                "Install with: pip install 'chronoagent[experiments]'"
            ) from exc

        matplotlib.use("Agg")  # non-interactive backend

        n_clean = self.result.n_clean_steps
        n_poison = self.result.n_poisoned_steps
        total = n_clean + n_poison
        stat_map = {s.label: s for s in self.result.signal_stats}

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(
            f"Signal Validation — {self.result.attack_type} "
            f"(n_clean={n_clean}, n_poison={n_poison})",
            fontsize=13,
        )

        ax_flat: list[Any] = list(axes.flat)

        for i, (sa, ax) in enumerate(zip(self.signal_analyses, ax_flat, strict=True)):
            stat = stat_map[sa.label]
            clean_col = self.result.clean_matrix[:, i]
            poison_col = self.result.poisoned_matrix[:, i]
            full = np.concatenate([clean_col, poison_col])
            steps = np.arange(total)

            # Background shading
            ax.axvspan(0, n_clean, alpha=0.08, color="steelblue", label="Clean phase")
            ax.axvspan(n_clean, total, alpha=0.08, color="tomato", label="Poisoned phase")

            # Signal line
            ax.plot(steps, full, color="black", linewidth=1.2, label="Signal")

            # Ground-truth phase boundary
            ax.axvline(
                n_clean, color="steelblue", linestyle="--", linewidth=1.0, label="Phase boundary"
            )

            # PELT changepoint
            if sa.pelt_breakpoint >= 0:
                ax.axvline(
                    sa.pelt_breakpoint,
                    color="darkorange",
                    linestyle="-",
                    linewidth=1.5,
                    label=f"PELT CP={sa.pelt_breakpoint}",
                )

            # Detection threshold
            ax.axhline(
                sa.threshold,
                color="gray",
                linestyle=":",
                linewidth=1.0,
                label=f"Threshold={sa.threshold:.3f}",
            )

            # First threshold crossing marker
            if sa.first_crossing >= 0:
                ax.scatter(
                    [sa.first_crossing],
                    [full[sa.first_crossing]],
                    color="red",
                    zorder=5,
                    s=40,
                    label=f"1st cross={sa.first_crossing}",
                )

            awt_str = str(sa.awt) if sa.pelt_breakpoint >= 0 else "—"
            ax.set_title(
                f"{sa.label}\nd={stat.cohens_d:.2f}  AWT={awt_str}",
                fontsize=9,
            )
            ax.set_xlabel("Step", fontsize=8)
            ax.set_ylabel("Value", fontsize=8)
            ax.tick_params(labelsize=7)

            if i == 0:
                ax.legend(fontsize=6, loc="upper left")

        plt.tight_layout()

        attack_slug = self.result.attack_type.lower().replace("styleattack", "")
        plot_name = f"signals_{attack_slug}.{self.config.plot_format}"
        plot_path = output_dir / plot_name

        plt.savefig(
            plot_path,
            dpi=self.config.plot_dpi if self.config.plot_format != "svg" else None,
            bbox_inches="tight",
        )
        plt.close(fig)
