"""Analysis subpackage for ChronoAgent experiments.

This subpackage contains the post-processing and visualisation code for
both Phase 1 (signal validation, GO/NO-GO ruling) and Phase 10 (paper
figures + tables for the research experiment suite).

Public re-exports preserve the historic
``from chronoagent.experiments.analysis import AnalysisConfig, SignalAnalyzer``
import path so the existing CLI hookup keeps working unchanged after
the Phase 10 task 10.7 refactor that converted the single-file
``analysis.py`` into a subpackage.

* :mod:`chronoagent.experiments.analysis.phase1` ships the legacy
  Phase 1 :class:`SignalAnalyzer` and :class:`AnalysisConfig` (Cohen's
  d ranking, PELT changepoint detection, AWT estimation, decision
  table). Imported as the package's public symbols below.
* :mod:`chronoagent.experiments.analysis.plots` ships the Phase 10
  figure functions consumed by tasks 10.7 / 10.8 / 10.9: signal drift
  viz, health score comparison, AWT box plot, allocation efficiency
  over time, ROC curve, ablation bar chart.
"""

from chronoagent.experiments.analysis.phase1 import (
    AnalysisConfig,
    SignalAnalyzer,
)

__all__ = ["AnalysisConfig", "SignalAnalyzer"]
