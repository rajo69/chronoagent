"""Phase 10 experiment baselines.

Each baseline in this subpackage exposes the same per-step decision
interface the Phase 10 runner consumes, so the metrics module in
:mod:`chronoagent.experiments.metrics` can score baseline runs and
full-system runs with the same code path.

* :mod:`chronoagent.experiments.baselines.sentinel` -- reactive
  threshold baseline (task 10.3).
* :mod:`chronoagent.experiments.baselines.no_monitoring` -- round-robin
  baseline with no monitoring at all (task 10.4, to be added).
"""
