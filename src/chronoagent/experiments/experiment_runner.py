"""Phase 10 experiment runner (task 10.6).

Orchestrates one full Phase 10 experiment end-to-end: loads an
:class:`~chronoagent.experiments.config_schema.ExperimentConfig`, runs
``cfg.num_runs`` independent reproducible runs at per-run-derived seeds,
dispatches each run through the correct comparator based on the
experiment's name and ablation block, computes the four 10.2 metrics per
run, and aggregates (mean / std / 95% CI) across runs.

High-level per-run flow::

    seed = cfg.per_run_seed(run_index)
    signal_matrix = signal_matrix_factory(
        attack_type=cfg.attack.type,
        seed=seed,
        injection_step=cfg.attack.injection_step,
        num_prs=cfg.num_prs,
        n_poison_docs=cfg.attack.n_poison_docs,
    )
    detector = _dispatch_detector(cfg)
    decisions = detector.run(signal_matrix)
    run_result = _compute_metrics(decisions, cfg)

Dispatch rule (locked in at 10.5):

* ``cfg.name == "baseline_sentinel"`` -> real
  :class:`~chronoagent.experiments.baselines.sentinel.SentinelBaseline`.
* ``cfg.ablation.health is False`` -> real
  :class:`~chronoagent.experiments.baselines.no_monitoring.NoMonitoringBaseline`.
* Otherwise (main, generalisation, three ablations) ->
  :class:`~chronoagent.experiments.full_system_detector.FullSystemDetector`
  wired with the ``bocpd`` / ``forecaster`` / ``integrity`` flags.

The signal matrix is produced by a pluggable factory so tests can inject
a fast deterministic fake without spinning up ChromaDB + MockBackend per
run. The default factory wraps the existing Phase 1
:class:`~chronoagent.experiments.runner.SignalValidationRunner` (which
ships real signals from the same code path Phase 1 validated) and
truncates its symmetric clean + poisoned matrices into an asymmetric
``(num_prs, NUM_SIGNALS)`` matrix with the label flip at row
``injection_step``.

Persistence:

* ``<output_dir>/<experiment_name>/runs.csv`` - one row per run with the
  per-run metrics and provenance (seed, detector name, injection step,
  first-flagged step).
* ``<output_dir>/<experiment_name>/aggregate.json`` - one JSON document
  with the config provenance plus per-metric mean / std / 95% CI,
  matching the shape of :class:`AggregateResult`.

The runner does NOT import any FastAPI, SQLAlchemy, or ChromaDB code
directly; every backend dependency is encapsulated in the injected
signal matrix factory so the runner test file can drive 100% of the
runner logic with pure numpy inputs.
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as student_t

from chronoagent.experiments.baselines.no_monitoring import (
    NO_MONITORING_AGENT_ID,
    NoMonitoringBaseline,
)
from chronoagent.experiments.baselines.sentinel import (
    SENTINEL_AGENT_ID,
    SentinelBaseline,
    SentinelConfig,
)
from chronoagent.experiments.full_system_detector import (
    FULL_SYSTEM_AGENT_ID,
    FullSystemConfig,
    FullSystemDetector,
)
from chronoagent.experiments.metrics import (
    advance_warning_time,
    allocation_efficiency,
    detection_auroc,
    detection_f1,
)
from chronoagent.monitor.collector import NUM_SIGNALS

if TYPE_CHECKING:
    from chronoagent.experiments.config_schema import (
        AttackType,
        ExperimentConfig,
    )


BASELINE_SENTINEL_NAME: Literal["baseline_sentinel"] = "baseline_sentinel"

# Confidence level for the 95% CI on the per-metric aggregator. Pinned
# as a module constant so tests can reference it without guessing.
CI_CONFIDENCE: float = 0.95

# Dispatch label that ends up on :attr:`RunResult.detector_name` so
# downstream analysis (and the CSV) can cleanly identify which
# comparator produced the row.
_DETECTOR_FULL_SYSTEM: str = "full_system_detector"
_DETECTOR_SENTINEL: str = "sentinel_baseline"
_DETECTOR_NO_MONITORING: str = "no_monitoring_baseline"


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class _DecisionLike(Protocol):
    """Structural protocol for a single per-step comparator decision.

    Every concrete Phase 10 detector decision (Sentinel, NoMonitoring,
    FullSystem) exposes these attributes. The runner treats a decision
    stream as a sequence of objects that satisfy this protocol and
    projects to dict form (``{"success": ..., "agent_id": ...}``)
    before handing to :func:`allocation_efficiency`.
    """

    step_index: int
    success: bool
    agent_id: str


@runtime_checkable
class _ScoredDecisionLike(_DecisionLike, Protocol):
    """Decision stream that carries a continuous ``score`` for AUROC."""

    score: float
    flagged: bool


@runtime_checkable
class _DetectorLike(Protocol):
    """Every comparator exposes ``run(signal_matrix) -> list[Decision]``.

    The return type is ``Sequence[Any]`` rather than
    ``Sequence[_DecisionLike]`` so that concrete decision subclasses
    (which satisfy the protocol structurally) can be returned without
    running into Sequence invariance. The runner projects each element
    through ``getattr(d, 'success', ...)`` etc. so the structural
    contract is enforced at projection time in ``_compute_metrics``.
    """

    def run(self, signal_matrix: NDArray[np.float64]) -> Sequence[Any]: ...


@runtime_checkable
class SignalMatrixFactory(Protocol):
    """Pluggable producer of ``(num_prs, NUM_SIGNALS)`` signal matrices.

    The runner calls the factory once per run with the per-run seed and
    the experiment's attack parameters. Implementations may wrap the
    Phase 1 :class:`~chronoagent.experiments.runner.SignalValidationRunner`,
    synthesize a deterministic matrix for tests, or load from a
    pre-recorded trace.
    """

    def __call__(
        self,
        *,
        attack_type: AttackType,
        seed: int,
        injection_step: int,
        num_prs: int,
        n_poison_docs: int,
    ) -> NDArray[np.float64]: ...


# ---------------------------------------------------------------------------
# Default signal matrix factory (wraps the Phase 1 runner)
# ---------------------------------------------------------------------------


def default_signal_matrix_factory(
    *,
    attack_type: AttackType,
    seed: int,
    injection_step: int,
    num_prs: int,
    n_poison_docs: int,
) -> NDArray[np.float64]:
    """Produce a ``(num_prs, NUM_SIGNALS)`` matrix via the Phase 1 runner.

    Wraps :meth:`~chronoagent.experiments.runner.SignalValidationRunner.create`
    with symmetric clean + poisoned phase lengths of
    ``max(injection_step, num_prs - injection_step)`` so the phase
    lengths requested by the experiment (``injection_step`` clean rows
    followed by ``num_prs - injection_step`` poisoned rows) both fit
    inside the produced matrices. The results are then truncated into
    an asymmetric matrix with the label flip at row ``injection_step``.

    The import of the Phase 1 runner is deferred to function scope so
    unit tests that inject a fake factory never pay the cost of loading
    ChromaDB / MockBackend at collection time.
    """
    from chronoagent.experiments.runner import SignalValidationRunner

    if attack_type == "none":
        raise ValueError(
            "default_signal_matrix_factory requires attack_type in "
            "{'minja', 'agentpoison'}; 'none' is not yet supported"
        )
    clean_len = int(injection_step)
    poison_len = int(num_prs) - clean_len
    if clean_len <= 0 or poison_len <= 0:
        raise ValueError(
            f"injection_step={injection_step} and num_prs={num_prs} must "
            "leave at least one row in each of the clean and poisoned phases"
        )

    n_steps = max(clean_len, poison_len)
    runner = SignalValidationRunner.create(
        attack=attack_type,
        n_steps=n_steps,
        n_poison_docs=n_poison_docs,
        n_calibration=clean_len,
        seed=seed,
        pr_seed=seed,
    )
    result = runner.run()
    clean_matrix = result.clean_matrix[:clean_len]
    poison_matrix = result.poisoned_matrix[:poison_len]
    full: NDArray[np.float64] = np.vstack([clean_matrix, poison_matrix]).astype(np.float64)
    if full.shape != (num_prs, NUM_SIGNALS):
        raise RuntimeError(
            f"default_signal_matrix_factory produced shape {full.shape}, "
            f"expected ({num_prs}, {NUM_SIGNALS})"
        )
    return full


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawRunRecord:
    """Per-run raw signal matrix and decision stream.

    Captured by the runner only when ``collect_raw=True`` was passed to
    :class:`ExperimentRunner`. The Phase 10 plotting layer (task 10.7)
    consumes these records to draw signal-drift, allocation-efficiency,
    health-score, and ROC figures that need per-step data the
    :class:`AggregateResult` summary alone does not carry.

    Attributes:
        run_index: Zero-indexed run number, matching the corresponding
            :class:`RunResult` so callers can join on it.
        seed: Per-run seed used to produce this record. Mirrors
            :attr:`RunResult.seed` for the same reason.
        signal_matrix: ``(num_prs, NUM_SIGNALS)`` float64 matrix the
            runner fed to the detector for this run. Stored as-is so
            the plot layer can re-key signal columns by
            :data:`~chronoagent.monitor.collector.SIGNAL_LABELS`.
        decisions: One dict per row, projected from the detector's
            decision stream by :func:`_project_decision_for_raw` so
            every detector flavour (Sentinel, NoMonitoring, FullSystem)
            ends up with the same JSON-friendly schema. Every dict
            carries at least ``step_index``, ``score``, ``flagged``,
            ``success``, ``agent_id``; sub-score fields like
            ``bocpd_score`` are present only when the source decision
            had them set.
    """

    run_index: int
    seed: int
    signal_matrix: NDArray[np.float64]
    decisions: list[dict[str, Any]]


@dataclass(frozen=True)
class RunResult:
    """Per-run metrics and provenance.

    Attributes:
        run_index: Zero-indexed run number in ``range(cfg.num_runs)``.
        seed: The per-run seed derived via ``cfg.per_run_seed(run_index)``.
        detector_name: Stable label identifying which comparator
            produced this row. One of ``"full_system_detector"``,
            ``"sentinel_baseline"``, ``"no_monitoring_baseline"``.
        injection_step: Label-flip row from ``cfg.attack.injection_step``.
            Rows ``0..injection_step-1`` are ground-truth clean; rows
            ``injection_step..num_prs-1`` are ground-truth poisoned.
        num_prs: Total rows in this run's signal matrix.
        first_flagged_step: Earliest step the detector flagged, or
            ``None`` if the detector never flagged in this run. Paper
            aggregations skip runs where this is ``None``.
        advance_warning_time: ``injection_step - first_flagged_step``
            per Pivot A, or ``None`` if never flagged. Can be negative
            (detector was late) or zero (concurrent detection).
        allocation_efficiency_score: Cumulative success rate in ``[0, 1]``.
        detection_auroc_score: AUROC vs the clean/poisoned ground truth.
            May be ``nan`` for single-class runs.
        detection_f1_score: F1 vs the same ground truth. May be ``nan``
            for empty runs, but in practice every Phase 10 run has at
            least one row per label.
        latency_ms: Mean per-step wall-clock latency in milliseconds
            for the detector pipeline on this run. Measured end-to-end
            from signal matrix ingestion through decision emission.
    """

    run_index: int
    seed: int
    detector_name: str
    injection_step: int
    num_prs: int
    first_flagged_step: int | None
    advance_warning_time: int | None
    allocation_efficiency_score: float
    detection_auroc_score: float
    detection_f1_score: float
    latency_ms: float


@dataclass(frozen=True)
class MetricAggregate:
    """Mean, std, and 95% CI for one metric over ``num_runs`` runs."""

    mean: float
    std: float
    ci_low: float
    ci_high: float
    n: int


@dataclass(frozen=True)
class AggregateResult:
    """Aggregated experiment result across all runs.

    Attributes:
        name: Experiment name from the config (propagated into the
            output filename).
        detector_name: The dispatch label used by every run. Pinned as
            a top-level field because dispatch is config-determined and
            therefore stable across runs.
        num_runs: Number of runs aggregated.
        injection_step: Ground-truth injection row (same across runs).
        num_prs: Signal-matrix row count (same across runs).
        runs: Per-run results in ``run_index`` order.
        advance_warning_time: Aggregate over runs where the detector
            fired. ``n`` may be less than ``num_runs``.
        allocation_efficiency_score: Aggregate over all runs
            (always defined).
        detection_auroc_score: Aggregate skipping nan per-run values.
        detection_f1_score: Aggregate skipping nan per-run values.
        latency_ms: Aggregate per-step latency in milliseconds.
    """

    name: str
    detector_name: str
    num_runs: int
    injection_step: int
    num_prs: int
    runs: list[RunResult]
    advance_warning_time: MetricAggregate
    allocation_efficiency_score: MetricAggregate
    detection_auroc_score: MetricAggregate
    detection_f1_score: MetricAggregate
    latency_ms: MetricAggregate
    provenance: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------


def _aggregate_metric(values: Sequence[float]) -> MetricAggregate:
    """Mean / std / 95% CI over a list of per-run values.

    NaN values are dropped before aggregation. An empty (or all-nan)
    input returns ``MetricAggregate(nan, nan, nan, nan, 0)`` so the
    runner can always serialise a complete JSON document regardless of
    which runs fired.
    """
    filtered = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    arr = np.asarray(filtered, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return MetricAggregate(
            mean=float("nan"),
            std=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=0,
        )
    mean = float(np.mean(arr))
    if n == 1:
        # Single sample: no std, no CI; report mean and mark CI as nan
        # to avoid implying a confidence bound from one observation.
        return MetricAggregate(
            mean=mean,
            std=0.0,
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=1,
        )
    std = float(np.std(arr, ddof=1))
    # Student-t 95% CI: t_{0.975, n-1} * std / sqrt(n)
    t_crit = float(student_t.ppf(0.5 + CI_CONFIDENCE / 2.0, df=n - 1))
    half_width = t_crit * std / math.sqrt(n)
    return MetricAggregate(
        mean=mean,
        std=std,
        ci_low=mean - half_width,
        ci_high=mean + half_width,
        n=n,
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _dispatch_detector(cfg: ExperimentConfig) -> tuple[_DetectorLike, str]:
    """Return a fresh detector instance and its stable label for ``cfg``.

    The dispatch rule is locked in at 10.5:

    * ``cfg.name == "baseline_sentinel"`` -> Sentinel baseline.
    * ``cfg.ablation.health is False`` -> no-monitoring baseline.
    * Otherwise -> :class:`FullSystemDetector` wired with the bocpd /
      forecaster / integrity ablation flags.
    """
    if cfg.name == BASELINE_SENTINEL_NAME:
        sentinel_cfg = SentinelConfig(calibration_steps=cfg.attack.injection_step)
        return SentinelBaseline(sentinel_cfg), _DETECTOR_SENTINEL
    if not cfg.ablation.health:
        return NoMonitoringBaseline(), _DETECTOR_NO_MONITORING
    full_cfg = FullSystemConfig(
        bocpd_hazard_lambda=cfg.system.bocpd_hazard_lambda,
        calibration_steps=cfg.attack.injection_step,
    )
    return FullSystemDetector(cfg.ablation, full_cfg), _DETECTOR_FULL_SYSTEM


# ---------------------------------------------------------------------------
# Per-run metric computation
# ---------------------------------------------------------------------------


def _compute_metrics(
    decisions: Sequence[_DecisionLike],
    *,
    run_index: int,
    seed: int,
    detector_name: str,
    injection_step: int,
    num_prs: int,
    latency_ms: float,
) -> RunResult:
    """Turn a decision stream into a :class:`RunResult`.

    Ground-truth labels: ``[False]*injection_step + [True]*(num_prs - injection_step)``.
    The 10.2 metric functions accept either a ``Sequence[bool]`` or a
    mapping-style audit row; the projection below uses the mapping path
    so the ``success`` key is explicit rather than relying on
    dataclass duck typing.
    """
    if len(decisions) != num_prs:
        raise RuntimeError(f"detector emitted {len(decisions)} decisions, expected {num_prs}")

    y_true = [False] * injection_step + [True] * (num_prs - injection_step)

    alloc_rows = [{"success": bool(d.success), "agent_id": d.agent_id} for d in decisions]
    alloc_score = allocation_efficiency(alloc_rows)

    # ``flagged`` and ``score`` are optional attributes. No-monitoring
    # decisions never carry a score or flagged field, so we default
    # them to constant 0 / False (the detector never flags).
    y_pred = [bool(getattr(d, "flagged", False)) for d in decisions]
    y_scores = [float(getattr(d, "score", 0.0)) for d in decisions]

    auroc = detection_auroc(y_true, y_scores)
    f1 = detection_f1(y_true, y_pred)

    first_flagged: int | None = None
    for d in decisions:
        if bool(getattr(d, "flagged", False)):
            first_flagged = int(d.step_index)
            break

    awt: int | None
    if first_flagged is None:
        awt = None
    else:
        awt = advance_warning_time(injection_step=injection_step, detection_step=first_flagged)

    return RunResult(
        run_index=run_index,
        seed=seed,
        detector_name=detector_name,
        injection_step=injection_step,
        num_prs=num_prs,
        first_flagged_step=first_flagged,
        advance_warning_time=awt,
        allocation_efficiency_score=float(alloc_score),
        detection_auroc_score=float(auroc),
        detection_f1_score=float(f1),
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ExperimentRunner:
    """Driver for one Phase 10 experiment.

    The runner is stateless across runs: each call to
    :meth:`run` loops over ``cfg.num_runs`` seeds, constructs a fresh
    detector via :func:`_dispatch_detector`, pulls a fresh signal matrix
    from the injected :class:`SignalMatrixFactory`, computes the four
    10.2 metrics, and aggregates.

    Args:
        cfg: Parsed :class:`~chronoagent.experiments.config_schema.ExperimentConfig`.
        signal_matrix_factory: Optional pluggable factory. Defaults to
            :func:`default_signal_matrix_factory` which wraps the real
            Phase 1 runner. Tests inject a deterministic fake.
        collect_raw: When ``True``, populate :attr:`raw_runs` during
            :meth:`run` with the per-run signal matrix + projected
            decision stream so the Phase 10 plot layer (task 10.7) can
            persist and visualise the raw signals. Defaults to
            ``False`` to keep the inner-loop test path lean: storing
            the matrix is cheap but storing decisions on long
            experiments adds noticeable memory pressure.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        signal_matrix_factory: SignalMatrixFactory | None = None,
        collect_raw: bool = False,
    ) -> None:
        self._cfg: ExperimentConfig = cfg
        self._factory: SignalMatrixFactory = (
            signal_matrix_factory
            if signal_matrix_factory is not None
            else default_signal_matrix_factory
        )
        self._collect_raw: bool = collect_raw
        self._raw_runs: list[RawRunRecord] = []

    @property
    def raw_runs(self) -> list[RawRunRecord]:
        """Per-run raw records collected during :meth:`run`.

        Empty unless the runner was constructed with ``collect_raw=True``.
        Returns the live list so callers should not mutate it.
        """
        return self._raw_runs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> AggregateResult:
        """Execute every run and return the aggregate."""
        runs: list[RunResult] = []
        detector_label: str | None = None
        if self._collect_raw:
            self._raw_runs = []
        for run_index in range(self._cfg.num_runs):
            run_result, label = self._run_single(run_index)
            runs.append(run_result)
            detector_label = label
        assert detector_label is not None  # num_runs >= 1 enforced by the schema

        awt_values = [
            float(r.advance_warning_time) for r in runs if r.advance_warning_time is not None
        ]
        alloc_values = [r.allocation_efficiency_score for r in runs]
        auroc_values = [r.detection_auroc_score for r in runs]
        f1_values = [r.detection_f1_score for r in runs]
        latency_values = [r.latency_ms for r in runs]

        return AggregateResult(
            name=self._cfg.name,
            detector_name=detector_label,
            num_runs=self._cfg.num_runs,
            injection_step=self._cfg.attack.injection_step,
            num_prs=self._cfg.num_prs,
            runs=runs,
            advance_warning_time=_aggregate_metric(awt_values),
            allocation_efficiency_score=_aggregate_metric(alloc_values),
            detection_auroc_score=_aggregate_metric(auroc_values),
            detection_f1_score=_aggregate_metric(f1_values),
            latency_ms=_aggregate_metric(latency_values),
            provenance=self._build_provenance(),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_single(self, run_index: int) -> tuple[RunResult, str]:
        seed = self._cfg.per_run_seed(run_index)
        detector, detector_label = _dispatch_detector(self._cfg)
        signal_matrix = self._factory(
            attack_type=self._cfg.attack.type,
            seed=seed,
            injection_step=self._cfg.attack.injection_step,
            num_prs=self._cfg.num_prs,
            n_poison_docs=self._cfg.attack.n_poison_docs,
        )
        t0 = time.perf_counter()
        decisions = list(detector.run(signal_matrix))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_step_ms = elapsed_ms / len(decisions) if decisions else 0.0
        run_result = _compute_metrics(
            decisions,
            run_index=run_index,
            seed=seed,
            detector_name=detector_label,
            injection_step=self._cfg.attack.injection_step,
            num_prs=self._cfg.num_prs,
            latency_ms=per_step_ms,
        )
        if self._collect_raw:
            projected = [_project_decision_for_raw(d) for d in decisions]
            self._raw_runs.append(
                RawRunRecord(
                    run_index=run_index,
                    seed=seed,
                    signal_matrix=signal_matrix,
                    decisions=projected,
                )
            )
        return run_result, detector_label

    def _build_provenance(self) -> dict[str, Any]:
        cfg = self._cfg
        return {
            "name": cfg.name,
            "seed": cfg.seed,
            "num_runs": cfg.num_runs,
            "num_prs": cfg.num_prs,
            "attack": {
                "type": cfg.attack.type,
                "target": cfg.attack.target,
                "injection_step": cfg.attack.injection_step,
                "n_poison_docs": cfg.attack.n_poison_docs,
                "strategy": cfg.attack.strategy,
            },
            "ablation": {
                "forecaster": cfg.ablation.forecaster,
                "bocpd": cfg.ablation.bocpd,
                "health": cfg.ablation.health,
                "integrity": cfg.ablation.integrity,
            },
            "system": {
                "llm_backend": cfg.system.llm_backend,
                "bocpd_hazard_lambda": cfg.system.bocpd_hazard_lambda,
                "health_threshold": cfg.system.health_threshold,
                "integrity_threshold": cfg.system.integrity_threshold,
            },
            "baseline_agent_ids": {
                "sentinel": SENTINEL_AGENT_ID,
                "no_monitoring": NO_MONITORING_AGENT_ID,
                "full_system": FULL_SYSTEM_AGENT_ID,
            },
        }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _project_decision_for_raw(decision: Any) -> dict[str, Any]:
    """Turn a detector decision into a JSON-friendly dict.

    Every Phase 10 detector flavour (Sentinel, NoMonitoring,
    FullSystem) carries the core ``step_index``, ``success``, ``agent_id``
    attributes. ``score`` and ``flagged`` are present on the scored
    detectors but missing from NoMonitoring; the projection defaults
    them to ``0.0`` and ``False`` so the persisted JSON has a uniform
    schema regardless of which comparator produced the row.

    Sub-score fields (``bocpd_score``, ``forecaster_score``,
    ``integrity_score``) are only present on
    :class:`~chronoagent.experiments.full_system_detector.FullSystemDecision`
    and are written through verbatim when set so the plot layer can
    inspect per-channel contributions.
    """
    base: dict[str, Any] = {
        "step_index": int(decision.step_index),
        "success": bool(decision.success),
        "agent_id": str(decision.agent_id),
        "score": float(getattr(decision, "score", 0.0)),
        "flagged": bool(getattr(decision, "flagged", False)),
    }
    for optional_field in ("bocpd_score", "forecaster_score", "integrity_score"):
        value = getattr(decision, optional_field, None)
        if value is not None:
            base[optional_field] = float(value)
    return base


def write_experiment_results(
    aggregate: AggregateResult,
    output_dir: Path | str,
    raw_runs: Sequence[RawRunRecord] | None = None,
) -> tuple[Path, Path]:
    """Write ``runs.csv`` and ``aggregate.json`` under ``output_dir/<name>/``.

    Args:
        aggregate: The :class:`AggregateResult` to persist.
        output_dir: Filesystem root under which the per-experiment
            directory is created.
        raw_runs: Optional sequence of :class:`RawRunRecord` instances
            (typically ``runner.raw_runs`` after ``runner.run()``).
            When supplied, the runner also writes
            ``raw/run_<i>_signals.npy`` and ``raw/run_<i>_decisions.json``
            per run so the Phase 10 plot layer can read per-step data
            without re-executing the experiment.

    Returns:
        ``(runs_csv_path, aggregate_json_path)`` so callers (and tests)
        can read the files back for verification. Raw files (when
        written) live under ``output_dir/<name>/raw/``.
    """
    base = Path(output_dir) / aggregate.name
    base.mkdir(parents=True, exist_ok=True)
    runs_path = base / "runs.csv"
    json_path = base / "aggregate.json"

    # runs.csv: flat per-run fields
    fieldnames = [
        "run_index",
        "seed",
        "detector_name",
        "injection_step",
        "num_prs",
        "first_flagged_step",
        "advance_warning_time",
        "allocation_efficiency_score",
        "detection_auroc_score",
        "detection_f1_score",
        "latency_ms",
    ]
    with runs_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for run in aggregate.runs:
            row = asdict(run)
            writer.writerow(row)

    # aggregate.json: the full AggregateResult minus the per-run rows
    # (those live in the CSV already), plus the provenance dict.
    agg_payload: dict[str, Any] = {
        "name": aggregate.name,
        "detector_name": aggregate.detector_name,
        "num_runs": aggregate.num_runs,
        "injection_step": aggregate.injection_step,
        "num_prs": aggregate.num_prs,
        "metrics": {
            "advance_warning_time": asdict(aggregate.advance_warning_time),
            "allocation_efficiency_score": asdict(aggregate.allocation_efficiency_score),
            "detection_auroc_score": asdict(aggregate.detection_auroc_score),
            "detection_f1_score": asdict(aggregate.detection_f1_score),
            "latency_ms": asdict(aggregate.latency_ms),
        },
        "provenance": aggregate.provenance,
    }
    cleaned = _nan_to_none(agg_payload)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(cleaned, fh, indent=2, allow_nan=False)

    if raw_runs:
        raw_dir = base / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for record in raw_runs:
            sig_path = raw_dir / f"run_{record.run_index:03d}_signals.npy"
            dec_path = raw_dir / f"run_{record.run_index:03d}_decisions.json"
            np.save(sig_path, record.signal_matrix)
            decisions_payload = {
                "run_index": record.run_index,
                "seed": record.seed,
                "decisions": record.decisions,
            }
            with dec_path.open("w", encoding="utf-8") as fh:
                json.dump(_nan_to_none(decisions_payload), fh, indent=2, allow_nan=False)

    return runs_path, json_path


def _nan_to_none(obj: Any) -> Any:
    """Walk a JSON-able structure and replace NaN / inf floats with None.

    ``json.dump`` emits ``NaN`` as the literal string ``NaN`` (invalid
    JSON by the spec) unless ``allow_nan=False``, in which case it
    raises. Preprocessing the payload here lets the runner write strict
    JSON (``null`` in place of NaN) without depending on a custom
    encoder; ``json.dump(default=...)`` does NOT intercept floats
    because they are already "serialisable" from its perspective.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, tuple):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, np.floating):
        return _nan_to_none(float(obj))
    if isinstance(obj, np.integer):
        return int(obj)
    return obj
