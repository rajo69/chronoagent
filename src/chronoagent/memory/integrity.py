"""Memory integrity detection (Phase 6 task 6.1).

:class:`MemoryIntegrityModule` inspects a set of retrieved documents and
returns a per-document and aggregated suspicion score.  Four orthogonal
detection signals are combined:

1. **Embedding outlier score** -- cosine distance between the doc's stored
   embedding and the centroid of a fitted clean baseline.  Distances larger
   than three times the baseline's 99th percentile saturate the signal.
2. **Freshness anomaly score** -- linearly ramps from 1.0 to 0.0 over the
   first ten percent of :attr:`freshness_window_seconds`, so brand-new docs
   that surface immediately after insertion are mildly suspicious.  Future
   dated docs (relative to ``now_fn``) saturate the signal.
3. **Retrieval frequency score** -- z-score of the doc's lifetime retrieval
   count against the mean and standard deviation across all tracked docs,
   linearly mapped onto ``[0, 1]`` with ``z = retrieval_spike_z`` saturating.
4. **Content embedding mismatch score** -- the doc text is re-embedded via
   the injected backend and the cosine distance between the recomputed and
   stored embedding is rescaled to ``[0, 1]``.  This is the primary defence
   against the MINJA / AgentPoison style attacks implemented in
   :mod:`chronoagent.memory.poisoning`, where an attacker injects an embedding
   that does not match the document text it ships with.

The four signals are combined as a weighted sum and compared against
``flag_threshold`` to produce a per-doc ``flagged`` bit.  Weights are
normalised at construction time so callers can pass un-normalised
preferences.

This module is intentionally self-contained: it owns its baseline state and
retrieval-history bookkeeping, and does not touch ChromaDB or
:class:`~chronoagent.memory.store.MemoryStore` directly.  The plumbing that
hands stored embeddings + metadata into :meth:`check_retrieval` is the
caller's responsibility, which keeps the detection logic trivially testable.
Task 6.2 will replace the centroid-distance baseline with an ``IsolationForest``
fitted via the same :meth:`fit_baseline` entry point.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chronoagent.agents.backends.base import LLMBackend

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RetrievedDoc:
    """A single retrieved document supplied to :meth:`MemoryIntegrityModule.check_retrieval`.

    The caller assembles these from whatever store backs the agent (typically
    :class:`~chronoagent.memory.store.MemoryStore`).  The integrity module
    deliberately requires the stored embedding so the content-embedding
    mismatch signal can be computed without coupling to ChromaDB.

    Attributes:
        doc_id: ChromaDB document identifier.
        text: Document content as stored.
        embedding: Stored embedding vector (any iterable of floats).
        distance_to_query: Cosine distance to the active query.  Optional;
            kept for downstream consumers (e.g. quarantine UI in task 6.3).
        metadata: Per-document metadata dict.  ``created_at`` (Unix epoch
            seconds, int or float) drives the freshness signal.
    """

    doc_id: str
    text: str
    embedding: Sequence[float]
    distance_to_query: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocSignal:
    """Per-document integrity signals and aggregate score.

    Attributes:
        doc_id: ChromaDB document identifier.
        embedding_outlier: Score in ``[0, 1]`` from the centroid-distance
            check; ``0.0`` when no baseline has been fitted.
        freshness_anomaly: Score in ``[0, 1]`` from the freshness ramp.
        retrieval_frequency: Score in ``[0, 1]`` from the retrieval-history
            z-score.
        content_embedding_mismatch: Score in ``[0, 1]`` from re-embedding the
            document text and comparing to the stored vector.
        aggregate: Weighted sum of the four signals, in ``[0, 1]``.
        flagged: ``True`` iff ``aggregate >= flag_threshold``.
    """

    doc_id: str
    embedding_outlier: float
    freshness_anomaly: float
    retrieval_frequency: float
    content_embedding_mismatch: float
    aggregate: float
    flagged: bool


@dataclass
class IntegrityResult:
    """Aggregated outcome of an integrity check on a single retrieval.

    Attributes:
        query: The query string that produced the inspected docs.
        signals: Per-document :class:`DocSignal`, aligned with input order.
        flagged_ids: Document IDs whose aggregate score met the threshold.
        max_aggregate: Largest aggregate score across all inspected docs.
            ``0.0`` if no docs were supplied.
        timestamp: Unix epoch seconds when the check ran (from ``now_fn``).
    """

    query: str
    signals: list[DocSignal]
    flagged_ids: list[str]
    max_aggregate: float
    timestamp: float

    @property
    def is_clean(self) -> bool:
        """``True`` iff no document was flagged."""
        return not self.flagged_ids


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Default signal weights, biased towards content-embedding mismatch because it
#: is the most direct evidence of an attacker-crafted record.  Callers may
#: override via the ``weights`` constructor argument; values are normalised to
#: sum to 1.0 so absolute scale does not matter.
DEFAULT_WEIGHTS: dict[str, float] = {
    "embedding_outlier": 0.25,
    "freshness_anomaly": 0.15,
    "retrieval_frequency": 0.20,
    "content_embedding_mismatch": 0.40,
}

_REQUIRED_WEIGHT_KEYS: frozenset[str] = frozenset(DEFAULT_WEIGHTS.keys())


# ---------------------------------------------------------------------------
# MemoryIntegrityModule
# ---------------------------------------------------------------------------


class MemoryIntegrityModule:
    """Detects anomalous retrieved documents using four orthogonal signals.

    The module is stateful in two narrow ways:

    * A clean-embedding **baseline** (centroid + 99th-percentile radius) used
      by the embedding-outlier signal.  Fitted via :meth:`fit_baseline`.
    * A **retrieval history** counter used by the retrieval-frequency signal.
      Updated automatically inside :meth:`check_retrieval`; can also be
      pre-seeded via :meth:`record_retrievals` (e.g. when warming the module
      from persisted ChromaDB metadata).

    Args:
        backend: LLM backend used to re-embed document text for the
            content-embedding mismatch signal.  Must be deterministic for
            the signal to be meaningful.
        flag_threshold: Aggregate score at which a document is flagged.
            Must lie in ``[0, 1]``.
        weights: Optional un-normalised weights, keyed by signal name.  Must
            cover all four signals.  Defaults to :data:`DEFAULT_WEIGHTS`.
        freshness_window_seconds: Width of the freshness ramp.  Documents
            older than this contribute zero to the freshness signal; the
            ramp covers the leading ten percent of the window.
        retrieval_history_max: Maximum number of distinct doc IDs tracked in
            the retrieval-history counter.  When the counter exceeds this
            cap, the least-frequent IDs are evicted.
        retrieval_spike_z: z-score that maps to a saturated retrieval-
            frequency signal of ``1.0``.  Must be strictly positive.
        now_fn: Injectable wall-clock source.  Defaults to :func:`time.time`.

    Raises:
        ValueError: If any numeric argument is outside its valid range or if
            ``weights`` is missing one of the required signal keys.
    """

    def __init__(
        self,
        backend: LLMBackend,
        *,
        flag_threshold: float = 0.6,
        weights: dict[str, float] | None = None,
        freshness_window_seconds: float = 30.0 * 86400.0,
        retrieval_history_max: int = 10_000,
        retrieval_spike_z: float = 3.0,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        if not 0.0 <= flag_threshold <= 1.0:
            raise ValueError(f"flag_threshold must be in [0, 1] (got {flag_threshold})")
        if freshness_window_seconds <= 0.0:
            raise ValueError(
                f"freshness_window_seconds must be positive (got {freshness_window_seconds})"
            )
        if retrieval_history_max <= 0:
            raise ValueError(
                f"retrieval_history_max must be positive (got {retrieval_history_max})"
            )
        if retrieval_spike_z <= 0.0:
            raise ValueError(f"retrieval_spike_z must be positive (got {retrieval_spike_z})")

        self._backend = backend
        self._flag_threshold = float(flag_threshold)
        self._weights = self._normalise_weights(weights or DEFAULT_WEIGHTS)
        self._freshness_window = float(freshness_window_seconds)
        self._retrieval_history_max = int(retrieval_history_max)
        self._retrieval_spike_z = float(retrieval_spike_z)
        self._now_fn = now_fn

        # Baseline state -- populated by fit_baseline.
        self._baseline_centroid: NDArray[np.float64] | None = None
        self._baseline_radius: float = 1.0

        # Retrieval-history bookkeeping.
        self._retrieval_counts: Counter[str] = Counter()
        self._total_retrievals: int = 0

    # ------------------------------------------------------------------
    # Read-only introspection (useful for tests and observability)
    # ------------------------------------------------------------------

    @property
    def flag_threshold(self) -> float:
        """Aggregate score at which a document is flagged."""
        return self._flag_threshold

    @property
    def weights(self) -> dict[str, float]:
        """Normalised signal weights summing to 1.0."""
        return dict(self._weights)

    @property
    def baseline_fitted(self) -> bool:
        """``True`` iff :meth:`fit_baseline` has been called with data."""
        return self._baseline_centroid is not None

    @property
    def total_retrievals(self) -> int:
        """Lifetime number of (doc_id, retrieval) events recorded."""
        return self._total_retrievals

    # ------------------------------------------------------------------
    # Baseline fitting
    # ------------------------------------------------------------------

    def fit_baseline(self, embeddings: Sequence[Sequence[float]]) -> None:
        """Fit the embedding-outlier baseline from a set of clean embeddings.

        The centroid is computed in unit-norm space (so the metric is cosine
        distance) and the 99th percentile of in-sample centroid distances
        becomes the "normal radius".  An outlier with distance equal to three
        times the radius saturates the signal at ``1.0``.

        Calling :meth:`fit_baseline` with an empty sequence clears any
        previously fitted baseline; the embedding-outlier signal then
        contributes zero to every aggregate score.

        Args:
            embeddings: Iterable of clean embedding vectors.  Vectors of any
                non-zero norm are accepted; zero vectors are ignored.
        """
        if not embeddings:
            self._baseline_centroid = None
            self._baseline_radius = 1.0
            return

        arr = np.asarray(embeddings, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"embeddings must be 2-D (got shape {arr.shape})")

        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        nonzero = (norms > 0.0).reshape(-1)
        if not bool(nonzero.any()):
            self._baseline_centroid = None
            self._baseline_radius = 1.0
            return

        unit = arr[nonzero] / norms[nonzero]
        centroid = unit.mean(axis=0)
        c_norm = float(np.linalg.norm(centroid))
        if c_norm > 0.0:
            centroid = centroid / c_norm

        sims = unit @ centroid
        dists = 1.0 - sims  # cosine distance in [0, 2]
        radius = float(np.quantile(dists, 0.99)) if dists.size > 1 else float(dists.max())

        self._baseline_centroid = centroid
        # Floor the radius so a perfectly-tight baseline does not divide by zero.
        self._baseline_radius = max(radius, 1e-3)

    # ------------------------------------------------------------------
    # Retrieval-history bookkeeping
    # ------------------------------------------------------------------

    def record_retrievals(self, doc_ids: Sequence[str]) -> None:
        """Increment retrieval counts for *doc_ids*.

        Called automatically inside :meth:`check_retrieval`, but exposed for
        callers that need to seed history from a persisted store.

        Args:
            doc_ids: Document identifiers that surfaced in a retrieval.
        """
        if not doc_ids:
            return
        for did in doc_ids:
            self._retrieval_counts[did] += 1
        self._total_retrievals += len(doc_ids)
        self._evict_history_if_needed()

    def _evict_history_if_needed(self) -> None:
        """Trim the retrieval-history counter to ``retrieval_history_max``."""
        excess = len(self._retrieval_counts) - self._retrieval_history_max
        if excess <= 0:
            return
        # most_common() sorts descending; the tail holds the least-frequent IDs.
        for did, _ in self._retrieval_counts.most_common()[-excess:]:
            del self._retrieval_counts[did]

    # ------------------------------------------------------------------
    # Public detection entry point
    # ------------------------------------------------------------------

    def check_retrieval(
        self,
        query: str,
        docs: Sequence[RetrievedDoc],
    ) -> IntegrityResult:
        """Score every doc and return the aggregated integrity result.

        The retrieval is also recorded in the retrieval-history counter so
        the next call's frequency signal sees it.

        Args:
            query: Query string that produced *docs*.  Stored on the result
                for downstream auditing; not used by the detection signals
                themselves.
            docs: Documents to inspect.  May be empty.

        Returns:
            :class:`IntegrityResult` with one :class:`DocSignal` per input
            document, the list of flagged IDs, and the maximum aggregate
            score across all docs.
        """
        ts = float(self._now_fn())

        if not docs:
            return IntegrityResult(
                query=query,
                signals=[],
                flagged_ids=[],
                max_aggregate=0.0,
                timestamp=ts,
            )

        # Re-embed every doc text in a single backend call -- this is the
        # expensive operation, so batch it once up front.
        recomputed = np.asarray(
            self._backend.embed([d.text for d in docs]),
            dtype=np.float64,
        )

        signals: list[DocSignal] = []
        for i, doc in enumerate(docs):
            stored = np.asarray(doc.embedding, dtype=np.float64)
            re_emb = recomputed[i]

            outlier = self._embedding_outlier_score(stored)
            freshness = self._freshness_score(doc.metadata, ts)
            frequency = self._retrieval_frequency_score(doc.doc_id)
            mismatch = self._content_mismatch_score(stored, re_emb)

            aggregate = (
                outlier * self._weights["embedding_outlier"]
                + freshness * self._weights["freshness_anomaly"]
                + frequency * self._weights["retrieval_frequency"]
                + mismatch * self._weights["content_embedding_mismatch"]
            )
            aggregate = float(min(1.0, max(0.0, aggregate)))

            signals.append(
                DocSignal(
                    doc_id=doc.doc_id,
                    embedding_outlier=outlier,
                    freshness_anomaly=freshness,
                    retrieval_frequency=frequency,
                    content_embedding_mismatch=mismatch,
                    aggregate=aggregate,
                    flagged=aggregate >= self._flag_threshold,
                )
            )

        # Update retrieval history *after* scoring so the current call sees
        # the previous distribution, not its own contribution.
        self.record_retrievals([d.doc_id for d in docs])

        flagged_ids = [s.doc_id for s in signals if s.flagged]
        max_agg = max((s.aggregate for s in signals), default=0.0)

        return IntegrityResult(
            query=query,
            signals=signals,
            flagged_ids=flagged_ids,
            max_aggregate=max_agg,
            timestamp=ts,
        )

    # ------------------------------------------------------------------
    # Individual signal computations
    # ------------------------------------------------------------------

    def _embedding_outlier_score(self, stored: NDArray[np.float64]) -> float:
        """Cosine distance to the fitted centroid, mapped onto ``[0, 1]``.

        Returns ``0.0`` when no baseline has been fitted (so this signal
        contributes nothing until :meth:`fit_baseline` is called) and
        ``1.0`` for a zero-norm input vector (which should never occur in
        practice but would otherwise produce a NaN).
        """
        if self._baseline_centroid is None:
            return 0.0
        norm = float(np.linalg.norm(stored))
        if norm == 0.0:
            return 1.0
        unit = stored / norm
        sim = float(unit @ self._baseline_centroid)
        dist = 1.0 - sim  # cosine distance in [0, 2]
        # Saturate at three times the in-sample 99th percentile distance.
        ratio = dist / max(self._baseline_radius * 3.0, 1e-6)
        return float(min(1.0, max(0.0, ratio)))

    def _freshness_score(self, metadata: dict[str, Any], now: float) -> float:
        """Linear ramp over the leading ten percent of the freshness window.

        Future-dated documents (more than 60 s ahead of ``now`` to absorb
        clock skew) saturate the signal.  Documents without a ``created_at``
        metadata field contribute zero so unannotated stores degrade
        gracefully.
        """
        ts_raw = metadata.get("created_at")
        if ts_raw is None:
            return 0.0
        try:
            ts = float(ts_raw)
        except (TypeError, ValueError):
            return 0.0

        if ts > now + 60.0:
            return 1.0

        age = now - ts
        if age < 0.0:
            return 1.0
        ramp_window = self._freshness_window * 0.1
        if age >= ramp_window or ramp_window <= 0.0:
            return 0.0
        return float(1.0 - age / ramp_window)

    def _retrieval_frequency_score(self, doc_id: str) -> float:
        """z-score of the doc's retrieval count, mapped to ``[0, 1]``.

        Returns ``0.0`` when fewer than two distinct documents have been
        observed (the variance is ill-defined) or when the population
        standard deviation is zero.
        """
        n_unique = len(self._retrieval_counts)
        if n_unique < 2 or self._total_retrievals == 0:
            return 0.0
        count = self._retrieval_counts.get(doc_id, 0)
        if count == 0:
            return 0.0

        mean = self._total_retrievals / n_unique
        var = sum((c - mean) ** 2 for c in self._retrieval_counts.values()) / n_unique
        if var <= 0.0:
            return 0.0
        std = math.sqrt(var)
        z = (count - mean) / std
        if z <= 0.0:
            return 0.0
        return float(min(1.0, z / self._retrieval_spike_z))

    @staticmethod
    def _content_mismatch_score(
        stored: NDArray[np.float64],
        recomputed: NDArray[np.float64],
    ) -> float:
        """Rescaled cosine distance between stored and recomputed embeddings.

        Cosine similarity in ``[-1, 1]`` is mapped to a distance in
        ``[0, 1]`` via ``(1 - sim) / 2``.  A perfect match yields ``0.0``;
        an antipodal pair yields ``1.0``.
        """
        a_norm = float(np.linalg.norm(stored))
        b_norm = float(np.linalg.norm(recomputed))
        if a_norm == 0.0 or b_norm == 0.0:
            return 1.0
        sim = float((stored / a_norm) @ (recomputed / b_norm))
        dist = (1.0 - sim) / 2.0
        return float(min(1.0, max(0.0, dist)))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_weights(weights: dict[str, float]) -> dict[str, float]:
        """Validate and L1-normalise *weights*.

        Raises:
            ValueError: If a required signal key is missing, if any weight
                is negative, or if the weights do not sum to a positive value.
        """
        missing = _REQUIRED_WEIGHT_KEYS - weights.keys()
        if missing:
            raise ValueError(f"weights missing required keys: {sorted(missing)}")
        for key in _REQUIRED_WEIGHT_KEYS:
            if weights[key] < 0.0:
                raise ValueError(f"weight {key!r} must be non-negative (got {weights[key]})")
        total = float(sum(weights[k] for k in _REQUIRED_WEIGHT_KEYS))
        if total <= 0.0:
            raise ValueError("weights must sum to a positive value")
        return {k: float(weights[k]) / total for k in _REQUIRED_WEIGHT_KEYS}
