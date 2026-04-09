"""Shannon entropy of retrieval similarity score distributions (task 1.4).

Computes per-step memory query entropy — signal #6 in the behavioural signal
set.  Entropy is normalised to ``[0, 1]`` so scores from runs with different
``k`` values remain comparable.

Interpretation:
    * **0.0** — all retrieval weight on one document (perfectly sharp, or only
      one result returned).
    * **1.0** — all k documents equally similar to the query (uniform
      distribution = maximum entropy).

Under memory-poisoning attacks the expected effect is a **spike** toward 1.0,
because injected documents introduce semantically similar decoys that flatten
the similarity distribution.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Floor applied to similarity scores before normalisation to avoid log(0).
_EPS: float = 1e-12


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def retrieval_entropy(scores: NDArray[np.float64]) -> float:
    """Shannon entropy of a similarity-score distribution, normalised to [0, 1].

    Converts raw similarity scores to a probability simplex by clipping
    negatives to zero and L1-normalising.  Then computes Shannon entropy and
    divides by ``log(k)`` (the maximum achievable entropy for *k* outcomes)
    to yield a value in ``[0, 1]``.

    Args:
        scores: 1-D float64 array of similarity scores for the top-k retrieved
            documents in one retrieval call.  Scores may be cosine similarities
            or any non-negative relevance measure.  Negative values are treated
            as zero weight (irrelevant documents).

    Returns:
        Normalised entropy in ``[0.0, 1.0]``.  Returns ``0.0`` for empty
        arrays, single-element arrays, and arrays whose positive mass sums to
        zero (all scores ≤ 0).

    Examples:
        >>> import numpy as np
        >>> retrieval_entropy(np.array([1.0, 1.0, 1.0]))  # uniform → max
        1.0
        >>> retrieval_entropy(np.array([1.0, 0.0, 0.0]))  # delta → min
        0.0
        >>> retrieval_entropy(np.array([0.7, 0.2, 0.1]))  # partial spread
        0.8...
    """
    s = np.asarray(scores, dtype=np.float64).ravel()

    k = s.shape[0]
    if k <= 1:
        return 0.0

    # Clip negatives; treat them as zero relevance.
    s = np.maximum(s, 0.0)

    total = s.sum()
    if total <= _EPS:
        return 0.0

    # Probability simplex.
    p = s / total

    # Shannon entropy H = -Σ p_i log(p_i), ignoring zero-weight terms.
    nonzero = p > _EPS
    h: float = -float(np.sum(p[nonzero] * np.log(p[nonzero])))

    # Normalise by log(k) — maximum possible entropy for k outcomes.
    h_max: float = np.log(float(k))
    if h_max <= _EPS:
        return 0.0

    return float(np.clip(h / h_max, 0.0, 1.0))


def step_entropy(score_batches: list[NDArray[np.float64]]) -> float:
    """Mean normalised entropy across multiple retrieval calls in one step.

    A pipeline step may involve several retrieval calls (e.g. one per agent).
    This helper pools them into a single per-step entropy value by averaging.

    Args:
        score_batches: List of 1-D score arrays, one per retrieval call.
            May be empty.

    Returns:
        Mean of :func:`retrieval_entropy` across all batches.  Returns
        ``0.0`` for an empty list.
    """
    if not score_batches:
        return 0.0
    values = [retrieval_entropy(b) for b in score_batches]
    return float(np.mean(values))
