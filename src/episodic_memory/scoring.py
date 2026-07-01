"""Confidence-weighted utility scoring.

The utility score answers: "how much should we trust this judgment, given
how often it was adopted versus corrected?"

A naive ratio ``adoption / (adoption + correction)`` fails the core promise
of episodic memory: a judgment adopted once and one adopted a hundred times
both score 1.0, so "repeatedly validated" judgments never actually rank
higher. We use the **Wilson score lower bound** for a binomial proportion
instead. It rewards both a high adoption ratio *and* a large sample size,
so evidence accumulates — which is exactly the flywheel the library sells.

Reference: Wilson (1927); popularised for ranking by Evan Miller,
"How Not To Sort By Average Rating".
"""

from __future__ import annotations

import math

# Default z corresponds to ~85% confidence. Chosen (rather than the textbook
# 1.96 / 95%) so that a handful of adoptions produce a usefully non-trivial
# score; the flywheel needs early signal, not statistical publication rigor.
DEFAULT_Z = 1.44


def utility_score(
    adoption_count: int,
    correction_count: int,
    z: float = DEFAULT_Z,
) -> float:
    """Wilson lower-bound utility in ``[0, 1)``.

    Args:
        adoption_count: Times this judgment was adopted (positive signal).
        correction_count: Times it was corrected (negative signal).
        z: Confidence z-score. Larger = more conservative (wider interval,
            lower bound pulled down for the same evidence).

    Returns:
        ``0.0`` when there is no evidence at all. Otherwise a score that
        rises with the adoption ratio *and* with the total sample size, so
        a well-tested judgment outranks a barely-tested one at equal ratio.
    """
    n = adoption_count + correction_count
    if n == 0:
        return 0.0

    phat = adoption_count / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = phat + z2 / (2 * n)
    margin = z * math.sqrt((phat * (1.0 - phat) + z2 / (4 * n)) / n)
    return (centre - margin) / denom
