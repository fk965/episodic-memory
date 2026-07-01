"""Tests for the utility scoring function.

The v0.2.0 change: utility is a *confidence-weighted* score (Wilson lower
bound), not a raw adoption ratio. This means a judgment adopted 100 times
must rank above one adopted only once, even though both have a raw ratio of
1.0 — which is what the README's "repeatedly validated" promise requires.
"""

from __future__ import annotations

from episodic_memory.scoring import utility_score


class TestUtilityScore:
    def test_unverified_is_zero(self):
        assert utility_score(0, 0) == 0.0

    def test_bounded_between_zero_and_one(self):
        for a, c in [(1, 0), (100, 0), (5, 3), (0, 10), (50, 50)]:
            s = utility_score(a, c)
            assert 0.0 <= s < 1.0

    def test_magnitude_increases_confidence_at_same_ratio(self):
        # Same 100% adoption ratio, but more evidence -> higher score.
        low = utility_score(1, 0)
        mid = utility_score(4, 0)
        high = utility_score(100, 0)
        assert low < mid < high

    def test_more_corrections_lower_score(self):
        assert utility_score(5, 0) > utility_score(5, 3) > utility_score(5, 10)

    def test_adoption_raises_correction_lowers_monotonic(self):
        # Adding an adoption never lowers the score; adding a correction
        # never raises it.
        base = utility_score(5, 5)
        assert utility_score(6, 5) > base
        assert utility_score(5, 6) < base

    def test_pure_corrections_near_zero(self):
        # No adoptions at all -> score should be very low (below any
        # judgment with at least one adoption).
        assert utility_score(0, 5) < utility_score(1, 5)

    def test_z_parameter_controls_conservatism(self):
        # A larger z (wider confidence interval) is more conservative,
        # pulling the lower bound down for the same evidence.
        conservative = utility_score(4, 0, z=2.58)
        lenient = utility_score(4, 0, z=1.0)
        assert conservative < lenient
