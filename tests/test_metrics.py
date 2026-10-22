"""Deterministic unit tests for ``dida.metrics``.

Every fixture below is synthetic and constructed in memory: known token
sequences and hand-computed probability tables, so the expected values
are exact (no tolerances needed for the endpoint cases). The tests pin
the input/output conventions documented in ``dida/metrics.py`` — strict
shape validation for exact match and count-or-probability inputs with
internal normalization for the distributional distance — because the
benchmark harness (issue #4) relies on those conventions for its
bit-for-bit repeatability guarantee.
"""

import pytest

from dida.metrics import (
    empirical_distribution,
    exact_match_accuracy,
    total_variation_distance,
)


class TestExactMatchAccuracy:
    """Exact-match metric: the headline metric for unique-answer tasks."""

    def test_all_correct_returns_exactly_one(self):
        predictions = [[1, 2, 3], [4, 5, 6]]
        targets = [[1, 2, 3], [4, 5, 6]]
        assert exact_match_accuracy(predictions, targets) == 1.0

    def test_none_correct_returns_exactly_zero(self):
        predictions = [[9, 9, 9], [8, 8, 8]]
        targets = [[1, 2, 3], [4, 5, 6]]
        assert exact_match_accuracy(predictions, targets) == 0.0

    def test_partial_credit_is_exact_fraction(self):
        # Two of four sequences match exactly -> exactly 0.5.
        predictions = [[1, 2], [0, 0], [5, 6], [0, 0]]
        targets = [[1, 2], [3, 4], [5, 6], [7, 8]]
        assert exact_match_accuracy(predictions, targets) == 0.5

    def test_single_token_difference_fails_the_whole_sequence(self):
        # Exact match gives no partial credit within a sequence: one
        # wrong token makes the whole sequence incorrect.
        predictions = [[1, 2, 99]]
        targets = [[1, 2, 3]]
        assert exact_match_accuracy(predictions, targets) == 0.0

    def test_tuple_sequences_compare_by_value(self):
        # Any sequence-like container must compare by value, not type.
        predictions = [(1, 2, 3)]
        targets = [[1, 2, 3]]
        assert exact_match_accuracy(predictions, targets) == 1.0

    def test_empty_inputs_raise(self):
        with pytest.raises(ValueError):
            exact_match_accuracy([], [])

    def test_outer_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            exact_match_accuracy([[1, 2]], [[1, 2], [3, 4]])

    def test_inner_length_mismatch_raises(self):
        # Strict by contract: a truncated generation is a harness bug,
        # not a quiet zero.
        with pytest.raises(ValueError):
            exact_match_accuracy([[1, 2]], [[1, 2, 3]])


class TestEmpiricalDistribution:
    """Empirical distributions over pattern statistics."""

    def test_uniform_samples_normalize_to_equal_probability(self):
        samples = [[1, 2], [3, 4]]
        distribution = empirical_distribution(samples)
        assert distribution == {(1, 2): 0.5, (3, 4): 0.5}

    def test_repeated_samples_accumulate_counts(self):
        samples = [[1, 2], [1, 2], [3, 4]]
        distribution = empirical_distribution(samples)
        assert distribution == {(1, 2): pytest.approx(2.0 / 3.0), (3, 4): pytest.approx(1.0 / 3.0)}

    def test_custom_pattern_statistic_defines_support(self):
        # The sorted-multiset statistic groups permutations together,
        # which is the multi-answer-task use case from METHODS.md 4.2.
        samples = [[1, 2], [2, 1], [3, 3]]
        distribution = empirical_distribution(samples, key=lambda s: tuple(sorted(s)))
        assert distribution == {(1, 2): pytest.approx(2.0 / 3.0), (3, 3): pytest.approx(1.0 / 3.0)}

    def test_probabilities_sum_to_one(self):
        samples = [[i % 3] for i in range(10)]
        distribution = empirical_distribution(samples)
        assert sum(distribution.values()) == pytest.approx(1.0)

    def test_empty_samples_raise(self):
        with pytest.raises(ValueError):
            empirical_distribution([])


class TestTotalVariationDistance:
    """Total variation distance between empirical distributions."""

    def test_identical_distributions_return_exactly_zero(self):
        p = {"a": 0.5, "b": 0.5}
        assert total_variation_distance(p, dict(p)) == 0.0

    def test_disjoint_supports_return_exactly_one(self):
        p = {"a": 1.0}
        q = {"b": 1.0}
        assert total_variation_distance(p, q) == 1.0

    def test_known_intermediate_value(self):
        # Union support {a, b, c}: |0.5-0.5| + |0.5-0| + |0-0.5| = 1.0,
        # halved -> 0.5.
        p = {"a": 0.5, "b": 0.5}
        q = {"a": 0.5, "c": 0.5}
        assert total_variation_distance(p, q) == 0.5

    def test_raw_counts_are_normalized_internally(self):
        # Counts and probabilities describing the same distribution must
        # give the same distance.
        p_counts = {"a": 2.0, "b": 2.0}
        p_probs = {"a": 0.5, "b": 0.5}
        q = {"a": 0.25, "b": 0.75}
        assert total_variation_distance(p_counts, q) == total_variation_distance(p_probs, q)

    def test_distance_is_symmetric(self):
        p = {"a": 0.9, "b": 0.1}
        q = {"a": 0.2, "b": 0.8}
        assert total_variation_distance(p, q) == total_variation_distance(q, p)

    def test_empty_distribution_raises(self):
        with pytest.raises(ValueError):
            total_variation_distance({}, {"a": 1.0})

    def test_negative_mass_raises(self):
        with pytest.raises(ValueError):
            total_variation_distance({"a": -1.0, "b": 2.0}, {"a": 1.0})

    def test_zero_total_mass_raises(self):
        with pytest.raises(ValueError):
            total_variation_distance({"a": 0.0}, {"a": 1.0})


class TestDeterminism:
    """The metrics are pure functions: identical inputs, identical outputs."""

    def test_repeated_calls_return_identical_results(self):
        predictions = [[1, 2], [3, 4], [5, 6]]
        targets = [[1, 2], [0, 0], [5, 6]]
        first = exact_match_accuracy(predictions, targets)
        second = exact_match_accuracy(predictions, targets)
        assert first == second

    def test_repeated_distribution_calls_return_identical_results(self):
        samples = [[1, 1], [2, 2], [1, 1]]
        first = empirical_distribution(samples)
        second = empirical_distribution(samples)
        assert first == second
