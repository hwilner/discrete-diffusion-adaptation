"""Quality metrics for the speed/quality benchmark harness.

This module implements the two quality metrics selected in
``docs/METHODS.md`` section 4.2 for the benchmark harness tracked in
issue #4. Both metrics are reported side by side; exact match is the
headline metric on tasks with a single correct answer, and the
distributional distance is the headline metric on multi-answer tasks
where mode collapse is the risk. If the two ever disagree on a
conclusion, the disagreement is reported rather than resolved post hoc.

Input/output conventions
------------------------
* Generated outputs and ground-truth targets are sequences of token
  sequences (for example ``list[list[int]]``). Tokens only need to be
  hashable and equality-comparable; the metrics never inspect token
  semantics.
* Empirical distributions are plain ``dict`` mappings from a hashable
  pattern statistic (for example a tuple of tokens) to non-negative
  counts or probabilities. Inputs are re-normalized internally, so
  callers may pass raw counts.
* All metrics are pure, deterministic functions of their inputs: no
  randomness, no global state, no I/O. Bit-for-bit identical inputs
  therefore always yield bit-for-bit identical outputs, which is what
  makes the harness-level repeatability guarantee testable.

Everything here is synthetic-data evaluation machinery only; no
empirical benchmark results are produced by this module itself.
"""

from collections.abc import Hashable as _HashableABC
from typing import Any, Callable, Dict, Hashable, Iterable, Sequence

__all__ = [
    "exact_match_accuracy",
    "empirical_distribution",
    "total_variation_distance",
]


def exact_match_accuracy(
    predictions: Sequence[Sequence[Any]],
    targets: Sequence[Sequence[Any]],
) -> float:
    """Compute the fraction of generated sequences exactly equal to target.

    Exact match is the headline quality metric for unique-answer
    synthetic tasks (METHODS.md section 4.2, Option A): a generation is
    correct only if every token matches the ground-truth pattern, so the
    metric is the plain fraction of exact hits with no partial credit.

    Args:
        predictions: Generated token sequences, one per task instance.
        targets: Ground-truth token sequences, aligned index-by-index
            with ``predictions``.

    Returns:
        A float in ``[0.0, 1.0]``: ``correct / len(targets)``. Returns
        ``1.0`` exactly when every pair matches and ``0.0`` exactly when
        none do, so deterministic fixtures can assert exact values.

    Raises:
        ValueError: If the inputs are empty, have different lengths, or
            contain a pair whose sequence lengths differ. Strict
            validation is deliberate: silently scoring a truncated
            generation as "wrong" would hide harness bugs, so shape
            mismatches fail loudly instead.
    """
    if len(predictions) == 0 or len(targets) == 0:
        raise ValueError("exact_match_accuracy requires non-empty inputs.")
    if len(predictions) != len(targets):
        raise ValueError(
            "predictions and targets must have the same number of "
            "sequences (got %d vs %d)." % (len(predictions), len(targets))
        )
    correct = 0
    for index, (prediction, target) in enumerate(zip(predictions, targets)):
        if len(prediction) != len(target):
            raise ValueError(
                "sequence length mismatch at index %d (%d vs %d); exact "
                "match requires equal lengths." % (index, len(prediction), len(target))
            )
        # Convert to list so any sequence-like container (tuple, tensor
        # row converted upstream, etc.) compares by value, not identity.
        if list(prediction) == list(target):
            correct += 1
    return correct / float(len(targets))


def empirical_distribution(
    samples: Iterable[Sequence[Hashable]],
    key: Callable[[Sequence[Hashable]], Hashable] = None,
) -> Dict[Hashable, float]:
    """Build a normalized empirical distribution over pattern statistics.

    Multi-answer synthetic tasks (METHODS.md section 4.2, Option B) are
    evaluated distributionally: many outputs are valid, so quality means
    matching the task's valid-output distribution rather than one
    answer. This helper turns a batch of generated sequences into the
    normalized frequency distribution that
    :func:`total_variation_distance` consumes.

    Args:
        samples: Iterable of generated token sequences. Must yield at
            least one item.
        key: Optional callable mapping one sample to the hashable
            pattern statistic that defines the distribution's support
            (for example "the sorted token multiset"). Defaults to the
            identity statistic ``tuple(sample)``, i.e. distributions
            over exact output sequences.

    Returns:
        A ``dict`` mapping each observed statistic to its empirical
        probability (probabilities sum to ``1.0`` up to floating-point
        rounding). Iteration order follows first-observation order, but
        callers must not rely on it; the metric treats the mapping as an
        unordered support.

    Raises:
        ValueError: If ``samples`` yields no items (a distribution over
            an empty sample set is undefined).
    """
    counts: Dict[Hashable, float] = {}
    total = 0
    for sample in samples:
        statistic = tuple(sample) if key is None else key(sample)
        if not isinstance(statistic, _HashableABC):
            raise ValueError("pattern statistic must be hashable.")
        counts[statistic] = counts.get(statistic, 0.0) + 1.0
        total += 1
    if total == 0:
        raise ValueError("empirical_distribution requires at least one sample.")
    return {statistic: count / float(total) for statistic, count in counts.items()}


def total_variation_distance(
    p: Dict[Hashable, float],
    q: Dict[Hashable, float],
) -> float:
    """Compute the total variation distance between two distributions.

    Total variation is the declared distributional distance for the
    benchmark harness (METHODS.md section 4.2 names "total variation
    over pattern statistics" as the example). It is symmetric, bounded
    in ``[0.0, 1.0]``, equals ``0.0`` exactly for identical supports and
    probabilities, and equals ``1.0`` exactly for disjoint supports, so
    deterministic fixtures can assert both endpoints exactly.

    Args:
        p: Mapping from pattern statistic to non-negative count or
            probability. Re-normalized internally, so raw counts are
            accepted. Must have positive total mass.
        q: Same convention as ``p``.

    Returns:
        ``0.5 * sum_k |p_k - q_k|`` over the union of the two supports
        as a float in ``[0.0, 1.0]``.

    Raises:
        ValueError: If either mapping is empty, contains a negative
            value, or has zero total mass (a distribution with no mass
            cannot be normalized).
    """
    p_norm = _normalized(p, "p")
    q_norm = _normalized(q, "q")
    support = set(p_norm) | set(q_norm)
    return 0.5 * sum(
        abs(p_norm.get(statistic, 0.0) - q_norm.get(statistic, 0.0))
        for statistic in support
    )


def _normalized(distribution: Dict[Hashable, float], name: str) -> Dict[Hashable, float]:
    """Validate and normalize a distribution mapping to unit total mass.

    Args:
        distribution: Mapping from hashable statistic to non-negative
            count or probability.
        name: Human-readable argument name used in error messages.

    Returns:
        A new ``dict`` with the same support and values scaled to sum
        to ``1.0``.

    Raises:
        ValueError: If the mapping is empty, has a negative entry, or
            has zero total mass.
    """
    if not distribution:
        raise ValueError("%s must be a non-empty distribution." % name)
    total = 0.0
    for statistic, value in distribution.items():
        if value < 0:
            raise ValueError(
                "%s contains a negative mass at %r; counts and "
                "probabilities must be non-negative." % (name, statistic)
            )
        total += value
    if total <= 0.0:
        raise ValueError("%s has zero total mass and cannot be normalized." % name)
    return {statistic: value / total for statistic, value in distribution.items()}
