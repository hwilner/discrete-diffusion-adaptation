"""Deterministic speed/quality benchmark harness runner.

This module provides the checked-in-configuration machinery required by
the speed/quality benchmark harness (issue #4, child issue #19):
documented seeds, a fixed step-budget grid, a deterministic synthetic
task fixture, a sampler-agnostic runner, and a CLI entry point.

Design contract
---------------
* **Fixed before inspection.** Seeds, the step-budget grid, the task
  specification, and the quality-metric set live in a checked-in Python
  config module (``configs/speed_quality_v1.py`` — a plain module rather
  than JSON/YAML so no parser dependency is needed and each choice is
  documented inline). They were fixed before any benchmark results are
  examined and must not be tuned post hoc.
* **Sampler-agnostic.** A sampler is any callable
  ``sampler(task, step_budget) -> SamplerOutput``. The runner never
  imports a model; real baseline samplers land in a sibling issue and
  plug in through this protocol without touching the harness.
* **Bit-for-bit repeatable.** Task instances depend only on
  ``(seed, instance_index)`` through an explicitly documented integer
  seed-mixing rule (never on the step budget, so a sweep varies budgets
  over identical tasks), records are emitted in a fixed iteration order,
  and serialization uses sorted keys with no timestamps. Given the same
  config and the same deterministic sampler, two runs produce identical
  bytes.

Metrics
-------
Both quality metrics from ``docs/METHODS.md`` section 4.2 are recorded
per ``(seed, step_budget)`` cell: the exact-match rate (headline for
this unique-answer task) and the total variation distance between the
cell's empirical output distribution and its valid-output distribution
(headline for multi-answer tasks). Metrics are computed by
``dida.metrics``; this module only wires them to the grid.

Boundary
--------
This is harness machinery only: no baseline implementations, no Pareto
curve report (a separate child issue), no text-leg tasks, no analysis
of results, and no empirical claims. The built-in
:class:`DiagnosticEchoSampler` is a wiring fixture that proves the
harness end to end; it is deliberately not a baseline and must never be
reported as one.

CLI
---
``python -m dida.benchmark --config configs/speed_quality_v1.py [--out PATH]``
writes one JSON record per ``(seed, step_budget)`` cell (sorted keys,
newline-delimited) to ``PATH`` or stdout.
"""

import argparse
import importlib.util
import json
import random
import sys
from typing import Any, Callable, Dict, List, NamedTuple, Sequence, Tuple

from dida.metrics import exact_match_accuracy, empirical_distribution, total_variation_distance

__all__ = [
    "BenchmarkConfig",
    "BenchmarkTask",
    "SamplerOutput",
    "DiagnosticEchoSampler",
    "load_config",
    "make_task",
    "run_benchmark",
    "main",
]

# --- Task registry -----------------------------------------------------------
# Only the unique-answer shift-by-one task is implemented; unknown task
# types are rejected at config load so an unsupported name can never
# silently change the meaning of a recorded run.
_TASK_TYPES = ("shift-by-one",)

# Quality metrics the runner knows how to compute, keyed by the names
# used in the checked-in config. Both come from METHODS.md section 4.2.
_METRIC_NAMES = ("exact_match", "total_variation")

# Required config keys and their expected Python types. The loader is
# strict: a missing key, an unknown extra key, or a wrong type is a
# ValueError, so a typo in a hand-edited config fails loudly instead of
# silently falling back to a default.
_CONFIG_SCHEMA = {
    "config_name": str,
    "task_type": str,
    "sequence_length": int,
    "vocab_size": int,
    "seeds": list,
    "step_budgets": list,
    "samples_per_cell": int,
    "quality_metrics": list,
}


class BenchmarkConfig(NamedTuple):
    """Validated, immutable benchmark harness configuration.

    Attributes:
        config_name: Human-readable identifier copied into every record
            so results are self-describing.
        task_type: Synthetic task family; currently only
            ``"shift-by-one"``.
        sequence_length: Number of tokens per task sequence.
        vocab_size: Token vocabulary size for the task generator.
        seeds: Fixed RNG seeds, one per replication arm.
        step_budgets: Fixed grid of forward-pass budgets to sweep.
        samples_per_cell: Task instances per ``(seed, step_budget)`` cell.
        quality_metrics: Metric names to record per cell; a subset of
            ``("exact_match", "total_variation")``.
    """

    config_name: str
    task_type: str
    sequence_length: int
    vocab_size: int
    seeds: Tuple[int, ...]
    step_budgets: Tuple[int, ...]
    samples_per_cell: int
    quality_metrics: Tuple[str, ...]


class BenchmarkTask(NamedTuple):
    """One fully materialized, deterministic synthetic task instance.

    Attributes:
        seed: Config seed this instance belongs to.
        instance_index: Index of the instance within its cell.
        source_tokens: The input sequence shown to the sampler.
        target_tokens: The unique correct output sequence.
    """

    seed: int
    instance_index: int
    source_tokens: Tuple[int, ...]
    target_tokens: Tuple[int, ...]


class SamplerOutput(NamedTuple):
    """What a sampler must return for one task instance.

    Attributes:
        tokens: The generated token sequence.
        forward_passes: Number of model forward passes the sampler used.
            The harness records this verbatim as the speed axis of the
            speed/quality tradeoff, so samplers must count honestly.
    """

    tokens: Tuple[int, ...]
    forward_passes: int


def load_config(path: str) -> BenchmarkConfig:
    """Load and strictly validate a benchmark harness configuration.

    Args:
        path: Filesystem path to a Python config module exposing a
            ``CONFIG`` dict with the schema documented in the module
            docstring and ``docs/BENCHMARK.md``.

    Returns:
        A validated :class:`BenchmarkConfig` with list fields converted
        to immutable tuples.

    Raises:
        ValueError: On any schema violation: missing or unknown keys,
            wrong types, empty ``seeds``/``step_budgets``/
            ``quality_metrics``, non-integer or negative entries,
            non-positive ``sequence_length``/``vocab_size``/
            ``samples_per_cell``, an unknown ``task_type``, or an
            unknown metric name. Strictness is deliberate: the config is
            the pre-registered protocol, so malformed input must fail
            loudly rather than run with silent defaults.
        FileNotFoundError: If ``path`` does not exist.
    """
    # Load the config module directly from its path so the CLI works
    # from any checkout location without requiring installation.
    spec = importlib.util.spec_from_file_location("dida_benchmark_config", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load benchmark config from %r." % (path,))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # raises FileNotFoundError if missing
    raw = getattr(module, "CONFIG", None)
    if not isinstance(raw, dict):
        raise ValueError("benchmark config module must define a CONFIG dict.")

    missing = sorted(key for key in _CONFIG_SCHEMA if key not in raw)
    if missing:
        raise ValueError("benchmark config is missing keys: %s" % ", ".join(missing))
    unknown = sorted(key for key in raw if key not in _CONFIG_SCHEMA)
    if unknown:
        raise ValueError(
            "benchmark config has unknown keys: %s (the schema is strict so "
            "typos cannot silently change the protocol)" % ", ".join(unknown)
        )
    for key, expected_type in _CONFIG_SCHEMA.items():
        # bool is a subclass of int in Python; exclude it explicitly so
        # `true` is never accepted where an integer count is required.
        if expected_type is int:
            valid = isinstance(raw[key], int) and not isinstance(raw[key], bool)
        else:
            valid = isinstance(raw[key], expected_type)
        if not valid:
            raise ValueError(
                "config key %r must be of type %s." % (key, expected_type.__name__)
            )

    seeds = tuple(raw["seeds"])
    step_budgets = tuple(raw["step_budgets"])
    quality_metrics = tuple(raw["quality_metrics"])

    if not seeds:
        raise ValueError("config key 'seeds' must be a non-empty list.")
    if not step_budgets:
        raise ValueError("config key 'step_budgets' must be a non-empty list.")
    if not quality_metrics:
        raise ValueError("config key 'quality_metrics' must be a non-empty list.")
    for seed in seeds:
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("every seed must be a non-negative integer (got %r)." % (seed,))
    for budget in step_budgets:
        if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
            raise ValueError("every step budget must be a positive integer (got %r)." % (budget,))
    for metric in quality_metrics:
        if metric not in _METRIC_NAMES:
            raise ValueError(
                "unknown quality metric %r; supported: %s." % (metric, ", ".join(_METRIC_NAMES))
            )
    if raw["task_type"] not in _TASK_TYPES:
        raise ValueError(
            "unknown task type %r; supported: %s." % (raw["task_type"], ", ".join(_TASK_TYPES))
        )
    if raw["sequence_length"] <= 0:
        raise ValueError("'sequence_length' must be a positive integer.")
    if raw["vocab_size"] <= 1:
        raise ValueError("'vocab_size' must be greater than 1.")
    if raw["samples_per_cell"] <= 0:
        raise ValueError("'samples_per_cell' must be a positive integer.")

    return BenchmarkConfig(
        config_name=raw["config_name"],
        task_type=raw["task_type"],
        sequence_length=raw["sequence_length"],
        vocab_size=raw["vocab_size"],
        seeds=seeds,
        step_budgets=step_budgets,
        samples_per_cell=raw["samples_per_cell"],
        quality_metrics=quality_metrics,
    )


def _derive_seed(seed: int, instance_index: int) -> int:
    """Mix a config seed and an instance index into one integer seed.

    The mixing rule is part of the pre-registered protocol, so it is
    documented here rather than hidden in a hash: ``seed * 1000003 +
    instance_index`` (1000003 is an arbitrary fixed large prime that
    decorrelates adjacent seeds). The rule uses only integer arithmetic,
    which is deterministic across Python versions and processes — unlike
    ``hash()`` on non-numeric types, which is salted per process.

    Args:
        seed: The config-level seed for the replication arm.
        instance_index: The instance index within the cell.

    Returns:
        A deterministic non-negative integer suitable for
        ``random.Random``.
    """
    return seed * 1000003 + instance_index


def make_task(config: BenchmarkConfig, seed: int, instance_index: int) -> BenchmarkTask:
    """Materialize one deterministic synthetic task instance.

    The ``shift-by-one`` task draws ``sequence_length`` source tokens
    uniformly from ``range(vocab_size)``; the unique correct target is
    each source token shifted by one modulo the vocabulary,
    ``target[i] = (source[i] + 1) % vocab_size``. A unique answer makes
    the task exact-match-headline per the METHODS.md section 4.2
    selection rule.

    Crucially, the instance depends only on ``(seed, instance_index)`` —
    never on the step budget — so every budget in a sweep is evaluated
    on identical tasks, which is what makes the resulting speed/quality
    comparison a controlled sweep rather than a resampling.

    Args:
        config: The validated benchmark configuration.
        seed: The config seed for this replication arm.
        instance_index: The instance index within the cell.

    Returns:
        A fully materialized :class:`BenchmarkTask`.
    """
    rng = random.Random(_derive_seed(seed, instance_index))
    source = tuple(rng.randrange(config.vocab_size) for _ in range(config.sequence_length))
    target = tuple((token + 1) % config.vocab_size for token in source)
    return BenchmarkTask(
        seed=seed,
        instance_index=instance_index,
        source_tokens=source,
        target_tokens=target,
    )


class DiagnosticEchoSampler:
    """Deterministic wiring-fixture sampler; NOT a benchmark baseline.

    Returns the task's target tokens verbatim and reports
    ``step_budget`` forward passes. Its only purpose is to exercise the
    harness end to end (repeatability tests, CLI smoke runs): with it,
    exact match is exactly 1.0 and the distributional distance exactly
    0.0, so any wiring bug shows up as a deviation from those known
    values. Real baseline samplers arrive with the sibling baseline
    issue and plug into the same protocol.
    """

    def __call__(self, task: BenchmarkTask, step_budget: int) -> SamplerOutput:
        """Echo the target tokens and report the full budget as spent.

        Args:
            task: The task instance to "solve".
            step_budget: The forward-pass budget for this cell, reported
                verbatim so the speed axis wiring is exercised.

        Returns:
            A :class:`SamplerOutput` with the target tokens and
            ``step_budget`` forward passes.
        """
        return SamplerOutput(tokens=task.target_tokens, forward_passes=step_budget)


def run_benchmark(
    config: BenchmarkConfig,
    sampler: Callable[[BenchmarkTask, int], SamplerOutput],
) -> List[Dict[str, Any]]:
    """Run a sampler over the full seed x budget grid and collect records.

    Iteration order is fixed (seeds, then step budgets, in config order),
    and each record carries the config name so results are
    self-describing. The function performs no I/O and no timing, which
    keeps the output a pure function of ``(config, sampler)``.

    Args:
        config: The validated benchmark configuration.
        sampler: Any callable following the documented protocol
            ``sampler(task, step_budget) -> SamplerOutput``.

    Returns:
        A list of per-cell records, one per ``(seed, step_budget)``
        combination, each with keys: ``config_name``, ``task_type``,
        ``seed``, ``step_budget``, ``samples``, ``forward_passes`` (the
        per-instance counts in instance order), and one entry per
        configured quality metric (``exact_match`` -> exact-match rate,
        ``total_variation`` -> distance between the cell's empirical
        output distribution and its valid-output distribution).

    Raises:
        ValueError: If the sampler returns a malformed
            :class:`SamplerOutput` (wrong token count or a negative
            forward-pass count). Failing loudly here protects the
            downstream metrics' strict input contract.
    """
    records: List[Dict[str, Any]] = []
    for seed in config.seeds:
        # Materialize the cell's tasks once per seed: identical across
        # budgets by construction (see make_task).
        tasks = [make_task(config, seed, index) for index in range(config.samples_per_cell)]
        for step_budget in config.step_budgets:
            outputs = [_validated_output(sampler(task, step_budget), task) for task in tasks]
            predictions = [list(output.tokens) for output in outputs]
            targets = [list(task.target_tokens) for task in tasks]

            record: Dict[str, Any] = {
                "config_name": config.config_name,
                "task_type": config.task_type,
                "seed": seed,
                "step_budget": step_budget,
                "samples": len(tasks),
                "forward_passes": [output.forward_passes for output in outputs],
            }
            if "exact_match" in config.quality_metrics:
                record["exact_match"] = exact_match_accuracy(predictions, targets)
            if "total_variation" in config.quality_metrics:
                # Multi-answer metric applied to this unique-answer task:
                # the valid-output distribution is the empirical
                # distribution of the targets, so the distance reduces
                # to one minus the mass the sampler places on correct
                # answers. Both metrics are always recorded side by side
                # per the METHODS.md 4.2 selection rule.
                record["total_variation"] = total_variation_distance(
                    empirical_distribution(predictions),
                    empirical_distribution(targets),
                )
            records.append(record)
    return records


def _validated_output(output: SamplerOutput, task: BenchmarkTask) -> SamplerOutput:
    """Validate one sampler output against the harness contract.

    Args:
        output: The sampler's returned value.
        task: The task instance the sampler was given (used for error
            context).

    Returns:
        The output, with tokens converted to a plain tuple.

    Raises:
        ValueError: If the token count does not match the task's target
            length or the forward-pass count is not a positive integer.
    """
    tokens = tuple(output.tokens)
    if len(tokens) != len(task.target_tokens):
        raise ValueError(
            "sampler returned %d tokens for task (seed=%d, instance=%d) with "
            "target length %d."
            % (len(tokens), task.seed, task.instance_index, len(task.target_tokens))
        )
    if not isinstance(output.forward_passes, int) or output.forward_passes <= 0:
        raise ValueError(
            "sampler reported invalid forward_passes=%r for task (seed=%d, "
            "instance=%d); the speed axis must be a positive integer count."
            % (output.forward_passes, task.seed, task.instance_index)
        )
    return SamplerOutput(tokens=tokens, forward_passes=output.forward_passes)


def serialize_records(records: Sequence[Dict[str, Any]]) -> str:
    """Serialize benchmark records deterministically as JSON lines.

    Keys are sorted and no timestamps or host information are included,
    so identical ``(config, sampler)`` inputs always produce identical
    bytes — this is the bit-for-bit repeatability guarantee the
    repeatability test asserts.

    Args:
        records: Records as returned by :func:`run_benchmark`.

    Returns:
        One JSON object per line, trailing newline included.
    """
    return "".join(
        json.dumps(record, sort_keys=True) + "\n" for record in records
    )


def main(argv: Sequence[str] = None) -> int:
    """CLI entry point: run the harness from a checked-in config.

    Uses the :class:`DiagnosticEchoSampler` wiring fixture; pluggable
    baseline samplers arrive with the sibling baseline issue and
    integrate through :func:`run_benchmark` directly.

    Args:
        argv: Argument vector excluding the program name; ``None`` reads
            ``sys.argv``. Accepts ``--config PATH`` (required) and
            ``--out PATH`` (optional; defaults to stdout).

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        prog="python -m dida.benchmark",
        description=(
            "Run the deterministic speed/quality benchmark harness from a "
            "checked-in config and emit one JSON record per (seed, "
            "step_budget) cell."
        ),
    )
    parser.add_argument("--config", required=True, help="path to the benchmark config module")
    parser.add_argument("--out", default=None, help="output path for JSON lines (default: stdout)")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    payload = serialize_records(run_benchmark(config, DiagnosticEchoSampler()))
    if args.out is None:
        sys.stdout.write(payload)
    else:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
