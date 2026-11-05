"""Pareto-curve report artifact for the speed/quality benchmark harness.

This module implements the Pareto-output child issue of the benchmark
harness (issue #4): it consumes the per-cell JSON records produced by
``dida.benchmark`` and emits a single reproducible report artifact that
records forward-pass count against the approved quality metrics — the
speed/quality Pareto curve named in ``docs/ROADMAP.md`` Phase 1.

Design contract
---------------
* **Consumes records, never re-runs.** The input is the harness's JSON
  lines output (one record per ``(seed, step_budget)`` cell, see
  ``dida.benchmark.serialize_records``). This module performs no
  sampling and no model work; it aggregates and formats what the
  harness recorded.
* **Does not change the parent protocol.** Seeds, the step-budget grid,
  the task specification, and the metric set come from the records'
  own ``config_name`` fields; this module validates that every record
  belongs to one config and refuses mixed inputs rather than silently
  merging protocols.
* **Bit-for-bit repeatable.** Aggregation iterates in a fixed order
  (ascending step budget, then ascending seed), and serialization uses
  sorted JSON keys with no timestamps or host information, so identical
  record bytes always produce an identical report.

Report contents
---------------
The report is one JSON object with: the source config identity, the
recorded grid, the per-budget curve (mean forward-pass count and, for
each configured quality metric, the across-seed mean/min/max), and the
Pareto frontier per metric (the step budgets no other budget dominates
on the speed/quality trade). Metric directions are explicit:
``exact_match`` is maximized, ``total_variation`` is minimized.

Boundary
--------
This is reporting machinery only: it draws no conclusions about any
sampler, ranks nothing, and changes no pre-registered metric, seed, or
budget. A report built from the harness's diagnostic wiring fixture
proves the plumbing end to end but is not a benchmark result and must
never be cited as one.

CLI
---
``python -m dida.pareto --records results.jsonl [--out report.json]``
reads harness records and writes the report (to ``--out`` or stdout).
"""

import argparse
import json
import sys
from typing import Any, Dict, List, Sequence

__all__ = [
    "parse_records",
    "load_records",
    "aggregate_curve",
    "pareto_frontier",
    "build_report",
    "serialize_report",
    "main",
]

# Optimization direction per approved quality metric (docs/METHODS.md
# section 4.2). The frontier test needs the direction because a smaller
# total variation distance is better while a higher exact-match rate is
# better; making the direction explicit per metric keeps the dominance
# rule below free of metric-specific special cases.
_METRIC_DIRECTIONS = {"exact_match": "maximize", "total_variation": "minimize"}

# Keys every harness record must carry beyond the metric values. The
# strict schema mirrors the harness config loader's philosophy: a
# malformed record fails loudly instead of being silently repaired,
# because the records are the pre-registered protocol's output.
_REQUIRED_KEYS = ("config_name", "task_type", "seed", "step_budget", "samples", "forward_passes")


def _is_int(value: Any) -> bool:
    """Return True for real integers, rejecting bools.

    ``bool`` is a subclass of ``int`` in Python, and JSON ``true`` would
    otherwise pass an ``isinstance(value, int)`` check; counts and seeds
    must be genuine integers, so the exclusion is explicit.

    Args:
        value: The value to test.

    Returns:
        True iff ``value`` is an ``int`` and not a ``bool``.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_record(record: Any, line_number: int) -> Dict[str, Any]:
    """Validate one harness record against the strict input schema.

    Args:
        record: The parsed JSON object for one line.
        line_number: 1-based line number in the source, used in error
            messages so a malformed file is easy to locate.

    Returns:
        The record unchanged (validation only; no normalization).

    Raises:
        ValueError: On any schema violation: not an object, missing or
            unknown keys, wrong types, a negative seed, a non-positive
            step budget or sample count, a ``forward_passes`` list that
            is empty, mistyped, non-positive, or inconsistent with
            ``samples``, or a metric value that is non-numeric or falls
            outside ``[0.0, 1.0]`` (both approved metrics are bounded
            rates/distances, so out-of-range values indicate corruption).
    """
    if not isinstance(record, dict):
        raise ValueError("line %d: each record must be a JSON object." % line_number)
    missing = sorted(key for key in _REQUIRED_KEYS if key not in record)
    if missing:
        raise ValueError(
            "line %d: record is missing keys: %s" % (line_number, ", ".join(missing))
        )
    metrics_present = sorted(key for key in _METRIC_DIRECTIONS if key in record)
    if not metrics_present:
        raise ValueError(
            "line %d: record carries no quality metric values; expected at "
            "least one of: %s" % (line_number, ", ".join(sorted(_METRIC_DIRECTIONS)))
        )
    unknown = sorted(
        key for key in record if key not in _REQUIRED_KEYS and key not in _METRIC_DIRECTIONS
    )
    if unknown:
        raise ValueError(
            "line %d: record has unknown keys: %s (the schema is strict so a "
            "typo cannot silently change the report)" % (line_number, ", ".join(unknown))
        )

    if not isinstance(record["config_name"], str):
        raise ValueError("line %d: 'config_name' must be a string." % line_number)
    if not isinstance(record["task_type"], str):
        raise ValueError("line %d: 'task_type' must be a string." % line_number)
    if not _is_int(record["seed"]) or record["seed"] < 0:
        raise ValueError("line %d: 'seed' must be a non-negative integer." % line_number)
    if not _is_int(record["step_budget"]) or record["step_budget"] <= 0:
        raise ValueError("line %d: 'step_budget' must be a positive integer." % line_number)
    if not _is_int(record["samples"]) or record["samples"] <= 0:
        raise ValueError("line %d: 'samples' must be a positive integer." % line_number)

    forward_passes = record["forward_passes"]
    if not isinstance(forward_passes, list) or not forward_passes:
        raise ValueError(
            "line %d: 'forward_passes' must be a non-empty list of positive "
            "integers." % line_number
        )
    if len(forward_passes) != record["samples"]:
        raise ValueError(
            "line %d: 'forward_passes' has %d entries but 'samples' is %d; the "
            "harness records one pass count per instance."
            % (line_number, len(forward_passes), record["samples"])
        )
    for count in forward_passes:
        if not _is_int(count) or count <= 0:
            raise ValueError(
                "line %d: every forward-pass count must be a positive integer "
                "(got %r)." % (line_number, count)
            )

    for metric in metrics_present:
        value = record[metric]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(
                "line %d: metric %r must be a number." % (line_number, metric)
            )
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "line %d: metric %r is %r, outside [0.0, 1.0]; both approved "
                "metrics are bounded, so this record is corrupt."
                % (line_number, metric, value)
            )
    return record


def parse_records(text: str) -> List[Dict[str, Any]]:
    """Parse and validate harness JSON-lines output into records.

    Beyond per-record validation, this enforces set-level invariants:
    every record must belong to the same config and task, carry the same
    metric key set, and occupy a distinct ``(seed, step_budget)`` cell.
    Merging two protocols or double-counting a cell would both silently
    corrupt the curve, so they are loud errors.

    Args:
        text: The JSON-lines payload emitted by the harness (one JSON
            object per line, as produced by
            ``dida.benchmark.serialize_records``).

    Returns:
        The validated records as a list of dicts, in input order.

    Raises:
        ValueError: If the payload is empty or blank, any line is not
            valid JSON, any record fails validation, records mix config
            names, task types, or metric sets, or two records share a
            ``(seed, step_budget)`` cell.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("no records: the harness payload is empty.")
    records: List[Dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "line %d: invalid JSON (%s)." % (line_number, exc)
            ) from exc
        records.append(_validate_record(parsed, line_number))

    config_names = {record["config_name"] for record in records}
    if len(config_names) != 1:
        raise ValueError(
            "records mix config names %s; one report covers exactly one "
            "pre-registered config." % (sorted(config_names),)
        )
    task_types = {record["task_type"] for record in records}
    if len(task_types) != 1:
        raise ValueError(
            "records mix task types %s; a curve must come from one task "
            "family." % (sorted(task_types),)
        )
    metric_sets = {
        tuple(sorted(key for key in _METRIC_DIRECTIONS if key in record))
        for record in records
    }
    if len(metric_sets) != 1:
        raise ValueError(
            "records carry inconsistent metric key sets %s; every cell must "
            "record the same approved metrics." % (sorted(metric_sets),)
        )
    cells = [(record["seed"], record["step_budget"]) for record in records]
    if len(set(cells)) != len(cells):
        raise ValueError(
            "duplicate (seed, step_budget) cells found; each cell must appear "
            "exactly once or aggregation would double-count it."
        )
    return records


def load_records(path: str) -> List[Dict[str, Any]]:
    """Read a harness JSON-lines file and return validated records.

    Args:
        path: Filesystem path to the harness output file.

    Returns:
        The validated records as a list of dicts, in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file content fails :func:`parse_records`
            validation.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return parse_records(handle.read())


def aggregate_curve(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate per-cell records into the speed/quality curve.

    Groups records by step budget and aggregates across seeds in a fixed
    order (budgets ascending, seeds ascending), so the output is a pure
    deterministic function of the input records. The speed axis is the
    mean recorded forward-pass count — taken from the records, not
    assumed equal to the budget, because samplers may finish under
    budget and the harness records the count verbatim.

    Args:
        records: Validated records as returned by :func:`parse_records`.

    Returns:
        One entry per step budget, ascending, each a dict with keys:
        ``step_budget``; ``mean_forward_passes`` (mean over every
        per-instance count in the budget's cells); ``per_seed`` (a list
        of ``{"seed", "mean_forward_passes", <metric...>}`` dicts in
        ascending seed order); and ``metrics`` (per metric name, the
        across-seed ``mean``/``min``/``max`` and its optimization
        ``direction``).
    """
    metrics = sorted(key for key in _METRIC_DIRECTIONS if key in records[0])
    budgets = sorted({record["step_budget"] for record in records})
    curve: List[Dict[str, Any]] = []
    for budget in budgets:
        cells = sorted(
            (record for record in records if record["step_budget"] == budget),
            key=lambda record: record["seed"],
        )
        all_passes = [count for cell in cells for count in cell["forward_passes"]]
        per_seed = []
        for cell in cells:
            entry: Dict[str, Any] = {
                "seed": cell["seed"],
                "mean_forward_passes": sum(cell["forward_passes"]) / float(len(cell["forward_passes"])),
            }
            for metric in metrics:
                entry[metric] = cell[metric]
            per_seed.append(entry)
        metric_summary = {}
        for metric in metrics:
            values = [cell[metric] for cell in cells]
            metric_summary[metric] = {
                "direction": _METRIC_DIRECTIONS[metric],
                "mean": sum(values) / float(len(values)),
                "min": min(values),
                "max": max(values),
            }
        curve.append(
            {
                "step_budget": budget,
                "mean_forward_passes": sum(all_passes) / float(len(all_passes)),
                "per_seed": per_seed,
                "metrics": metric_summary,
            }
        )
    return curve


def pareto_frontier(curve: Sequence[Dict[str, Any]], metric: str) -> List[int]:
    """Return the step budgets on the Pareto frontier for one metric.

    A point ``A`` dominates a point ``B`` on the speed/quality trade
    when ``A`` uses no more forward passes and is no worse on the
    metric, with at least one of the two strict (quality compared in
    the metric's documented direction: higher exact match is better,
    lower total variation is better). The frontier is the set of
    budgets no other budget dominates. Points tied on both axes
    dominate neither each other nor themselves, so exact ties are all
    kept — dropping a tie would be an arbitrary, undocumented choice.

    Args:
        curve: The aggregated curve as returned by
            :func:`aggregate_curve`.
        metric: The metric whose frontier is requested; must be one of
            the approved metric names present in the curve.

    Returns:
        The non-dominated step budgets in ascending order.

    Raises:
        ValueError: If ``metric`` is not an approved metric name or is
            absent from the curve.
    """
    if metric not in _METRIC_DIRECTIONS:
        raise ValueError(
            "unknown quality metric %r; supported: %s."
            % (metric, ", ".join(sorted(_METRIC_DIRECTIONS)))
        )
    if any(metric not in point["metrics"] for point in curve):
        raise ValueError("metric %r is not present in every curve point." % metric)

    maximize = _METRIC_DIRECTIONS[metric] == "maximize"

    def dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        # Quality is oriented so that a larger oriented value is always
        # better regardless of the metric's natural direction.
        quality_a = a["metrics"][metric]["mean"]
        quality_b = b["metrics"][metric]["mean"]
        if not maximize:
            quality_a, quality_b = -quality_a, -quality_b
        passes_a = a["mean_forward_passes"]
        passes_b = b["mean_forward_passes"]
        no_worse = passes_a <= passes_b and quality_a >= quality_b
        strictly_better = passes_a < passes_b or quality_a > quality_b
        return no_worse and strictly_better

    frontier = []
    for point in curve:
        if not any(other is not point and dominates(other, point) for other in curve):
            frontier.append(point["step_budget"])
    return sorted(frontier)


def build_report(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the complete Pareto-curve report artifact from records.

    Args:
        records: Validated records as returned by :func:`parse_records`
            (validation guarantees one config, one task type, and a
            consistent metric set, so the report is self-describing).

    Returns:
        The report as a JSON-ready dict with keys: ``artifact`` (the
        fixed artifact identifier ``"pareto-curve-report"``),
        ``config_name``, ``task_type``, ``num_records``, ``seeds`` and
        ``step_budgets`` (the recorded grid, ascending), ``curve`` (the
        aggregated speed/quality curve), and ``pareto_frontier`` (per
        metric, the non-dominated step budgets).
    """
    curve = aggregate_curve(records)
    metrics = sorted(key for key in _METRIC_DIRECTIONS if key in records[0])
    return {
        "artifact": "pareto-curve-report",
        "config_name": records[0]["config_name"],
        "task_type": records[0]["task_type"],
        "num_records": len(records),
        "seeds": sorted({record["seed"] for record in records}),
        "step_budgets": [point["step_budget"] for point in curve],
        "curve": curve,
        "pareto_frontier": {metric: pareto_frontier(curve, metric) for metric in metrics},
    }


def serialize_report(report: Dict[str, Any]) -> str:
    """Serialize a report deterministically as pretty-printed JSON.

    Keys are sorted and no timestamps or host information are included,
    so identical records always produce identical bytes — the same
    bit-for-bit repeatability contract as the harness itself. Two-space
    indentation keeps the single-artifact report human-readable.

    Args:
        report: The report dict as returned by :func:`build_report`.

    Returns:
        The JSON text, trailing newline included.
    """
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def main(argv: Sequence[str] = None) -> int:
    """CLI entry point: build the Pareto-curve report from harness records.

    Args:
        argv: Argument vector excluding the program name; ``None`` reads
            ``sys.argv``. Accepts ``--records PATH`` (required; the JSON
            lines file written by ``python -m dida.benchmark``) and
            ``--out PATH`` (optional; defaults to stdout).

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        prog="python -m dida.pareto",
        description=(
            "Build the reproducible Pareto-curve report (forward-pass count "
            "vs the approved quality metrics) from speed/quality harness "
            "records."
        ),
    )
    parser.add_argument(
        "--records", required=True, help="path to the harness JSON-lines records file"
    )
    parser.add_argument("--out", default=None, help="output path for the report (default: stdout)")
    args = parser.parse_args(argv)

    payload = serialize_report(build_report(load_records(args.records)))
    if args.out is None:
        sys.stdout.write(payload)
    else:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
