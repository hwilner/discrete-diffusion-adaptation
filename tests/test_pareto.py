"""Deterministic tests for ``dida.pareto`` (Pareto-curve report artifact).

These tests pin the guarantees the Pareto-output child issue of the
benchmark harness (issue #4) exists to provide:

1. **Strict record validation** — the report consumes the harness's
   JSON-lines records; malformed, mixed-protocol, or double-counted
   input fails loudly rather than silently producing a corrupt curve.
2. **Correct aggregation and frontier computation** — per-budget
   across-seed mean/min/max are hand-verifiable, and the Pareto
   frontier respects each metric's documented direction.
3. **Bit-for-bit repeatability** — the same records always produce the
   same report bytes, through the library and through the CLI.

All fixtures are synthetic and in memory; the CLI tests write only to
pytest's ``tmp_path``. No models, no external data, no result claims.
"""

import json

import pytest

from dida.benchmark import (
    DiagnosticEchoSampler,
    load_config,
    run_benchmark,
    serialize_records,
)
from dida.pareto import (
    aggregate_curve,
    build_report,
    load_records,
    main,
    pareto_frontier,
    parse_records,
    serialize_report,
)

# Path to the checked-in pre-registered config, resolved relative to the
# repo root (tests run from the root per pyproject's testpaths config).
CONFIG_PATH = "configs/speed_quality_v1.py"


def _record(seed, budget, passes, exact_match, total_variation, config_name="test-config"):
    """Build one valid harness-shaped record dict for focused tests.

    Args:
        seed: Replication-arm seed for the record's cell.
        budget: The cell's step budget.
        passes: Per-instance forward-pass counts; a bare int is expanded
            to a one-element list.
        exact_match: The cell's exact-match rate, or None to omit the
            key (for metric-set tests).
        total_variation: The cell's total variation distance, or None to
            omit the key.
        config_name: Config identifier copied into the record.

    Returns:
        A record dict that passes validation unless the caller's
        arguments deliberately break it.
    """
    if isinstance(passes, int):
        passes = [passes]
    record = {
        "config_name": config_name,
        "task_type": "shift-by-one",
        "seed": seed,
        "step_budget": budget,
        "samples": len(passes),
        "forward_passes": passes,
    }
    if exact_match is not None:
        record["exact_match"] = exact_match
    if total_variation is not None:
        record["total_variation"] = total_variation
    return record


def _records_text(records):
    """Render record dicts as the harness's sorted-keys JSON lines.

    Args:
        records: The record dicts to serialize.

    Returns:
        One JSON object per line with sorted keys, matching
        ``dida.benchmark.serialize_records`` byte conventions.
    """
    return "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)


def _two_seed_records():
    """Build a small hand-computable two-seed, two-budget record set.

    Metric values are chosen so every aggregate is exactly representable
    in floating point (halves and quarters), letting the aggregation
    tests assert exact equality rather than tolerances.

    Returns:
        A list of four valid records covering seeds 0 and 1 crossed
        with budgets 2 and 4.
    """
    return [
        _record(0, 2, [2, 2], 0.5, 0.5),
        _record(1, 2, [2, 4], 1.0, 0.0),
        _record(0, 4, 4, 0.75, 0.25),
        _record(1, 4, 4, 0.25, 0.75),
    ]


class TestParseRecordsValidation:
    """Strict input schema: every malformed shape is a loud ValueError."""

    def test_empty_payload_raises(self):
        with pytest.raises(ValueError):
            parse_records("")

    def test_blank_lines_only_raises(self):
        with pytest.raises(ValueError):
            parse_records("\n  \n")

    def test_invalid_json_line_raises(self):
        with pytest.raises(ValueError):
            parse_records("{not json}\n")

    def test_non_object_line_raises(self):
        with pytest.raises(ValueError):
            parse_records("[1, 2, 3]\n")

    def test_missing_key_raises(self):
        record = _record(0, 2, 2, 0.5, 0.5)
        del record["seed"]
        with pytest.raises(ValueError):
            parse_records(_records_text([record]))

    def test_unknown_extra_key_raises(self):
        # Strictness rationale: an unknown key is almost always a typo,
        # and silently ignoring it could change the report's meaning.
        record = _record(0, 2, 2, 0.5, 0.5)
        record["exact_mach"] = 0.5
        with pytest.raises(ValueError):
            parse_records(_records_text([record]))

    def test_record_without_any_metric_raises(self):
        record = _record(0, 2, 2, None, None)
        with pytest.raises(ValueError):
            parse_records(_records_text([record]))

    def test_negative_seed_raises(self):
        with pytest.raises(ValueError):
            parse_records(_records_text([_record(-1, 2, 2, 0.5, 0.5)]))

    def test_zero_step_budget_raises(self):
        with pytest.raises(ValueError):
            parse_records(_records_text([_record(0, 0, 2, 0.5, 0.5)]))

    def test_boolean_step_budget_raises(self):
        # bool is an int subclass; the schema must reject it explicitly.
        with pytest.raises(ValueError):
            parse_records(_records_text([_record(0, True, 2, 0.5, 0.5)]))

    def test_forward_passes_length_must_match_samples(self):
        record = _record(0, 2, [2, 2], 0.5, 0.5)
        record["samples"] = 3  # two pass counts cannot cover three samples
        with pytest.raises(ValueError):
            parse_records(_records_text([record]))

    def test_non_positive_forward_pass_count_raises(self):
        with pytest.raises(ValueError):
            parse_records(_records_text([_record(0, 2, [2, 0], 0.5, 0.5)]))

    def test_metric_outside_unit_interval_raises(self):
        # Both approved metrics are bounded in [0, 1]; out-of-range
        # values mean the record is corrupt, not a surprising result.
        with pytest.raises(ValueError):
            parse_records(_records_text([_record(0, 2, 2, 1.5, 0.5)]))

    def test_mixed_config_names_raise(self):
        records = _two_seed_records()
        records[1]["config_name"] = "other-config"
        with pytest.raises(ValueError):
            parse_records(_records_text(records))

    def test_mixed_task_types_raise(self):
        records = _two_seed_records()
        records[1]["task_type"] = "copy"
        with pytest.raises(ValueError):
            parse_records(_records_text(records))

    def test_inconsistent_metric_sets_raise(self):
        records = _two_seed_records()
        del records[2]["total_variation"]
        with pytest.raises(ValueError):
            parse_records(_records_text(records))

    def test_duplicate_cell_raises(self):
        records = _two_seed_records()
        # Re-adding the seed-0/budget-2 cell would double-count it in
        # the across-seed aggregation, so duplicates are rejected.
        records.append(_record(0, 2, 2, 0.5, 0.5))
        with pytest.raises(ValueError):
            parse_records(_records_text(records))

    def test_valid_payload_round_trips(self):
        records = _two_seed_records()
        parsed = parse_records(_records_text(records))
        assert parsed == records


class TestAggregateCurve:
    """Per-budget aggregation across seeds, with exact hand-computed values."""

    def test_budgets_ascending_with_exact_means(self):
        curve = aggregate_curve(parse_records(_records_text(_two_seed_records())))
        assert [point["step_budget"] for point in curve] == [2, 4]
        # Budget 2: exact-match values 0.5 and 1.0 across the two seeds.
        assert curve[0]["metrics"]["exact_match"]["mean"] == 0.75
        assert curve[0]["metrics"]["exact_match"]["min"] == 0.5
        assert curve[0]["metrics"]["exact_match"]["max"] == 1.0
        assert curve[0]["metrics"]["exact_match"]["direction"] == "maximize"
        # Budget 4: total variation values 0.25 and 0.75.
        assert curve[1]["metrics"]["total_variation"]["mean"] == 0.5
        assert curve[1]["metrics"]["total_variation"]["direction"] == "minimize"

    def test_speed_axis_uses_recorded_pass_counts_not_budget(self):
        # Seed 1 at budget 2 recorded [2, 4] passes (mean 3.0); the
        # report must aggregate what was recorded, not assume the budget.
        curve = aggregate_curve(parse_records(_records_text(_two_seed_records())))
        assert curve[0]["mean_forward_passes"] == 2.5  # (2 + 2 + 2 + 4) / 4
        assert curve[1]["mean_forward_passes"] == 4.0

    def test_per_seed_breakdown_in_ascending_seed_order(self):
        curve = aggregate_curve(parse_records(_records_text(_two_seed_records())))
        seeds = [entry["seed"] for entry in curve[0]["per_seed"]]
        assert seeds == [0, 1]
        assert curve[0]["per_seed"][1]["mean_forward_passes"] == 3.0
        assert curve[0]["per_seed"][1]["exact_match"] == 1.0

    def test_iteration_order_of_input_does_not_change_output(self):
        records = _two_seed_records()
        shuffled = [records[2], records[0], records[3], records[1]]
        assert aggregate_curve(parse_records(_records_text(shuffled))) == aggregate_curve(
            parse_records(_records_text(records))
        )


class TestParetoFrontier:
    """Frontier computation respects dominance and metric direction."""

    def _curve_with_dominated_middle(self):
        # Budget 4 is dominated for exact match: budget 2 is cheaper AND
        # strictly better (0.9 > 0.8), so 4 must drop off the frontier.
        records = [
            _record(0, 1, 1, 0.5, 0.4),
            _record(0, 2, 2, 0.9, 0.4),
            _record(0, 4, 4, 0.8, 0.1),
            _record(0, 8, 8, 1.0, 0.2),
        ]
        return aggregate_curve(parse_records(_records_text(records)))

    def test_dominated_point_is_excluded(self):
        curve = self._curve_with_dominated_middle()
        assert pareto_frontier(curve, "exact_match") == [1, 2, 8]

    def test_minimize_metric_uses_lower_is_better(self):
        # Total variation is minimized: budget 2 (0.4) is dominated by
        # budget 4 (0.1) only on price, and budget 1 (0.4 at one pass)
        # ties budget 2's quality at lower cost, so the frontier is
        # {1, 4}; budget 8 (0.2) is worse than 4 at higher cost.
        curve = self._curve_with_dominated_middle()
        assert pareto_frontier(curve, "total_variation") == [1, 4]

    def test_exact_ties_are_all_kept(self):
        # Identical (passes, quality) points dominate neither each other
        # nor themselves; dropping one would be an undocumented choice.
        records = [
            _record(0, 2, 2, 0.5, 0.5),
            _record(0, 4, 2, 0.5, 0.5),
        ]
        curve = aggregate_curve(parse_records(_records_text(records)))
        assert pareto_frontier(curve, "exact_match") == [2, 4]

    def test_unknown_metric_raises(self):
        curve = aggregate_curve(parse_records(_records_text(_two_seed_records())))
        with pytest.raises(ValueError):
            pareto_frontier(curve, "accuracy")

    def test_metric_absent_from_curve_raises(self):
        records = [_record(0, 2, 2, 0.5, None)]  # only exact_match recorded
        curve = aggregate_curve(parse_records(_records_text(records)))
        with pytest.raises(ValueError):
            pareto_frontier(curve, "total_variation")


class TestBuildReport:
    """The report artifact is self-describing and reproducible."""

    def test_report_identity_fields(self):
        report = build_report(parse_records(_records_text(_two_seed_records())))
        assert report["artifact"] == "pareto-curve-report"
        assert report["config_name"] == "test-config"
        assert report["task_type"] == "shift-by-one"
        assert report["num_records"] == 4
        assert report["seeds"] == [0, 1]
        assert report["step_budgets"] == [2, 4]
        assert set(report["pareto_frontier"]) == {"exact_match", "total_variation"}

    def test_serialization_is_byte_identical_across_builds(self):
        records = parse_records(_records_text(_two_seed_records()))
        assert serialize_report(build_report(records)) == serialize_report(build_report(records))

    def test_serialized_report_uses_sorted_keys(self):
        report = build_report(parse_records(_records_text(_two_seed_records())))
        parsed = json.loads(serialize_report(report))
        assert list(parsed.keys()) == sorted(parsed.keys())


class TestCli:
    """CLI entry point: same report bytes to a file or stdout, twice over."""

    def _write_records(self, tmp_path):
        path = tmp_path / "records.jsonl"
        path.write_text(_records_text(_two_seed_records()), encoding="utf-8")
        return str(path)

    def test_cli_writes_byte_identical_reports_across_runs(self, tmp_path):
        records_path = self._write_records(tmp_path)
        first = tmp_path / "first.json"
        second = tmp_path / "second.json"
        assert main(["--records", records_path, "--out", str(first)]) == 0
        assert main(["--records", records_path, "--out", str(second)]) == 0
        assert first.read_bytes() == second.read_bytes()

    def test_cli_stdout_mode(self, tmp_path, capsys):
        records_path = self._write_records(tmp_path)
        assert main(["--records", records_path]) == 0
        captured = capsys.readouterr()
        report = json.loads(captured.out)
        assert report["artifact"] == "pareto-curve-report"

    def test_cli_requires_records_argument(self):
        with pytest.raises(SystemExit):
            main([])

    def test_cli_missing_records_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            main(["--records", str(tmp_path / "absent.jsonl")])


class TestHarnessIntegration:
    """End-to-end wiring: real harness records flow into the report.

    The diagnostic echo sampler is a wiring fixture (exact match 1.0,
    total variation 0.0 at every budget, passes equal to the budget), so
    the cheapest budget dominates every other point on both metrics and
    the frontier must be exactly ``[1]``. Any wiring break — records not
    parsing, passes not recorded, frontier inverted — fails this test.
    """

    def test_echo_sampler_records_yield_degenerate_flat_curve(self, tmp_path):
        config = load_config(CONFIG_PATH)
        records_path = tmp_path / "records.jsonl"
        records_path.write_text(
            serialize_records(run_benchmark(config, DiagnosticEchoSampler())),
            encoding="utf-8",
        )
        report = build_report(load_records(str(records_path)))
        assert report["config_name"] == config.config_name
        assert report["step_budgets"] == list(config.step_budgets)
        for point in report["curve"]:
            assert point["metrics"]["exact_match"]["mean"] == 1.0
            assert point["metrics"]["total_variation"]["mean"] == 0.0
            assert point["mean_forward_passes"] == float(point["step_budget"])
        assert report["pareto_frontier"]["exact_match"] == [1]
        assert report["pareto_frontier"]["total_variation"] == [1]
