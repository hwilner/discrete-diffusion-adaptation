"""Deterministic tests for ``dida.benchmark`` (harness config + runner).

These tests pin the two guarantees the speed/quality harness (issue #4,
child issue #19) exists to provide:

1. **Pre-registration integrity** — the checked-in config fixes the
   seeds, the step-budget grid, the task spec, and the metric set; the
   loader is strict so a malformed or typo'd config fails loudly.
2. **Bit-for-bit repeatability** — given the same config and the same
   deterministic sampler, two runs produce byte-identical output.

All fixtures are synthetic and in memory; the CLI tests write only to
pytest's ``tmp_path``. No models, no external data, no result claims.
"""

import json
import random

import pytest

from dida.benchmark import (
    BenchmarkConfig,
    DiagnosticEchoSampler,
    SamplerOutput,
    load_config,
    main,
    make_task,
    run_benchmark,
    serialize_records,
)

# Path to the checked-in pre-registered config, resolved relative to the
# repo root (tests run from the root per pyproject's testpaths config).
CONFIG_PATH = "configs/speed_quality_v1.py"


def _small_config(**overrides):
    """Build a minimal valid config dict for focused validation tests.

    Args:
        **overrides: Keys to replace in the baseline dict.

    Returns:
        A config dict that passes validation unless an override breaks it.
    """
    baseline = {
        "config_name": "test-config",
        "task_type": "shift-by-one",
        "sequence_length": 4,
        "vocab_size": 8,
        "seeds": [0],
        "step_budgets": [1, 2],
        "samples_per_cell": 2,
        "quality_metrics": ["exact_match", "total_variation"],
    }
    baseline.update(overrides)
    return baseline


def _write_config(tmp_path, config_dict):
    """Write a config dict as a temporary Python config module.

    The dict is rendered with ``repr`` so any Python literal (including
    the booleans used by the strict-typing tests) round-trips exactly.
    """
    path = tmp_path / "config.py"
    path.write_text("CONFIG = %r\n" % (config_dict,), encoding="utf-8")
    return str(path)


class TestCheckedInConfig:
    """The checked-in config is the pre-registered protocol; pin it."""

    def test_checked_in_config_loads_and_validates(self):
        config = load_config(CONFIG_PATH)
        assert isinstance(config, BenchmarkConfig)

    def test_checked_in_config_pins_seeds_and_budgets(self):
        # The pre-registered seeds and step-budget grid are fixed before
        # results are examined; this test fails if they ever drift.
        config = load_config(CONFIG_PATH)
        assert config.seeds == (0, 1, 2)
        assert config.step_budgets == (1, 2, 4, 8, 16)
        assert config.samples_per_cell == 4
        assert config.task_type == "shift-by-one"
        assert config.sequence_length == 16
        assert config.vocab_size == 128
        assert config.quality_metrics == ("exact_match", "total_variation")


class TestConfigValidation:
    """Strict schema: every malformed shape is a loud ValueError."""

    def test_missing_key_raises(self, tmp_path):
        config = _small_config()
        del config["seeds"]
        with pytest.raises(ValueError):
            load_config(_write_config(tmp_path, config))

    def test_unknown_extra_key_raises(self, tmp_path):
        # Strictness rationale: an unknown key is almost always a typo,
        # and silently ignoring it would change the recorded protocol.
        with pytest.raises(ValueError):
            load_config(_write_config(tmp_path, _small_config(seed=1)))

    def test_empty_seeds_raise(self, tmp_path):
        with pytest.raises(ValueError):
            load_config(_write_config(tmp_path, _small_config(seeds=[])))

    def test_negative_seed_raises(self, tmp_path):
        with pytest.raises(ValueError):
            load_config(_write_config(tmp_path, _small_config(seeds=[-1])))

    def test_boolean_budget_raises(self, tmp_path):
        # bool is an int subclass; the loader must reject it explicitly.
        with pytest.raises(ValueError):
            load_config(_write_config(tmp_path, _small_config(step_budgets=[True])))

    def test_zero_budget_raises(self, tmp_path):
        with pytest.raises(ValueError):
            load_config(_write_config(tmp_path, _small_config(step_budgets=[0])))

    def test_unknown_task_type_raises(self, tmp_path):
        with pytest.raises(ValueError):
            load_config(_write_config(tmp_path, _small_config(task_type="copy")))

    def test_unknown_metric_raises(self, tmp_path):
        with pytest.raises(ValueError):
            load_config(_write_config(tmp_path, _small_config(quality_metrics=["accuracy"])))

    def test_wrong_scalar_type_raises(self, tmp_path):
        with pytest.raises(ValueError):
            load_config(_write_config(tmp_path, _small_config(vocab_size="128")))


class TestTaskFixture:
    """The shift-by-one task generator is deterministic and budget-free."""

    def test_same_seed_and_index_give_identical_tasks(self):
        config = load_config(CONFIG_PATH)
        first = make_task(config, seed=1, instance_index=2)
        second = make_task(config, seed=1, instance_index=2)
        assert first == second

    def test_different_seeds_give_different_tasks(self):
        config = load_config(CONFIG_PATH)
        first = make_task(config, seed=0, instance_index=0)
        second = make_task(config, seed=1, instance_index=0)
        assert first.source_tokens != second.source_tokens

    def test_task_matches_documented_mixing_rule(self):
        # Independently recompute the expected tokens from the documented
        # mixing rule seed * 1000003 + instance_index, so a change to the
        # rule (which would invalidate the pre-registered protocol) fails
        # this test even if the implementation stays self-consistent.
        config = load_config(CONFIG_PATH)
        task = make_task(config, seed=2, instance_index=3)
        rng = random.Random(2 * 1000003 + 3)
        expected_source = tuple(rng.randrange(128) for _ in range(16))
        expected_target = tuple((token + 1) % 128 for token in expected_source)
        assert task.source_tokens == expected_source
        assert task.target_tokens == expected_target

    def test_target_is_unique_shift_of_source(self):
        config = load_config(CONFIG_PATH)
        task = make_task(config, seed=0, instance_index=0)
        assert task.target_tokens == tuple((t + 1) % 128 for t in task.source_tokens)


class _ShiftByTwoSampler:
    """Deterministic always-wrong sampler: emits (source + 2) % vocab.

    Because the unique correct answer is (source + 1) % vocab and the
    vocabulary has more than one token, every emission is guaranteed
    wrong, which pins exact match at exactly 0.0 and the distributional
    distance at exactly 1.0.
    """

    def __call__(self, task, step_budget):
        wrong = tuple((token + 2) % 8 for token in task.source_tokens)
        return SamplerOutput(tokens=wrong, forward_passes=step_budget)


class TestRunner:
    """Runner wiring: metric correctness and repeatability."""

    def _config(self):
        return load_config(CONFIG_PATH)

    def test_echo_sampler_scores_perfect_on_both_metrics(self):
        records = run_benchmark(self._config(), DiagnosticEchoSampler())
        for record in records:
            assert record["exact_match"] == 1.0
            assert record["total_variation"] == 0.0

    def test_forward_pass_axis_records_the_budget_per_instance(self):
        config = self._config()
        records = run_benchmark(config, DiagnosticEchoSampler())
        for record in records:
            assert record["forward_passes"] == [record["step_budget"]] * config.samples_per_cell

    def test_record_grid_covers_every_seed_and_budget(self):
        config = self._config()
        records = run_benchmark(config, DiagnosticEchoSampler())
        cells = {(record["seed"], record["step_budget"]) for record in records}
        assert cells == {(seed, budget) for seed in config.seeds for budget in config.step_budgets}
        for record in records:
            assert record["config_name"] == config.config_name
            assert record["task_type"] == "shift-by-one"
            assert record["samples"] == config.samples_per_cell

    def test_repeatability_is_bit_for_bit(self):
        config = self._config()
        first = serialize_records(run_benchmark(config, DiagnosticEchoSampler()))
        second = serialize_records(run_benchmark(config, DiagnosticEchoSampler()))
        assert first == second

    def test_always_wrong_sampler_scores_zero_and_max_distance(self, tmp_path):
        config = load_config(_write_config(tmp_path, _small_config()))
        records = run_benchmark(config, _ShiftByTwoSampler())
        for record in records:
            assert record["exact_match"] == 0.0
            assert record["total_variation"] == 1.0

    def test_malformed_token_count_raises(self, tmp_path):
        class ShortSampler:
            def __call__(self, task, step_budget):
                return SamplerOutput(tokens=(0,), forward_passes=step_budget)

        with pytest.raises(ValueError):
            run_benchmark(load_config(_write_config(tmp_path, _small_config())), ShortSampler())

    def test_invalid_forward_pass_count_raises(self, tmp_path):
        class ZeroPassSampler:
            def __call__(self, task, step_budget):
                return SamplerOutput(tokens=task.target_tokens, forward_passes=0)

        with pytest.raises(ValueError):
            run_benchmark(load_config(_write_config(tmp_path, _small_config())), ZeroPassSampler())


class TestCli:
    """CLI entry point: same bytes to a file or stdout, twice over."""

    def test_cli_writes_byte_identical_output_across_runs(self, tmp_path):
        first = tmp_path / "first.jsonl"
        second = tmp_path / "second.jsonl"
        assert main(["--config", CONFIG_PATH, "--out", str(first)]) == 0
        assert main(["--config", CONFIG_PATH, "--out", str(second)]) == 0
        assert first.read_bytes() == second.read_bytes()

    def test_cli_output_is_sorted_keys_json_lines(self, tmp_path):
        out = tmp_path / "out.jsonl"
        main(["--config", CONFIG_PATH, "--out", str(out)])
        lines = out.read_text(encoding="utf-8").splitlines()
        config = load_config(CONFIG_PATH)
        assert len(lines) == len(config.seeds) * len(config.step_budgets)
        for line in lines:
            record = json.loads(line)
            # json.dumps(..., sort_keys=True) must hold line by line.
            assert list(record.keys()) == sorted(record.keys())
            assert record["exact_match"] == 1.0

    def test_cli_stdout_mode(self, capsys):
        assert main(["--config", CONFIG_PATH]) == 0
        captured = capsys.readouterr()
        assert len(captured.out.splitlines()) == 15

    def test_cli_requires_config_argument(self):
        with pytest.raises(SystemExit):
            main([])
