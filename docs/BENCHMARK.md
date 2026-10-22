# Speed/Quality Benchmark Harness

This document describes the deterministic benchmark harness whose
configuration and runner were added under issue #4 (child issues #19
and #20). It records what is fixed, how to run the harness, and what
the harness deliberately does not do.

## What is fixed (pre-registered)

The harness protocol lives in the checked-in config
`configs/speed_quality_v1.py` and was fixed before any results are
examined. The config is a plain Python module (rather than JSON/YAML)
so it needs no parser dependency beyond the standard library and each
choice is documented inline:

- **Seeds:** `0, 1, 2` (three replication arms).
- **Step-budget grid:** `1, 2, 4, 8, 16` forward passes.
- **Task:** `shift-by-one` — synthetic token sequences of length 16
  over a vocabulary of 128; the unique correct output is each source
  token shifted by one modulo the vocabulary. A unique answer makes
  exact match the headline quality metric per the `docs/METHODS.md`
  section 4.2 selection rule.
- **Samples per cell:** 4 task instances per `(seed, step_budget)`
  cell. Instances depend only on `(seed, instance_index)` — never on
  the budget — so every budget is evaluated on identical tasks.
- **Quality metrics:** both metrics from section 4.2 are recorded per
  cell — the exact-match rate and the total variation distance between
  the cell's empirical output distribution and its valid-output
  distribution. If the two ever disagree on a conclusion, the
  disagreement is reported rather than resolved post hoc.

The config loader (`dida.benchmark.load_config`) is strict: a missing
key, an unknown extra key, or a wrong type is a `ValueError`, so a typo
cannot silently change the recorded protocol. The seed-mixing rule
(`seed * 1000003 + instance_index`) is documented in
`dida/benchmark.py` and pinned by test, because changing it would
change the pre-registered task instances.

## Running the harness

```bash
python -m dida.benchmark --config configs/speed_quality_v1.py --out results.jsonl
```

One JSON record per `(seed, step_budget)` cell is written (sorted keys,
no timestamps), so the same config and the same deterministic sampler
always produce byte-identical output. Omit `--out` to write to stdout.

## Sampler protocol

The runner is sampler-agnostic. A sampler is any callable:

```python
sampler(task, step_budget) -> SamplerOutput(tokens, forward_passes)
```

`sampler` receives a fully materialized `BenchmarkTask` and the cell's
budget, and returns the generated tokens plus the number of model
forward passes it used. The forward-pass count is the speed axis of the
speed/quality tradeoff and is recorded verbatim, so samplers must count
honestly. Malformed outputs (wrong token count, non-positive pass
count) raise `ValueError` at the harness boundary.

The built-in `DiagnosticEchoSampler` used by the CLI is a **wiring
fixture, not a baseline**: it echoes the target tokens (exact match
1.0, total variation 0.0) and reports the budget as spent, which proves
the harness end to end. Real baseline samplers (MaskGIT cosine,
Fast-dLLM confidence threshold) arrive with the sibling baseline issue
and plug into the same protocol without touching the harness.

## Boundary and limitations

- Harness machinery only: **no benchmark results are analyzed here**,
  and no Pareto-curve report is produced (that is the separate
  Pareto-output child issue, which consumes these records).
- No text-leg tasks and no changes to model training, matching the
  parent issue boundary.
- The metrics in `dida/metrics.py` are pure deterministic functions
  with documented input/output conventions (sequences of token
  sequences for exact match; count-or-probability mappings, internally
  normalized, for total variation). See the module docstring for the
  full contract.
