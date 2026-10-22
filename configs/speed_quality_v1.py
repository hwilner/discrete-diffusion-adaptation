"""Pre-registered speed/quality benchmark harness configuration (v1).

This module is the checked-in protocol for the speed/quality benchmark
harness (issue #4, child issue #19). Every value here was fixed before
any benchmark results are examined and must not be tuned post hoc;
changing any value invalidates comparability with runs recorded under
this config name. See ``docs/BENCHMARK.md`` for the full documentation.

The config is a plain Python module (rather than JSON/YAML) so it needs
no parser dependency beyond the standard library and so each choice can
be documented inline. ``dida.benchmark.load_config`` validates every
field strictly: a missing key, an unknown extra key, or a wrong type is
a loud ``ValueError``.
"""

CONFIG = {
    # Human-readable identifier copied into every emitted record so
    # results are self-describing.
    "config_name": "speed-quality-v1",

    # Synthetic task family. "shift-by-one": the unique correct output
    # is each source token shifted by one modulo the vocabulary, so the
    # task has exactly one correct answer and exact match is the
    # headline quality metric per docs/METHODS.md section 4.2.
    "task_type": "shift-by-one",

    # Task shape: 16-token sequences over a 128-token vocabulary.
    "sequence_length": 16,
    "vocab_size": 128,

    # Fixed replication-arm seeds. Task instances are derived from
    # (seed, instance_index) only, so every step budget is evaluated on
    # identical tasks.
    "seeds": [0, 1, 2],

    # Fixed forward-pass budgets swept by the harness (speed axis).
    "step_budgets": [1, 2, 4, 8, 16],

    # Task instances per (seed, step_budget) cell.
    "samples_per_cell": 4,

    # Both quality metrics from docs/METHODS.md section 4.2 are recorded
    # side by side per cell; disagreements are reported, never resolved
    # post hoc.
    "quality_metrics": ["exact_match", "total_variation"],
}
