# Project Board

## Status: Ready to start

This is the in-repo mirror of the issue backlog. Phases and rationale live in
`docs/ROADMAP.md`; open design choices in `docs/METHODS.md`; how to pick up
work in `CONTRIBUTING.md`. Sizes: `[XS]` ≈ an hour or two, `[S]` ≈ a day,
`[L]` ≈ multi-day epic.

### Ready now (no unresolved dependencies)

| Issue | Title | Size |
|-------|-------|------|
| #5 | Implement MaskGIT cosine + Fast-dLLM confidence-threshold remasking baselines behind one scheduler API | S |
| #4 | Speed/quality benchmark harness: forward-pass count vs synthetic-task quality with seeds | S |
| #6 | Synthetic text task suite with strong position dependencies (arithmetic carries, bracket matching) + data generators | S |
| #12 | pyproject.toml packaging + pytest config cleanup | XS |
| #16 | Example script: end-to-end image-token generation walkthrough with comments | XS |
| #18 | docs/API_REFERENCE.md for DiDACore/DiscreteDiffusionScheduler/DiDAInference | S |

### Epic: Phase 1 — Baselines and Pareto evaluation harness (#1)

| Issue | Title | Size | Blocked by |
|-------|-------|------|------------|
| #5 | MaskGIT cosine + Fast-dLLM confidence-threshold remasking baselines behind one scheduler API | S | — |
| #4 | Speed/quality benchmark harness with seeds | S | — |
| #7 | Pre-registered comparison report | XS | #5, #4 |

### Epic: Phase 2 — Synthetic text modality leg, educational reproduction (#2)

| Issue | Title | Size | Blocked by |
|-------|-------|------|------------|
| #6 | Synthetic text task suite + data generators | S | — |
| #8 | Text-leg hybrid masks + adaptation smoke test vs causal AR baseline | S | #6 |
| #9 | Per-modality adaptation quality report | XS | #8 |

### Epic: Phase 3 (novel research) — LCRS, Learned Confidence Remasking Scheduler (#3)

| Issue | Title | Size | Blocked by |
|-------|-------|------|------------|
| #10 | Oracle unmask-order generator via greedy per-step search | S | #1 |
| #11 | LCRS policy module with distillation training + tests | S | #10 |
| #13 | Step-budget prediction head + per-sample adaptive stopping | S | #11 |
| #14 | Evaluation: Pareto vs baselines, difficulty-adaptivity correlation, input ablations | S | #11, #13, #1 |
| #15 | Cross-modal transfer experiment (text↔image) | S | #2, #11 |
| #17 | Results report against success criteria with honest negatives | S | #14, #15 |

### Standalone (no epic)

| Issue | Title | Size | Blocked by |
|-------|-------|------|------------|
| #12 | pyproject.toml packaging + pytest config cleanup | XS | — |
| #16 | Example script: end-to-end image-token generation walkthrough | XS | — |
| #18 | docs/API_REFERENCE.md for the public API | S | — |

## Suggested contribution paths

- **First-time contributor (docs/packaging):** #12 → #16 or #18. Low risk,
  immediate value, and a tour of the codebase.
- **ML engineer (Phase 1):** #5 and #4 in either order → #7. You will build
  the measuring stick every later claim is checked against.
- **ML engineer (Phase 2):** #6 → #8 → #9. Touches masks, data, and an
  honest parity check vs an AR baseline.
- **Researcher (Phase 3, novel):** start once Phase 1 lands: #10 → #11 →
  #13, then #14/#15, and synthesize in #17. Read `docs/ROADMAP.md` Phase 3
  and `docs/METHODS.md` first — the risks and pre-registered criteria are
  part of the task.

## Rules

1. **Tests pass.** `pytest` green before review; new behavior gets new tests;
   existing tests are never weakened to fit a change.
2. **Honest negatives.** Null and negative results that meet acceptance
   criteria are successes. Metrics and seeds fixed in an issue are
   pre-registered and cannot change after results are seen.
3. **No overclaiming.** Synthetic-data educational repo; the ~20× speedup is
   Emu3.5's result (arXiv:2510.26583), not ours. Phase 2 is reproduction,
   not novelty.
4. **Respect boundaries.** Out-of-scope work needs its own issue first.

## Owner action: real GitHub Project

This file is a mirror. The repo owner should create a GitHub Project (Board
view) linked to this repository, add the original issue range plus the linked child cards below, and add fields for
**Size** (XS/S/L), **Phase** (0/1/2/3/standalone), and **Blocked by**. Keep
this file in sync when issues are added or closed.

## Live GitHub Project

The public [contribution board](https://github.com/users/hwilner/projects/9) mirrors the active issue backlog. **Implementation, tests, documentation, evaluation, and scoped extensions to the original project are welcome from all contributors.** Choose a `Ready` card only after reading its boundary and acceptance criteria; the GitHub issue body remains the source of truth for scope and dependencies.


## Refined contributor child cards

The original broad cards below remain **open parent/integration tasks**. They were not deleted, replaced, closed, or assigned. Each small linked child is an unassigned, focused contribution unit; contributors should claim one child rather than duplicate parent work.

| Parent task | Linked child issue | Focus |
| --- | --- | --- |
| #4 | [#19](https://github.com/hwilner/discrete-diffusion-adaptation/issues/19) | [XS] Benchmark configuration and deterministic runner for speed-quality harness |
| #4 | [#20](https://github.com/hwilner/discrete-diffusion-adaptation/issues/20) | [XS] Exact-match and distributional-distance metrics with tests |
| #4 | [#21](https://github.com/hwilner/discrete-diffusion-adaptation/issues/21) | [S] Pareto-curve output and CLI integration for benchmark harness |
| #5 | [#22](https://github.com/hwilner/discrete-diffusion-adaptation/issues/22) | [XS] Common remasking scheduler interface and compatibility tests |
| #5 | [#23](https://github.com/hwilner/discrete-diffusion-adaptation/issues/23) | [XS] MaskGIT cosine remasking baseline with tests |
| #5 | [#24](https://github.com/hwilner/discrete-diffusion-adaptation/issues/24) | [XS] Fast-dLLM confidence-threshold remasking baseline with tests |
| #6 | [#25](https://github.com/hwilner/discrete-diffusion-adaptation/issues/25) | [XS] Seedable synthetic-text task generator interface and deterministic fixture |
| #6 | [#26](https://github.com/hwilner/discrete-diffusion-adaptation/issues/26) | [XS] Arithmetic-carry synthetic-text generator with invariants |
| #6 | [#27](https://github.com/hwilner/discrete-diffusion-adaptation/issues/27) | [XS] Bracket-matching synthetic-text generator with invariants |
| #8 | [#28](https://github.com/hwilner/discrete-diffusion-adaptation/issues/28) | [S] Text-leg hybrid masks with causal-equivalence tests |
| #8 | [#29](https://github.com/hwilner/discrete-diffusion-adaptation/issues/29) | [XS] Reproducible DiDA-versus-causal-AR smoke-test runner and result capture |
| #10 | [#30](https://github.com/hwilner/discrete-diffusion-adaptation/issues/30) | [XS] Deterministic greedy oracle unmask-order search engine |
| #10 | [#31](https://github.com/hwilner/discrete-diffusion-adaptation/issues/31) | [XS] Oracle trajectory schema, serialization, and tests |
| #10 | [#32](https://github.com/hwilner/discrete-diffusion-adaptation/issues/32) | [XS] Oracle-versus-baseline structured-task sanity-check script |
| #11 | [#33](https://github.com/hwilner/discrete-diffusion-adaptation/issues/33) | [XS] Scheduler-compatible LCRS policy module with shape and determinism tests |
| #11 | [#34](https://github.com/hwilner/discrete-diffusion-adaptation/issues/34) | [S] Oracle-trajectory dataset loader and seedable LCRS distillation loop |
| #11 | [#35](https://github.com/hwilner/discrete-diffusion-adaptation/issues/35) | [XS] LCRS tiny-batch overfit smoke test |
| #14 | [#36](https://github.com/hwilner/discrete-diffusion-adaptation/issues/36) | [XS] Pre-registered Pareto comparison for LCRS and baseline schedulers |
| #14 | [#37](https://github.com/hwilner/discrete-diffusion-adaptation/issues/37) | [XS] Difficulty-versus-allocated-steps correlation with confidence interval |
| #14 | [#38](https://github.com/hwilner/discrete-diffusion-adaptation/issues/38) | [XS] Scheduler-input ablation runs and result table |
| #15 | [#39](https://github.com/hwilner/discrete-diffusion-adaptation/issues/39) | [XS] Text-to-image zero-shot transfer run and diagnostics |
| #15 | [#40](https://github.com/hwilner/discrete-diffusion-adaptation/issues/40) | [XS] Image-to-text zero-shot transfer run and diagnostics |
| #15 | [#41](https://github.com/hwilner/discrete-diffusion-adaptation/issues/41) | [XS] Shared cross-modal retention computation and summary |
