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
view) linked to this repository, add issues #1–#18, and add fields for
**Size** (XS/S/L), **Phase** (0/1/2/3/standalone), and **Blocked by**. Keep
this file in sync when issues are added or closed.
