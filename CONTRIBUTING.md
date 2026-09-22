# Contributing

Thanks for your interest in contributing! This repository is an educational,
from-scratch implementation of Discrete Diffusion Adaptation (DiDA) with an
active research roadmap (see `docs/ROADMAP.md`). Contributions of all sizes
are welcome — from documentation fixes to research features.

## How to find work

1. Open `docs/PROJECT.md` — the in-repo task board listing issues by epic,
   size, and dependency.
2. Pick an issue marked **READY** (no unresolved "Blocked by").
3. Comment on the issue to say you're starting, so effort isn't duplicated.

Every backlog issue is a **task card** with the same structure:

- **Size** — `[XS]` (an hour or two), `[S]` (a day), `[L]` (multi-day epic).
- **Acceptance criteria** — what "done" means; check every box before
  opening a PR.
- **Boundary** — what the task explicitly does *not* cover. Respect it;
  out-of-scope work belongs in a follow-up issue.
- **Dependencies** — issues that must land first.

If you want to work on something not in the backlog, open an issue first so
scope can be agreed before code is written.

## Development setup

```bash
git clone https://github.com/hwilner/discrete-diffusion-adaptation.git
cd discrete-diffusion-adaptation
pip install -r requirements.txt
pip install -e .
pytest            # all 18 tests should pass
```

## Branch and PR conventions

- Branch from `main`: `task/<issue-number>-<short-slug>`,
  e.g. `task/7-maskgit-baseline`.
- One issue per PR. Keep PRs small enough to review in one sitting.
- Fill in the PR template, including the **task-key comment**
  (`Task: #<issue number>`) so the PR links back to its card.
- All tests must pass (`pytest`) before requesting review. Add tests for new
  behavior; do not delete or weaken existing tests to make a change pass.
- Follow the existing code style: plain PyTorch, Google-style docstrings.

## Integrity rules

This repo holds itself to research-integrity standards:

- **Pre-registered metrics.** If an experiment issue specifies seeds,
  metrics, or success criteria up front, they cannot be changed after results
  are seen.
- **Honest negatives.** A null or negative result that satisfies the
  acceptance criteria is a *successful* contribution. Never tune metrics or
  seeds post hoc to manufacture a positive one.
- **No overclaiming.** This is a synthetic-data, educational codebase.
  Documentation and PR descriptions must not imply real-image quality or
  real-world speedups; the ~20× figure belongs to the Emu3.5 paper.
- **Citations from the verified pack only.** When citing papers in docs, use
  the exact titles, authors, years, and arXiv IDs already used in
  `docs/INTRODUCTION.md`.

## License

By contributing, you agree that your contributions are licensed under the
project's MIT License.
