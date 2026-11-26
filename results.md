# Results

**Generated 2026-11-26 by executing this repository's own test suite against
the current `main`.**

## Verification run

| Metric | Value |
|---|---|
| Tests collected | **133** |
| Tests passed | **133** |
| Tests failed | 0 |
| Errors | 0 |
| Skipped | 0 |
| Wall clock | 29.7 s |
| Environment | Python 3.11.2, torch 2.15.0+cpu, CPU-only, no GPU |

Reproduce with:

```bash
python -m pytest -q
```

The previous reproduce command set `PYTHONPATH=$PWD:$PWD/src`, but this
repository has no `src/` directory: the package lives at the repository
root. `python -m pytest -q` puts the working directory on `sys.path`, so it
runs the suite as it is actually laid out.

## What this establishes

- The code imports cleanly and **133 tests pass with 0 failures**.
- The suite now contains checks that fail when the repository's own claims
  drift: `tests/test_repo_consistency.py` pins the package version against
  `pyproject.toml` and pins the documented test count against a real
  collection run, so the numbers in this table cannot go stale unnoticed.
- The three defects listed below are covered by tests that were confirmed to
  fail against the pre-fix code before they were confirmed to pass against
  the fixed code.

## What this does NOT establish

- **No scientific claim is verified by a passing unit test.** These tests check
  that functions behave as specified -- not that a biological or computational
  hypothesis holds.
- **No benchmark, metric, or comparative result was reproduced.** The unit tests
  are not experiments.
- **Coverage was not measured.** A passing suite can still leave large areas
  untested.
- **Statistical significance was not assessed.** No experiment was run, so there
  is nothing to test.
- **The DiDA mechanism itself was not validated.** The suite exercises the
  code paths on synthetic tensors; it says nothing about image quality, about
  the Emu3.5 speedup, or about any result claimed in the README or `docs/`.

## Negative and unverified results

Recorded explicitly, so they are not mistaken for successes:

- **No empirical result was reproduced for this repository.** Any finding
  described in the README or `docs/` was not re-derived here.
- **The test suite passing is not evidence the research is correct.** It is
  evidence the code is internally consistent.
- **The guidance fix is verified structurally, not empirically.** The tests
  show that the conditioning prefix now reaches the denoiser and that the
  guidance term is no longer identically zero. They do not show that guided
  sampling produces better images, because no model is trained here.

## Defects found and fixed on 2026-11-26

Each defect below was confirmed by running the code before the fix, and each
new check was confirmed to **fail** against the pre-fix code before it was
confirmed to pass against the fixed code.

| Defect | Evidence before the fix | Fix |
|---|---|---|
| Classifier-free guidance was a no-op: the "conditional" and "unconditional" logits came from identical inputs, so `cond - uncond` was identically zero for every `guidance_scale`, and the `condition` argument was never passed to the model | `max abs(cond - uncond) = 0.0`; the guided logits were bit-identical to the conditional logits at scales 0.0, 1.0, 7.5 and 50.0 | The conditioning tokens form a fixed prefix; the conditional pass lets the generated tokens see it and the unconditional pass hides it, so the guidance term is the difference the prefix actually makes (`dida/sampler.py`, `dida/core.py`) |
| `CTMCUniformizationSampler.sample_num_jumps` ignored its `generator` argument, so a seeded run was not reproducible | Two identically seeded generators returned different counts | The counts are drawn with `torch.poisson`, which honours the generator (`dida/ctmc.py`) |
| `CTMCUniformizationSampler.sample_terminal_state` crashed on a zero-length interval, and silently accepted a negative one | `ValueError: math domain error` from `log(0)` | A zero-length interval returns the start states (the chain cannot move); a negative interval is rejected (`dida/ctmc.py`) |
| The repository's statements about itself were stale: `pyproject.toml` declared version `0.1.0` while the package reported `0.2.0`, and the documentation stated 18, 30 and 79 tests where the suite collects 133 | The version and test-count checks in `tests/test_repo_consistency.py` failed | The version is mirrored and pinned by a test; the documented test count is pinned against a real collection run (`pyproject.toml`, `README.md`, `CONTRIBUTING.md`, `docs/TESTING.md`) |

## Reproducing

```bash
python -m pytest -q
```
