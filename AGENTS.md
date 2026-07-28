# AGENTS.md

## Overview

`elsim` is a pure Python library for simulating elections: generate voter-candidate utilities, convert them to ballots with strategic rules, and count them with different voting methods. It is used to reproduce published voting-theory results (Merrill 1984, Weber 1977, etc.). No servers, databases, or services required. Runtime deps are NumPy and SciPy; Numba is an optional speedup.

## Quick commands

```sh
# Install dev dependencies (editable)
pip install -e ".[test,fast]"

# Run full test suite (includes doctests; see addopts in pyproject.toml)
pytest

# Run a single test file
pytest tests/test_irv.py

# Lint (blocking — same gate as CI)
ruff check . --select=E9,F63,F7,F82

# Lint (full ruleset from ruff.toml; informational)
ruff check .

# Coverage report
pytest --cov=./ --cov-report html

# Run an example script (needs `pip install -e ".[examples]"`)
python examples/<script>.py
```

Optional: `pre-commit install` sets up local hooks mirroring the CI lint gates (config in `.pre-commit-config.yaml`). CI is the source of truth.

## Architecture

Three-step pipeline: `elections` → `strategies` → `methods`.

- `elsim/elections.py` — generates voter-candidate utility/ranking matrices (spatial models, impartial culture, etc.)
- `elsim/strategies.py` — converts utilities to ballots (honest rankings, scores, approval strategies)
- `elsim/methods/` — one file per voting method (IRV, Borda, STAR, etc.), each exporting a function named after the method (e.g. `irv()`, `borda()`, `star()`). Winner-only functions return `int | None` (`None` on unresolved ties).

Private internal helpers live in `elsim/methods/_common.py` (Numba JIT wrappers, tally primitives, tiebreak logic). Not part of the public API.

## Boundaries

- **Never** add utility functions, plotting helpers, or analysis helpers to the public interface. `elsim/methods/__init__.py` exports voting-method functions only.
- **Never** import example-script utilities from the library. Example scripts may have their own local utility modules; the library does not depend on them.
- **Never** hand-edit generated artifacts: `docs/_autosummary/`, `htmlcov/`, `examples/results/`, coverage output. The Sphinx site builds from README + docstrings; edit those sources instead.
- **Ask first** before adding dependencies. Runtime deps are NumPy and SciPy only; Numba must stay optional (it doesn't support every platform).

## Key conventions

- **Docstrings:** numpydoc format. Doctests run as part of the test suite (`--doctest-modules`) — keep them correct and deterministic (pass `random_state` in examples).
- **Numpy alias:** `elsim/elections.py` and `elsim/strategies.py` import numpy as `_np` to keep it out of the public namespace. `elsim/methods/` modules use plain `np`; exports are pinned by `__all__` in `__init__.py`, so it doesn't leak. Either way, numpy must not appear in the public interface.
- **Numba:** Soft dependency. Performance paths use `@njit(cache=True, nogil=True)`. If Numba is absent, a no-op `njit` decorator is used. Check `elsim.methods._common.numba_enabled` for the flag.
- **Random state:** All public functions accept `random_state=None|int|Generator` and use `np.random.default_rng()`.
- **Tiebreakers:** Most methods accept `tiebreaker=None` (returns `None` on ties), `'random'`, or `'order'`.
- **Data types:** Candidates are 0-indexed integer IDs. Ballots are numpy arrays (`uint8` where possible). Rows = voters.
- **Line length:** 79 characters (PEP8).
- **Target Python:** 3.8+ (`ruff.toml` target-version).
- **Internal helper prefix:** Private functions in source modules use `_` prefix (e.g. `_tally_at_rank_idx`, `_get_tiebreak`).

## Testing notes

- CI tests across Python 3.8–3.14, with and without Numba.
- When Numba is installed, Hypothesis deadline is relaxed to 5000ms (configured in `tests/conftest.py`).
- Property-based tests use Hypothesis (`@given` with `lists(permutations(...))` for ballot generation).
- `tests/test_methods.py` has parametrized cross-cutting tests (unanimity, degenerate cases, invalid tiebreakers) that run against all methods.
- Coverage omits `tests/` directory (`.coveragerc`).

## Caveats

- When Numba is not installed, importing `elsim` emits `UserWarning: Numba not installed, … code will run slower`. This is expected and harmless.
- The first `@njit` call (especially `prange`-based Condorcet methods) triggers a compile that can take >1 s. Hypothesis deadlines are relaxed in CI (see Testing notes), and fixtures in tests amortize the cost by pre-warming the JIT. Do not add a second Hypothesis test that compiles the same function — reuse the existing fixture instead.

## Code change guidelines

### Documentation

When changing code, update **all** relevant documentation:

- **Code comments and docstrings** — keep them accurate and up-to-date.
- **`README.md`** — update if you change something documented there.
- **`AGENTS.md`** — update this file if the development guidelines change or there is something non-obvious that an agent needs to know in future jobs.

### Testing

- **Always write or update unit tests** when changing code. New functions/methods need tests; bug fixes need regression tests.
- **Every test function must have a docstring** explaining what behavior it verifies and why. Someone who breaks the test must be able to understand what they broke and what the intended behavior is.
- Run `pytest` before pushing. Monitor CI until it passes.

### Commits

- **Make every commit a small, self-contained, working unit that completes one coherent idea—and nothing else** (i.e., both atomic and logical). Unrelated edits belong in separate commits even when each is small (e.g. a workflow trigger change and a pytest marker are two commits). This includes documentation and tests for that idea—keep them in the same commit as the code they describe, not in a later commit for a different feature, so reviewers can read commit-by-commit and `git revert <commit>` undoes one idea cleanly.
- **Write comprehensive commit messages.** The subject line is a concise summary; the body must explain the problem being solved, the chosen approach, and any trade-offs. Provide the *context* that makes the diff understandable—why each change exists and what it achieves. Avoid meta-commentary about the commit itself (e.g., "fixing my commit according to instructions"). Keep process discussion in chat.
- **Use Conventional Commits** (e.g., `feat:`, `fix:`, `docs:`, `test:`, `chore:`) to categorize changes and enable automated changelog generation.
- **Authorship:** When AI writes code, the AI is author and human is coauthor. Use `Co-authored-by:` trailer in commit message.
- **Do not commit unless explicitly asked.**

### Comments

- **Code comments** explain *why*: the intent, non-obvious reasoning, edge cases, and business logic. If a comment is needed to restate what the code does, rewrite the code to be clearer instead. Historical context that explains current behavior is acceptable. Remove meta-commentary about the development process (e.g., "fixing my commit according to instructions" or "now following the directions"). Keep process discussions in chat, not in comments or commit messages.
- Don't delete or omit comments while changing things. Comments are just as important as code.

### PRs and Issues

- All changes must be submitted as PRs so they can be revised independently.
- **Prefer small, reviewable PRs.** Split large efforts into stacked PRs with a clear merge order. Each PR should have one scope; the description should list commits and what each one does so reviewers can read commit-by-commit.
- Check if there are any Issues related to the change you are making, and if so, mention it in the PR and write `Fixes #…` in the relevant commit message, so that the Issue will be auto-closed on merge.
- **PR descriptions** should stand alone for a reviewer who has not read the issue or agent chat. Use short sections: **Background** (what should work), **Problem** (what is wrong), **Visible symptoms** (what users or CI observe), **What this PR changes** (scope and non-goals), **Tests** (what was added or updated). Add **Related work** only when stacked PRs or merge order matter. Split unrelated fixes into separate PRs; cross-link siblings when you do.

## Done checklist

- [ ] `pytest` passes locally
- [ ] Blocking lint passes (`ruff check . --select=E9,F63,F7,F82`)
- [ ] New logic has tests with docstrings
- [ ] Affected docs updated (docstrings, README, AGENTS.md)
- [ ] CI is green (monitor until it passes)
