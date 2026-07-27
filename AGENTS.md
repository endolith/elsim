# AGENTS.md

## Quick commands

```sh
# Install dev dependencies (editable)
pip install -e ".[test,fast]"

# Run full test suite (includes doctests)
pytest

# Run a single test file
pytest tests/test_irv.py

# Lint (blocking — same as CI)
ruff check . --select=E9,F63,F7,F82

# Lint (full, informational)
ruff check .

# Coverage report
pytest --cov=./ --cov-report html
```

## Architecture

Three-step pipeline: `elections` → `strategies` → `methods`.

- `elsim/elections.py` — generates voter-candidate utility matrices (spatial models, impartial culture, etc.)
- `elsim/strategies.py` — converts utilities to ballots (honest rankings, scores, approval strategies)
- `elsim/methods/` — one file per voting method (IRV, Borda, STAR, etc.), each exporting a function named after the method (e.g. `irv()`, `borda()`, `star()`)

Private internal helpers live in `elsim/methods/_common.py` (Numba JIT wrappers, tally primitives, tiebreak logic). Not part of the public API. Example scripts may have their own local utility modules, but these are not imported by the library.

Don't add utility functions, plotting helpers, or analysis helpers to the public interface. Winner-only functions return `int | None`.

## Key conventions

- **Docstrings:** numpydoc format. Doctests are run via `pytest --doctest-modules` — keep them correct.
- **Numpy alias:** Source modules import numpy as `_np`; tests use `np`.
- **Numba:** Soft dependency. Performance paths use `@njit(cache=True, nogil=True)`. If Numba is absent, a no-op `njit` decorator is used. Check `elsim.methods._common.numba_enabled` for the flag.
- **Random state:** All public functions accept `random_state=None|int|Generator` and use `np.random.default_rng()`.
- **Tiebreakers:** Most methods accept `tiebreaker=None` (returns `None` on ties), `'random'`, or `'order'`.
- **Data types:** Candidates are 0-indexed integer IDs. Ballots are numpy arrays (`uint8` where possible). Rows = voters.
- **Line length:** 127 characters (ruff).
- **Target Python:** 3.8+ (`ruff.toml` target-version).
- **Internal helper prefix:** Private functions in source modules use `_` prefix (e.g. `_tally_at_rank_idx`, `_get_tiebreak`). Doctests are exempt from this rule.

## Testing notes

- CI tests across Python 3.8–3.14, with and without Numba.
- When Numba is installed, Hypothesis deadline is relaxed to 5000ms (configured in `tests/conftest.py`).
- Property-based tests use Hypothesis (`@given` with `lists(permutations(...))` for ballot generation).
- `tests/test_methods.py` has parametrized cross-cutting tests (unanimity, degenerate cases, invalid tiebreakers) that run against all methods.
- Coverage omits `tests/` directory (`.coveragerc`).

## Code change guidelines

### Commits

- **Make every commit a small, self-contained, working unit that completes one coherent idea — and nothing else.** Unrelated edits belong in separate commits even when each is small. Tests and docs go in the same commit as the code they describe, so reviewers can read commit-by-commit and `git revert <commit>` undoes one idea cleanly.
- **Write comprehensive commit messages.** The subject line is a concise summary; the body must explain the problem being solved, the chosen approach, and any trade-offs. Provide the context that makes the diff understandable — why each change exists and what it achieves. Avoid meta-commentary about the commit itself.
- **Use Conventional Commits** (`feat:`, `fix:`, `docs:`, `test:`, `chore:`) to categorize changes and enable automated changelog generation.
- **Authorship:** When AI assists with code, the human is author and AI is coauthor. Use `Co-authored-by:` trailer in commit message.

### Comments

- **Code comments explain *why*:** the intent, non-obvious reasoning, edge cases, and business logic. If a comment is needed to restate what the code does, rewrite the code to be clearer instead. Historical context that explains current behavior is acceptable.
- Don't delete or omit comments while changing things. Comments are just as important as code.

### Testing

- **Always write or update unit tests** when changing code. New functions need tests; bug fixes need regression tests.
- **Every test function must have a docstring** explaining what behavior it verifies and why. Someone who breaks the test must be able to understand what they broke and what the intended behavior is.
- Run `pytest` before pushing. Monitor CI until it passes.
