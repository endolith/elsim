# AGENTS.md

## Cursor Cloud specific instructions

This is a pure Python library (no servers, databases, or services). All development tasks run locally via the Python interpreter.

### Key commands

- **Install (dev):** `pip install -e ".[test,examples]"` — installs the package in editable mode with test and example dependencies.
- **Lint (blocking):** `ruff check --select=E9,F63,F7,F82 .` — the subset enforced by pre-commit (must pass).
- **Lint (full):** `ruff check .` — full ruleset from `ruff.toml`; informational only (exit-zero in pre-commit).
- **Test:** `pytest` — runs all unit tests + doctests with coverage (options configured in `pyproject.toml`).
- **Examples:** `python examples/<script>.py` — run any example script (requires `tabulate`, `joblib` from `examples` extra).

### Caveats

- Numba (`[fast]` extra) is not installed in the Cloud VM. The library works without it — you will see a `UserWarning: Numba not installed, Condorcet code will run slower` warning at import time. This is expected and harmless.
- `~/.local/bin` must be on `PATH` for `pytest` and `ruff` to be found (the update script installs to user site-packages).
