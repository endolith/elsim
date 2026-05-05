"""Pytest configuration."""

from hypothesis import settings


def pytest_configure(config):
    # Relax Hypothesis's per-example deadline only when Numba is present: first
    # @njit compile can exceed ~1.5 s on a large ballot (Hypothesis default 200 ms
    # then flakes). Without Numba, keep Hypothesis's defaults.
    from elsim.methods import _common

    if not _common.numba_enabled:
        return
    settings.register_profile("elsim", deadline=5000)
    settings.load_profile("elsim")
