"""Pytest configuration."""

from hypothesis import settings


def pytest_configure(config):
    # ``deadline`` is a per-*example* wall clock cap (Hypothesis), not “the whole
    # test never finishes”. Each ``@given`` run still stops after ``max_examples``
    # etc.; only exceptionally slow or hung single examples are affected.
    #
    # Numba's first @njit compile on a large ballot exceeded ~1.5 s here once;
    # Hypothesis's default 200 ms then flakes. Use a generous ceiling (ms) so JIT
    # warm-up fits but a pathological hang still trips eventually.
    settings.register_profile("elsim", deadline=5000)
    settings.load_profile("elsim")
