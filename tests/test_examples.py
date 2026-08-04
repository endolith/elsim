"""
Run the example simulation scripts in full and check their results.

Each script is executed as a whole (``runpy``) and its computed ``table`` is
read directly, rather than parsing printed output.  These tests are slow, so
they are marked "slow" and skipped by default (run with ``pytest -m slow``),
and they require the optional ``examples`` dependencies (joblib and
matplotlib).  Reference values and tolerances are described at
``REFERENCE_VALUES``.
"""
import contextlib
import io
import os
import pathlib
import runpy
import sys

import numpy as np
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')

pytest.importorskip('joblib')
pytest.importorskip('matplotlib')
pytest.importorskip('tabulate')

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / 'examples'

# Some example scripts import sibling modules from the examples directory
# (e.g. weber_1977_expressions).
sys.path.insert(0, str(EXAMPLES))

ALL_SCRIPTS = [
    'merrill_1984_fig_2c_2d.py',
    'merrill_1984_fig_2c_2d_updated.py',
    'merrill_1984_fig_4a_4b.py',
    'merrill_1984_fig_4a_4b_updated.py',
    'merrill_1984_table_1_fig_1.py',
    'merrill_1984_table_2.py',
    'merrill_1984_table_3_fig_3.py',
    'merrill_1984_table_4.py',
    'weber_1977_effectiveness_table.py',
    'weber_1977_table_4.py',
]

# Reference results for the example scripts.
#
# For merrill_1984_table_1, merrill_1984_table_3, weber_1977_effectiveness
# and weber_1977_table_4 the scripts reproduce the published tables within
# ~2 pp, so the reference values are taken from the papers.
#
# merrill_1984_table_2 and merrill_1984_table_4 do not match the published
# tables (unresolved discrepancies of up to ~5 pp and ~9 pp), so those two are
# checked against the "Typical result" tables in their docstrings instead.
# weber_1977_effectiveness_table's last row (255 candidates) is taken from its
# docstring too, because the paper only gives the m -> infinity limit there.
#
# Values are keyed by method and appear in the same order as the script's
# columns (n_cands, n_voters, or condition).
REFERENCE_VALUES = {
    'merrill_1984_table_1_fig_1.py': {
        'Plurality': (100.0, 79.1, 69.4, 62.1, 52.0, 42.6),
        'Runoff': (100.0, 96.2, 90.1, 83.6, 73.5, 61.3),
        'Hare': (100.0, 96.2, 92.7, 89.1, 84.8, 77.9),
        'Approval': (100.0, 76.0, 69.8, 67.1, 63.7, 61.3),
        'Borda': (100.0, 90.8, 87.3, 86.2, 85.3, 84.3),
        'Coombs': (100.0, 96.3, 93.4, 90.2, 86.1, 81.1),
        'Black': (100.0, 100.0, 100.0, 100.0, 100.0, 100.0),
        'SU max': (100.0, 84.4, 80.2, 77.9, 77.2, 77.8),
        'CW': (100.0, 91.6, 83.4, 75.8, 64.3, 52.5),
    },
    'merrill_1984_table_2.py': {
        'Plurality': (57.5, 65.8, 62.2, 78.4, 21.7, 24.4, 27.2, 41.3),
        'Runoff': (80.1, 87.3, 81.6, 93.6, 35.4, 42.2, 41.5, 61.5),
        'Hare': (79.2, 86.7, 84.0, 95.4, 35.9, 46.8, 41.0, 69.9),
        'Approval': (73.8, 77.8, 76.9, 85.4, 71.5, 76.4, 73.8, 82.7),
        'Borda': (87.1, 89.3, 88.2, 92.3, 83.7, 86.3, 85.2, 89.4),
        'Coombs': (97.8, 97.3, 97.9, 98.2, 93.5, 92.3, 93.8, 94.5),
        'Black': (100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0),
        'SU max': (82.9, 85.8, 85.3, 90.8, 78.1, 81.5, 80.8, 87.1),
        'CW': (99.7, 99.7, 99.7, 99.6, 98.9, 98.6, 98.7, 98.5),
    },
    'merrill_1984_table_3_fig_3.py': {
        'Plurality': (100.0, 83.0, 75.0, 69.2, 62.8, 53.3),
        'Runoff': (100.0, 89.5, 83.8, 80.5, 75.6, 67.6),
        'Hare': (100.0, 89.5, 84.7, 82.4, 80.5, 74.9),
        'Approval': (100.0, 95.4, 91.1, 89.1, 87.8, 87.0),
        'Borda': (100.0, 94.8, 94.1, 94.4, 95.4, 95.9),
        'Coombs': (100.0, 89.7, 86.7, 85.1, 83.1, 82.4),
        'Black': (100.0, 93.1, 91.9, 92.0, 93.1, 94.3),
    },
    'merrill_1984_table_4.py': {
        'Plurality': (72.1, 79.1, 80.4, 92.4, 4.0, 6.3, 25.2, 52.9),
        'Runoff': (90.5, 94.2, 92.0, 97.5, 36.6, 43.6, 53.3, 75.3),
        'Hare': (91.7, 94.7, 94.3, 98.4, 46.4, 57.7, 58.7, 83.6),
        'Approval': (96.2, 97.0, 96.8, 98.5, 95.6, 96.8, 95.8, 98.0),
        'Borda': (97.8, 98.6, 98.3, 99.4, 96.6, 97.7, 97.4, 99.0),
        'Coombs': (97.0, 97.5, 97.7, 98.7, 94.0, 94.3, 95.0, 96.7),
        'Black': (97.3, 97.8, 98.0, 99.0, 95.5, 96.1, 96.5, 98.0),
    },
    'weber_1977_effectiveness_table.py': {
        'Standard': (81.65, 75.00, 69.28, 64.55, 60.61, 49.79, 12.78),
        'Vote-for-half': (81.65, 75.00, 80.00, 79.06, 81.32, 82.99, 86.37),
        'Borda': (81.65, 86.60, 89.44, 91.29, 92.58, 95.35, 99.80),
    },
    'weber_1977_table_4.py': {
        'Standard': (1.2500, 1.8333, 2.3889, 2.9167, 5.5975, 8.2245,
                     10.8328, 13.4328, 16.0190),
        'Borda': (1.2917, 1.8750, 2.4236, 2.9765, 5.6706, 8.3206,
                  10.9472, 13.5588, 16.1597),
        'Approval': (1.2917, 1.8646, 2.4213, 2.9726, 5.6719, 8.3245,
                     10.9531, 13.5662, 16.1684),
    },
}

# Absolute tolerance per script: percentage points for the Merrill and Weber
# effectiveness tables, utility units for weber_1977_table_4.
TOLERANCES = {
    'merrill_1984_table_1_fig_1.py': 3.5,
    'merrill_1984_table_2.py': 3.0,
    'merrill_1984_table_3_fig_3.py': 3.0,
    'merrill_1984_table_4.py': 3.0,
    'weber_1977_effectiveness_table.py': 4.0,
    'weber_1977_table_4.py': 0.2,
}


def _run(name):
    """Run an example script in full; return its module globals."""
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        return runpy.run_path(str(EXAMPLES / name))


def _table_rows(ns):
    """Return an example script's ``table`` as {label: np.array}."""
    table = ns['table']
    if isinstance(table, dict):
        return {k: np.asarray(v, dtype=float) for k, v in table.items()}
    return {row[0]: np.asarray(row[1:], dtype=float) for row in table}


@pytest.mark.slow
@pytest.mark.parametrize('name', ALL_SCRIPTS)
def test_example(name):
    table = _table_rows(_run(name))
    for method, expected in REFERENCE_VALUES.get(name, {}).items():
        got = table[method][:len(expected)]
        np.testing.assert_allclose(got, expected, atol=TOLERANCES[name])
