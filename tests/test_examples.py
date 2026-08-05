"""
Run the example simulation scripts in full and check their results.

Each script is executed as a whole (in a subprocess) and its computed
``table`` is read directly, rather than parsing printed output.  These tests
are slow, so they are marked "slow" and skipped by default (run with
``pytest -m slow``), and they require the optional ``examples`` dependencies
(joblib and matplotlib).

Each example script that has expected output defines ``reference_table`` (the
values to check its computed ``table`` against) and ``tolerance`` (the
absolute tolerance for the comparison), so the script doubles as a test.  See
issue #91.
"""
import os
import pathlib
import pickle
import subprocess
import sys

import numpy as np
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')

pytest.importorskip('joblib')
pytest.importorskip('matplotlib')
pytest.importorskip('tabulate')

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / 'examples'

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

def _run(name, tmp_path):
    """Run an example script in full (subprocess); return its globals."""
    script = (EXAMPLES / name).read_text()
    out = tmp_path / 'result.pkl'
    script += (f'\nimport pickle\n'
               f'pickle.dump((table, reference_table, tolerance), '
               f'open({str(out)!r}, "wb"))\n')
    variant = tmp_path / name
    variant.write_text(script)
    env = {**os.environ, 'MPLBACKEND': 'Agg', 'PYTHONPATH': str(EXAMPLES)}
    subprocess.run([sys.executable, str(variant)], env=env,
                   capture_output=True, timeout=1200, check=True)
    with open(out, 'rb') as f:
        return pickle.load(f)


def _method_values(rows):
    """Return an example script's ``table``/``reference_table`` rows as
    {method: np.array}.

    ``rows`` is either a list of [method, *values] rows (Merrill-style
    tables) or a dict mapping method to values (Weber-style tables).
    """
    if isinstance(rows, dict):
        return {k: np.asarray(v, dtype=float) for k, v in rows.items()}
    return {row[0]: np.asarray(row[1:], dtype=float) for row in rows}


def _reference_values(reference):
    """Return an example script's ``reference_table`` as {method: np.array}.

    Values may be plain sequences in column order, or dicts keyed by column
    label (sorted by key into column order, e.g. ``merrill_table_1``).
    """
    out = {}
    for method, values in reference.items():
        if isinstance(values, dict):
            out[method] = np.asarray(
                [v for _, v in sorted(values.items())], dtype=float)
        else:
            out[method] = np.asarray(values, dtype=float)
    return out


def _is_nested(table):
    """True if ``table`` is a figure script's dict of {fig: [rows]}."""
    return isinstance(table, dict) and all(
        isinstance(v, list) for v in table.values())


def _assert_close(name, got, expected, tolerance):
    """Check one computed row against its reference and report clear errors."""
    assert got, f'{name}: produced an empty table'
    for method, expected_row in expected.items():
        assert method in got, (
            f'{name}: computed table is missing row {method!r}')
        assert len(got[method]) == len(expected_row), (
            f'{name}: row {method!r} has {len(got[method])} values, '
            f'expected {len(expected_row)}')
        np.testing.assert_allclose(got[method], expected_row,
                                   atol=tolerance)


@pytest.mark.slow
@pytest.mark.parametrize('name', ALL_SCRIPTS)
def test_example(name, tmp_path):
    """Run each example script and check its ``table`` against the
    ``reference_table``/``tolerance`` defined in the script itself."""
    table, reference_table, tolerance = _run(name, tmp_path)
    if _is_nested(table):
        for fig, rows in table.items():
            got = _method_values(rows)
            expected = _reference_values(reference_table[fig])
            _assert_close(f'{name} ({fig})', got, expected, tolerance)
    else:
        got = _method_values(table)
        expected = _reference_values(reference_table)
        _assert_close(name, got, expected, tolerance)
