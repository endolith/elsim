"""
Smoke tests for the example simulation scripts.

The examples are stochastic Monte Carlo verification scripts (they
intentionally do not fix a global seed), so exact output values are not
asserted here.  Instead each script is run at a small election count with an
injected seed and must complete successfully and emit non-empty tabulated
results.

This guards the joblib batching pattern used by the Monte Carlo examples
(``batch_size`` / ``Parallel`` / result aggregation) against regressions such
as a broken batch-coverage assertion, an import error, or an aggregation bug.
See ``examples/README.md``.

These tests require the optional ``examples`` dependencies (joblib and
matplotlib) and are skipped when they are not installed.
"""
import pathlib
import re
import subprocess
import sys

import pytest

pytest.importorskip('joblib')
pytest.importorskip('matplotlib')

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / 'examples'

# n_elections must remain a multiple of batch_size (100) so that the
# batch-coverage assertion in each script holds.
N_ELECTIONS = 200

SEED_BLOCK = '''
import random as _r
import elsim.elections as _e
_r.seed(42)
np.random.seed(42)
_e.elections_rng = np.random.default_rng(42)
'''

AFFECTED_SCRIPTS = [
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


def _make_variant(name):
    """Return a copy of an example script with a reduced election count and an
    injected seed."""
    source = (EXAMPLES / name).read_text()
    source = re.sub(r'^n_elections\s*=\s*[\d_]+',
                    f'n_elections = {N_ELECTIONS}', source, flags=re.M)
    idx = source.index('import numpy as np')
    line_end = source.index('\n', idx) + 1
    return source[:line_end] + SEED_BLOCK + source[line_end:]


@pytest.mark.parametrize('name', AFFECTED_SCRIPTS)
def test_example_runs(name, tmp_path):
    variant = tmp_path / name
    variant.write_text(_make_variant(name))

    env = {'MPLBACKEND': 'Agg',
           'PYTHONPATH': str(EXAMPLES),
           'PATH': '/usr/bin:/bin'}
    result = subprocess.run([sys.executable, str(variant)],
                            capture_output=True, text=True, env=env,
                            timeout=300)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip(), f'{name} produced no output'
