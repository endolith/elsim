"""Monte Carlo check: extremal vote-for-or-against-k matches Weber's SUE."""
import importlib.util
from pathlib import Path

import numpy as np

from elsim.elections import random_utilities
from elsim.methods import combined_approval, utility_winner
from elsim.strategies import vote_for_or_against_k


def _load_weber_exprs():
    path = Path(__file__).resolve().parents[1] / 'examples' / 'weber_1977_expressions.py'
    spec = importlib.util.spec_from_file_location('weber_1977_expressions', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mc_sue(m, k, *, n_voters, n_elections, seed):
    rng = np.random.default_rng(seed)
    uw_sum = meth_sum = 0.0
    for _ in range(n_elections):
        u = random_utilities(n_voters, m, random_state=rng)
        b = vote_for_or_against_k(u, k, rng=rng)
        w = combined_approval(b, 'random')
        row = u.sum(axis=0)
        uw_sum += row[utility_winner(u)]
        meth_sum += row[w]
    avg = n_voters * n_elections / 2
    return (meth_sum - avg) / (uw_sum - avg)


def test_mc_vote_for_or_against_k_near_weber_theory():
    weber = _load_weber_exprs()
    n_voters = 2500
    n_elections = 3000
    tol = 0.028
    cases = ((3, 1), (4, 2), (6, 3), (10, 4))
    for m, k in cases:
        mc = _mc_sue(m, k, n_voters=n_voters, n_elections=n_elections, seed=17 + m + k)
        th = weber.eff_vote_for_or_against_k(m, k)
        assert abs(mc - th) < tol, (m, k, mc, th)
