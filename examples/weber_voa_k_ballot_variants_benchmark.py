"""
Benchmark: Monte Carlo SUE vs Weber's vote-for-or-against-k closed form.

Weber (1978), Cowles DP 498, gives ``eff_vote_for_or_against_k(m, k)`` under
impartial culture in the infinite-voter limit.  This script compares that
formula to several **finite-voter** combined-approval ballot rules that have
been suggested as readings of "vote for S" / "vote against S".

Run from the repository root::

    PYTHONPATH=. python examples/weber_voa_k_ballot_variants_benchmark.py

Or from ``examples/``::

    PYTHONPATH=.. python weber_voa_k_ballot_variants_benchmark.py

Adjust ``--n-voters`` and ``--n-elections`` for smoother estimates (slower).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from elsim.elections import random_utilities
from elsim.methods import combined_approval, utility_winner
from elsim.strategies import vote_for_k, vote_for_or_against_k
from weber_1977_expressions import (
    best_vote_for_or_against_k,
    eff_best_vote_for_or_against_k,
    eff_vote_for_or_against_k,
)


def _jittered(utilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    u = np.asarray(utilities, dtype=np.float64)
    return u + rng.random(u.shape) * (np.finfo(np.float64).eps * 64)


def ballot_simultaneous_top_plus_bottom_minus(
    utilities: np.ndarray, k: int, rng: np.random.Generator | int | None,
) -> np.ndarray:
    """+1 on utility-top-k and -1 on utility-bottom-k on the same ballot."""
    rng = np.random.default_rng(rng)
    n_voters, _ = utilities.shape
    uj = _jittered(utilities, rng)
    top_k = np.argpartition(uj, -k, axis=1)[:, -k:]
    bot_k = np.argpartition(uj, k - 1, axis=1)[:, :k]
    ballots = np.zeros((n_voters, utilities.shape[1]), dtype=np.int8)
    rows = np.arange(n_voters)[:, np.newaxis]
    ballots[rows, top_k] = 1
    ballots[rows, bot_k] = -1
    return ballots


def ballot_coin_plus_top_or_minus_same_top(
    utilities: np.ndarray, k: int, rng: np.random.Generator | int | None,
) -> np.ndarray:
    """Fair coin: +1 on top-k or -1 on the same top-k (disapprove favorites)."""
    rng = np.random.default_rng(rng)
    n_voters, _ = utilities.shape
    uj = _jittered(utilities, rng)
    top_k = np.argpartition(uj, -k, axis=1)[:, -k:]
    signs = (1 - 2 * rng.integers(2, size=n_voters, dtype=np.int8))[:, np.newaxis]
    ballots = np.zeros((n_voters, utilities.shape[1]), dtype=np.int8)
    rows = np.arange(n_voters)[:, np.newaxis]
    ballots[rows, top_k] = signs
    return ballots


def ballot_uniform_subset_coin(
    utilities: np.ndarray, k: int, rng: np.random.Generator | int | None,
) -> np.ndarray:
    """Uniform random k-subset; fair +1 or -1 on that subset (utilities unused)."""
    rng = np.random.default_rng(rng)
    n_voters, n_cands = utilities.shape
    keys = rng.random((n_voters, n_cands))
    subset = np.argpartition(keys, -k, axis=1)[:, -k:]
    signs = (1 - 2 * rng.integers(2, size=n_voters, dtype=np.int8))[:, np.newaxis]
    ballots = np.zeros((n_voters, n_cands), dtype=np.int8)
    rows = np.arange(n_voters)[:, np.newaxis]
    ballots[rows, subset] = signs
    return ballots


def ballot_vote_for_k_plus_only(
    utilities: np.ndarray, k: int, rng: np.random.Generator | int | None,
) -> np.ndarray:
    """Baseline: approval vote-for-k (+1 on top-k only), as int8 for combined_approval."""
    a = vote_for_k(utilities, k)
    return a.astype(np.int8)


def mc_social_utility_efficiency(
    ballot_fn,
    n_cands: int,
    k: int,
    *,
    n_voters: int,
    n_elections: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    uw_sum = 0.0
    meth_sum = 0.0
    for _ in range(n_elections):
        u = random_utilities(n_voters, n_cands, random_state=rng)
        b = ballot_fn(u, k, rng)
        winner = combined_approval(b, tiebreaker='random')
        row = u.sum(axis=0)
        uw_sum += row[utility_winner(u)]
        meth_sum += row[winner]
    average = n_voters * n_elections / 2
    return (meth_sum - average) / (uw_sum - average)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n-voters', type=int, default=5000)
    p.add_argument('--n-elections', type=int, default=3500)
    p.add_argument('--seed', type=int, default=11)
    args = p.parse_args()

    strategies = (
        ('repo coin +top OR -bottom', vote_for_or_against_k),
        ('simult +top & -bottom', ballot_simultaneous_top_plus_bottom_minus),
        ('coin +top OR -same top', ballot_coin_plus_top_or_minus_same_top),
        ('uniform S, ignore u', ballot_uniform_subset_coin),
        ('vote-for-k (+1 top)', ballot_vote_for_k_plus_only),
    )

    cases = ((3, 1), (4, 1), (4, 2), (5, 2), (6, 3), (10, 4))

    print('Monte Carlo SUE minus Weber eff_vote_for_or_against_k (percentage points).\n'
          f'n_voters={args.n_voters}, n_elections={args.n_elections}\n')

    header = ['m', 'k', 'Weber %'] + [name for name, _ in strategies]
    rows = []
    for m, k in cases:
        theory = 100 * eff_vote_for_or_against_k(m, k)
        row = [m, k, f'{theory:.2f}']
        for name, fn in strategies:
            seed = args.seed + m * 31 + k + sum(map(ord, name[:6]))
            mc = 100 * mc_social_utility_efficiency(
                fn, m, k,
                n_voters=args.n_voters,
                n_elections=args.n_elections,
                seed=seed,
            )
            row.append(f'{mc - theory:+.2f}')
        rows.append(row)

    widths = [max(len(str(row[i])) for row in [header] + rows) for i in range(len(header))]
    fmt = '  '.join('{{:{}}}'.format(w) for w in widths)

    print(fmt.format(*[str(h) for h in header]))
    print('  '.join('-' * w for w in widths))
    for row in rows:
        print(fmt.format(*[str(x) for x in row]))

    print('\nSame comparison at Weber best k per m (theory vs MC, percent SUE):\n')
    for m in (4, 6, 10):
        kb = best_vote_for_or_against_k(m)
        th = 100 * eff_best_vote_for_or_against_k(m)
        sub = []
        for name, fn in strategies[:2]:
            seed = args.seed + m * 97 + sum(map(ord, name[:6]))
            mc = 100 * mc_social_utility_efficiency(
                fn, m, kb,
                n_voters=args.n_voters,
                n_elections=args.n_elections,
                seed=seed,
            )
            sub.append(f'{name:26s} MC={mc:6.2f}%  delta={mc - th:+6.2f} pp')
        print(f'm={m}  best k={kb}  Weber best={th:.2f}%')
        for line in sub:
            print('  ', line)


if __name__ == '__main__':
    main()
