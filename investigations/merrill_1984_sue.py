"""Experiments for the Merrill (1984) social-utility-efficiency mismatch.

This is a research script for issue #88, not part of the library or the
tested example suite.

It reports two things:

* ``table4``: the SUE values for the eight spatial conditions in Table 4,
  under several definitions of the "random candidate" baseline.
* ``fig4``: repeated paper-sized samples for Figures 4a/4b, to check whether
  Merrill's even/odd ripple is within Monte Carlo noise.
"""
import argparse
import random

import numpy as np

from elsim.elections import normal_electorate
from elsim.methods import (approval, black, borda, coombs, fptp, irv,
                           runoff)
from elsim.strategies import approval_optimal, honest_rankings

METHOD_NAMES = ('Plurality', 'Runoff', 'Hare', 'Approval', 'Borda',
                'Coombs', 'Black')
METHODS = (fptp, runoff, irv, approval, borda, coombs, black)

# Merrill (1984), Table 4 (201 voters, 5 candidates).  Spatial columns only.
PAPER_TABLE4 = {
    (1.0, 0.5, 2): (64, 86, 88, 96, 98, 96, 97),
    (1.0, 0.5, 4): (75, 92, 92, 96, 98, 96, 98),
    (1.0, 0.0, 2): (74, 88, 91, 97, 98, 96, 98),
    (1.0, 0.0, 4): (93, 98, 98, 98, 99, 98, 99),
    (0.5, 0.5, 2): (-1, 28, 40, 96, 97, 92, 96),
    (0.5, 0.5, 4): (0, 47, 59, 96, 97, 92, 97),
    (0.5, 0.0, 2): (22, 48, 52, 95, 96, 92, 96),
    (0.5, 0.0, 4): (52, 75, 82, 98, 99, 94, 98),
}

# Values traced from Merrill's Figures 4.a/4.b by the example scripts.
PAPER_FIG4 = {
    1.0: {
        'Plurality': {2: 100.0, 3: 76.0, 4: 74.1, 5: 64.0, 7: 57.9},
        'Runoff': {2: 100.0, 3: 90.0, 4: 89.9, 5: 86.0, 7: 84.9},
        'Hare': {2: 100.0, 3: 90.0, 4: 89.9, 5: 88.1, 7: 88.9},
        'Approval': {2: 100.0, 3: 98.0, 4: 96.0, 5: 96.0, 7: 96.0},
        'Borda': {2: 100.0, 3: 98.0, 4: 97.1, 5: 98.0, 7: 97.9},
        'Coombs': {2: 100.0, 3: 95.0, 4: 96.0, 5: 96.0, 7: 96.0},
        'Black': {2: 100.0, 3: 96.0, 4: 96.1, 5: 97.1, 7: 98.0},
    },
    0.5: {
        'Plurality': {2: 100.0, 3: 41.1, 4: 27.0, 5: -1.0, 7: -9.0},
        'Runoff': {2: 100.0, 3: 65.1, 4: 53.0, 5: 28.0, 7: 13.9},
        'Hare': {2: 100.0, 3: 65.1, 4: 55.0, 5: 40.0, 7: 35.2},
        'Approval': {2: 100.0, 3: 94.0, 4: 96.0, 5: 96.0, 7: 95.1},
        'Borda': {2: 100.0, 3: 97.2, 4: 97.2, 5: 97.0, 7: 96.2},
        'Coombs': {2: 100.0, 3: 92.1, 4: 93.1, 5: 92.0, 7: 91.2},
        'Black': {2: 100.0, 3: 98.1, 4: 96.1, 5: 96.0, 7: 96.2},
    },
}

# Stable reference values from the current example scripts (500k elections).
REPO_FIG4 = {
    1.0: {
        'Plurality': {2: 100, 3: 84.7, 4: 77.1, 5: 72.1, 6: 68.1, 7: 64.8},
        'Runoff': {2: 100, 3: 94.2, 4: 92.0, 5: 90.4, 6: 88.9, 7: 87.4},
        'Hare': {2: 100, 3: 94.2, 4: 92.6, 5: 91.7, 6: 91.0, 7: 90.3},
        'Approval': {2: 100, 3: 98.7, 4: 97.3, 5: 96.2, 6: 95.6, 7: 95.2},
        'Borda': {2: 100, 3: 98.7, 4: 98.2, 5: 97.9, 6: 97.7, 7: 97.6},
        'Coombs': {2: 100, 3: 97.1, 4: 96.8, 5: 97.0, 6: 97.2, 7: 97.4},
        'Black': {2: 100, 3: 97.2, 4: 97.1, 5: 97.3, 6: 97.6, 7: 97.8},
    },
    0.5: {
        'Plurality': {2: 100, 3: 50.1, 4: 23.7, 5: 4.3, 6: -11.8, 7: -25.1},
        'Runoff': {2: 100, 3: 70.2, 4: 51.7, 5: 36.9, 6: 24.3, 7: 13.5},
        'Hare': {2: 100, 3: 70.2, 4: 55.9, 5: 46.7, 6: 39.7, 7: 34.6},
        'Approval': {2: 100, 3: 98.6, 4: 96.7, 5: 95.6, 6: 94.9, 7: 94.5},
        'Borda': {2: 100, 3: 97.9, 4: 97.1, 5: 96.6, 6: 96.4, 7: 96.3},
        'Coombs': {2: 100, 3: 94.9, 4: 94.1, 5: 94.0, 6: 94.0, 7: 94.1},
        'Black': {2: 100, 3: 95.5, 4: 95.2, 5: 95.5, 6: 95.8, 7: 96.2},
    },
}


def _normalize(utilities, mode):
    if mode == 'range':
        utilities = utilities - utilities.min(axis=1)[:, np.newaxis]
        utilities = utilities / utilities.max(axis=1)[:, np.newaxis]
    else:
        utilities = utilities - utilities.mean(axis=1)[:, np.newaxis]
        utilities = utilities / utilities.std(axis=1)[:, np.newaxis]
    return utilities


def _election(disp, corr, dims, n_voters, n_cands, rng, normalization):
    voters, cands = normal_electorate(n_voters, n_cands, dims=dims,
                                      corr=corr, disp=disp,
                                      random_state=rng)
    dists = np.sqrt(((voters[:, np.newaxis, :] - cands[np.newaxis, :, :])**2)
                    .sum(axis=-1))
    utilities = _normalize(-dists, normalization)
    rankings = honest_rankings(utilities)
    return utilities, rankings


def _winner_su(method, utilities, rankings):
    if method is approval:
        winner = approval(approval_optimal(utilities), tiebreaker='random')
    else:
        winner = method(rankings, tiebreaker='random')
    return utilities.sum(axis=0)[winner]


def _efficiencies(method_su, max_su, baselines):
    return {name: (method_su - baseline) / (max_su - baseline) * 100
            for name, baseline in baselines.items()}


def table4(n_elections, normalization, seed):
    random.seed(seed)
    rng = np.random.default_rng(seed)
    print(f'Table 4, {n_elections} elections, {normalization} normalization')
    for (disp, corr, dims), paper in PAPER_TABLE4.items():
        sums = np.zeros(len(METHODS))
        max_su = 0.0
        baselines = {name: 0.0 for name in
                     ('mean', 'median', 'trimmed', 'random_ballot')}
        for _ in range(n_elections):
            utilities, rankings = _election(disp, corr, dims, 201, 5, rng,
                                            normalization)
            su = utilities.sum(axis=0)
            max_su += su.max()
            sorted_su = np.sort(su)
            baselines['mean'] += su.mean()
            baselines['median'] += sorted_su[len(sorted_su) // 2]
            baselines['trimmed'] += sorted_su[1:-1].mean()
            voter = rng.integers(201)
            baselines['random_ballot'] += su[np.argmax(utilities[voter])]
            for i, method in enumerate(METHODS):
                sums[i] += _winner_su(method, utilities, rankings)
        sums /= n_elections
        max_su /= n_elections
        baselines = {k: v / n_elections for k, v in baselines.items()}
        print(f'\n  disp={disp} corr={corr} dims={dims}')
        print('    paper      ' + ' '.join(f'{v:6.0f}' for v in paper))
        for name, baseline in baselines.items():
            eff = (sums - baseline) / (max_su - baseline) * 100
            rmse = np.sqrt(np.mean((eff - np.array(paper))**2))
            print(f'    {name:10s} ' + ' '.join(f'{v:6.1f}' for v in eff)
                  + f'   RMSE={rmse:4.1f}')


def table4_perset(n_repeats, normalization, seed):
    """Merrill's design: 200 candidate sets x 5 electorates per set."""
    random.seed(seed)
    rng = np.random.default_rng(seed)
    print(f'Table 4 with per-candidate-set ratios, {n_repeats} repeats, '
          f'{normalization} normalization')
    for (disp, corr, dims), paper in PAPER_TABLE4.items():
        runs = np.zeros((n_repeats, len(METHODS)))
        for repeat in range(n_repeats):
            per_set = np.zeros((200, len(METHODS)))
            for s in range(200):
                _, cands = normal_electorate(0, 5, dims=dims, corr=corr,
                                             disp=disp, random_state=rng)
                sums = np.zeros(len(METHODS))
                max_su = 0.0
                random_su = 0.0
                for _ in range(5):
                    voters, _ = normal_electorate(201, 5, dims=dims,
                                                  corr=corr, disp=disp,
                                                  random_state=rng)
                    delta = voters[:, np.newaxis, :] - cands[np.newaxis, :, :]
                    dists = np.sqrt((delta**2).sum(axis=-1))
                    utilities = _normalize(-dists, normalization)
                    rankings = honest_rankings(utilities)
                    su = utilities.sum(axis=0)
                    max_su += su.max()
                    random_su += su.mean()
                    for i, method in enumerate(METHODS):
                        sums[i] += _winner_su(method, utilities, rankings)
                denom = max_su - random_su
                per_set[s] = (sums - random_su) / denom * 100
            runs[repeat] = per_set.mean(axis=0)
        eff = runs.mean(axis=0)
        rmse = np.sqrt(np.mean((eff - np.array(paper))**2))
        print(f'\n  disp={disp} corr={corr} dims={dims}')
        print('    paper      ' + ' '.join(f'{v:6.0f}' for v in paper))
        print('    per-set    ' + ' '.join(f'{v:6.1f}' for v in eff)
              + f'   RMSE={rmse:4.1f}')
        sd = runs.std(axis=0)
        print('    repeat sd  ' + ' '.join(f'{v:6.1f}' for v in sd))


def fig4(disp, n_elections, repeats, normalization, seed):
    random.seed(seed)
    rng = np.random.default_rng(seed)
    print(f'Figure 4.{"a" if disp == 1.0 else "b"}: disp={disp}, '
          f'{repeats} x {n_elections} elections')
    for n_cands in (2, 3, 4, 5, 6, 7):
        method_su = np.zeros((repeats, len(METHODS)))
        baselines = np.zeros(repeats)
        rand_ballot = np.zeros(repeats)
        max_su = np.zeros(repeats)
        for r in range(repeats):
            sums = np.zeros(len(METHODS))
            for _ in range(n_elections):
                utilities, rankings = _election(disp, 0.5, 2, 201, n_cands,
                                                rng, normalization)
                su = utilities.sum(axis=0)
                max_su[r] += su.max()
                baselines[r] += su.mean()
                voter = rng.integers(201)
                rand_ballot[r] += su[np.argmax(utilities[voter])]
                for i, method in enumerate(METHODS):
                    sums[i] += _winner_su(method, utilities, rankings)
            method_su[r] = sums / n_elections
            max_su[r] /= n_elections
            baselines[r] /= n_elections
            rand_ballot[r] /= n_elections
        eff = (method_su - baselines[:, None]) / (
            max_su - baselines)[:, None] * 100
        eff_rb = (method_su - rand_ballot[:, None]) / (
            max_su - rand_ballot)[:, None] * 100
        print(f'\n  n_cands={n_cands}')
        paper = np.array([PAPER_FIG4[disp][name].get(n_cands, np.nan)
                          for name in METHOD_NAMES], dtype=float)
        mean_su = method_su.mean(axis=0)
        mean_max = max_su.mean()
        known = ~np.isnan(paper)
        q = paper[known] / 100
        m = mean_su[known]
        # m = q*M + (1-q)*r, with M fixed at the sample max
        weights = 1 - q
        r_fit = np.sum(weights * (m - q*mean_max)) / np.sum(weights**2)
        fitted = (mean_su - r_fit) / (mean_max - r_fit) * 100
        rmse = np.sqrt(np.mean((fitted[known] - paper[known])**2))
        print(f'    fitted random baseline r={r_fit:6.1f} '
              f'(mean candidate={baselines.mean():6.1f}, '
              f'max={mean_max:6.1f}) RMSE={rmse:4.1f}')
        for i, name in enumerate(METHOD_NAMES):
            repo = REPO_FIG4[disp][name].get(n_cands)
            print(f'    {name:10s} sample={eff[:, i].mean():6.1f} '
                  f'±{eff[:, i].std():4.1f}  rb={eff_rb[:, i].mean():6.1f}'
                  f'  repo={repo}  paper={paper[i]:6.1f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('table4', 'table4_perset', 'fig4'))
    parser.add_argument('--n-elections', type=int, default=2000)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--normalization', choices=('range', 'z'),
                        default='range')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    if args.mode == 'table4':
        table4(args.n_elections, args.normalization, args.seed)
    elif args.mode == 'table4_perset':
        table4_perset(args.repeats, args.normalization, args.seed)
    else:
        for disp in (1.0, 0.5):
            fig4(disp, args.n_elections, args.repeats,
                 args.normalization, args.seed)


if __name__ == '__main__':
    main()
