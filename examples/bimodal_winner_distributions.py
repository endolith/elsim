"""
Winner distributions with a bimodal electorate (two partisan clusters).

Suggested in https://github.com/endolith/elsim/issues/20 — two groups of voters
(5k + 5k) centered at ±0.5 on one ideological dimension, candidates drawn in
two matching clusters, plus partisan primaries with optional lower primary /
runoff turnout than the general electorate.

Similar layout to distributions_by_method.py.
"""
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from seaborn import histplot, kdeplot

from elsim.elections import bimodal_electorate, normed_dist_utilities
from elsim.methods import (approval, black, closed_partisan_primary_runoff,
                           fptp, irv, open_partisan_primary, runoff, score, star,
                           three_two_one, top_two_runoff_reduced_turnout)
from elsim.strategies import (approval_optimal, honest_321_ratings,
                              honest_normed_scores, honest_rankings,
                              vote_for_k)

n_elections = 50_000
n_voters_each = 5_000
n_cands_each = 4
dims = 1
disp = 0.5
separation = 0.5

primary_each = 3_800
runoff_n = 8_500

tb = 'random'

batch_size = 100
n_batches = n_elections // batch_size
assert n_batches * batch_size == n_elections


def human_format(num):
    for unit in ['', 'k', 'M', 'B', 'T']:
        if abs(num) < 1000:
            return f"{num:.3g}{unit}"
        num /= 1000.0


def simulate_batch(seed):
    rng = np.random.default_rng(seed)
    winners = defaultdict(list)
    for _ in range(batch_size):
        v, c = bimodal_electorate(
            n_voters_each, n_cands_each, dims=dims, disp=disp,
            separation=separation, random_state=rng)

        utilities = normed_dist_utilities(v, c)
        rankings = honest_rankings(utilities)

        left_v = np.arange(0, n_voters_each)
        right_v = np.arange(n_voters_each, 2 * n_voters_each)
        all_v = np.arange(2 * n_voters_each)

        primary_left = rng.choice(left_v, size=primary_each, replace=False)
        primary_right = rng.choice(right_v, size=primary_each, replace=False)
        runoff_voters = rng.choice(all_v, size=runoff_n, replace=False)

        winner = rng.integers(0, c.shape[0])
        winners['Random winner (candidate distribution)'].append(c[winner][0])

        winner = fptp(rankings, tiebreaker=tb)
        winners['First past the post'].append(c[winner][0])

        winner = runoff(rankings, tiebreaker=tb)
        winners['Top-two runoff (contingent vote, full electorate)'].append(
            c[winner][0])

        winner = top_two_runoff_reduced_turnout(
            rankings, all_v, runoff_voters, tiebreaker=tb)
        if winner is None:
            continue
        winners['Top-two runoff (pairwise subset of voters)'].append(
            c[winner][0])

        winner = irv(rankings, tiebreaker=tb)
        winners['Instant-runoff (Hare)'].append(c[winner][0])

        ballots = honest_normed_scores(utilities, max_score=5, min_score=0)
        winner = score(ballots, tiebreaker=tb)
        winners['Score voting (0–5)'].append(c[winner][0])

        winner = star(ballots, tiebreaker=tb)
        winners['STAR voting (0–5)'].append(c[winner][0])

        winner = black(rankings, tiebreaker=tb)
        winners['Black (Condorcet + RCV)'].append(c[winner][0])

        winner = open_partisan_primary(
            rankings, n_cands_each, primary_left, primary_right, all_v,
            tiebreaker=tb)
        winners['Open partisan primary → general (subset primary electorate)'].append(
            c[winner][0])

        winner = closed_partisan_primary_runoff(
            rankings, n_cands_each, primary_left, primary_right,
            runoff_voters, tiebreaker=tb)
        winners['Closed partisan primary → runoff (subset runoff electorate)'].append(
            c[winner][0])

        ballots321 = honest_321_ratings(utilities)
        winner = three_two_one(ballots321, tiebreaker=tb)
        if winner is None:
            continue
        winners['3-2-1 voting'].append(c[winner][0])

        winner = approval(approval_optimal(utilities), tiebreaker=tb)
        winners['Approval (threshold at mean utility)'].append(c[winner][0])

        winner = approval(vote_for_k(utilities, 'half'), tiebreaker=tb)
        winners[f'Approval (vote-for-{n_cands_each // 2} per party pool)'].append(
            c[winner][0])

        winner = np.argmin(np.abs(c[:, 0]))
        winners['Nearest cluster center (ideal spatial winner)'].append(
            c[winner][0])

    return winners


jobs = [delayed(simulate_batch)(k) for k in range(n_batches)]
print(f'{len(jobs)} tasks total:')
results = Parallel(n_jobs=-3, verbose=5)(jobs)

winners = {k: [v for d in results for v in d[k]] for k in results[0]}

title = (f'{human_format(n_elections)} 1D elections, '
         f'{human_format(2 * n_voters_each)} bimodal voters '
         f'({human_format(n_voters_each)} per mode ±{separation}), '
         f'{n_cands_each} candidates per cluster, {disp} dispersion')

v, _ = bimodal_electorate(n_voters_each, 50, dims=dims, disp=disp,
                          separation=separation)

fig, ax = plt.subplots(nrows=len(winners), num=title, sharex=True,
                       constrained_layout=True, figsize=(7.5, 14))
fig.suptitle(title)
for n, method in enumerate(winners.keys()):
    histplot(winners[method], ax=ax[n], label='Winners', stat='density')
    ax[n].set_title(method, loc='left')
    ax[n].set_yticklabels([])
    ax[n].set_ylabel('')

    tmp = ax[n].twinx()
    kdeplot(v[:, 0], ax=tmp, ls=':', label='Voters')
    ax[n].plot([], [], ls=':', label='Voters')
    tmp.set_yticklabels([])
    tmp.set_ylabel('')

ax[0].set_xlim(-2.5, 2.5)
ax[0].legend()
