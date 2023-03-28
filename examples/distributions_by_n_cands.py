"""
Show how the winner distribution and bias of a voting method change with
number of candidates.
"""
import random
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from seaborn import kdeplot, histplot
from joblib import Parallel, delayed
from elsim.methods import fptp, irv, black, utility_winner, star
from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.strategies import honest_rankings, honest_normed_scores

n_elections = 100_000  # Several minutes
n_voters = 10_000
n_cands_list = [2, 3, 4, 5, 6, 7, 11, 15, 25]
cand_dist = 'normal'


# Do more than just one election per worker to improve efficiency
batch = 10
n_batches = n_elections // batch
assert n_batches * batch == n_elections


def human_format(num):
    for unit in ['', 'k', 'M', 'B', 'T']:
        if abs(num) < 1000:
            return f"{num:.3g}{unit}"
        num /= 1000.0


method = 'STAR'
title = f'{method}, {human_format(n_elections)} elections, '
title += f'{human_format(n_voters)} voters, '

# For plotting only
v, c = normal_electorate(n_voters, 1000, dims=1)

# if cand_dist == 'uniform':
#     # Replace with uniform distribution of candidates
#     # c = elections_rng.uniform(-2.5, 2.5, n_cands)
#     c = np.random.uniform(-2.5, 2.5, n_cands)
#     # Same shape
#     c = np.atleast_2d(c).T

title += cand_dist + ' candidates'


def func():
    winners = defaultdict(list)
    for n_cands in n_cands_list:
        for iteration in range(batch):
            v, c = normal_electorate(n_voters, n_cands, dims=1)

            if cand_dist == 'uniform':
                # Replace with uniform distribution of candidates
                # c = elections_rng.uniform(-2.5, 2.5, n_cands)
                c = np.random.uniform(-2.5, 2.5, n_cands)
                # Same shape
                c = np.atleast_2d(c).T

            if 'Random' not in method:
                utilities = normed_dist_utilities(v, c)

            if method in {'FPTP', 'Hare RCV', 'Condorcet RCV (Black)'}:
                rankings = honest_rankings(utilities)

            if method == 'Random Winner':  # Votes don't matter at all.
                winner = random.sample(range(n_cands), 1)[0]

            # Pick one voter and go with their choice.
            if method == 'Random Ballot':
                winning_voter = random.sample(range(n_voters), 1)[0]
                dists = abs(v[winning_voter] - c)
                winner = np.argmin(dists)

            if method == 'FPTP':
                winner = fptp(rankings, tiebreaker='random')

            if method == 'Hare RCV':
                winner = irv(rankings, tiebreaker='random')

            if method == 'STAR':
                ballots = honest_normed_scores(utilities)
                winner = star(ballots, tiebreaker='random')

            if method == 'Condorcet RCV (Black)':
                winner = black(rankings, tiebreaker='random')

            # (on normalized utilities though, so STAR can do better)
            if method == 'Utility Winner':
                winner = utility_winner(utilities, tiebreaker='random')

            winners[n_cands].append(c[winner][0])

    return winners


p = Parallel(n_jobs=-3, verbose=5)(delayed(func)() for i in range(n_batches))
winners = {k: [v for d in p for v in d[k]] for k in p[0]}

fig, ax = plt.subplots(nrows=len(winners), num=title, sharex=True,
                       constrained_layout=True, figsize=(9, 9))
ax[0].set_title(title)
for n, n_cands in enumerate(winners):
    histplot(winners[n_cands], ax=ax[n], label=f'{n_cands} cands',
             stat='density')
    kdeplot(v[:, 0], ax=ax[n], ls=':', label='Voters')
    ax[n].legend()
ax[0].set_xlim(-3, 3)
