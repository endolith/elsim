"""
Show how the winner distribution and bias of a voting method change with
varying dispersion of candidates.
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
n_cands = 7
cand_dist = 'normal'
u_width = 10
disps_list = np.geomspace(4, 0.25, 9)

# Do more than just one election per worker to improve efficiency
batch = 10
n_batches = n_elections // batch
assert n_batches * batch == n_elections


def human_format(num):
    for unit in ['', 'k', 'M', 'B', 'T']:
        if abs(num) < 1000:
            return f"{num:.3g}{unit}"
        num /= 1000.0


method = 'FPTP'
title = f'{method}, {human_format(n_elections)} 1D elections, '
title += f'{human_format(n_voters)} voters, '
title += f'{human_format(n_cands)} '
title += cand_dist + 'ly-distributed candidates'

# For plotting only
v, c = normal_electorate(n_voters, 1000, dims=1)


def simulate_batch():
    winners = defaultdict(list)
    for disp in disps_list:
        for iteration in range(batch):
            v, c = normal_electorate(n_voters, n_cands, dims=1, disp=disp)

            if cand_dist == 'uniform':
                # Replace with uniform distribution of candidates of same shape
                c = np.random.uniform(-u_width/2, +u_width/2, n_cands)
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

            winners[disp].append(c[winner][0])

    return winners


p = Parallel(n_jobs=-3, verbose=5)(delayed(simulate_batch)()
                                   for i in range(n_batches))
winners = {k: [v for d in p for v in d[k]] for k in p[0]}

fig, ax = plt.subplots(nrows=len(winners), num=title, sharex=True,
                       constrained_layout=True, figsize=(7.5, 9.5))
fig.suptitle(title)
for n, disp in enumerate(winners):
    histplot(winners[disp], ax=ax[n], label=f'{disp:.1f} dispersion',
             stat='density')
    ax[n].set_yticklabels([])  # Don't care about numbers
    ax[n].set_ylabel("")  # No "density"

    tmp = ax[n].twinx()
    kdeplot(v[:, 0], ax=tmp, ls=':', label='Voters')  # Label doesn't work
    ax[n].plot([], [], ls=':', label='Voters')  # Dummy label hack
    tmp.set_yticklabels([])  # Don't care about numbers
    tmp.set_ylabel("")  # No "density"

    ax[n].legend()
ax[0].set_xlim(-2.5, 2.5)
