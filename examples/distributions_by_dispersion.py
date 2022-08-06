"""
Show the distribution bias of different voting methods.

"""
import time
import random
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from seaborn import displot, kdeplot, histplot
from joblib import Parallel, delayed
from elsim.methods import (fptp, runoff, irv, approval, borda, coombs,
                           black, utility_winner, star)
from elsim.elections import (normal_electorate, normed_dist_utilities,
                             elections_rng)
from elsim.strategies import (honest_rankings, approval_optimal,
                              honest_normed_scores)

n_iter = 100_000  # Several minutes
n_voters = 10_000
n_cands = 7
cand_dist = 'normal'
u_width = 10
disps_list = np.geomspace(4, 0.25, 9)


# Do more than just one iteration per worker to improve efficiency
batch = 10
n_batches = n_iter // batch
assert n_batches * batch == n_iter

# ranked_methods = {'Plurality': fptp, 'Runoff': runoff, 'Hare': irv,
#                   'Borda': borda, 'Coombs': coombs, 'Black': black}

# rated_methods = {'SU max': utility_winner,
#                  'Approval': lambda utilities, tiebreaker:
#                      approval(approval_optimal(utilities), tiebreaker)}

start_time = time.monotonic()


def human_format(num):
    for unit in ['', 'k', 'M', 'B', 'T']:
        if abs(num) < 1000:
            return f"{num:.3g}{unit}"
        num /= 1000.0


title = f'1-dimensional, {human_format(n_iter)} iterations, '
title += f'{human_format(n_voters)} voters, '
title += f'{human_format(n_cands)} '
title += cand_dist + 'ly-distributed candidates'

# For plotting only
v, c = normal_electorate(n_voters, 1000, dims=1)

if cand_dist == 'uniform':
    # Replace with uniform distribution of candidates
    # c = elections_rng.uniform(-2.5, 2.5, n_cands)
    c = np.random.uniform(-u_width/2, +u_width/2, n_cands)
    # Same shape
    c = np.atleast_2d(c).T


def func():
    winners = defaultdict(list)
    for disp in disps_list:
        for iteration in range(batch):
            v, c = normal_electorate(n_voters, n_cands, dims=1, disp=disp)

            if cand_dist == 'uniform':
                # Replace with uniform distribution of candidates
                # c = elections_rng.uniform(-2.5, 2.5, n_cands)
                c = np.random.uniform(-2.5, 2.5, n_cands)
                # Same shape
                c = np.atleast_2d(c).T

            # # Random winner method.  Votes don't matter at all.
            # winner = random.sample(range(n_cands), 1)[0]
            # winners['RW'].append(c[winner][0])

            # # Random ballot method.  Pick one voter and go with their choice.
            # winning_voter = random.sample(range(n_voters), 1)[0]
            # dists = abs(v[winning_voter] - c)
            # winner = np.argmin(dists)
            # winners['RB'].append(c[winner][0])

            # # FPTP voting method.
            utilities = normed_dist_utilities(v, c)
            rankings = honest_rankings(utilities)
            winner = fptp(rankings, tiebreaker='random')
            winners[disp].append(c[winner][0])

            # # Instant-runoff
            # winner = irv(rankings, tiebreaker='random')
            # winners[disp].append(c[winner][0])

            # STAR voting
            # ballots = honest_normed_scores(utilities)
            # winner = star(ballots, tiebreaker='random')
            # winners[disp].append(c[winner][0])

            # Condorcet RCV
            # winner = black(rankings, tiebreaker='random')
            # winners[disp].append(c[winner][0])

            # # Utility winner (on normalized utilities though, so STAR can do better)
            # winner = utility_winner(utilities, tiebreaker='random')
            # winners['UW'] = c[winner][0]

    return winners


p = Parallel(n_jobs=-3, verbose=5)(delayed(func)() for i in range(n_batches))

winners = {k: [v for d in p for v in d[k]] for k in p[0]}

elapsed_time = time.monotonic() - start_time
print('Elapsed:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)), '\n')

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
