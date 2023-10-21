"""
Show the winner distributions and bias of different voting methods with uniform
distribution of voters and candidates.

Similar to Figure 3

The distributions of the winning position with k = 3, 4, 5 candidates and
continuous 1-Euclidean voters (both uniformly distributed) under plurality and
IRV.

from

Kiran Tomlinson, Johan Ugander, Jon Kleinberg (2023) Moderation in instant
runoff voting https://arxiv.org/abs/2303.09734
"""
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from seaborn import histplot

from elsim.elections import normed_dist_utilities
from elsim.methods import (approval, black, borda, coombs, fptp, irv, runoff,
                           star)
from elsim.strategies import (approval_optimal, honest_normed_scores,
                              honest_rankings, vote_for_k)

n_elections = 1_000_000  # Roughly 80 minutes on a 2019 6-core i7-9750H
n_voters = 1_000
n_cands = 5
cand_dist = 'uniform'
u_width = 5
disp = 1
vote_for = n_cands//2

# Simulate more than just one election per worker to improve efficiency
batch_size = 100
n_batches = n_elections // batch_size
assert n_batches * batch_size == n_elections


def ceildiv(a, b):
    return -(a // -b)


def human_format(num):
    for unit in ['', 'k', 'M', 'B', 'T']:
        if abs(num) < 1000:
            return f"{num:.3g}{unit}"
        num /= 1000.0


def simulate_batch():
    winners = defaultdict(list)
    for iteration in range(batch_size):
        # v, c = normal_electorate(n_voters, n_cands, dims=1, disp=disp)

        if cand_dist == 'uniform':
            # Replace with uniform distribution of candidates of same shape
            v = np.random.uniform(-u_width/2, +u_width/2, n_voters)
            v = np.atleast_2d(v).T
            c = np.random.uniform(-u_width/2, +u_width/2, n_cands)
            c = np.atleast_2d(c).T

        # FPTP voting method
        utilities = normed_dist_utilities(v, c)
        rankings = honest_rankings(utilities)
        winner = fptp(rankings, tiebreaker='random')
        winners['First Past The Post / Plurality'].append(c[winner][0])

        # Top-two runoff
        utilities = normed_dist_utilities(v, c)
        rankings = honest_rankings(utilities)
        winner = runoff(rankings, tiebreaker='random')
        winners['Top-Two Runoff/Primary / Two-Round System / '
                'Contingent Vote'].append(c[winner][0])

        # Instant-runoff
        winner = irv(rankings, tiebreaker='random')
        winners['Ranked-Choice Voting (Hare) / '
                'Alternative Vote / Instant-Runoff'].append(c[winner][0])

        # Approval voting
        winner = approval(approval_optimal(utilities), tiebreaker='random')
        winners['Approval Voting ("optimal" strategy)'].append(c[winner][0])

        # Approval voting
        winner = approval(vote_for_k(utilities, vote_for), tiebreaker='random')
        winners[f'Approval Voting "(Vote-for-{vote_for}"'
                ' strategy)'].append(c[winner][0])

        # STAR voting
        ballots = honest_normed_scores(utilities)
        winner = star(ballots, tiebreaker='random')
        winners['STAR Voting'].append(c[winner][0])

        # Borda count
        winner = borda(rankings, tiebreaker='random')
        winners['Borda count'].append(c[winner][0])

        # Coombs method
        winner = coombs(rankings, tiebreaker='random')
        winners["Coombs' method"].append(c[winner][0])

        # Condorcet RCV
        winner = black(rankings, tiebreaker='random')
        winners['Condorcet Ranked-Choice Voting (Black)'].append(c[winner][0])

        # Ideal winner method.  Votes don't matter at all; pick the center.
        winner = np.argmin(abs(c))
        winners['Best possible winner (nearest center)'].append(c[winner][0])

    return winners


jobs = [delayed(simulate_batch)(n_cands)] * n_batches
print(f'{len(jobs)} tasks total:')
results = Parallel(n_jobs=-3, verbose=5)(jobs)
winners = {k: [v for d in results for v in d[k]] for k in results[0]}

title = f'{human_format(n_elections)} 1D elections, '
title += f'{human_format(n_voters)} voters, '
title += f'{human_format(n_cands)} candidates'
title += ', both ' + cand_dist

fig, ax = plt.subplots(nrows=ceildiv(len(winners), 2), ncols=2, num=title,
                       sharex=True, constrained_layout=True,
                       figsize=(11, 9.5))
fig.suptitle(title)

ax = ax.T.flatten()  # Flatten the ax array for easier indexing

for n, method in enumerate(winners.keys()):
    histplot(winners[method], ax=ax[n], label='Winners', stat='density')
    ax[n].set_title(method, loc='left')
    ax[n].set_yticklabels([])  # Don't care about numbers
    ax[n].set_ylabel("")  # No "density"

    tmp = ax[n].twinx()
    tmp.set_yticklabels([])  # Don't care about numbers
    tmp.set_ylabel("")  # No "density"

ax[0].set_xlim(-2.5, 2.5)
ax[0].legend()

# These take so long and kernel crashes!
plt.savefig(title + '.png')

with open(title + '.pkl', "wb") as file:
    pickle.dump(winners, file)
