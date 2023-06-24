"""
Reproduce Figure 3

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
from elsim.methods import fptp, irv
from elsim.strategies import honest_rankings

n_elections = 1_000_000  # Roughly 6 minutes on a 2019 6-core i7-9750H
n_voters = 1_000
n_cands_list = (3, 4, 5)
disp = 1

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


def simulate_batch(n_cands):
    winners = defaultdict(list)
    for iteration in range(batch_size):

        # "voters and candidates come from the uniform distribution on [0, 1]"
        v = np.random.uniform(0, 1, n_voters)
        # But remove any run-to-run random bias, because we want to match ideal
        # points of the actual set of voters, not their expected value.
        v = (v - v.mean() + 0.5)[:, np.newaxis]

        c = np.random.uniform(0, 1, n_cands)
        c = (c - v.mean() + 0.5)[:, np.newaxis]

        # FPTP voting method
        utilities = normed_dist_utilities(v, c)
        rankings = honest_rankings(utilities)
        winner = fptp(rankings, tiebreaker='random')
        winners['Plurality'].append(c[winner][0])

        # Instant-runoff
        winner = irv(rankings, tiebreaker='random')
        winners['IRV'].append(c[winner][0])

    return winners


title = f'{human_format(n_elections)} 1D elections, '
title += f'{human_format(n_voters)} voters, '
title += 'both uniform'

fig, ax = plt.subplots(nrows=2, ncols=3, num=title,
                       sharex=True, constrained_layout=True,
                       figsize=(11, 5))
fig.suptitle(title)

for n_cands in n_cands_list:
    print(f'{n_batches} tasks total:')

    results = Parallel(n_jobs=-3, verbose=5)(delayed(simulate_batch)(n_cands)
                                             for i in range(n_batches))
    winners = {k: [v for d in results for v in d[k]] for k in results[0]}

    for n, method in enumerate(winners.keys()):
        histplot(winners[method], ax=ax[n, n_cands-3],
                 label='Winners', stat='density')
        ax[n, n_cands-3].set_title(f'{method}, {n_cands} candidates',
                                   loc='left')
        ax[n, n_cands-3].set_yticklabels([])  # Don't care about numbers
        ax[n, n_cands-3].set_ylabel("")  # No "density"

        tmp = ax[n, n_cands-3].twinx()
        tmp.set_yticklabels([])  # Don't care about numbers
        tmp.set_ylabel("")  # No "density"

    ax[0, n_cands-3].set_xlim(-0.05, 1.05)

    # "the probability that the winning candidate produced by IRV lies outside
    # the interval [1/6, 5/6] goes to 0 as the number of candidates k goes to
    # infinity"
    ax[1, n_cands-3].axvline(1/6, linestyle='--')
    ax[1, n_cands-3].axvline(5/6, linestyle='--')

ax[0, 0].legend()

# These take so long and kernel crashes!
plt.savefig(title + '.png')

with open(title + '.pkl', "wb") as file:
    pickle.dump(winners, file)
