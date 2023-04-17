"""
Show the winner distributions and bias of different voting methods.
"""
import pickle
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from seaborn import histplot
from joblib import Parallel, delayed
from elsim.methods import fptp, irv
from elsim.elections import normed_dist_utilities
from elsim.strategies import honest_rankings

n_elections = 1_000_000  # Roughly 80 minutes on a 2019 6-core i7-9750H
n_voters = 1_000
n_cands_list = (3, 4, 5)
cand_dist = 'uniform'
u_width = 5
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

        v = np.random.uniform(-u_width/2, +u_width/2, n_voters)
        v = np.atleast_2d(v).T
        c = np.random.uniform(-u_width/2, +u_width/2, n_cands)
        c = np.atleast_2d(c).T

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
title += 'both ' + cand_dist

fig, ax = plt.subplots(nrows=2, ncols=3, num=title,
                       sharex=True, constrained_layout=True,
                       figsize=(11, 5))
fig.suptitle(title)

for n_cands in n_cands_list:
    print(f'{n_batches} tasks total:')

    p = Parallel(n_jobs=-3, verbose=5)(delayed(simulate_batch)(n_cands)
                                       for i in range(n_batches))
    winners = {k: [v for d in p for v in d[k]] for k in p[0]}

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

    ax[0, n_cands-3].set_xlim(-2.5, 2.5)

    # These take so long and kernel crashes!
    plt.savefig(title + '.png')

    with open(title + '.pkl', "wb") as file:
        pickle.dump(winners, file)

ax[0, 0].legend()
