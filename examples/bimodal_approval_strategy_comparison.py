"""
Bimodal electorate: optimal approval vs vote-for-half (and vote-for-k).

Mirrors the Weber-style comparison requested in
https://github.com/endolith/elsim/issues/20 alongside distributions_by_method.py.
"""
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from seaborn import histplot, kdeplot

from elsim.elections import bimodal_electorate, normed_dist_utilities
from elsim.methods import approval
from elsim.strategies import approval_optimal, vote_for_k

n_elections = 50_000
n_voters_each = 5_000
n_cands_each = 4
dims = 1
disp = 0.5
separation = 0.5
vote_for = n_cands_each // 2

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

        winner = approval(approval_optimal(utilities), tiebreaker='random')
        winners['Approval (optimal / mean threshold)'].append(c[winner][0])

        winner = approval(vote_for_k(utilities, 'half'), tiebreaker='random')
        winners[f'Approval (vote-for-{vote_for})'].append(c[winner][0])

        winner = approval(vote_for_k(utilities, 1), tiebreaker='random')
        winners['Approval (vote-for-1)'].append(c[winner][0])

        winner = np.argmin(np.abs(c[:, 0]))
        winners['Nearest cluster center'].append(c[winner][0])

    return winners


jobs = [delayed(simulate_batch)(k) for k in range(n_batches)]
print(f'{len(jobs)} tasks total:')
results = Parallel(n_jobs=-3, verbose=5)(jobs)

winners = {k: [v for d in results for v in d[k]] for k in results[0]}

title = (f'Bimodal approval strategies, {human_format(n_elections)} elections, '
         f'{2 * n_voters_each} voters, ±{separation} separation')

v, _ = bimodal_electorate(n_voters_each, 50, dims=dims, disp=disp,
                          separation=separation)

fig, ax = plt.subplots(nrows=len(winners), num=title, sharex=True,
                       constrained_layout=True, figsize=(7.5, 8))
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
