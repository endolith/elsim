"""
Find worst-case scenarios with RCV.
"""
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from seaborn import histplot, kdeplot

from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.methods import fptp, irv
from elsim.strategies import honest_rankings

n_voters = 1_000
n_cands = 3
cand_dist = 'normal'


def human_format(num):
    for unit in ['', 'k', 'M', 'B', 'T']:
        if abs(num) < 1000:
            return f"{num:.3g}{unit}"
        num /= 1000.0


for trial in range(1000):
    v, c = normal_electorate(n_voters, n_cands, dims=1)
    utilities = normed_dist_utilities(v, c)
    rankings = honest_rankings(utilities)

    winner = fptp(rankings, tiebreaker='random')
    winner = irv(rankings, tiebreaker='random')


print(f'{n_batches} tasks total:')

# Create a list of jobs
jobs = []

# Add jobs to the list
for _ in range(n_elections):
    jobs.append(delayed(simulate_batch)(n_voters, n_cands, batch_size))

# Execute the jobs in parallel
results = Parallel(n_jobs=-3, verbose=5)(jobs)

winners = {k: [v for d in results for v in d[k]] for k in results[0]}

title = f'{method}, {human_format(n_elections)} 1D elections, '
title += f'{human_format(n_voters)} voters, '
title += f'{human_format(n_cands)} '
title += cand_dist + 'ly-distributed candidates'

# For plotting only
v, c = normal_electorate(n_voters, 1000, dims=1)

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
