"""
Show the winner distributions and bias of different voting methods.
"""
import random
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from seaborn import kdeplot, histplot
from joblib import Parallel, delayed
from elsim.methods import fptp, runoff, irv, black, star
from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.strategies import honest_rankings, honest_normed_scores

n_elections = 1_000_000  # Roughly 11 minutes on a 2019 6-core i7-9750H
n_voters = 1_000
n_cands = 7
cand_dist = 'normal'
u_width = 10
disp = 0.5

# Simulate more than just one election per worker to improve efficiency
batch_size = 100
n_batches = n_elections // batch_size
assert n_batches * batch_size == n_elections


def human_format(num):
    for unit in ['', 'k', 'M', 'B', 'T']:
        if abs(num) < 1000:
            return f"{num:.3g}{unit}"
        num /= 1000.0


def simulate_batch():
    winners = defaultdict(list)
    for _ in range(batch_size):
        v, c = normal_electorate(n_voters, n_cands, dims=1, disp=disp)

        if cand_dist == 'uniform':
            # Replace with uniform distribution of candidates of same shape
            c = np.random.uniform(-u_width/2, +u_width/2, n_cands)
            c = np.atleast_2d(c).T

        # Random winner method.  Votes don't matter at all.
        winner = random.sample(range(n_cands), 1)[0]
        winners['Random winner (candidate distribution)'].append(c[winner][0])

        # # Random ballot method.  Pick one voter and go with their choice.
        # winning_voter = random.sample(range(n_voters), 1)[0]
        # dists = abs(v[winning_voter] - c)
        # winner = np.argmin(dists)
        # winners['RB'].append(c[winner][0])

        # FPTP voting method
        utilities = normed_dist_utilities(v, c)
        rankings = honest_rankings(utilities)
        winner = fptp(rankings, tiebreaker='random')
        winners['First Past The Post / Plurality'].append(c[winner][0])

        # Top-two runoff
        utilities = normed_dist_utilities(v, c)
        rankings = honest_rankings(utilities)
        winner = runoff(rankings, tiebreaker='random')
        winners['Top-Two Runoff / Top-Two Primary / Two-Round System / '
                'Contingent Vote'].append(c[winner][0])

        # Instant-runoff
        winner = irv(rankings, tiebreaker='random')
        winners['Ranked-Choice Voting (Hare) / '
                'Alternative Vote / Instant-Runoff'].append(c[winner][0])

        # STAR voting
        ballots = honest_normed_scores(utilities)
        winner = star(ballots, tiebreaker='random')
        winners['STAR Voting'].append(c[winner][0])

        # Condorcet RCV
        winner = black(rankings, tiebreaker='random')
        winners['Condorcet Ranked-Choice Voting (Black)'].append(c[winner][0])

        # Ideal winner method.  Votes don't matter at all; pick the center.
        winner = np.argmin(abs(c))
        winners['Best possible winner (nearest center)'].append(c[winner][0])

    return winners


print(f'{n_batches} tasks total:')
p = Parallel(n_jobs=-3, verbose=5)(
    delayed(simulate_batch)() for _ in range(n_batches)
)
winners = {k: [v for d in p for v in d[k]] for k in p[0]}

title = f'{human_format(n_elections)} 1D elections, '
title += f'{human_format(n_voters)} voters, '
title += f'{human_format(n_cands)} '
title += f'{cand_dist}ly-dist. candidates'
title += f', {disp:.2f} dispersion'

# For plotting only
v, c = normal_electorate(n_voters, 1000, dims=1)

fig, ax = plt.subplots(nrows=len(winners), num=title, sharex=True,
                       constrained_layout=True, figsize=(7.5, 9.5))
fig.suptitle(title)
for n, method in enumerate(winners.keys()):
    histplot(winners[method], ax=ax[n], label='Winners', stat='density')
    ax[n].set_title(method, loc='left')
    ax[n].set_yticklabels([])  # Don't care about numbers
    ax[n].set_ylabel("")  # No "density"

    tmp = ax[n].twinx()
    kdeplot(v[:, 0], ax=tmp, ls=':', label='Voters')  # Label doesn't work
    ax[n].plot([], [], ls=':', label='Voters')  # Dummy label hack
    tmp.set_yticklabels([])  # Don't care about numbers
    tmp.set_ylabel("")  # No "density"

ax[0].set_xlim(-2.5, 2.5)
ax[0].legend()
