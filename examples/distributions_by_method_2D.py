"""
Show how the winner distribution and bias of a voting method changes with
varying dispersion of candidates, in a 2D opinion space.

Essentially a 2D and normally-distributed version of Figure 3

    The distributions of the winning position with k = 3, 4, 5 candidates

from

    Kiran Tomlinson, Johan Ugander, Jon Kleinberg (2023) Moderation in instant
    runoff voting https://arxiv.org/abs/2303.09734

or a single-winner version of Figure 3

    Histograms and sample elections for our rules and distributions

from

    Edith Elkind, et al. (2019) What Do Multiwinner Voting Rules Do?
    https://arxiv.org/abs/1901.09217

"""
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.methods import (approval, black, borda, coombs, fptp, irv, runoff,
                           star)
from elsim.strategies import (approval_optimal, honest_normed_scores,
                              honest_rankings, vote_for_k)

n_elections = 1_000_000
n_voters = 1_000
n_cands = 9
cand_dist = 'normal'
u_width = 5
disp = 0.5
dims = 2
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


def simulate_batch(n_cands):
    winners = defaultdict(list)
    for iteration in range(batch_size):
        v, c = normal_electorate(n_voters, n_cands, dims=dims, disp=disp)

        if cand_dist == 'uniform':
            # Replace with uniform distribution of candidates of same shape
            v = np.random.uniform(-u_width/2, +u_width/2, n_voters)
            v = np.atleast_2d(v).T
            c = np.random.uniform(-u_width/2, +u_width/2, n_cands)
            c = np.atleast_2d(c).T
            raise SystemExit

        # FPTP voting method
        utilities = normed_dist_utilities(v, c)
        rankings = honest_rankings(utilities)
        winner = fptp(rankings, tiebreaker='random')
        winners['First Past The Post / Plurality'].append(c[winner])

        # Top-two runoff
        utilities = normed_dist_utilities(v, c)
        rankings = honest_rankings(utilities)
        winner = runoff(rankings, tiebreaker='random')
        winners['Top-Two Runoff/Primary / Two-Round System / '
                'Contingent Vote'].append(c[winner])

        # Instant-runoff
        winner = irv(rankings, tiebreaker='random')
        winners['Ranked-Choice Voting (Hare) / '
                'Alternative Vote / Instant-Runoff'].append(c[winner])

        # Approval voting
        winner = approval(approval_optimal(utilities), tiebreaker='random')
        winners['Approval Voting ("optimal" strategy)'].append(c[winner])

        # Approval voting
        winner = approval(vote_for_k(utilities, vote_for), tiebreaker='random')
        winners[f'Approval Voting "(Vote-for-{vote_for}"'
                ' strategy)'].append(c[winner])

        # STAR voting
        ballots = honest_normed_scores(utilities)
        winner = star(ballots, tiebreaker='random')
        winners['STAR Voting'].append(c[winner])

        # Borda count
        winner = borda(rankings, tiebreaker='random')
        winners['Borda count'].append(c[winner])

        # Coombs method
        winner = coombs(rankings, tiebreaker='random')
        winners["Coombs' method"].append(c[winner])

        # Condorcet RCV
        winner = black(rankings, tiebreaker='random')
        winners['Condorcet Ranked-Choice Voting (Black)'].append(c[winner])

        # Ideal winner method.  Votes don't matter at all; pick the center.
        # winner = np.argmin(abs(c))  # 1D
        winner = np.argmin(np.sum(c**2, axis=1))  # 2D
        winners['Best possible winner (nearest center)'].append(c[winner])

        # This could just accumulate winner numbers and then get the coordinates later
        # no because c only exists here
        # Or it could just accumulate directly to the histogram heatmap in the job

    return winners


jobs = [delayed(simulate_batch)(n_cands)] * n_batches
print(f'{len(jobs)} tasks total:')
results = Parallel(n_jobs=-3, verbose=5)(jobs)
# winners = {k: [v for d in results for v in d[k]] for k in results[0]}
winners = {k: np.array([v for d in results for v in d[k]]) for k in results[0]}


# %% Plotting

title = f'{human_format(n_elections)} 2D elections, '
title += f'{human_format(n_voters)} voters, '
title += f'{human_format(n_cands)} candidates'
title += f', both {cand_dist}'
title += f', {disp} disp'

fig, ax = plt.subplots(nrows=ceildiv(len(winners), 4), ncols=4, num=title,
                       sharex=True, constrained_layout=True,
                       figsize=(11, 9.5))
fig.suptitle(title)

ax = ax.T.flatten()  # Flatten the ax array for easier indexing
max_lim = 1.5

for n, method in enumerate(winners.keys()):
    coordinates = winners[method]
    # Create a 2D histogram with specified range
    heatmap, xedges, yedges = np.histogram2d(coordinates[:, 0],
                                             coordinates[:, 1], bins=50,
                                             range=[[-max_lim, max_lim],
                                                    [-max_lim, max_lim]])

    # Calculate extent from xedges and yedges
    extent = [-max_lim, max_lim, -max_lim, max_lim]

    ax[n].imshow(heatmap.T, cmap="Blues", origin='lower',
                 aspect='auto', extent=extent)

    ax[n].set_xlim([-max_lim, max_lim])
    ax[n].set_ylim([-max_lim, max_lim])
    ax[n].set_aspect('equal')  # Set the aspect ratio to be equal (square)
    ax[n].set_title(method, loc='left')
    ax[n].tick_params(left=False, bottom=False)  # Remove tick marks
    ax[n].set_xticks([])  # Remove x-axis ticks
    ax[n].set_yticks([])  # Remove y-axis ticks
    ax[n].set_xlabel("")  # Remove x-axis label
    ax[n].set_ylabel("")  # Remove y-axis label

    # Add borders to the plots
    for spine in ax[n].spines.values():
        spine.set_visible(True)

# Hide the last axes if they are not used
for i in range(len(winners), len(ax)):
    fig.delaxes(ax[i])

plt.show()

# %% Save the figure

plt.savefig(title + '.png')

with open(title + '.pkl', "wb") as file:
    pickle.dump(winners, file)
