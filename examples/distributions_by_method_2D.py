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
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
# from colorcet import fire
from joblib import Parallel, delayed

from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.methods import (approval, black, borda, coombs, fptp, irv, runoff,
                           star)
from elsim.strategies import (approval_optimal, honest_normed_scores,
                              honest_rankings, vote_for_k)

try:
    import ehtplot.color  # Creates afmhot_u colormap
except ValueError:  # "colormap … is already registered."
    pass

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

        # Contrived candidate at exact center
        # c[0] = 0, 0

        if cand_dist == 'uniform':
            # Replace with uniform distribution of candidates of same shape
            v = np.random.uniform(-u_width/2, +u_width/2, n_voters)
            v = np.atleast_2d(v).T
            c = np.random.uniform(-u_width/2, +u_width/2, n_cands)
            c = np.atleast_2d(c).T
            raise SystemExit

        # Voter distribution
        winners['Voters'].append(v)

        # Candidate distribution
        winners['Candidates'].append(c)

        # Ideal winner method.  Votes don't matter at all; pick the center.
        # winner = np.argmin(abs(c))  # 1D
        winner = np.argmin(np.sum(c**2, axis=1))  # 2D
        winners['Best possible winner'].append(c[winner])

        # FPTP voting method
        utilities = normed_dist_utilities(v, c)
        rankings = honest_rankings(utilities)
        winner = fptp(rankings, tiebreaker='random')
        winners['First Past The Post'].append(c[winner])

        # Top-two runoff
        utilities = normed_dist_utilities(v, c)
        rankings = honest_rankings(utilities)
        winner = runoff(rankings, tiebreaker='random')
        winners['Top-Two Runoff/Primary'].append(c[winner])

        # Instant-runoff
        winner = irv(rankings, tiebreaker='random')
        winners['Ranked-Choice Voting (Hare)'].append(c[winner])

        # Approval voting
        winner = approval(approval_optimal(utilities), tiebreaker='random')
        winners['Approval Voting ("optimal")'].append(c[winner])

        # Approval voting
        winner = approval(vote_for_k(utilities, vote_for), tiebreaker='random')
        winners[f'Approval Voting "(Vote-for-{vote_for}")'].append(c[winner])

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
        winners['Condorcet RCV (Black)'].append(c[winner])

        # This could just accumulate winner numbers and then get the coordinates later
        # no because c only exists here
        # Or it could just accumulate directly to the histogram heatmap in the job

    return winners


title = f'{human_format(n_elections)} 2D elections, '
title += f'{human_format(n_voters)} voters, '
title += f'{human_format(n_cands)} candidates'
title += f', both {cand_dist}'
title += f', {disp:.1f} disp'

# Load from .pkl file if it exists
pkl_filename = title + '.pkl'
if os.path.exists(pkl_filename):
    print('Loading pickled simulation results')
    with open(pkl_filename, "rb") as file:
        winners = pickle.load(file)
else:
    print('Running simulations')
    jobs = [delayed(simulate_batch)(n_cands)] * n_batches
    print(f'{len(jobs)} tasks total:')
    results = Parallel(n_jobs=-3, verbose=5)(jobs)

    winners = {k: np.array([v for d in results for v in d[k]])
               for k in results[0]}

    # Save the generated data to .pkl file
    with open(pkl_filename, "wb") as file:
        pickle.dump(winners, file)

# %% Measure distributions

winners['Voters'] = winners['Voters'].reshape(-1, winners['Voters'].shape[-1])
winners['Candidates'] = winners['Candidates'].reshape(
    -1, winners['Candidates'].shape[-1])

winners_stats = {method: (np.mean(winners[method], axis=0),
                          np.std(winners[method], axis=0)
                          ) for method in winners.keys()}

assert np.allclose(winners_stats['Voters'][1], [1, 1], rtol=1e-2)

for method, (mean, std) in winners_stats.items():
    print(f"{method}:")
    print(f"{len(winners[method])} samples")
    print(f"Winner distribution mean: {mean[0]:.3f}, {mean[1]:.3f}")
    print(f"                     std: {std[0]:.3f}, {std[1]:.3f}")
    print()

# %% Plotting


def plot_distribution(ax, data, title, max_lim):
    heatmap, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=50,
                                             range=[[-max_lim, max_lim],
                                                    [-max_lim, max_lim]])
    extent = [-max_lim, max_lim, -max_lim, max_lim]
    ax.imshow(heatmap.T, cmap='afmhot_u', origin='lower',
              aspect='auto', extent=extent)
    ax.set_xlim([-max_lim, max_lim])
    ax.set_ylim([-max_lim, max_lim])
    ax.set_aspect('equal')  # Set the aspect ratio to be equal (square)
    ax.set_title(title, loc='left')
    ax.tick_params(left=False, bottom=False)  # Remove tick marks
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_xlabel("")  # Remove x-axis label
    ax.set_ylabel("")  # Remove y-axis label
    for spine in ax.spines.values():
        spine.set_visible(True)


fig, ax = plt.subplots(nrows=ceildiv(len(winners), 4), ncols=4, num=title,
                       sharex=True, constrained_layout=True,
                       figsize=(11, 9.5))
fig.suptitle(title)

ax = ax.T.flatten()  # Flatten the ax array for easier indexing
max_lim = 1.5

for n, method in enumerate(winners.keys()):
    plot_distribution(ax[n], winners[method], method, max_lim)

    # Add standard deviation text in the lower right corner
    std = winners_stats[method][1]
    ax[n].text(0.98, 0.02, f'std: ({std[0]:.2f}, {std[1]:.2f})',
               verticalalignment='bottom', horizontalalignment='right',
               transform=ax[n].transAxes, color='white', fontsize=8)

# Hide the last axes if they are not used
for i in range(len(winners), len(ax)):
    fig.delaxes(ax[i])

plt.show()

# %% Save the figure

plt.savefig(title + '.png')
