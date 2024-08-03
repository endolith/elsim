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

bins = 50
range_lim = [-1.5, 1.5]


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

    histograms = {}
    variances = {}

    for method, points in winners.items():
        points = np.vstack(points)  # Concatenate all points
        histograms[method] = np.histogram2d(points[:, 0], points[:, 1],
                                            bins=bins,
                                            range=[range_lim, range_lim])[0]
        variances[method] = np.var(points, axis=0)  # x, y

    return histograms, variances


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
        aggregated_histograms, aggregated_variances = pickle.load(file)
else:
    print('Running simulations')
    jobs = [delayed(simulate_batch)(n_cands)] * n_batches
    print(f'{len(jobs)} tasks total:')
    results = Parallel(n_jobs=-3, verbose=5)(jobs)
    del jobs

    aggregated_histograms = defaultdict(lambda: np.zeros((bins, bins)))
    aggregated_variances = defaultdict(lambda: np.zeros(2))

    for result in results:
        histograms, variances = result
        for key in histograms:
            # Technically requires Bessel's correction since we are sampling
            # an infinite number of elections(?), but negligible for large N.
            aggregated_histograms[key] += histograms[key]
            # Sum all variances for calculating mean
            aggregated_variances[key] += variances[key]
    del results  # n_batches * bins**2 * n methods = GBs of RAM

    # Average the variances
    for key in aggregated_variances:
        # Technically the overall variance of a union of chunks has an
        # inter-chunk variance term as well, but since ours are all zero-mean,
        # I don't think it matters.
        aggregated_variances[key] /= n_batches

    # Convert defaultdicts to regular dictionaries before pickling
    aggregated_histograms = dict(aggregated_histograms)
    aggregated_variances = dict(aggregated_variances)

    # Save the generated data to .pkl file
    with open(pkl_filename, "wb") as file:
        pickle.dump((aggregated_histograms, aggregated_variances), file)


# %% Measure distributions

winners_stats = {method: (np.sqrt(aggregated_variances[method])
                          ) for method in aggregated_histograms.keys()}

for method, std in winners_stats.items():
    print(f"{method}:")
    print(f"Winner distribution std: {std[0]:.3f}")
    print()

# %% Plotting


def plot_distribution(ax, histogram, title, max_lim):
    extent = [-max_lim, max_lim, -max_lim, max_lim]
    ax.imshow(histogram.T, cmap='afmhot_u', origin='lower',
              aspect='auto', extent=extent, interpolation='none')
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


fig, ax = plt.subplots(nrows=ceildiv(len(aggregated_histograms), 4), ncols=4,
                       num=title, sharex=True, constrained_layout=True,
                       figsize=(11, 9.5))
fig.suptitle(title)

ax = ax.T.flatten()  # Flatten the ax array for easier indexing
max_lim = 1.5

for n, method in enumerate(aggregated_histograms.keys()):
    plot_distribution(ax[n], aggregated_histograms[method], method, max_lim)

    # Add standard deviation text in the lower right corner
    std = winners_stats[method][1]
    ax[n].text(0.98, 0.02, f'std: {std:.3f}',
               verticalalignment='bottom', horizontalalignment='right',
               transform=ax[n].transAxes, color='white', fontsize=9)

# Hide the last axes if they are not used
for i in range(len(aggregated_histograms), len(ax)):
    fig.delaxes(ax[i])

plt.show()

# %% Save the figure

plt.savefig(title + '.png')
