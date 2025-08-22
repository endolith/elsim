"""
Shared utilities for RCV collapse analysis examples.

This module provides common functions used across different RCV collapse scenarios:

Core Utilities:
- Election data calculation and analysis (utilities, rankings, tallies)
- Candidate evaluation methods (head-to-head wins, geometric positioning)
- Data processing helpers (array indexing, unique pattern counting)

Visualization:
- Voter distribution plotting with candidate-colored regions
- Election result charts (FPTP tallies, candidate positions)
- Consistent plot styling and axis setup

Helper Functions:
- Data formatting and display (tables, number formatting)
- Index-to-label conversion for candidate identification

Used by:
- collapse_finder.py: Simple 3-candidate RCV elimination
- collapse_finder final five version.py: Two-stage FPTP+RCV hybrid system
"""

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from elsim.elections import normed_dist_utilities
from elsim.methods import ranked_election_to_matrix
from elsim.strategies import honest_rankings


def human_format(num):
    """Format large numbers with K, M, B, T suffixes."""
    for unit in ['', 'k', 'M', 'B', 'T']:
        if abs(num) < 1000:
            return f"{num:.3g}{unit}"
        num /= 1000.0


def print_candidates_and_tallies(c, tallies):
    """Print candidates and their tallies in a formatted table."""
    # If the two lists have different lengths, raise an error.
    assert len(c) == len(tallies)

    table = [
        ["Cand pos:"] + [f"{i:.3f}" for i in c[:, 0]],
        ["Tallies:"] + list(tallies)
    ]

    print(tabulate(table, tablefmt='pipe', numalign="center"))


def top_n_indices(arr, n):
    """Return indices of top n elements in array."""
    return arr.argsort()[-n:][::-1]


def bottom_n_indices(arr, n):
    """Return indices of bottom n elements in array."""
    return arr.argsort()[:n]


def closest_to_origin_indices(arr, n):
    """Return indices of n elements closest to origin."""
    dist = np.linalg.norm(arr, axis=1)
    return dist.argsort()[:n]


def count_unique_rows(election):
    """Count unique ranking patterns in an election."""
    # We need to ensure rows are viewed as single items
    rows_as_tuples = map(tuple, election)

    # Use np.unique to find unique rows and their counts
    unique_rows, counts = np.unique(list(rows_as_tuples), return_counts=True,
                                    axis=0)

    # Zip together the unique rows and their counts for easy viewing
    result = list(zip(unique_rows, counts))

    return result


def count_wins(matrix):
    """
    Count the number of candidates beaten by each candidate.

    Parameters
    ----------
    matrix : ndarray
        A pairwise comparison matrix of candidate vs candidate defeats.

    Returns
    -------
    wins : list
        A list of the number of candidates beaten by each candidate.
    """
    n_cands = matrix.shape[0]
    wins = []
    for i in range(n_cands):
        wins.append(sum(matrix[i, j] > matrix[j, i] for j in range(n_cands)))
    return wins


def calculate_election_data(v, c):
    """Calculate utilities, rankings, election matrix, and first preference tallies."""
    utilities = normed_dist_utilities(v, c)
    rankings = honest_rankings(utilities)
    election = np.asarray(rankings)
    first_preferences = election[:, 0]
    tallies = np.bincount(first_preferences)
    return utilities, rankings, election, first_preferences, tallies


def find_best_candidates(election, n_best):
    """Find the n_best candidates based on head-to-head wins."""
    wins = count_wins(ranked_election_to_matrix(election))
    return top_n_indices(np.array(wins), n_best)


def indices_to_letters(indices, letters=None):
    """Convert candidate indices to letter labels."""
    if letters is None:
        letters = [chr(65 + i) for i in range(max(indices) + 1)]
    return " > ".join(letters[i] for i in indices)


def gaussian(x, mu, sigma):
    """
    Return a normal distribution pdf with center `mu` and standard deviation `sigma`.
    """
    return np.exp(-(x-mu)**2/(2*sigma**2))


def setup_plot_axes(fig, ax_hist, x_max):
    """Setup common plot axes with consistent styling."""
    # Now define the inset axes
    ax_fptp = ax_hist.inset_axes([0.08, 0.65, 0.25, 0.3])  # [x, y, width, height]

    # Adjust axis parameters for visibility
    for ax in [ax_fptp]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_tick_params(width=0.5)
        ax.yaxis.set_tick_params(width=0.5)

    ax_hist.set_ylim([-0.07, 0.5])
    ax_hist.set_xlim([-x_max, x_max])

    return ax_fptp


def plot_candidate_positions(ax_hist, pos, colors, letters=None):
    """Plot candidate positions on the histogram axis."""
    if letters is None:
        letters = [chr(65 + i) for i in range(len(pos))]

    # Each candidate has a position and a color
    for n in range(len(pos)):
        ax_hist.plot(pos[n], -0.02, '^', markersize=10, color=colors[n])
        ax_hist.text(pos[n], -0.04, letters[n], color=colors[n],
                     ha='center', va='top')


def plot_voter_distribution(ax_hist, pos, colors, x_max):
    """Plot the voter distribution colored by candidate regions."""
    # Generate x values
    x = np.linspace(-x_max, x_max, 300)

    pos_sorted = np.sort(pos)
    colors_sorted = np.array(colors)[np.argsort(pos)]

    midlines = (pos_sorted[1:] + pos_sorted[:-1])/2
    regions = np.searchsorted(x, midlines)
    bnds = [0, *regions, len(x)-2]

    for n, color in enumerate(colors_sorted):
        i_lo = bnds[n]
        i_hi = bnds[n+1]+1
        xx = x[i_lo: i_hi]
        ax_hist.fill_between(xx, gaussian(xx, 0, 1)*1/np.sqrt(2*np.pi), 0,
                             color=color)


def plot_fptp_results(ax_fptp, tallies, n_voters, colors, labels):
    """Plot FPTP results as a bar chart."""
    ax_fptp.bar(range(len(tallies)), tallies/n_voters*100,
                tick_label=labels, color=colors)
    ax_fptp.set_ylabel('1st rankings [%]')
