"""
Find worst-case scenarios with RCV.
"""

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.methods import ranked_election_to_matrix
from elsim.strategies import honest_rankings

n_voters = 1_000
n_cands = 2
cand_dist = 'normal'


def human_format(num):
    for unit in ['', 'k', 'M', 'B', 'T']:
        if abs(num) < 1000:
            return f"{num:.3g}{unit}"
        num /= 1000.0


def print_candidates_and_tallies(c, tallies):
    # If the two lists have different lengths, raise an error.
    assert len(c) == len(tallies)

    table = [
        ["Cand pos:"] + [f"{i:.3f}" for i in c[:, 0]],
        ["Tallies:"] + list(tallies)
    ]

    print(tabulate(table, tablefmt='pipe', numalign="center"))


# ChatGPT
def top_n_indices(arr, n):
    return arr.argsort()[-n:][::-1]


def bottom_n_indices(arr, n):
    return arr.argsort()[:n]


# ChatGPT
def closest_to_origin_indices(arr, n):
    dist = np.linalg.norm(arr, axis=1)
    return dist.argsort()[:n]


def count_unique_rows(election):
    # We need to ensure rows are viewed as single items
    rows_as_tuples = map(tuple, election)

    # Use np.unique to find unique rows and their counts
    unique_rows, counts = np.unique(list(rows_as_tuples), return_counts=True,
                                    axis=0)

    # Zip together the unique rows and their counts for easy viewing
    result = list(zip(unique_rows, counts))

    return result


# ChatGPT
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


n_elections = 50_000
n_failures = 0
for trial in range(n_elections):
    v, c = normal_electorate(n_voters, n_cands, dims=1, disp=1, random_state=1)
    c = np.array([[0, +0.7]]).T
    c = np.sort(c, axis=0)  # just for ease of viewing
    original_c = c

    utilities = normed_dist_utilities(v, c)
    rankings = honest_rankings(utilities)
    election = np.asarray(rankings)
    first_preferences = election[:, 0]
    tallies = np.bincount(first_preferences)
    original_matrix = ranked_election_to_matrix(election)
    original_tallies = tallies
    print('Final three:')
    print_candidates_and_tallies(c, tallies)

    # Find the 1 best candidate
    # best_indices = closest_to_origin_indices(c, 1)
    wins = count_wins(ranked_election_to_matrix(election))
    best_indices = top_n_indices(np.array(wins), 1)

    # Eliminate the lowest-voted again
    loser = np.argmin(tallies)
    print(f'{loser} eliminated')

    # # To find worst-case scenario, eliminated needs to be in best set
    # if loser not in set(best_indices):
    #     continue

    # # Eliminate and do it a fourth time
    # c = np.delete(c, loser, axis=0)
    # utilities = normed_dist_utilities(v, c)
    # rankings = honest_rankings(utilities)
    # election = np.asarray(rankings)
    # first_preferences = election[:, 0]
    # tallies = np.bincount(first_preferences)
    # print('Final two:')
    # print_candidates_and_tallies(c, tallies)

    # print(f'After {trial} trials')

    break

print(original_c)
# plt.plot(original_c[:, 0], [1]*n_cands, '|')
# plt.xlim(-max(abs(original_c))*1.1, max(abs(original_c))*1.1)

# # Call the function with your 'election' array
# result = count_unique_rows(original_election)
# for row, count in result:
#     print(f"Row: {row}, Count: {count}")


letters = [ 'F', 'R']


# Define a function to convert indices to letters
def indices_to_letters(indices):
    return " > ".join(letters[i] for i in indices)


# Call the function with your 'election' array
# result = count_unique_rows(original_election)
# for row, count in result:
#     print(f"{count:4}: {indices_to_letters(row)}")

x_max = +2.5
pos = original_c[:, 0]

colors = ['#3333FF', '#242851', '#E81B23']  # Wikipedia
colors = ['#0044C9', '#182742', '#D71F27']  # Websites
colors = [ 'tab:gray', 'tab:red']  # Previous suck


def gaussian(x, mu, sigma):
    """
    Return a normal distribution pdf with center `mu` and standard deviation
    `sigma`
    """
    return np.exp(-(x-mu)**2/(2*sigma**2))


# Generate x values
x = np.linspace(-x_max, x_max, 300)
# Define the figure
fig, ax_hist = plt.subplots(figsize=(8, 4))  # Adjust as necessary

# Now define the inset axes
# [x, y, width, height]
ax_fptp = ax_hist.inset_axes([0.08, 0.65, 0.25, 0.3])
# ax_wins = ax_hist.inset_axes([0.74, 0.65, 0.25, 0.3])

# Adjust axis parameters for visibility
for ax in [ax_fptp, ]:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(width=0.5)
    ax.yaxis.set_tick_params(width=0.5)

# ax.grid(True)
ax_hist.set_ylim([-0.07, 0.5])
ax_hist.set_xlim([-x_max, x_max])

# Each candidate has a position and a color
for n in range(n_cands):
    ax_hist.plot(pos[n], -0.02, '^', markersize=10, color=colors[n])
    ax_hist.text(pos[n], -0.04, letters[n], color=colors[n],
                 ha='center', va='top')


pos_sorted = np.sort(pos)
colors_sorted = np.array(colors)[np.argsort(pos)]

midlines = (pos_sorted[1:] + pos_sorted[:-1])/2

regions = np.searchsorted(x, midlines)

bnds = [0, *regions, len(x)-2]
for n, color in enumerate(colors_sorted):
    i_lo = bnds[n]
    i_hi = bnds[n+1]+1
    # print(i_lo, i_hi, x[i_lo], x[i_hi], end='|')
    xx = x[i_lo: i_hi]
    # print(xx.min(), xx.max(), len(xx), np.sum(gaussian(xx, wmu.value,
    #                                                    wsigma.value)))
    ax_hist.fill_between(xx, gaussian(xx, 0, 1)*1/np.sqrt(2*np.pi), 0,
                         color=color)

# To verify the colorful normal matches the actual voters
# plt.hist(v, density=True, bins=300, alpha=0.5)


# Plurality results bar chart in percent
ax_fptp.bar(range(n_cands), original_tallies/n_voters*100,
            tick_label=[letters[n] for n in range(n_cands)], color=colors)
# ax_fptp.set_ylim(0, 100)
ax_fptp.set_ylabel('1st rankings [%]')

# ax_fav.bar(range(n_cands), original_utilities*100,
#            tick_label=[letters[n] for n in range(n_cands)], color=colors)
# ax_fav.set_ylabel('Favorability [%]')


# def plot_wins(wins, ax, colors='b', gap=0.1):
#     """
#     Plot number of wins as discrete blocks stacked on top of each other with symmetrical gaps.

#     Parameters
#     ----------
#     wins : list
#         A list of the number of wins for each candidate.
#     ax : matplotlib axis
#         The axis to plot on.
#     colors : str or list
#         The colors to use for the bars.
#     gap : float
#         The gap to leave between blocks. Default is 0.2.
#     """
#     n_cands = len(wins)
#     block_height = 1 - gap  # The height of each block, accounting for the gap.
#     total_height = block_height + gap  # Total height including the gap below each block.
#     for n in range(n_cands):
#         for i in range(int(wins[n])):
#             # The bottom parameter is adjusted to include the gap below each block.
#             ax.bar(n, block_height, bottom=i * total_height,
#                    color=colors if isinstance(colors, str) else colors[n],
#                    edgecolor='black', linewidth=1, width=1-gap)
#     ax.set_xticks(range(n_cands))
#     ax.set_xticklabels([letters[n] for n in range(n_cands)])
#     ax.set_xlim(-0.5, n_cands-0.5)  # Set fixed x-axis limits
#     ax.set_ylabel('Head-to-head wins')

# # Use the function
# wins = count_wins(original_matrix)
# plot_wins(wins, ax_wins, colors)

# # To ensure the aspect ratio is set such that blocks are always squares, regardless of the figure dimensions
# ax_wins.set_aspect('equal', adjustable='datalim')

plt.tight_layout()
plt.show()
