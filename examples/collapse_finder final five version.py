"""
Find worst-case scenarios with RCV.
"""

import matplotlib.pyplot as plt
import numpy as np
from collapse_utils import (bottom_n_indices, calculate_election_data,
                            count_unique_rows, find_best_candidates,
                            indices_to_letters, plot_candidate_positions,
                            plot_fptp_results, plot_voter_distribution,
                            print_candidates_and_tallies, setup_plot_axes)
from palettable.cartocolors.qualitative import Prism_9 as cmap

from elsim.elections import normal_electorate

n_voters = 1_000
n_cands = 9
cand_dist = 'normal'


n_elections = 50_000
n_failures = 0
for trial in range(n_elections):
    v, c = normal_electorate(n_voters, n_cands, dims=1, disp=1)
    c = np.sort(c, axis=0)  # just for ease of viewing
    original_c = c

    # First remove the least tallied candidates in FPTP primary
    utilities, rankings, election, first_preferences, tallies = calculate_election_data(v, c)
    original_utilities = utilities.sum(axis=0)
    original_utilities /= original_utilities.max()
    original_election = election
    original_matrix = ranked_election_to_matrix(election)

    # Get first preferences from election array
    first_preferences = election[:, 0]

    # Tally all first preferences (with index of tally = candidate ID)
    tallies = np.bincount(first_preferences)
    original_tallies = tallies

    # Find the set of 5 candidates who have the highest tally
    n_finalists = 5
    n_losers = n_cands - n_finalists
    loser_indices = bottom_n_indices(tallies, n_losers)
    original_loser_indices = loser_indices
    best_indices = find_best_candidates(original_election, 4)

    if set(loser_indices) != set(best_indices):
        continue

    print('\n===================')
    print(f'{n_losers} best candidates eliminated in FPTP primary.')
    print(f'Found after {trial} trials')
    print_candidates_and_tallies(c, tallies)
    # print(f'Candidate positions: {[f"{i:.1f}" for i in c[:, 0]]}')
    # print(f'Tallies: {tallies}')
    print(f'Least tallied:     {set(loser_indices)}')
    print(f'Closest to origin: {set(best_indices)}')

    # Remaining candidates proceed to RCV general
    c = np.delete(c, loser_indices, axis=0)

    # RCV elimination rounds - start with 5 candidates, eliminate down to 2
    n_remaining = 5
    found_worst_case = False

    for round_num in range(3):  # 3 rounds: 5->4, 4->3, 3->2
        utilities, rankings, election, first_preferences, tallies = calculate_election_data(v, c)

        print(f'Final {n_remaining}:')
        print_candidates_and_tallies(c, tallies)

        # Find the best candidates (n_remaining - 1 best)
        best_indices = find_best_candidates(election, n_remaining - 1)

        # Eliminate the lowest-voted
        loser = np.argmin(tallies)
        print(f'{loser} eliminated')

        # To find worst-case scenario, eliminated needs to be in best set
        if loser not in set(best_indices):
            # This trial didn't produce a worst-case scenario, try next trial
            found_worst_case = False
            break

        # Remove the eliminated candidate
        c = np.delete(c, loser, axis=0)
        n_remaining -= 1

        # If we've reached 2 candidates, we found a worst-case scenario
        if n_remaining == 2:
            # Recalculate utilities and tallies for the final 2 candidates
            utilities, rankings, election, first_preferences, tallies = calculate_election_data(v, c)

            print('Final two:')
            print_candidates_and_tallies(c, tallies)
            print(f'After {trial} trials')
            found_worst_case = True
            break

    # If we found a worst-case scenario, exit the trial loop
    if found_worst_case:
        break


print(original_c)
# plt.plot(original_c[:, 0], [1]*n_cands, '|')
# plt.xlim(-max(abs(original_c))*1.1, max(abs(original_c))*1.1)

# # Call the function with your 'election' array
# result = count_unique_rows(original_election)
# for row, count in result:
#     print(f"Row: {row}, Count: {count}")


# Define a function to convert indices to letters
def indices_to_letters(indices):
    return " > ".join(chr(65 + i) for i in indices)


# Call the function with your 'election' array
result = count_unique_rows(original_election)
for row, count in result:
    print(f"{count:4}: {indices_to_letters(row)}")

x_max = +2.5
pos = original_c[:, 0]

# from palettable.tableau import Tableau_10 as cmap
# from palettable.tableau import GreenOrange_12 as cmap
# from palettable.tableau import TableauMedium_10 as cmap
# from palettable.mycarta import Cube1_9 as cmap
# from palettable.colorbrewer.qualitative import Set1_9 as cmap
# from palettable.colorbrewer.qualitative import Set3_9 as cmap
# from palettable.cartocolors.qualitative import Bold_9 as cmap
# from palettable.cartocolors.qualitative import Vivid_9 as cmap
# from palettable.cartocolors.qualitative import Antique_9 as cmap
from palettable.cartocolors.qualitative import Prism_9 as cmap
# from palettable.cartocolors.qualitative import Pastel_9 as cmap


colors = cmap.mpl_colors

# colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:n_cands]


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

# Setup axes using shared utilities
ax_fptp = setup_plot_axes(fig, ax_hist, x_max)
ax_wins = ax_hist.inset_axes([0.72, 0.53, 0.32, 0.45])  # [x, y, width, height]

# Adjust axis parameters for wins axis
ax_wins.spines['right'].set_visible(False)
ax_wins.spines['top'].set_visible(False)
ax_wins.xaxis.set_tick_params(width=0.5)
ax_wins.yaxis.set_tick_params(width=0.5)

# Plot candidate positions using shared utility
plot_candidate_positions(ax_hist, pos, colors)

# Plot voter distribution using shared utility
plot_voter_distribution(ax_hist, pos, colors, x_max)

# Plurality results bar chart in percent using shared utility
plot_fptp_results(ax_fptp, original_tallies, n_voters, colors,
                  [chr(65 + n) for n in range(n_cands)])

# ax_fav.bar(range(n_cands), original_utilities*100,
#            tick_label=[chr(65 + n) for n in range(n_cands)], color=colors)
# ax_fav.set_ylabel('Favorability [%]')


def plot_wins(wins, ax, colors='b', gap=0.1):
    """
    Plot number of wins as discrete blocks stacked on top of each other.

    Parameters
    ----------
    wins : list
        A list of the number of wins for each candidate.
    ax : matplotlib axis
        The axis to plot on.
    colors : str or list
        The colors to use for the bars.
    gap : float
        The gap to leave between blocks. Default is 0.1.
    """
    n_cands = len(wins)
    for n in range(n_cands):
        for i in range(int(wins[n])):
            ax.bar(n, 1 - gap, bottom=i + i * gap,
                   color=colors if isinstance(colors, str) else colors[n],
                   edgecolor='black', linewidth=1)
    ax.set_xticks(range(n_cands))
    ax.set_xticklabels([chr(65 + n) for n in range(n_cands)])
    ax.set_xlim(-0.5, n_cands-0.5)  # Set fixed x-axis limits
    ax.set_ylabel('Head-to-head wins')


# Use the function
wins = count_wins(original_matrix)
plot_wins(wins, ax_wins, colors)

# This is required to make the blocks in ax_wins square
ax_wins.set_aspect('equal')

plt.tight_layout()
plt.show()


plt.tight_layout()

plt.show()
