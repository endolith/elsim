"""
Find worst-case scenarios with RCV.
"""

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.strategies import honest_rankings

n_voters = 1_000
n_cands = 9
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


n_elections = 10_000
n_failures = 0
for trial in range(n_elections):
    v, c = normal_electorate(n_voters, n_cands, dims=1, disp=1)
    c = np.sort(c, axis=0)  # just for ease of viewing
    original_c = c

    # First remove the least tallied candidates in FPTP primary
    utilities = normed_dist_utilities(v, c)
    rankings = honest_rankings(utilities)
    election = np.asarray(rankings)

    # Get first preferences from election array
    first_preferences = election[:, 0]

    # Tally all first preferences (with index of tally = candidate ID)
    tallies = np.bincount(first_preferences)

    # Find the set of 5 candidates who have the highest tally
    n_finalists = 5
    n_losers = n_cands - n_finalists
    loser_indices = bottom_n_indices(tallies, n_losers)

    # Find the best candidates
    best_indices = closest_to_origin_indices(c, n_losers)

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
    utilities = normed_dist_utilities(v, c)
    rankings = honest_rankings(utilities)
    election = np.asarray(rankings)
    first_preferences = election[:, 0]
    tallies = np.bincount(first_preferences)
    print('Final five:')
    print_candidates_and_tallies(c, tallies)

    # Find the 3 best candidates
    best_indices = closest_to_origin_indices(c, 3)

    # Eliminate the lowest-voted
    loser = np.argmin(tallies)
    print(f'{loser} eliminated')

    # To find worst-case scenario, eliminated needs to be in best set
    if loser not in set(best_indices):
        continue

    # Eliminate and do it again
    c = np.delete(c, loser, axis=0)
    utilities = normed_dist_utilities(v, c)
    rankings = honest_rankings(utilities)
    election = np.asarray(rankings)
    first_preferences = election[:, 0]
    tallies = np.bincount(first_preferences)
    print('Final four:')
    print_candidates_and_tallies(c, tallies)

    # Find the 2 best candidates
    best_indices = closest_to_origin_indices(c, 2)

    # Eliminate the lowest-voted again
    loser = np.argmin(tallies)
    print(f'{loser} eliminated')

    # To find worst-case scenario, eliminated needs to be in best set
    if loser not in set(best_indices):
        continue

    # Eliminate and do it a third time
    c = np.delete(c, loser, axis=0)
    utilities = normed_dist_utilities(v, c)
    rankings = honest_rankings(utilities)
    election = np.asarray(rankings)
    first_preferences = election[:, 0]
    tallies = np.bincount(first_preferences)
    print('Final three:')
    print_candidates_and_tallies(c, tallies)

    # Find the 1 best candidate
    best_indices = closest_to_origin_indices(c, 1)

    # Eliminate the lowest-voted again
    loser = np.argmin(tallies)
    print(f'{loser} eliminated')

    # To find worst-case scenario, eliminated needs to be in best set
    if loser not in set(best_indices):
        continue

    # Eliminate and do it a fourth time
    c = np.delete(c, loser, axis=0)
    utilities = normed_dist_utilities(v, c)
    rankings = honest_rankings(utilities)
    election = np.asarray(rankings)
    first_preferences = election[:, 0]
    tallies = np.bincount(first_preferences)
    print('Final two:')
    print_candidates_and_tallies(c, tallies)

    print(f'After {trial} trials')
    print(original_c)
    plt.plot(original_c[:, 0], [1]*n_cands, '|')

    break
