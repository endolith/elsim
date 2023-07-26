"""
Find worst-case scenarios with RCV.
"""
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from seaborn import histplot, kdeplot
from tabulate import tabulate

from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.methods import fptp, irv
from elsim.strategies import honest_rankings

n_voters = 1_000 # 00
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
        ["Cand pos:"] + [f"{i:.1f}" for i in c[:, 0]],
        ["Tallies:"] + list(tallies)
    ]

    print(tabulate(table, tablefmt='pipe'))


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
    utilities = normed_dist_utilities(v, c)
    rankings = honest_rankings(utilities)

    # First remove the least 4 in FPTP ballot
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

    print(f'{n_losers} best candidates eliminated in FPTP primary.')
    print(f'Found after {trial} trials')
    print_candidates_and_tallies(c, tallies)
    # print(f'Candidate positions: {[f"{i:.1f}" for i in c[:, 0]]}')
    # print(f'Tallies: {tallies}')
    print(f'Least tallied:     {set(loser_indices)}')
    print(f'Closest to origin: {set(best_indices)}')

    break

print(n_failures/n_elections*100, "%")
# import numpy as np

# def closest_to_origin_indices(arr, n):
#     dist = np.linalg.norm(arr, axis=1)
#     return dist.argsort()[:n]

# coordinates = np.array([[-0.48410594, -1.32690993],
#                         [ 0.05752119, -0.1791397 ],
#                         [ 1.85980733, -0.78246966],
#                         [ 1.24417245,  0.35441942],
#                         [ 0.43939255,  0.96480142],
#                         [-1.1830837 ,  1.26436022],
#                         [-0.32579403, -0.9014799 ],
#                         [-1.46592261,  0.66729373],
#                         [-0.1399564 ,  0.92234018]])

# n = 3  # or whatever number you prefer

# indices = closest_to_origin_indices(coordinates, n)
# print(indices)


#     est_tally = max(tallies)
#     winners = _all_indices(tallies, highest_tally)

#     # Break any ties using specified method
#     tiebreak = _get_tiebreak(tiebreaker, _tiebreak_map)
#     # return tiebreak(winners)[0]

#     winner = fptp(rankings, tiebreaker='random')
#     winner = irv(rankings, tiebreaker='random')

# raise SystemExit
# # print(f'{n_batches} tasks total:')

# # Create a list of jobs
# jobs = []

# # Add jobs to the list
# for _ in range(n_elections):
#     jobs.append(delayed(simulate_batch)(n_voters, n_cands, batch_size))

# # Execute the jobs in parallel
# results = Parallel(n_jobs=-3, verbose=5)(jobs)

# winners = {k: [v for d in results for v in d[k]] for k in results[0]}

# title = f'{method}, {human_format(n_elections)} 1D elections, '
# title += f'{human_format(n_voters)} voters, '
# title += f'{human_format(n_cands)} '
# title += cand_dist + 'ly-distributed candidates'

# # For plotting only
# v, c = normal_electorate(n_voters, 1000, dims=1)

# fig, ax = plt.subplots(nrows=len(winners), num=title, sharex=True,
#                        constrained_layout=True, figsize=(7.5, 9.5))
# fig.suptitle(title)
# for n, disp in enumerate(winners):
#     histplot(winners[disp], ax=ax[n], label=f'{disp:.1f} dispersion',
#              stat='density')
#     ax[n].set_yticklabels([])  # Don't care about numbers
#     ax[n].set_ylabel("")  # No "density"

#     tmp = ax[n].twinx()
#     kdeplot(v[:, 0], ax=tmp, ls=':', label='Voters')  # Label doesn't work
#     ax[n].plot([], [], ls=':', label='Voters')  # Dummy label hack
#     tmp.set_yticklabels([])  # Don't care about numbers
#     tmp.set_ylabel("")  # No "density"

#     ax[n].legend()
# ax[0].set_xlim(-2.5, 2.5)
