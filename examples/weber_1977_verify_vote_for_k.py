"""
Reproduce Vote-for-k effectiveness for different k.

Weber, Robert J. (1978). "Comparison of Public Choice Systems".
Cowles Foundation Discussion Papers. Cowles Foundation for Research in
Economics. No. 498. https://cowles.yale.edu/publications/cfdp/cfdp-498

50k elections required for smooth lines.  Typical results with 1k voters:

|    |    1 |     2 |     3 |     4 |   half |
|---:|-----:|------:|------:|------:|-------:|
|  2 | 81.7 | nan   | nan   | nan   |   81.6 |
|  3 | 74.8 |  74.9 | nan   | nan   |   74.8 |
|  4 | 69.3 |  79.9 |  68.8 | nan   |   79.9 |
|  5 | 64.9 |  79.0 |  78.7 |  63.6 |   79.0 |
|  6 | 61.4 |  77.1 |  81.3 |  76.6 |   81.3 |
|  7 | 57.9 |  74.3 |  80.9 |  80.8 |   80.9 |
|  8 | 54.9 |  71.6 |  79.8 |  82.3 |   82.3 |
|  9 | 52.9 |  69.2 |  78.1 |  82.2 |   82.2 |
| 10 | 51.0 |  67.1 |  76.3 |  81.6 |   82.9 |

But the smooth lines are inaccurate, diverging slightly higher as the
number of candidates increases.

Increasing the number of voters to 100_000 improves the accuracy so that
it matches Weber's results (which used an infinite number of voters), but this
takes a very long time (hours) to simulate.

Typical results with 100k voters, 100k elections:

|    |    1 |     2 |     3 |     4 |   half |
|---:|-----:|------:|------:|------:|-------:|
|  2 | 81.6 | nan   | nan   | nan   |   81.6 |
|  3 | 75.0 |  75.2 | nan   | nan   |   75.0 |
|  4 | 69.4 |  80.1 |  69.2 | nan   |   80.1 |
|  5 | 64.4 |  79.1 |  79.1 |  64.9 |   79.1 |
|  6 | 60.8 |  76.6 |  81.4 |  76.6 |   81.4 |
|  7 | 57.1 |  74.0 |  81.1 |  81.1 |   81.1 |
|  8 | 54.4 |  71.3 |  79.7 |  82.3 |   82.3 |
|  9 | 51.8 |  68.5 |  77.8 |  82.0 |   82.0 |
| 10 | 49.7 |  66.4 |  75.9 |  81.5 |   83.0 |
"""
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from joblib import Parallel, delayed
from elsim.methods import (fptp, utility_winner, approval)
from elsim.elections import random_utilities
from elsim.strategies import honest_rankings, vote_for_k
from weber_1977_expressions import (eff_standard, eff_vote_for_k,
                                    eff_vote_for_half)

n_elections = 10_000  # Roughly 60 seconds
n_voters = 1_000
n_cands_list = np.arange(2, 11)

ranked_methods = {'Standard': fptp}

rated_methods = {'Vote-for-1': lambda utilities, tiebreaker:
                 approval(vote_for_k(utilities, 1), tiebreaker),
                 'Vote-for-2': lambda utilities, tiebreaker:
                 approval(vote_for_k(utilities, 2), tiebreaker),
                 'Vote-for-3': lambda utilities, tiebreaker:
                 approval(vote_for_k(utilities, 3), tiebreaker),
                 'Vote-for-4': lambda utilities, tiebreaker:
                 approval(vote_for_k(utilities, 4), tiebreaker),
                 'Vote-for-half': lambda utilities, tiebreaker:
                 approval(vote_for_k(utilities, 'half'), tiebreaker),
                 'Vote-for-(n-1)': lambda utilities, tiebreaker:
                 approval(vote_for_k(utilities, -1), tiebreaker),
                 }

utility_sums = {key: Counter() for key in (
    ranked_methods.keys() | rated_methods.keys() | {'UW'})}


def simulate_election():
    utility_sums = defaultdict(dict)
    for n_cands in n_cands_list:
        utilities = random_utilities(n_voters, n_cands)

        # Find the social utility winner and accumulate utilities
        UW = utility_winner(utilities)
        utility_sums['UW'][n_cands] = utilities.sum(axis=0)[UW]

        for name, method in rated_methods.items():
            try:
                winner = method(utilities, tiebreaker='random')
                utility_sums[name][n_cands] = utilities.sum(axis=0)[winner]
            except ValueError:
                # Skip junk cases like vote-for-2 with 2 candidates
                utility_sums[name][n_cands] = np.nan

        rankings = honest_rankings(utilities)
        for name, method in ranked_methods.items():
            winner = method(rankings, tiebreaker='random')
            utility_sums[name][n_cands] = utilities.sum(axis=0)[winner]

    return utility_sums


print(f'Doing {n_elections:,} elections (tasks), {n_voters:,} voters, '
      f'{n_cands_list} candidates')
p = Parallel(n_jobs=-3, verbose=5)(
    delayed(simulate_election)() for _ in range(n_elections)
)

for result in p:
    for method, d in result.items():
        for n_cands, value in d.items():
            utility_sums[method][n_cands] += value

plt.figure(f'Effectiveness, {n_voters} voters, {n_elections} elections')
plt.title('The Effectiveness of Several Voting Systems')
for name, method in (('Standard', eff_standard),
                     ('Vote-for-1', lambda m: eff_vote_for_k(m, 1)),
                     ('Vote-for-2', lambda m: eff_vote_for_k(m, 2)),
                     ('Vote-for-3', lambda m: eff_vote_for_k(m, 3)),
                     ('Vote-for-4', lambda m: eff_vote_for_k(m, 4)),
                     ('Vote-for-half', lambda m: eff_vote_for_half(m)),
                     ('Vote-for-(n-1)', lambda m: eff_vote_for_k(m, -1)),
                     ):
    eff = method(np.array(n_cands_list))*100
    plt.plot(n_cands_list[eff != 0], eff[eff != 0], ':', lw=0.8)

# Restart color cycle, so result colors match
plt.gca().set_prop_cycle(None)

table = {}

# Calculate Social Utility Efficiency from summed utilities
x_uw, y_uw = zip(*sorted(utility_sums['UW'].items()))
average_utility = n_voters * n_elections / 2
for method in ('Standard', 'Vote-for-1', 'Vote-for-2', 'Vote-for-3',
               'Vote-for-4', 'Vote-for-half', 'Vote-for-(n-1)'):
    x, y = zip(*sorted(utility_sums[method].items()))
    SUE = (np.array(y) - average_utility)/(np.array(y_uw) - average_utility)
    plt.plot(x, SUE*100, '-', label=method)
    table[method.split('-')[-1]] = SUE*100

print(tabulate(table, 'keys', showindex=[str(x) for x in n_cands_list],
               tablefmt="pipe", floatfmt='.1f'))

plt.plot([], [], 'k:', lw=0.8, label='Weber')  # Dummy plot for label
plt.legend()
plt.grid(True, color='0.7', linestyle='-', which='major', axis='both')
plt.grid(True, color='0.9', linestyle='-', which='minor', axis='both')
plt.ylim(45, 85)
plt.xlim(2, 10)
plt.show()
