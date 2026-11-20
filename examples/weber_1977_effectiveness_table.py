"""
Reproduce table: "The Effectiveness of Several Voting Systems".

from p. 19 of "Reproducing Voting Systems".

Weber, Robert J. (1978). "Comparison of Public Choice Systems".
Cowles Foundation Discussion Papers. Cowles Foundation for Research in
Economics. No. 498. https://cowles.yale.edu/publications/cfdp/cfdp-498

Typical result with n_elections = 100_000:

|     |   Standard |   Vote-for-half |   Borda |
|----:|-----------:|----------------:|--------:|
|   2 |      81.37 |           81.71 |   81.41 |
|   3 |      75.10 |           75.00 |   86.53 |
|   4 |      69.90 |           79.92 |   89.47 |
|   5 |      65.02 |           79.09 |   91.34 |
|   6 |      61.08 |           81.20 |   92.61 |
|  10 |      50.78 |           82.94 |   95.35 |
| 255 |      12.78 |           86.37 |   99.80 |
"""
# TODO: Standard is consistently ~1% high, while Borda is very accurate
# TODO: Best Vote-for-or-against-k is not implemented yet
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from weber_1977_expressions import eff_borda, eff_standard, eff_vote_for_half

from elsim.elections import random_utilities
from elsim.methods import approval, borda, fptp
from elsim.strategies import honest_rankings, vote_for_k
from elsim.studies import random_society_utility_updates

n_elections = 2_000  # Roughly 60 seconds on a 2019 6-core i7-9750H
n_voters = 1_000
n_cands_list = (2, 3, 4, 5, 6, 10, 255)

ranked_methods = {'Standard': fptp, 'Borda': borda}

rated_methods = {'Vote-for-half': lambda utilities, tiebreaker:
                 approval(vote_for_k(utilities, 'half'), tiebreaker)}

utility_sums = {key: Counter() for key in (ranked_methods.keys() |
                                           rated_methods.keys() | {'UW'})}

start_time = time.monotonic()

for _ in range(n_elections):
    for n_cands in n_cands_list:
        utilities = random_utilities(n_voters, n_cands)

        rankings = honest_rankings(utilities)

        delta = random_society_utility_updates(
            utilities, rankings, ranked_methods, rated_methods,
            tiebreaker='random',
            uw_key='UW',
            utility_winner_tiebreaker=None,
        )
        for name, value in delta.items():
            utility_sums[name][n_cands] += value

elapsed_time = time.monotonic() - start_time
print('Elapsed:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)), '\n')

plt.figure(f'Effectiveness, {n_voters} voters, {n_elections} elections')
plt.title('The Effectiveness of Several Voting Systems')
for name, method in (('Standard', eff_standard),
                     ('Vote-for-half', eff_vote_for_half),
                     ('Borda', eff_borda)):
    plt.plot(n_cands_list, method(np.array(n_cands_list))*100, ':', lw=0.8)

# Restart color cycle, so result colors match
plt.gca().set_prop_cycle(None)

table = {}

# Calculate Social Utility Efficiency from summed utilities
x_uw, y_uw = zip(*sorted(utility_sums['UW'].items()))
average_utility = n_voters * n_elections / 2
for method in ('Standard', 'Vote-for-half', 'Borda'):
    x, y = zip(*sorted(utility_sums[method].items()))
    SUE = (np.array(y) - average_utility)/(np.array(y_uw) - average_utility)
    plt.plot(x, SUE*100, '-', label=method)
    table.update({method: SUE*100})

print(tabulate(table, 'keys', showindex=n_cands_list,
               tablefmt="pipe", floatfmt='.2f'))

plt.plot([], [], 'k:', lw=0.8, label='Weber')  # Dummy plot for label
plt.legend()
plt.grid(True, color='0.7', linestyle='-', which='major', axis='both')
plt.grid(True, color='0.9', linestyle='-', which='minor', axis='both')
plt.ylim(40, 102)
plt.xlim(1.8, 10.2)
plt.show()
