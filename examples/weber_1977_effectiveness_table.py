"""
Reproduce table: "The Effectiveness of Several Voting Systems".

from p. 19 of "Reproducing Voting Systems".

Weber, Robert J. (1978). "Comparison of Public Choice Systems".
Cowles Foundation Discussion Papers. Cowles Foundation for Research in
Economics. No. 498. https://cowles.yale.edu/publications/cfdp/cfdp-498

Typical result with n = 100_000:

|     |   Standard |   Borda |
|----:|-----------:|--------:|
|   2 |      81.37 |   81.41 |
|   3 |      75.10 |   86.53 |
|   4 |      69.90 |   89.47 |
|   5 |      65.02 |   91.34 |
|   6 |      61.08 |   92.61 |
|  10 |      50.78 |   95.35 |
| 255 |      12.78 |   99.80 |
"""
# TODO: Standard is consistently ~1% high, while Borda is very accurate
import time
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from elsim.methods import (fptp, borda, utility_winner)
from elsim.elections import random_utilities
from elsim.strategies import honest_rankings
from weber_1977_expressions import eff_standard, eff_borda

n = 2_000
n_voters = 1000
n_cands_list = (2, 3, 4, 5, 6, 10, 255)

ranked_methods = {'Standard': fptp, 'Borda': borda}

count = {key: Counter() for key in (ranked_methods.keys() | {'UW'})}

start_time = time.monotonic()

for iteration in range(n):
    for n_cands in n_cands_list:
        utilities = random_utilities(n_voters, n_cands)

        # Find the social utility winner and accumulate utilities
        UW = utility_winner(utilities)
        count['UW'][n_cands] += utilities.sum(axis=0)[UW]

        rankings = honest_rankings(utilities)
        for name, method in ranked_methods.items():
            winner = method(rankings, tiebreaker='random')
            count[name][n_cands] += utilities.sum(axis=0)[winner]

elapsed_time = time.monotonic() - start_time
print('Elapsed:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)), '\n')


plt.figure(f'Effectiveness, {n_voters} voters, {n} iterations')
plt.title('The Effectiveness of Several Voting Systems')
for name, method in (('Standard', eff_standard), ('Borda', eff_borda)):
    plt.plot(n_cands_list, method(np.array(n_cands_list))*100, ':', lw=0.8)

# Restart color cycle, so result colors match
plt.gca().set_prop_cycle(None)

table = {}

# Calculate Social Utility Efficiency from summed utilities
x_uw, y_uw = zip(*sorted(count['UW'].items()))
average_utility = n_voters * n / 2
for method in ('Standard', 'Borda'):
    x, y = zip(*sorted(count[method].items()))
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