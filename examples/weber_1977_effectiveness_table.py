"""
Reproduce table: "The Effectiveness of Several Voting Systems".

from p. 19 of "Reproducing Voting Systems".

Weber, Robert J. (1978). "Comparison of Public Choice Systems".
Cowles Foundation Discussion Papers. Cowles Foundation for Research in
Economics. No. 498. https://cowles.yale.edu/publications/cfdp/cfdp-498

Typical result with n = 100_000:

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
# TODO: Best Vote-for-or-against-k is not implemneted yet
import time
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from joblib import Parallel, delayed
from elsim.methods import (utility_winner,
                           approval)
from elsim.elections import random_utilities
from elsim.strategies import (vote_for_or_against_k)
from weber_1977_expressions import eff_vote_for_or_against_k

n = 20_000
n_voters = 1_000 # Increased voters for stability
n_cands_list = (2, 3, 4, 5, 6, 10) # 10 is key

# We will dynamically run Vfoak-k inside the loop

count = defaultdict(lambda: defaultdict(int)) # accumulated utility
# We also need to track social utility winner (UW) for normalization

print(f'Doing {n:,} iterations, {n_voters:,} voters, '
      f'{n_cands_list} candidates')

start_time = time.monotonic()


def func():
    # Local accumulation for this batch
    batch_count = defaultdict(lambda: defaultdict(float))
    for n_cands in n_cands_list:
        utilities = random_utilities(n_voters, n_cands)

        # Find the social utility winner and accumulate utilities
        UW = utility_winner(utilities)
        batch_count['UW'][n_cands] = utilities.sum(axis=0)[UW]

        # VFOAK: iterate all valid k
        max_k = n_cands // 2
        for k in range(1, max_k + 1):
            try:
                election = vote_for_or_against_k(utilities, k)
                winner = approval(election, tiebreaker='random')
                batch_count[f'Vfoak-{k}'][n_cands] = utilities.sum(axis=0)[winner]
            except ValueError:
                pass

    return batch_count


p = Parallel(n_jobs=-3, verbose=5)(delayed(func)() for i in range(n))

for result in p:
    for method, d in result.items():
        for n_cands, value in d.items():
            count[method][n_cands] += value


elapsed_time = time.monotonic() - start_time
print('Elapsed:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)), '\n')

plt.figure(f'Effectiveness, {n_voters} voters, {n} iterations')
plt.title('The Effectiveness of Several Voting Systems')

# Restart color cycle, so result colors match
plt.gca().set_prop_cycle(None)

table = {}

# Calculate Social Utility Efficiency from summed utilities
x_uw, y_uw = zip(*sorted(count['UW'].items()))
average_utility = n_voters * n / 2

# Calculate Best Vote-for-or-against-k SUE
best_vfoak_sue = []
sorted_n_cands = sorted(n_cands_list)
print("\nDebug Best VFOAK selection:")
for n_cands in sorted_n_cands:
    max_util = -np.inf
    best_k = -1

    # Iterate valid k for this n_cands
    max_k = n_cands // 2
    for k in range(1, max_k + 1):
        method_name = f'Vfoak-{k}'
        if n_cands in count[method_name]:
             if count[method_name][n_cands] > max_util:
                 max_util = count[method_name][n_cands]
                 best_k = k

    if max_util == -np.inf:
        # Should not happen if k=1 is always valid for m>=2
        best_vfoak_sue.append(np.nan)
    else:
        uw_util = count['UW'][n_cands]
        sue = (max_util - average_utility)/(uw_util - average_utility)
        best_vfoak_sue.append(sue * 100)
        # Debug output
        if n_cands in (3, 4, 10):
            print(f"  m={n_cands}: Best k={best_k} (SUE={sue*100:.2f}%)")

table['Best Vote-for-or-against-k'] = best_vfoak_sue

print(tabulate(table, 'keys', showindex=n_cands_list,
               tablefmt="pipe", floatfmt='.2f'))

plt.plot([], [], 'k:', lw=0.8, label='Weber')  # Dummy plot for label
plt.legend()
plt.grid(True, color='0.7', linestyle='-', which='major', axis='both')
plt.grid(True, color='0.9', linestyle='-', which='minor', axis='both')
plt.ylim(40, 102)
plt.xlim(1.8, 10.2)
