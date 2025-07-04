"""
Reproduce Table 3 and Figure 3

Social Utility Efficiencies under Spatial-Model Assumptions
(201 voters, 5 candidates, two dimensions, correlation = .5)

Results with 1_000_000 elections:

| Dispersion |   Plurality |   Runoff |   Hare |   Approval |   Borda |   Coombs |   Black |
|-----------:|------------:|---------:|-------:|-----------:|--------:|---------:|--------:|
|        0.5 |        84.7 |     84.9 |   84.0 |       90.1 |    91.1 |     97.8 |    97.8 |
|        1.0 |        72.9 |     73.1 |   70.8 |       83.3 |    83.7 |     95.6 |    95.6 |
|        2.0 |        57.9 |     58.1 |   54.6 |       73.3 |    73.5 |     91.6 |    91.6 |

from

S. Merrill III, "A Comparison of Efficiency of Multicandidate
Electoral Systems", American Journal of Political Science, vol. 28,
no. 1, pp. 23-48, 1984.  :doi:`10.2307/2110786`
"""
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tabulate import tabulate

from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.methods import (approval, black, borda, coombs, fptp, irv, runoff,
                           utility_winner)
from elsim.strategies import approval_optimal, honest_rankings

n_elections = 50_000  # Roughly 30 seconds on a 2019 6-core i7-9750H
n_voters = 201
n_cands = 5
corr = 0.5
D = 2
disp_list = (0.5, 1.0, 2.0)

# Simulate more than just one election per worker to improve efficiency
batch_size = 100
n_batches = n_elections // batch_size
assert n_batches * batch_size == n_elections

ranked_methods = {'Plurality': fptp, 'Runoff': runoff, 'Hare': irv,
                  'Borda': borda, 'Coombs': coombs, 'Black': black}

rated_methods = {'SU max': utility_winner,
                 'Approval': lambda utilities, tiebreaker:
                     approval(approval_optimal(utilities), tiebreaker)}


def simulate_batch():
    utility_sums = {key: Counter() for key in (ranked_methods.keys() |
                                               rated_methods.keys())}
    
    for iteration in range(batch_size):
        for disp in disp_list:
            v, c = normal_electorate(n_voters, n_cands, dims=D, corr=corr,
                                     disp=disp)
            utilities = normed_dist_utilities(v, c)

            # Find the social utility winner and accumulate utilities
            UW = utility_winner(utilities)
            utility_sums['SU max'][disp] += utilities.sum(axis=0)[UW]

            for name, method in rated_methods.items():
                if name != 'SU max':  # Already handled above
                    winner = method(utilities, tiebreaker='random')
                    utility_sums[name][disp] += utilities.sum(axis=0)[winner]

            rankings = honest_rankings(utilities)
            for name, method in ranked_methods.items():
                winner = method(rankings, tiebreaker='random')
                utility_sums[name][disp] += utilities.sum(axis=0)[winner]
    
    return utility_sums


start_time = time.monotonic()

jobs = [delayed(simulate_batch)()] * n_batches
print(f'{len(jobs)} tasks total:')
results = Parallel(n_jobs=-3, verbose=5)(jobs)

# Merge results from all batches
utility_sums = {key: Counter() for key in (ranked_methods.keys() |
                                           rated_methods.keys())}
for result in results:
    for method in utility_sums:
        for disp in disp_list:
            utility_sums[method][disp] += result[method][disp]

elapsed_time = time.monotonic() - start_time
print('Elapsed:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)), '\n')

plt.figure(f'Utility ratios, {n_voters} voters, {n_cands} candidates, '
           f'{n_elections} elections')
plt.title('Social Utility Efficiencies under Spatial-Model Assumptions')

table = []

# Calculate Social Utility Efficiency from summed utilities
x_uw, y_uw = zip(*sorted(utility_sums['SU max'].items()))
random_utility = n_voters / 2  # Expected utility for random choice
for method in ('Plurality', 'Runoff', 'Hare', 'Approval', 'Borda', 'Coombs',
               'Black'):
    x, y = zip(*sorted(utility_sums[method].items()))
    utilities_per_election = np.array(y) / n_elections
    max_utilities_per_election = np.array(y_uw) / n_elections
    SUE = (utilities_per_election - random_utility) / (max_utilities_per_election - random_utility)
    plt.plot(x, SUE*100, '-', label=method)
    table.append([method, *SUE*100])

print(tabulate(table, ["Dispersion", *x], tablefmt="pipe", floatfmt='.1f'))

plt.legend()
plt.grid(True, color='0.7', linestyle='-', which='major', axis='both')
plt.grid(True, color='0.9', linestyle='-', which='minor', axis='both')
plt.xlim(0.4, 2.2)
plt.show()
