"""
Reproduce Table 2

Condorcet Efficiencies under […] Spatial Model Assumptions
(201 voters, 5 candidates)

from

S. Merrill III, "A Comparison of Efficiency of Multicandidate
Electoral Systems", American Journal of Political Science, vol. 28,
no. 1, pp. 23-48, 1984.  :doi:`10.2307/2110786`

(Not including random society column)

Typical result:

| Disp      |   1.0 |   1.0 |   1.0 |   1.0 |   0.5 |   0.5 |   0.5 |   0.5 |
| Corr      |   0.5 |   0.5 |   0.0 |   0.0 |   0.5 |   0.5 |   0.0 |   0.0 |
| Dims      |     2 |     4 |     2 |     4 |     2 |     4 |     2 |     4 |
|:----------|------:|------:|------:|------:|------:|------:|------:|------:|
| Plurality |  57.5 |  65.8 |  62.2 |  78.4 |  21.7 |  24.4 |  27.2 |  41.3 |
| Runoff    |  80.1 |  87.3 |  81.6 |  93.6 |  35.4 |  42.2 |  41.5 |  61.5 |
| Hare      |  79.2 |  86.7 |  84.0 |  95.4 |  35.9 |  46.8 |  41.0 |  69.9 |
| Approval  |  73.8 |  77.8 |  76.9 |  85.4 |  71.5 |  76.4 |  73.8 |  82.7 |
| Borda     |  87.1 |  89.3 |  88.2 |  92.3 |  83.7 |  86.3 |  85.2 |  89.4 |
| Coombs    |  97.8 |  97.3 |  97.9 |  98.2 |  93.5 |  92.3 |  93.8 |  94.5 |
| Black     | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| SU max    |  82.9 |  85.8 |  85.3 |  90.8 |  78.1 |  81.5 |  80.8 |  87.1 |
| CW        |  99.7 |  99.7 |  99.7 |  99.6 |  98.9 |  98.6 |  98.7 |  98.5 |

Many of these values match the paper closely, but some are consistently off by
up to 4%.
"""
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tabulate import tabulate

from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.methods import (approval, black, borda, condorcet, coombs, fptp,
                           irv, runoff, utility_winner)
from elsim.strategies import approval_optimal, honest_rankings

n_elections = 10_000  # Roughly 60 seconds on a 2019 6-core i7-9750H
n_voters = 201
n_cands = 5

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
    condorcet_winner_count = {key: Counter() for key in (
        ranked_methods.keys() | rated_methods.keys() | {'CW'})}
    
    for iteration in range(batch_size):
        for n_cands in n_cands_list:
            utilities = random_utilities(n_voters, n_cands)
            rankings = honest_rankings(utilities)

            # If there is a Condorcet winner, analyze election, otherwise skip
            # it
            CW = condorcet(rankings)
            if CW is not None:
                condorcet_winner_count['CW'][n_cands] += 1

                for name, method in ranked_methods.items():
                    if method(rankings, tiebreaker='random') == CW:
                        condorcet_winner_count[name][n_cands] += 1

                for name, method in rated_methods.items():
                    if method(utilities, tiebreaker='random') == CW:
                        condorcet_winner_count[name][n_cands] += 1
    
    return condorcet_winner_count


start_time = time.monotonic()

jobs = [delayed(simulate_batch)()] * n_batches
print(f'{len(jobs)} tasks total:')
results = Parallel(n_jobs=-3, verbose=5)(jobs)

# Merge results from all batches
condorcet_winner_count = {key: Counter() for key in (
    ranked_methods.keys() | rated_methods.keys() | {'CW'})}
for result in results:
    for method in condorcet_winner_count:
        for n_cands in n_cands_list:
            condorcet_winner_count[method][n_cands] += result[method][n_cands]

elapsed_time = time.monotonic() - start_time
print('Elapsed:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)), '\n')

plt.figure(f'Condorcet efficiency, {n_voters} voters, {n_elections} elections')
plt.title('Condorcet Efficiency under Impartial Culture Assumptions')

table = []

# Of those elections with CW, likelihood that method chooses CW
x_cw, y_cw = zip(*sorted(condorcet_winner_count['CW'].items()))
for method in ('Plurality', 'Runoff', 'Hare', 'Approval', 'Borda', 'Coombs',
               'Black'):
    x, y = zip(*sorted(condorcet_winner_count[method].items()))
    CE = np.array(y)/y_cw
    plt.plot(x, CE*100, '-', label=method)
    table.append([method, *CE*100])

print(tabulate(table, ["Candidates", *x], tablefmt="pipe", floatfmt='.1f'))

plt.legend()
plt.grid(True, color='0.7', linestyle='-', which='major', axis='both')
plt.grid(True, color='0.9', linestyle='-', which='minor', axis='both')
plt.xlim(2.8, 7.2)
plt.show()
