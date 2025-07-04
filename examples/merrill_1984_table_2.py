"""
Reproduce Table 2

Condorcet Efficiency under Impartial Culture Assumptions

Results with 1_000_000 elections:

| Candidates |   Plurality |   Runoff |   Hare |   Approval |   Borda |   Coombs |   Black |
|-----------:|------------:|---------:|-------:|-----------:|--------:|---------:|--------:|
|          3 |        91.2 |     93.8 |   95.9 |       85.2 |    98.9 |     94.3 |   100.0 |
|          4 |        79.0 |     79.5 |   82.2 |       73.1 |    97.2 |     84.6 |   100.0 |
|          5 |        68.6 |     67.4 |   69.7 |       62.8 |    95.0 |     74.5 |   100.0 |
|          6 |        59.7 |     56.8 |   58.7 |       54.1 |    92.4 |     65.3 |   100.0 |
|          7 |        52.2 |     47.9 |   49.4 |       46.5 |    89.4 |     57.1 |   100.0 |

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

from elsim.elections import random_utilities
from elsim.methods import (approval, black, borda, condorcet, coombs, fptp, irv,
                           runoff, utility_winner)
from elsim.strategies import approval_optimal, honest_rankings

n_elections = 50_000  # Roughly 60 seconds on a 2019 6-core i7-9750H
n_voters = 201
n_cands_list = (3, 4, 5, 6, 7)

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
