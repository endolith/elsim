"""
Reproduce Table 4 from p. 17 of Comparison of Voting Systems.

The expected social utility of the elected candidate, under three voting
systems.

from

Weber, Robert J. (1978). "Comparison of Public Choice Systems".
Cowles Foundation Discussion Papers. Cowles Foundation for Research in
Economics. No. 498. https://cowles.yale.edu/publications/cfdp/cfdp-498

Typical result with n_elections = 1_000_000:

|    |   Standard |   Borda |   Approval |
|---:|-----------:|--------:|-----------:|
|  2 |     1.2505 |  1.2922 |     1.2920 |
|  3 |     1.8330 |  1.8748 |     1.8644 |
|  4 |     2.3889 |  2.4236 |     2.4210 |
|  5 |     2.9171 |  2.9768 |     2.9727 |
| 10 |     5.5970 |  5.6701 |     5.6714 |
| 15 |     8.2248 |  8.3207 |     8.3246 |
| 20 |    10.8325 | 10.9470 |    10.9537 |
| 25 |    13.4320 | 13.5583 |    13.5659 |
| 30 |    16.0177 | 16.1577 |    16.1669 |

"""
import time
from collections import Counter

import numpy as np
from joblib import Parallel, delayed
from tabulate import tabulate

from elsim.elections import random_utilities
from elsim.methods import approval, borda, fptp
from elsim.strategies import approval_optimal, honest_rankings

n_elections = 30_000  # Roughly 30 seconds on a 2019 6-core i7-9750H
n_voters_list = (2, 3, 4, 5, 10, 15, 20, 25, 30)
n_cands = 3

# Simulate more than just one election per worker to improve efficiency
batch_size = 100
n_batches = n_elections // batch_size
assert n_batches * batch_size == n_elections

ranked_methods = {'Standard': fptp, 'Borda': borda}
rated_methods = {'Approval': lambda utilities, tiebreaker:
                 approval(approval_optimal(utilities), tiebreaker)}


def simulate_batch():
    utility_sums = {key: Counter() for key in (ranked_methods.keys() |
                                               rated_methods.keys())}
    
    for iteration in range(batch_size):
        for n_voters in n_voters_list:
            utilities = random_utilities(n_voters, n_cands)

            for name, method in rated_methods.items():
                winner = method(utilities, tiebreaker='random')
                utility_sums[name][n_voters] += utilities.sum(axis=0)[winner]

            rankings = honest_rankings(utilities)
            for name, method in ranked_methods.items():
                winner = method(rankings, tiebreaker='random')
                utility_sums[name][n_voters] += utilities.sum(axis=0)[winner]
    
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
        for n_voters in n_voters_list:
            utility_sums[method][n_voters] += result[method][n_voters]

elapsed_time = time.monotonic() - start_time
print('Elapsed:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)), '\n')

table = {}

# Calculate the expected utilities over all elections
for method in ('Standard', 'Borda', 'Approval'):
    x, y = zip(*sorted(utility_sums[method].items()))
    table.update({method: np.array(y) / n_elections})

print(tabulate(table, 'keys', showindex=n_voters_list,
               tablefmt="pipe", floatfmt='.4f'))
