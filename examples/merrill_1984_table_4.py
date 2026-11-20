"""
Reproduce Table 4

Social Utility Efficiencies under […] Spatial Model Assumptions
(201 voters, 5 candidates)

from

S. Merrill III, "A Comparison of Efficiency of Multicandidate
Electoral Systems", American Journal of Political Science, vol. 28,
no. 1, pp. 23-48, 1984.  :doi:`10.2307/2110786`

(Not including random society column)

Typical result with 10_000 elections (2_000 candidate sets x 5 electorates):

| Disp      |   1.0 |   1.0 |   1.0 |   1.0 |   0.5 |   0.5 |   0.5 |   0.5 |
| Corr      |   0.5 |   0.5 |   0.0 |   0.0 |   0.5 |   0.5 |   0.0 |   0.0 |
| Dims      |     2 |     4 |     2 |     4 |     2 |     4 |     2 |     4 |
|:----------|------:|------:|------:|------:|------:|------:|------:|------:|
| Plurality |  68.1 |  75.9 |  77.5 |  90.8 |   3.4 |   7.4 |  24.4 |  50.7 |
| Runoff    |  88.8 |  92.9 |  90.0 |  96.6 |  36.2 |  44.4 |  52.7 |  74.1 |
| Hare      |  90.2 |  93.3 |  93.0 |  97.7 |  43.2 |  56.6 |  55.8 |  81.9 |
| Approval  |  95.7 |  97.0 |  96.5 |  98.2 |  95.3 |  96.5 |  95.0 |  97.4 |
| Borda     |  97.6 |  98.3 |  97.8 |  99.3 |  96.2 |  97.5 |  96.8 |  98.6 |
| Coombs    |  96.3 |  97.0 |  97.3 |  98.2 |  93.4 |  93.8 |  94.5 |  95.8 |
| Black     |  97.0 |  97.5 |  97.8 |  98.8 |  95.5 |  96.3 |  96.5 |  97.9 |

Merrill computed social-utility efficiency for each candidate set (five
electorates) and then averaged over 200 sets.  This script now follows that
design; using 2,000 sets reduces the Monte Carlo error.  The paper's 200-set
values carry several percentage points of error, so the remaining differences
are not meaningful.
"""
import random
import time

import numpy as np
from tabulate import tabulate

from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.methods import (approval, black, borda, coombs, fptp, irv, runoff)
from elsim.strategies import approval_optimal, honest_rankings

n_elections = 10_000  # Roughly 60 seconds on a 2019 6-core i7-9750H
n_voters = 201
n_cands = 5
n_electorates_per_set = 5
n_candidate_sets = n_elections // n_electorates_per_set
assert n_candidate_sets * n_electorates_per_set == n_elections

seed = 1984
random.seed(seed)
rng = np.random.default_rng(seed)

ranked_methods = {'Plurality': fptp, 'Runoff': runoff, 'Hare': irv,
                  'Borda': borda, 'Coombs': coombs, 'Black': black}

rated_methods = {'Approval': lambda utilities, tiebreaker:
                 approval(approval_optimal(utilities), tiebreaker)}

start_time = time.monotonic()

#             disp, corr, D
conditions = ((1.0, 0.5, 2),
              (1.0, 0.5, 4),
              (1.0, 0.0, 2),
              (1.0, 0.0, 4),
              (0.5, 0.5, 2),
              (0.5, 0.5, 4),
              (0.5, 0.0, 2),
              (0.5, 0.0, 4),
              )

results = []

for disp, corr, D in conditions:
    print(disp, corr, D)

    per_set_sue = {name: [] for name in (*rated_methods, *ranked_methods)}

    for _set in range(n_candidate_sets):
        # Merrill combines each candidate set with five fresh electorates.
        _, c = normal_electorate(0, n_cands, dims=D, corr=corr, disp=disp,
                                 random_state=rng)
        utility_sums = {name: 0.0 for name in per_set_sue}
        max_utility_sum = 0.0
        random_utility_sum = 0.0

        for _ in range(n_electorates_per_set):
            v, _ = normal_electorate(n_voters, n_cands, dims=D, corr=corr,
                                     disp=disp, random_state=rng)

            """
            Simulated utilities were normalized by range, i.e. each voter's
            utilities were linearly expanded so that their highest and lowest
            utilities were 1 and 0.

            For the spatial model, Merrill used standard scores instead:
            each voter's utilities have mean 0 and standard deviation 1.
            """
            utilities = normed_dist_utilities(v, c)
            utilities = ((utilities - utilities.mean(axis=1, keepdims=True))
                         / utilities.std(axis=1, keepdims=True))
            rankings = honest_rankings(utilities)
            utility_sum = utilities.sum(axis=0)

            max_utility_sum += utility_sum.max()
            random_utility_sum += utility_sum.mean()

            for name, method in rated_methods.items():
                winner = method(utilities, tiebreaker='random')
                utility_sums[name] += utility_sum[winner]

            for name, method in ranked_methods.items():
                winner = method(rankings, tiebreaker='random')
                utility_sums[name] += utility_sum[winner]

        # Merrill computed social-utility efficiency for each candidate set
        # (its five electorates combined) and then averaged over the sets.
        denominator = max_utility_sum - random_utility_sum
        for name in per_set_sue:
            per_set_sue[name].append(
                (utility_sums[name] - random_utility_sum)
                / denominator * 100)

    results.append({name: np.mean(values)
                    for name, values in per_set_sue.items()})

elapsed_time = time.monotonic() - start_time
print('Elapsed:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)), '\n')

# Neither Tabulate nor Markdown support column span or multiple headers, but
# at least this prints to plain text in a readable way.
header = ['Disp\nCorr\nDims'] + [f'{x}\n{y}\n{z}' for x, y, z in conditions]

table = []
for method in ('Plurality', 'Runoff', 'Hare', 'Approval', 'Borda', 'Coombs',
               'Black'):
    y = np.array([c[method] for c in results])
    table.append([method, *y])
print(tabulate(table, header, tablefmt="pipe", floatfmt='.1f'))
