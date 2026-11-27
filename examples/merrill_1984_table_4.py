"""
Reproduce Table 4

Social Utility Efficiencies under […] Spatial Model Assumptions
(201 voters, 5 candidates)

from

S. Merrill III, "A Comparison of Efficiency of Multicandidate
Electoral Systems", American Journal of Political Science, vol. 28,
no. 1, pp. 23-48, 1984.  :doi:`10.2307/2110786`

(Not including random society column)

Typical result with 100_000 elections:

| Disp      |   1.0 |   1.0 |   1.0 |   1.0 |   0.5 |   0.5 |   0.5 |   0.5 |
| Corr      |   0.5 |   0.5 |   0.0 |   0.0 |   0.5 |   0.5 |   0.0 |   0.0 |
| Dims      |     2 |     4 |     2 |     4 |     2 |     4 |     2 |     4 |
|:----------|------:|------:|------:|------:|------:|------:|------:|------:|
| Plurality |  73.3 |  79.3 |  81.0 |  92.7 |   4.5 |   7.8 |  27.3 |  54.5 |
| Runoff    |  90.9 |  94.4 |  92.3 |  97.8 |  38.3 |  45.5 |  55.2 |  76.3 |
| Hare      |  92.2 |  94.6 |  94.6 |  98.6 |  47.9 |  59.1 |  59.9 |  84.7 |
| Approval  |  96.1 |  96.9 |  96.7 |  98.5 |  95.7 |  96.9 |  95.9 |  97.9 |
| Borda     |  97.9 |  98.7 |  98.3 |  99.4 |  96.7 |  97.9 |  97.3 |  98.9 |
| Coombs    |  97.2 |  97.6 |  97.9 |  98.8 |  94.6 |  94.9 |  95.5 |  97.0 |
| Black     |  97.5 |  98.1 |  98.3 |  99.1 |  96.1 |  96.7 |  97.0 |  98.3 |

Each voter's range-normalized utilities are then standardized to mean 0 and
standard deviation 1, matching Merrill's spatial-model procedure.

Many of these values match the paper closely, but some are consistently off by
up to 9%.
"""
import time
from collections import Counter
from random import randint

import numpy as np
from tabulate import tabulate

from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.methods import (approval, black, borda, coombs, fptp, irv, runoff,
                           utility_winner)
from elsim.strategies import approval_optimal, honest_rankings

n_elections = 10_000  # Roughly 60 seconds on a 2019 6-core i7-9750H
n_voters = 201
n_cands = 5

ranked_methods = {'Plurality': fptp, 'Runoff': runoff, 'Hare': irv,
                  'Borda': borda, 'Coombs': coombs, 'Black': black}

rated_methods = {'SU max': utility_winner,
                 'Approval': lambda utilities, tiebreaker:
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

    utility_sums = Counter()

    for _iteration in range(n_elections):
        v, c = normal_electorate(n_voters, n_cands, dims=D, corr=corr,
                                 disp=disp)

        # "Simulated utilities were normalized by range, that is, each
        # voter's set of utilities were linearly expanded so that the highest
        # and lowest utilities for each voter were 1 and 0, respectively."
        # Then standardize each voter's utilities to mean 0 and standard
        # deviation 1, as Merrill did for the spatial-model simulations.
        utilities = normed_dist_utilities(v, c)
        utilities = ((utilities - utilities.mean(axis=1, keepdims=True))
                     / utilities.std(axis=1, keepdims=True))
        rankings = honest_rankings(utilities)

        # Pick a random winner and accumulate utilities
        RW = randint(0, n_cands - 1)
        utility_sums['RW'] += utilities.sum(axis=0)[RW]

        for name, method in rated_methods.items():
            winner = method(utilities, tiebreaker='random')
            utility_sums[name] += utilities.sum(axis=0)[winner]

        for name, method in ranked_methods.items():
            winner = method(rankings, tiebreaker='random')
            utility_sums[name] += utilities.sum(axis=0)[winner]

    results.append(utility_sums)

elapsed_time = time.monotonic() - start_time
print('Elapsed:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)), '\n')

# Neither Tabulate nor Markdown support column span or multiple headers, but
# at least this prints to plain text in a readable way.
header = ['Disp\nCorr\nDims'] + [f'{x}\n{y}\n{z}' for x, y, z in conditions]

# Calculate Social Utility Efficiency from summed utilities
y_uw = np.array([c['SU max'] for c in results])
y_rw = np.array([c['RW'] for c in results])

table = []
for method in ('Plurality', 'Runoff', 'Hare', 'Approval', 'Borda', 'Coombs',
               'Black'):
    y = np.array([c[method] for c in results])
    SUE = (y - y_rw)/(y_uw - y_rw)
    table.append([method, *(SUE*100)])
print(tabulate(table, header, tablefmt="pipe", floatfmt='.1f'))
