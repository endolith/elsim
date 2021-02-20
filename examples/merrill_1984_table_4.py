"""
Reproduce Table 4

Social Utility Efficiencies under Random Society and
Spatial Model Assumptions
(201 voters, 5 candidates)

from

S. Merrill III, "A Comparison of Efficiency of Multicandidate
Electoral Systems", American Journal of Political Science, vol. 28,
no. 1, pp. 23-48, 1984.  :doi:`10.2307/2110786`

Typical result with 100_000 simulations:

| Disp      |   1.0 |   1.0 |   1.0 |   1.0 |   0.5 |   0.5 |   0.5 |   0.5 |
| Corr      |   0.5 |   0.5 |   0.0 |   0.0 |   0.5 |   0.5 |   0.0 |   0.0 |
| Dims      |     2 |     4 |     2 |     4 |     2 |     4 |     2 |     4 |
|:----------|------:|------:|------:|------:|------:|------:|------:|------:|
| Plurality |  72.1 |  79.1 |  80.4 |  92.4 |   4.0 |   6.3 |  25.2 |  52.9 |
| Runoff    |  90.5 |  94.2 |  92.0 |  97.5 |  36.6 |  43.6 |  53.3 |  75.3 |
| Hare      |  91.7 |  94.7 |  94.3 |  98.4 |  46.4 |  57.7 |  58.7 |  83.6 |
| Approval  |  96.2 |  97.0 |  96.8 |  98.5 |  95.6 |  96.8 |  95.8 |  98.0 |
| Borda     |  97.8 |  98.6 |  98.3 |  99.4 |  96.6 |  97.7 |  97.4 |  99.0 |
| Coombs    |  97.0 |  97.5 |  97.7 |  98.7 |  94.0 |  94.3 |  95.0 |  96.7 |
| Black     |  97.3 |  97.8 |  98.0 |  99.0 |  95.5 |  96.1 |  96.5 |  98.0 |

Many of these values match the paper closely, but some are consistently off by
up to 9%.
"""
import time
from collections import Counter
from random import randint
import numpy as np
from tabulate import tabulate
from elsim.methods import (fptp, runoff, irv, approval, borda, coombs,
                           black, utility_winner)
from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.strategies import honest_rankings, approval_optimal

n = 10_000
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

    count = Counter()

    for iteration in range(n):
        v, c = normal_electorate(n_voters, n_cands, dims=D, corr=corr,
                                 disp=disp)

        """
        "Simulated utilities were normalized by range, that is, each voter's
        set of utilities were linearly expanded so that the highest and lowest
        utilities for each voter were 1 and 0, respectively."

        TODO: standard scores vs normalized don't matter for the ranked systems
        and don't affect approval much

        but This is necessary for the SU Maximizer results to match Merrill's.
        """
        utilities = normed_dist_utilities(v, c)
        rankings = honest_rankings(utilities)

        # Pick a random winner and accumulate utilities
        RW = randint(0, n_cands - 1)
        count['RW'] += utilities.sum(axis=0)[RW]

        for name, func in rated_methods.items():
            winner = func(utilities, tiebreaker='random')
            count[name] += utilities.sum(axis=0)[winner]

        for name, func in ranked_methods.items():
            winner = func(rankings, tiebreaker='random')
            count[name] += utilities.sum(axis=0)[winner]

    results.append(count)

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
