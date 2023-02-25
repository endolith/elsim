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
import numpy as np
from tabulate import tabulate
from elsim.methods import (fptp, runoff, irv, approval, borda, coombs,
                           black, utility_winner, condorcet)
from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.strategies import honest_rankings, approval_optimal

n_elections = 10_000  # Roughly 60 seconds
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

    condorcet_winner_count = Counter()

    for iteration in range(n_elections):
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

        # If there is a Condorcet winner, analyze election, otherwise skip it
        CW = condorcet(rankings)
        if CW is not None:
            condorcet_winner_count['CW'] += 1

            for name, method in ranked_methods.items():
                if method(rankings, tiebreaker='random') == CW:
                    condorcet_winner_count[name] += 1

            for name, method in rated_methods.items():
                if method(utilities, tiebreaker='random') == CW:
                    condorcet_winner_count[name] += 1

    results.append(condorcet_winner_count)

elapsed_time = time.monotonic() - start_time
print('Elapsed:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)), '\n')

# Neither Tabulate nor Markdown support column span or multiple headers, but
# at least this prints to plain text in a readable way.
header = ['Disp\nCorr\nDims'] + [f'{x}\n{y}\n{z}' for x, y, z in conditions]

# Of those elections with CW, likelihood that method chooses CW
table = []
y_cw = np.array([c['CW'] for c in results])
for method in ('Plurality', 'Runoff', 'Hare', 'Approval', 'Borda', 'Coombs',
               'Black', 'SU max'):
    y = np.array([c[method] for c in results])
    table.append([method, *(y/y_cw*100)])

# Likelihood of Condorcet Winner (normalized by n elections)
table.append(['CW', *(y_cw / n_elections * 100)])

print(tabulate(table, header, tablefmt="pipe", floatfmt='.1f'))
