"""
Add modern voting methods to Figures 4.a and 4.b

Social Utility Efficiency under Spatial-Model Assumptions
(201 voters, two dimensions, correlation = .5, relative dispersion = .5 or 1.0)

from

S. Merrill III, "A Comparison of Efficiency of Multicandidate
Electoral Systems", American Journal of Political Science, vol. 28,
no. 1, pp. 23-48, 1984.  :doi:`10.2307/2110786`

Results with 100_000 elections:

4.a

| Method          |     2 |     3 |    4 |    5 |    6 |    7 |
|:----------------|------:|------:|-----:|-----:|-----:|-----:|
| Score           | 100.0 | 100.0 | 99.9 | 99.9 | 99.9 | 99.8 |
| STAR            | 100.0 |  97.5 | 97.8 | 98.3 | 98.6 | 98.8 |
| Borda           | 100.0 |  98.8 | 98.2 | 97.9 | 97.6 | 97.6 |
| Condorcet RCV   | 100.0 |  97.1 | 97.0 | 97.3 | 97.6 | 97.8 |
| Coombs          | 100.0 |  97.0 | 96.8 | 97.0 | 97.2 | 97.4 |
| Approval (opt.) | 100.0 |  98.7 | 97.3 | 96.2 | 95.5 | 95.2 |
| Hare RCV        | 100.0 |  94.2 | 92.5 | 91.6 | 91.0 | 90.4 |
| Top-2 Runoff    | 100.0 |  94.2 | 91.9 | 90.4 | 88.9 | 87.5 |
| Plurality       | 100.0 |  84.8 | 77.4 | 72.0 | 68.2 | 64.9 |

4.b

| Method          |     2 |     3 |    4 |    5 |     6 |     7 |
|:----------------|------:|------:|-----:|-----:|------:|------:|
| Score           | 100.0 | 100.0 | 99.9 | 99.7 |  99.5 |  99.3 |
| STAR            | 100.0 |  96.2 | 96.7 | 97.2 |  97.6 |  97.8 |
| Borda           | 100.0 |  97.9 | 97.1 | 96.6 |  96.4 |  96.3 |
| Condorcet RCV   | 100.0 |  95.5 | 95.2 | 95.5 |  95.8 |  96.2 |
| Coombs          | 100.0 |  95.0 | 94.2 | 94.1 |  94.0 |  94.1 |
| Approval (opt.) | 100.0 |  98.6 | 96.7 | 95.5 |  94.9 |  94.6 |
| Hare RCV        | 100.0 |  70.4 | 55.9 | 46.7 |  39.5 |  35.0 |
| Top-2 Runoff    | 100.0 |  70.4 | 51.8 | 37.3 |  23.9 |  13.6 |
| Plurality       | 100.0 |  50.5 | 23.9 |  4.7 | -12.2 | -24.7 |

The general trend is similar to Merrill's, but there are significant
discrepancies.  It is smoother, so maybe the original just had lower number of
simulations.
"""
import time
from collections import Counter
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from elsim.methods import (fptp, runoff, irv, approval, borda, coombs,
                           black, utility_winner, score, star)
from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.strategies import (honest_rankings, approval_optimal,
                              honest_normed_scores)

n_elections = 5_000  # Roughly 30 seconds each
n_voters = 201
n_cands_list = (2, 3, 4, 5, 6, 7)
corr = 0.5
D = 2

ranked_methods = {'Plurality': fptp, 'Top-2 Runoff': runoff, 'Hare RCV': irv,
                  'Borda': borda, 'Coombs': coombs, 'Condorcet RCV': black}

rated_methods = {'SU max': utility_winner,
                 'Approval (opt.)': lambda utilities, tiebreaker:
                     approval(approval_optimal(utilities), tiebreaker),
                     'Score': lambda utilities, tiebreaker:
                         score(honest_normed_scores(utilities, 5),
                               tiebreaker),
                     'STAR': lambda utilities, tiebreaker:
                         star(honest_normed_scores(utilities, 5),
                              tiebreaker),
                 }

for fig, disp, ymin in (('4.a', 1.0, 55),
                        ('4.b', 0.5, 0)):

    utility_sums = {key: Counter() for key in (ranked_methods.keys() |
                                               rated_methods.keys() |
                                               {'SU max', 'RW'})}
    start_time = time.monotonic()

    for iteration in range(n_elections):
        for n_cands in n_cands_list:
            v, c = normal_electorate(n_voters, n_cands, dims=D, corr=corr,
                                     disp=disp)
            utilities = normed_dist_utilities(v, c)
            rankings = honest_rankings(utilities)

            # Pick a random winner and accumulate utilities
            RW = randint(0, n_cands - 1)
            utility_sums['RW'][n_cands] += utilities.sum(axis=0)[RW]

            for name, method in rated_methods.items():
                winner = method(utilities, tiebreaker='random')
                utility_sums[name][n_cands] += utilities.sum(axis=0)[winner]

            for name, method in ranked_methods.items():
                winner = method(rankings, tiebreaker='random')
                utility_sums[name][n_cands] += utilities.sum(axis=0)[winner]

    elapsed_time = time.monotonic() - start_time
    print('Elapsed:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),
          '\n')

    plt.figure(f'Figure {fig}. {n_voters} voters, {n_elections} elections',
               figsize=(8, 6.5))
    plt.title(f'Figure {fig}: Social Utility Efficiency under Spatial-Model '
              f'Assumptions [Disp: {disp}]')

    table = []

    # Calculate Social Utility Efficiency from summed utilities
    x_uw, y_uw = zip(*sorted(utility_sums['SU max'].items()))
    x_rw, y_rw = zip(*sorted(utility_sums['RW'].items()))
    for method in ('Score', 'STAR', 'Borda', 'Condorcet RCV', 'Coombs',
                   'Approval (opt.)', 'Hare RCV', 'Top-2 Runoff', 'Plurality'):
        x, y = zip(*sorted(utility_sums[method].items()))
        SUE = (np.array(y) - y_rw) / (np.array(y_uw) - y_rw)
        plt.plot(x, SUE*100, '-', label=method)
        table.append([method, *SUE*100])

    print(tabulate(table, ["Method", *x], tablefmt="pipe", floatfmt='.1f'))
    print()

    plt.legend()
    plt.grid(True, color='0.7', linestyle='-', which='major', axis='both')
    plt.grid(True, color='0.9', linestyle='-', which='minor', axis='both')
    plt.ylim(85, 100.5)  # or ymin
    plt.xlim(1.8, 7.2)
