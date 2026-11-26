"""
Reproduce Figures 4.a and 4.b

Social Utility Efficiency under Spatial-Model Assumptions
(201 voters, two dimensions, correlation = .5, relative dispersion = .5 or 1.0)

from

S. Merrill III, "A Comparison of Efficiency of Multicandidate
Electoral Systems", American Journal of Political Science, vol. 28,
no. 1, pp. 23-48, 1984.  :doi:`10.2307/2110786`

Typical result with 10_000 elections:

4.a

| Method    |     2 |    3 |    4 |    5 |    6 |    7 |
|:----------|------:|-----:|-----:|-----:|-----:|-----:|
| Black     | 100.0 | 97.2 | 97.1 | 97.6 | 97.7 | 98.0 |
| Coombs    | 100.0 | 97.2 | 96.9 | 97.2 | 97.3 | 97.4 |
| Borda     | 100.0 | 98.7 | 98.2 | 97.8 | 97.7 | 97.5 |
| Approval  | 100.0 | 98.7 | 97.3 | 96.1 | 95.5 | 95.1 |
| Hare      | 100.0 | 94.2 | 92.6 | 92.3 | 91.2 | 90.4 |
| Runoff    | 100.0 | 94.2 | 92.3 | 90.9 | 89.2 | 88.0 |
| Plurality | 100.0 | 84.5 | 77.8 | 72.8 | 68.7 | 65.1 |

4.b

| Method    |     2 |    3 |    4 |    5 |     6 |     7 |
|:----------|------:|-----:|-----:|-----:|------:|------:|
| Black     | 100.0 | 95.6 | 95.4 | 96.0 |  96.2 |  96.5 |
| Coombs    | 100.0 | 95.1 | 94.4 | 94.4 |  94.3 |  94.2 |
| Borda     | 100.0 | 97.9 | 97.1 | 96.9 |  96.5 |  96.4 |
| Approval  | 100.0 | 98.7 | 96.7 | 95.9 |  94.8 |  94.4 |
| Hare      | 100.0 | 70.2 | 55.6 | 48.2 |  39.7 |  34.4 |
| Runoff    | 100.0 | 70.2 | 51.7 | 38.1 |  24.7 |  14.2 |
| Plurality | 100.0 | 50.0 | 23.2 |  5.9 | -13.1 | -26.9 |

Each voter's range-normalized utilities are standardized to mean 0 and
standard deviation 1, matching Merrill's spatial-model procedure.

The general trend is similar to Merrill's, but there are significant
discrepancies.  It is smoother, so maybe the original just had lower number of
simulations.
"""
import time
from collections import Counter
from random import randint

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.methods import (approval, black, borda, coombs, fptp, irv, runoff,
                           utility_winner)
from elsim.strategies import approval_optimal, honest_rankings

n_elections = 10_000  # Roughly 30 seconds each on a 2019 6-core i7-9750H
n_voters = 201
n_cands_list = (2, 3, 4, 5, 6, 7)
corr = 0.5
D = 2

ranked_methods = {'Plurality': fptp, 'Runoff': runoff, 'Hare': irv,
                  'Borda': borda, 'Coombs': coombs, 'Black': black}

rated_methods = {'SU max': utility_winner,
                 'Approval': lambda utilities, tiebreaker:
                     approval(approval_optimal(utilities), tiebreaker)}

# Plot Merrill's results as dotted lines for comparison (traced from plots)
merrill_fig_4a = {
    'Borda':     {2: 100.0, 3: 98.0, 4: 97.1, 5: 98.0, 7: 97.9},
    'Black':     {2: 100.0, 3: 96.0, 4: 96.1, 5: 97.1, 7: 98.0},
    'Approval':  {2: 100.0, 3: 98.0, 4: 96.0, 5: 96.0, 7: 96.0},
    'Coombs':    {2: 100.0, 3: 95.0, 4: 96.0, 5: 96.0, 7: 96.0},
    'Hare':      {2: 100.0, 3: 90.0, 4: 89.9, 5: 88.1, 7: 88.9},
    'Runoff':    {2: 100.0, 3: 90.0, 4: 89.9, 5: 86.0, 7: 84.9},
    'Plurality': {2: 100.0, 3: 76.0, 4: 74.1, 5: 64.0, 7: 57.9},
}

merrill_fig_4b = {
    'Borda':     {2: 100.0, 3: 97.2, 4: 97.2, 5: 97.0, 7: 96.2},
    'Black':     {2: 100.0, 3: 98.1, 4: 96.1, 5: 96.0, 7: 96.2},
    'Approval':  {2: 100.0, 3: 94.0, 4: 96.0, 5: 96.0, 7: 95.1},
    'Coombs':    {2: 100.0, 3: 92.1, 4: 93.1, 5: 92.0, 7: 91.2},
    'Hare':      {2: 100.0, 3: 65.1, 4: 55.0, 5: 40.0, 7: 35.2},
    'Runoff':    {2: 100.0, 3: 65.1, 4: 53.0, 5: 28.0, 7: 13.9},
    'Plurality': {2: 100.0, 3: 41.1, 4: 27.0, 5: -1.0, 7: -9},
}

for fig, disp, ymin, orig in (('4.a', 1.0, 55, merrill_fig_4a),
                              ('4.b', 0.5, 0, merrill_fig_4b)):

    utility_sums = {key: Counter() for key in (ranked_methods.keys() |
                                               rated_methods.keys() |
                                               {'SU max', 'RW'})}
    start_time = time.monotonic()

    for _iteration in range(n_elections):
        for n_cands in n_cands_list:
            v, c = normal_electorate(n_voters, n_cands, dims=D, corr=corr,
                                     disp=disp)
            utilities = normed_dist_utilities(v, c)
            # Merrill's spatial-model simulations standardized each voter's
            # utilities to mean 0 and standard deviation 1.
            utilities = ((utilities - utilities.mean(axis=1, keepdims=True))
                         / utilities.std(axis=1, keepdims=True))
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

    plt.figure(f'Figure {fig}. {n_voters} voters, {n_elections} elections')
    plt.title(f'Figure {fig}: Social Utility Efficiency under Spatial-Model '
              'Assumptions')

    # Restart color cycle, so result colors match
    plt.gca().set_prop_cycle(None)

    for method in ('Black', 'Coombs', 'Borda', 'Approval', 'Hare', 'Runoff',
                   'Plurality'):
        x, y = zip(*sorted(orig[method].items()))
        plt.plot(x, y, ':', lw=0.8)

    # Restart color cycle, so result colors match
    plt.gca().set_prop_cycle(None)

    table = []

    # Calculate Social Utility Efficiency from summed utilities
    x_uw, y_uw = zip(*sorted(utility_sums['SU max'].items()))
    x_rw, y_rw = zip(*sorted(utility_sums['RW'].items()))
    for method in ('Black', 'Coombs', 'Borda', 'Approval', 'Hare', 'Runoff',
                   'Plurality'):
        x, y = zip(*sorted(utility_sums[method].items()))
        SUE = (np.array(y) - y_rw) / (np.array(y_uw) - y_rw)
        plt.plot(x, SUE*100, '-', label=method)
        table.append([method, *SUE*100])

    print(tabulate(table, ["Method", *x], tablefmt="pipe", floatfmt='.1f'))
    print()

    plt.plot([], [], 'k:', lw=0.8, label='Merrill')  # Dummy plot for label
    plt.legend()
    plt.grid(True, color='0.7', linestyle='-', which='major', axis='both')
    plt.grid(True, color='0.9', linestyle='-', which='minor', axis='both')
    plt.ylim(ymin, 102)
    plt.xlim(1.8, 7.2)
    plt.show()
