"""
Reproduce Figures 4.a and 4.b

Social Utility Efficiency under Spatial-Model Assumptions
(201 voters, two dimensions, correlation = .5, relative dispersion = .5 or 1.0)

from

S. Merrill III, "A Comparison of Efficiency of Multicandidate
Electoral Systems", American Journal of Political Science, vol. 28,
no. 1, pp. 23-48, 1984.  :doi:`10.2307/2110786`

Results with 100_000 iterations:

4.a

| Method    |     2 |    3 |    4 |    5 |    7 |
|:----------|------:|-----:|-----:|-----:|-----:|
| Black     | 100.0 | 97.1 | 97.1 | 97.3 | 97.8 |
| Coombs    | 100.0 | 97.0 | 96.8 | 97.0 | 97.4 |
| Borda     | 100.0 | 98.7 | 98.2 | 97.9 | 97.7 |
| Approval  | 100.0 | 98.7 | 97.4 | 96.2 | 95.3 |
| Hare      | 100.0 | 94.1 | 92.7 | 91.7 | 90.2 |
| Runoff    | 100.0 | 94.1 | 92.0 | 90.5 | 87.4 |
| Plurality | 100.0 | 84.4 | 77.3 | 72.0 | 64.8 |

4.b

| Method    |     2 |    3 |    4 |    5 |     7 |
|:----------|------:|-----:|-----:|-----:|------:|
| Black     | 100.0 | 95.4 | 95.2 | 95.5 |  96.1 |
| Coombs    | 100.0 | 94.9 | 94.1 | 94.0 |  94.1 |
| Borda     | 100.0 | 97.9 | 97.1 | 96.6 |  96.2 |
| Approval  | 100.0 | 98.5 | 96.8 | 95.5 |  94.5 |
| Hare      | 100.0 | 70.2 | 55.8 | 46.7 |  34.7 |
| Runoff    | 100.0 | 70.1 | 51.4 | 36.8 |  13.6 |
| Plurality | 100.0 | 50.0 | 23.2 |  4.0 | -24.7 |

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
                           black, utility_winner)
from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.strategies import honest_rankings, approval_optimal

n = 10_000
n_voters = 201
n_cands_list = (2, 3, 4, 5, 7)
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

    count = {key: Counter() for key in (ranked_methods.keys() |
                                        rated_methods.keys() |
                                        {'SU max', 'RW'})}
    start_time = time.monotonic()

    for iteration in range(n):
        for n_cands in n_cands_list:
            v, c = normal_electorate(n_voters, n_cands, dims=D, corr=corr,
                                     disp=disp)
            utilities = normed_dist_utilities(v, c)
            rankings = honest_rankings(utilities)

            # Pick a random winner and accumulate utilities
            RW = randint(0, n_cands - 1)
            count['RW'][n_cands] += utilities.sum(axis=0)[RW]

            for name, method in rated_methods.items():
                winner = method(utilities, tiebreaker='random')
                count[name][n_cands] += utilities.sum(axis=0)[winner]

            for name, method in ranked_methods.items():
                winner = method(rankings, tiebreaker='random')
                count[name][n_cands] += utilities.sum(axis=0)[winner]

    elapsed_time = time.monotonic() - start_time
    print('Elapsed:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),
          '\n')

    plt.figure(f'Figure {fig}. {n_voters} voters, {n} iterations')
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
    x_uw, y_uw = zip(*sorted(count['SU max'].items()))
    x_rw, y_rw = zip(*sorted(count['RW'].items()))
    for method in ('Black', 'Coombs', 'Borda', 'Approval', 'Hare', 'Runoff',
                   'Plurality'):
        x, y = zip(*sorted(count[method].items()))
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
