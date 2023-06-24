"""
Reproduce Figures 2.c and 2.d

Condorcet Efficiency under Spatial-Model Assumptions
(201 voters, two dimensions, correlation = .5, relative dispersion = .5 or 1.0)

from

S. Merrill III, "A Comparison of Efficiency of Multicandidate
Electoral Systems", American Journal of Political Science, vol. 28,
no. 1, pp. 23-48, 1984.  :doi:`10.2307/2110786`

Results with 500_000 elections:

2.c

| Method    |     2 |     3 |     4 |     5 |     6 |     7 |
|:----------|------:|------:|------:|------:|------:|------:|
| Black     | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Coombs    | 100.0 |  99.4 |  98.6 |  97.8 |  96.9 |  96.0 |
| Borda     | 100.0 |  91.4 |  89.2 |  87.1 |  85.7 |  84.6 |
| Approval  | 100.0 |  85.9 |  79.8 |  73.9 |  70.1 |  66.8 |
| Hare      | 100.0 |  94.1 |  86.6 |  78.9 |  71.7 |  65.2 |
| Runoff    | 100.0 |  94.1 |  87.1 |  79.7 |  72.8 |  66.1 |
| Plurality | 100.0 |  80.6 |  67.6 |  57.4 |  49.3 |  42.6 |

2.d

| Method    |     2 |     3 |     4 |     5 |     6 |     7 |
|:----------|------:|------:|------:|------:|------:|------:|
| Black     | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Coombs    | 100.0 |  98.2 |  95.9 |  93.4 |  90.9 |  88.4 |
| Borda     | 100.0 |  89.2 |  86.3 |  83.8 |  82.1 |  80.8 |
| Approval  | 100.0 |  84.0 |  76.9 |  71.5 |  67.8 |  64.7 |
| Hare      | 100.0 |  72.2 |  50.3 |  35.8 |  26.0 |  19.7 |
| Runoff    | 100.0 |  72.2 |  50.6 |  35.3 |  24.4 |  16.9 |
| Plurality | 100.0 |  55.9 |  34.7 |  21.5 |  13.5 |   8.5 |

These look generally like Merrill's, but are smoother, and there are some
discrepancies, as great as 7%. This may just be random variation from not
running as many simulations, however the Coombs results are consistently
high.
"""
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.methods import (approval, black, borda, condorcet, coombs, fptp,
                           irv, runoff, utility_winner)
from elsim.strategies import approval_optimal, honest_rankings

n_elections = 10_000  # Roughly 30 seconds each
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
merrill_fig_2c = {
    'Black':     {2: 100.0, 3: 100.0, 4: 100.0, 5: 100.0, 7: 100.0},
    'Coombs':    {2: 100.0, 3: 99.1,  4: 98.1,  5: 97.1,  7: 93.1},
    'Borda':     {2: 100.0, 3: 91.0,  4: 87.1,  5: 86.0,  7: 84.0},
    'Approval':  {2: 100.0, 3: 86.0,  4: 77.1,  5: 74.0,  7: 66.0},
    'Hare':      {2: 100.0, 3: 94.1,  4: 88.1,  5: 78.0,  7: 66.0},
    'Runoff':    {2: 100.0, 3: 94.1,  4: 88.1,  5: 80.1,  7: 64.0},
    'Plurality': {2: 100.0, 3: 79.1,  4: 68.9,  5: 57.0,  7: 50.0},
    }

merrill_fig_2d = {
    'Black':     {2: 100.0, 3: 100.0, 4: 100.0, 5: 100.0, 7: 100.0},
    'Coombs':    {2: 100.0, 3: 97.2,  4: 93.2,  5: 90.0,  7: 81.2},
    'Borda':     {2: 100.0, 3: 88.3,  4: 88.0,  5: 84.0,  7: 76.0},
    'Approval':  {2: 100.0, 3: 84.2,  4: 81.3,  5: 73.0,  7: 62.0},
    'Hare':      {2: 100.0, 3: 69.4,  4: 52.1,  5: 34.0,  7: 21.0},
    'Runoff':    {2: 100.0, 3: 69.4,  4: 53.3,  5: 31.0,  7: 16.9},
    'Plurality': {2: 100.0, 3: 51.3,  4: 36.2,  5: 21.0,  7: 7.8},
    }

for fig, disp, ymin, orig in (('2.c', 1.0, 50, merrill_fig_2c),
                              ('2.d', 0.5, 0, merrill_fig_2d)):

    condorcet_winner_count = {key: Counter() for key in (
        ranked_methods.keys() | rated_methods.keys() | {'CW'})}
    start_time = time.monotonic()

    for iteration in range(n_elections):
        for n_cands in n_cands_list:
            v, c = normal_electorate(n_voters, n_cands, dims=D, corr=corr,
                                     disp=disp)
            utilities = normed_dist_utilities(v, c)
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

    elapsed_time = time.monotonic() - start_time
    print('Elapsed:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),
          '\n')

    plt.figure(f'Figure {fig}. {n_voters} voters, {n_elections} elections')
    plt.title(f'Figure {fig}: Condorcet Efficiency under Spatial-Model '
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

    # Of those elections with CW, likelihood that method chooses CW
    x_cw, y_cw = zip(*sorted(condorcet_winner_count['CW'].items()))
    for method in ('Black', 'Coombs', 'Borda', 'Approval', 'Hare', 'Runoff',
                   'Plurality'):
        x, y = zip(*sorted(condorcet_winner_count[method].items()))
        CE = np.array(y)/y_cw
        plt.plot(x, CE*100, '-', label=method)
        table.append([method, *CE*100])

    print(tabulate(table, ["Method", *x], tablefmt="pipe", floatfmt='.1f'))
    print()

    plt.plot([], [], 'k:', lw=0.8, label='Merrill')  # Dummy plot for label
    plt.legend()
    plt.grid(True, color='0.7', linestyle='-', which='major', axis='both')
    plt.grid(True, color='0.9', linestyle='-', which='minor', axis='both')
    plt.ylim(ymin, 102)
    plt.xlim(1.8, 7.2)
    plt.show()
