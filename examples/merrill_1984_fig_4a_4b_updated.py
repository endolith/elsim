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

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from joblib import Parallel, delayed

from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.methods import (approval, black, borda, coombs, fptp, irv, runoff,
                           score, star, utility_winner)
from elsim.strategies import (approval_optimal, honest_normed_scores,
                              honest_rankings)

n_elections = 5_000  # Roughly 30 seconds each on a 2019 6-core i7-9750H
n_voters = 201
n_cands_list = (2, 3, 4, 5, 6, 7)
corr = 0.5
D = 2

# Simulate more than just one election per worker to improve efficiency
batch_size = 100
n_batches = n_elections // batch_size
assert n_batches * batch_size == n_elections

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

# Plot Merrill's results as dotted lines for comparison (traced from plots)
merrill_fig_4a = {
    'Black':     {1: 98.1, 2: 97.8, 3: 97.2, 4: 96.8},
    'Coombs':    {1: 98.1, 2: 97.8, 3: 97.2, 4: 96.8},
    'Borda':     {1: 91.4, 2: 91.1, 3: 90.4, 4: 89.8},
    'Approval':  {1: 90.4, 2: 90.1, 3: 89.4, 4: 88.8},
    'Hare':      {1: 85.1, 2: 84.0, 3: 81.6, 4: 79.6},
    'Runoff':    {1: 85.8, 2: 84.9, 3: 82.7, 4: 80.9},
    'Plurality': {1: 85.6, 2: 84.7, 3: 82.6, 4: 80.8},
}

merrill_fig_4b = {
    'Black':     {1: 96.0, 2: 95.6, 3: 95.0, 4: 94.5},
    'Coombs':    {1: 96.0, 2: 95.6, 3: 95.0, 4: 94.5},
    'Borda':     {1: 83.9, 2: 83.7, 3: 83.0, 4: 82.5},
    'Approval':  {1: 83.5, 2: 83.3, 3: 82.6, 4: 82.1},
    'Hare':      {1: 71.3, 2: 70.8, 3: 68.3, 4: 66.4},
    'Runoff':    {1: 73.8, 2: 73.1, 3: 70.6, 4: 68.7},
    'Plurality': {1: 73.6, 2: 72.9, 3: 70.4, 4: 68.5},
}

def simulate_batch(disp):
    utility_sums = {key: Counter() for key in (ranked_methods.keys() |
                                               rated_methods.keys() |
                                               {'SU max', 'RW'})}
    
    for iteration in range(batch_size):
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
    
    return utility_sums

for fig, disp, orig in (('4.a', 0.5, merrill_fig_4a),
                        ('4.b', 1.0, merrill_fig_4b)):

    start_time = time.monotonic()

    jobs = [delayed(simulate_batch)(disp)] * n_batches
    print(f'{len(jobs)} tasks total:')
    results = Parallel(n_jobs=-3, verbose=5)(jobs)
    
    # Merge results from all batches
    utility_sums = {key: Counter() for key in (ranked_methods.keys() |
                                               rated_methods.keys())}
    for result in results:
        for method in utility_sums:
            for n_cands in n_cands_list:
                utility_sums[method][n_cands] += result[method][n_cands]

    elapsed_time = time.monotonic() - start_time
    print('Elapsed:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),
          '\n')

    plt.figure(f'Figure {fig}. {n_voters} voters, {n_elections} elections',
               figsize=(8, 6.5))
    plt.title(f'Figure {fig}: Social Utility Efficiency under Spatial-Model '
              f'Assumptions [Disp: {disp}]')

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
    plt.show()
