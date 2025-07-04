"""
Reproduce Figures 4.a and 4.b

Social Utility Efficiency under Spatial-Model Assumptions
(201 voters, 5 candidates, correlation = .5, relative dispersion = .5 or 1.0)

from

S. Merrill III, "A Comparison of Efficiency of Multicandidate
Electoral Systems", American Journal of Political Science, vol. 28,
no. 1, pp. 23-48, 1984.  :doi:`10.2307/2110786`

Results with 500_000 elections:

4.a

| Dims |   Plurality |   Runoff |   Hare |   Approval |   Borda |   Coombs |   Black |
|-----:|------------:|---------:|-------:|-----------:|--------:|---------:|--------:|
|    1 |        85.6 |     85.8 |   85.1 |       90.4 |    91.4 |     97.8 |    97.8 |
|    2 |        84.7 |     84.9 |   84.0 |       90.1 |    91.1 |     97.8 |    97.8 |
|    3 |        82.6 |     82.7 |   81.6 |       89.4 |    90.4 |     97.6 |    97.6 |
|    4 |        80.8 |     80.9 |   79.6 |       88.8 |    89.8 |     97.4 |    97.4 |

4.b

| Dims |   Plurality |   Runoff |   Hare |   Approval |   Borda |   Coombs |   Black |
|-----:|------------:|---------:|-------:|-----------:|--------:|---------:|--------:|
|    1 |        73.6 |     73.8 |   71.3 |       83.5 |    83.9 |     95.7 |    95.7 |
|    2 |        72.9 |     73.1 |   70.8 |       83.3 |    83.7 |     95.6 |    95.6 |
|    3 |        70.4 |     70.6 |   68.3 |       82.6 |    83.0 |     95.2 |    95.2 |
|    4 |        68.5 |     68.7 |   66.4 |       82.1 |    82.5 |     94.9 |    94.9 |

These look generally like Merrill's, but are smoother, and there are some
discrepancies, as great as 3%. This may just be random variation from not
running as many simulations.
"""
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tabulate import tabulate

from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.methods import (approval, black, borda, coombs, fptp, irv, runoff,
                           utility_winner)
from elsim.strategies import approval_optimal, honest_rankings

n_elections = 50_000  # Roughly 30 seconds each on a 2019 6-core i7-9750H
n_voters = 201
n_cands = 5
corr = 0.5
dims_list = (1, 2, 3, 4)

# Simulate more than just one election per worker to improve efficiency
batch_size = 100
n_batches = n_elections // batch_size
assert n_batches * batch_size == n_elections

ranked_methods = {'Plurality': fptp, 'Runoff': runoff, 'Hare': irv,
                  'Borda': borda, 'Coombs': coombs, 'Black': black}

rated_methods = {'SU max': utility_winner,
                 'Approval': lambda utilities, tiebreaker:
                     approval(approval_optimal(utilities), tiebreaker)}

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
                                               rated_methods.keys())}
    
    for iteration in range(batch_size):
        for dims in dims_list:
            v, c = normal_electorate(n_voters, n_cands, dims=dims, corr=corr,
                                     disp=disp)
            utilities = normed_dist_utilities(v, c)

            # Find the social utility winner and accumulate utilities
            UW = utility_winner(utilities)
            utility_sums['SU max'][dims] += utilities.sum(axis=0)[UW]

            for name, method in rated_methods.items():
                if name != 'SU max':  # Already handled above
                    winner = method(utilities, tiebreaker='random')
                    utility_sums[name][dims] += utilities.sum(axis=0)[winner]

            rankings = honest_rankings(utilities)
            for name, method in ranked_methods.items():
                winner = method(rankings, tiebreaker='random')
                utility_sums[name][dims] += utilities.sum(axis=0)[winner]
    
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
            for dims in dims_list:
                utility_sums[method][dims] += result[method][dims]

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
    random_utility = n_voters / 2  # Expected utility for random choice
    for method in ('Plurality', 'Runoff', 'Hare', 'Approval', 'Borda', 'Coombs',
                   'Black'):
        x, y = zip(*sorted(utility_sums[method].items()))
        utilities_per_election = np.array(y) / n_elections
        max_utilities_per_election = np.array(y_uw) / n_elections
        SUE = (utilities_per_election - random_utility) / (max_utilities_per_election - random_utility)
        plt.plot(x, SUE*100, '-', label=method)
        table.append([method, *SUE*100])

    print(tabulate(table, ["Dims", *x], tablefmt="pipe", floatfmt='.1f'))
    print()

    plt.plot([], [], 'k:', lw=0.8, label='Merrill')  # Dummy plot for label
    plt.legend()
    plt.grid(True, color='0.7', linestyle='-', which='major', axis='both')
    plt.grid(True, color='0.9', linestyle='-', which='minor', axis='both')
    plt.ylim(60, 102)
    plt.xlim(0.8, 4.2)
    plt.show()
