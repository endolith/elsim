"""
Reproduce Figures 4.a and 4.b

Social Utility Efficiency under Spatial-Model Assumptions
(201 voters, two dimensions, correlation = .5, relative dispersion = .5 or 1.0)

from

S. Merrill III, "A Comparison of Efficiency of Multicandidate
Electoral Systems", American Journal of Political Science, vol. 28,
no. 1, pp. 23-48, 1984.  :doi:`10.2307/2110786`

Results with 500_000 elections:

4.a

| Method    |     2 |    3 |    4 |    5 |    6 |    7 |
|:----------|------:|-----:|-----:|-----:|-----:|-----:|
| Black     | 100.0 | 97.2 | 97.1 | 97.3 | 97.6 | 97.8 |
| Coombs    | 100.0 | 97.1 | 96.8 | 97.0 | 97.2 | 97.4 |
| Borda     | 100.0 | 98.7 | 98.2 | 97.9 | 97.7 | 97.6 |
| Approval  | 100.0 | 98.7 | 97.3 | 96.2 | 95.6 | 95.2 |
| Hare      | 100.0 | 94.2 | 92.6 | 91.7 | 91.0 | 90.3 |
| Runoff    | 100.0 | 94.2 | 92.0 | 90.4 | 88.9 | 87.4 |
| Plurality | 100.0 | 84.7 | 77.1 | 72.1 | 68.1 | 64.8 |

4.b

| Method    |     2 |    3 |    4 |    5 |     6 |     7 |
|:----------|------:|-----:|-----:|-----:|------:|------:|
| Black     | 100.0 | 95.5 | 95.2 | 95.5 |  95.8 |  96.2 |
| Coombs    | 100.0 | 94.9 | 94.1 | 94.0 |  94.0 |  94.1 |
| Borda     | 100.0 | 97.9 | 97.1 | 96.6 |  96.4 |  96.3 |
| Approval  | 100.0 | 98.6 | 96.7 | 95.6 |  94.9 |  94.5 |
| Hare      | 100.0 | 70.2 | 55.9 | 46.7 |  39.7 |  34.6 |
| Runoff    | 100.0 | 70.2 | 51.7 | 36.9 |  24.3 |  13.5 |
| Plurality | 100.0 | 50.1 | 23.7 |  4.3 | -11.8 | -25.1 |

The general trend is similar to Merrill's, but there are significant
discrepancies.  It is smoother, so maybe the original just had lower number of
simulations.
"""
import time
from random import randint

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.methods import (approval, black, borda, coombs, fptp, irv, runoff,
                           utility_winner)
from elsim.strategies import approval_optimal, honest_rankings

from plot_uncertainty import sue_ratio_curve_points_and_errors

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

    idx = {nc: i for i, nc in enumerate(n_cands_list)}
    nc_ct = len(n_cands_list)
    plot_keys = list(ranked_methods.keys()) + list(rated_methods.keys())
    A_elec = {m: np.zeros((nc_ct, n_elections)) for m in plot_keys}
    B_z = np.zeros((nc_ct, n_elections))
    start_time = time.monotonic()

    for iteration in range(n_elections):
        for n_cands in n_cands_list:
            j = idx[n_cands]
            v, c = normal_electorate(n_voters, n_cands, dims=D, corr=corr,
                                     disp=disp)
            utilities = normed_dist_utilities(v, c)
            rankings = honest_rankings(utilities)

            RW = randint(0, n_cands - 1)
            rw_sum = utilities.sum(axis=0)[RW]

            uw = utility_winner(utilities)
            su_sum = utilities.sum(axis=0)[uw]
            B_z[j, iteration] = su_sum - rw_sum

            for name, method in rated_methods.items():
                winner = method(utilities, tiebreaker='random')
                A_elec[name][j, iteration] = (
                    utilities.sum(axis=0)[winner] - rw_sum)

            for name, method in ranked_methods.items():
                winner = method(rankings, tiebreaker='random')
                A_elec[name][j, iteration] = (
                    utilities.sum(axis=0)[winner] - rw_sum)

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

    rng = np.random.default_rng(0)
    xp = list(n_cands_list)

    # SUE = sum(method total - RW total) / sum(SU max total - RW total)
    for method in ('Black', 'Coombs', 'Borda', 'Approval', 'Hare', 'Runoff',
                   'Plurality'):
        y, el, eh = sue_ratio_curve_points_and_errors(
            A_elec[method], B_z, rng=rng)
        plt.errorbar(xp, y, yerr=[el, eh], fmt='-', label=method,
                     capsize=2, elinewidth=0.8)
        table.append([method, *y])

    print(tabulate(table, ["Method", *xp], tablefmt="pipe", floatfmt='.1f'))
    print()

    plt.plot([], [], 'k:', lw=0.8, label='Merrill')  # Dummy plot for label
    plt.figtext(
        0.99,
        0.01,
        'Simulation error bars: 95% percentile bootstrap CI for '
        'sum(A_i)/sum(B_i) with paired election resampling '
        '(A_i = method utility sum minus random winner sum; '
        'B_i = SU-max sum minus random winner sum).',
        fontsize=7,
        ha='right',
        va='bottom',
    )
    plt.legend()
    plt.grid(True, color='0.7', linestyle='-', which='major', axis='both')
    plt.grid(True, color='0.9', linestyle='-', which='minor', axis='both')
    plt.ylim(ymin, 102)
    plt.xlim(1.8, 7.2)
    plt.show()
