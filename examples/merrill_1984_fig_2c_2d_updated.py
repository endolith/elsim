"""
Add modern voting methods to Figures 2.c and 2.d

Condorcet Efficiency under Spatial-Model Assumptions
(201 voters, two dimensions, correlation = .5, relative dispersion = .5 or 1.0)

from

S. Merrill III, "A Comparison of Efficiency of Multicandidate
Electoral Systems", American Journal of Political Science, vol. 28,
no. 1, pp. 23-48, 1984.  :doi:`10.2307/2110786`

Results with 100_000 elections:

2.c

| Method          |     2 |     3 |     4 |     5 |     6 |     7 |
|:----------------|------:|------:|------:|------:|------:|------:|
| Condorcet RCV   | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Coombs          | 100.0 |  99.4 |  98.6 |  97.8 |  96.9 |  96.0 |
| STAR            | 100.0 |  97.8 |  94.7 |  92.1 |  89.9 |  88.0 |
| Borda           | 100.0 |  91.4 |  89.2 |  87.1 |  85.8 |  84.8 |
| Score           | 100.0 |  88.7 |  84.6 |  82.7 |  81.3 |  79.9 |
| Approval (opt.) | 100.0 |  86.0 |  79.7 |  73.9 |  70.5 |  67.0 |
| Hare RCV        | 100.0 |  94.1 |  86.6 |  79.0 |  71.3 |  65.1 |
| Top-2 Runoff    | 100.0 |  94.1 |  87.1 |  79.9 |  72.8 |  65.9 |
| Plurality       | 100.0 |  80.6 |  67.8 |  57.3 |  49.1 |  42.7 |

2.d

| Method          |     2 |     3 |     4 |     5 |     6 |     7 |
|:----------------|------:|------:|------:|------:|------:|------:|
| Condorcet RCV   | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Coombs          | 100.0 |  98.3 |  95.7 |  93.4 |  90.8 |  88.5 |
| STAR            | 100.0 |  96.5 |  91.5 |  87.2 |  83.1 |  79.8 |
| Borda           | 100.0 |  89.2 |  86.4 |  83.9 |  82.2 |  80.7 |
| Score           | 100.0 |  86.2 |  80.5 |  77.4 |  74.5 |  72.1 |
| Approval (opt.) | 100.0 |  83.7 |  76.9 |  71.6 |  67.6 |  64.6 |
| Hare RCV        | 100.0 |  72.4 |  50.4 |  35.6 |  26.2 |  19.8 |
| Top-2 Runoff    | 100.0 |  72.4 |  50.5 |  35.3 |  24.5 |  17.1 |
| Plurality       | 100.0 |  56.3 |  34.8 |  21.5 |  13.5 |   8.6 |

These look generally like Merrill's, but are smoother, and there are some
discrepancies, as great as 7%. This may just be random variation from not
running as many simulations, however the Coombs results are consistently
high.
"""
import time
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from elsim.methods import (fptp, runoff, irv, approval, borda, coombs,
                           black, utility_winner, condorcet, score, star)
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

for fig, disp, ymin in (('2.c', 1.0, 50),
                        ('2.d', 0.5, 0)):

    count = {key: Counter() for key in (ranked_methods.keys() |
                                        rated_methods.keys() |
                                        {'CW'})}
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
                count['CW'][n_cands] += 1

                for name, method in ranked_methods.items():
                    if method(rankings, tiebreaker='random') == CW:
                        count[name][n_cands] += 1

                for name, method in rated_methods.items():
                    if method(utilities, tiebreaker='random') == CW:
                        count[name][n_cands] += 1

    elapsed_time = time.monotonic() - start_time
    print('Elapsed:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),
          '\n')

    plt.figure(f'Figure {fig}. {n_voters} voters, {n_elections} elections',
               figsize=(8, 6.5))
    plt.title(f'Figure {fig}: Condorcet Efficiency under Spatial-Model '
              f'Assumptions [Disp: {disp}]')

    table = []

    # Of those elections with CW, likelihood that method chooses CW
    x_cw, y_cw = zip(*sorted(count['CW'].items()))
    for method in ('Condorcet RCV', 'Coombs', 'STAR', 'Borda', 'Score',
                   'Approval (opt.)', 'Hare RCV', 'Top-2 Runoff', 'Plurality'):
        x, y = zip(*sorted(count[method].items()))
        CE = np.array(y)/y_cw
        plt.plot(x, CE*100, '-', label=method)
        table.append([method, *CE*100])

    print(tabulate(table, ["Method", *x], tablefmt="pipe", floatfmt='.1f'))
    print()

    plt.legend()
    plt.grid(True, color='0.7', linestyle='-', which='major', axis='both')
    plt.grid(True, color='0.9', linestyle='-', which='minor', axis='both')
    plt.ylim(ymin, 102)
    plt.xlim(1.8, 7.2)
