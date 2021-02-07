"""
Reproduce Table 3 and Figure 3

Efficiencies for Social Utility for a Random Society
(25 voters)

from

S. Merrill III, "A Comparison of Efficiency of Multicandidate
Electoral Systems", American Journal of Political Science, vol. 28,
no. 1, pp. 23-48, 1984.  :doi:`10.2307/2110786`

Typical result:

| Method    |     2 |    3 |    4 |    5 |    7 |   10 |
|:----------|------:|-----:|-----:|-----:|-----:|-----:|
| Plurality | 100.0 | 83.3 | 75.1 | 69.5 | 62.5 | 54.9 |
| Runoff    | 100.0 | 89.1 | 83.9 | 80.4 | 75.1 | 69.1 |
| Hare      | 100.0 | 89.0 | 84.8 | 82.5 | 79.9 | 77.3 |
| Approval  | 100.0 | 95.5 | 91.3 | 89.3 | 87.8 | 86.8 |
| Borda     | 100.0 | 94.7 | 94.3 | 94.4 | 95.3 | 96.2 |
| Coombs    | 100.0 | 90.2 | 86.8 | 85.2 | 84.0 | 82.9 |
| Black     | 100.0 | 92.9 | 92.0 | 92.1 | 93.2 | 94.6 |
"""
import time
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from elsim.methods import (fptp, runoff, irv, approval, borda, coombs,
                           black, utility_winner)
from elsim.elections import random_utilities
from elsim.strategies import honest_rankings, approval_optimal

n = 10_000
n_voters = 25
n_cands_list = (2, 3, 4, 5, 7, 10)

ranked_methods = {'Plurality': fptp, 'Runoff': runoff, 'Hare': irv,
                  'Borda': borda, 'Coombs': coombs, 'Black': black}

rated_methods = {'Approval': lambda utilities, tiebreaker:
                 approval(approval_optimal(utilities), tiebreaker)}

count = {key: Counter() for key in (ranked_methods.keys() |
                                    rated_methods.keys() | {'UW'})}

start_time = time.monotonic()

for iteration in range(n):
    for n_cands in n_cands_list:
        utilities = random_utilities(n_voters, n_cands)

        """
        "Simulated utilities were normalized by range, that is, each voter's
        set of utilities were linearly expanded so that the highest and lowest
        utilities for each voter were 1 and 0, respectively."
        """
        # TODO: Try the Standard Score normalization too?
        utilities -= utilities.min(1)[:, np.newaxis]
        utilities /= utilities.max(1)[:, np.newaxis]

        # Find the social utility winner and accumulate utilities
        UW = utility_winner(utilities)
        count['UW'][n_cands] += utilities.sum(0)[UW]

        for name, func in rated_methods.items():
            winner = func(utilities, tiebreaker='random')
            count[name][n_cands] += utilities.sum(0)[winner]

        rankings = honest_rankings(utilities)
        for name, func in ranked_methods.items():
            winner = func(rankings, tiebreaker='random')
            count[name][n_cands] += utilities.sum(0)[winner]


elapsed_time = time.monotonic() - start_time
print('Elapsed:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)), '\n')

# Plot Merrill's results as dotted lines for comparison
merrill_table_1 = {
    'Plurality': {2: 100.0, 3: 83.0, 4: 75.0, 5: 69.2, 7: 62.8, 10: 53.3},
    'Runoff':    {2: 100.0, 3: 89.5, 4: 83.8, 5: 80.5, 7: 75.6, 10: 67.6},
    'Hare':      {2: 100.0, 3: 89.5, 4: 84.7, 5: 82.4, 7: 80.5, 10: 74.9},
    'Approval':  {2: 100.0, 3: 95.4, 4: 91.1, 5: 89.1, 7: 87.8, 10: 87.0},
    'Borda':     {2: 100.0, 3: 94.8, 4: 94.1, 5: 94.4, 7: 95.4, 10: 95.9},
    'Coombs':    {2: 100.0, 3: 89.7, 4: 86.7, 5: 85.1, 7: 83.1, 10: 82.4},
    'Black':     {2: 100.0, 3: 93.1, 4: 91.9, 5: 92.0, 7: 93.1, 10: 94.3},
    }

plt.figure(f'Figure 3. {n_voters} voters, {n} iterations')
plt.title('Figure 3: Efficiencies for Social Utility for a Random Society')
for method in ('Plurality', 'Runoff', 'Hare', 'Approval', 'Borda', 'Coombs',
               'Black'):
    x, y = zip(*sorted(merrill_table_1[method].items()))
    plt.plot(x, y, ':', lw=0.8)

# Restart color cycle, so result colors match
plt.gca().set_prop_cycle(None)

table = []

# Calculate Social Utility Efficiency from summed utilities
x_uw, y_uw = zip(*sorted(count['UW'].items()))
for method in ('Plurality', 'Runoff', 'Hare', 'Approval', 'Borda', 'Coombs',
               'Black'):
    x, y = zip(*sorted(count[method].items()))
    SUE = (np.array(y) - n_voters * n / 2)/(np.array(y_uw) - n_voters * n / 2)
    plt.plot(x, SUE*100, '-', label=method)
    table.append([method, *(SUE*100)])

print(tabulate(table, ["Method", *x], tablefmt="pipe", floatfmt='.1f'))

plt.plot([], [], 'k:', lw=0.8, label='Merrill')  # Dummy plot for label
plt.legend()
plt.grid(True, color='0.7', linestyle='-', which='major', axis='both')
plt.grid(True, color='0.9', linestyle='-', which='minor', axis='both')
plt.ylim(40, 102)
plt.xlim(1.8, 10.2)
