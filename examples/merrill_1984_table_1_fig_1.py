"""
Reproduce Table 1 and Figure 1

Condorcet Efficiencies for a Random Society
(25 voters)

from

S. Merrill III, "A Comparison of Efficiency of Multicandidate
Electoral Systems", American Journal of Political Science, vol. 28,
no. 1, pp. 23-48, 1984.  :doi:`10.2307/2110786`

Typical result:

| Method    |     2 |     3 |     4 |     5 |     7 |    10 |
|:----------|------:|------:|------:|------:|------:|------:|
| Plurality | 100.0 |  79.1 |  68.4 |  61.8 |  51.4 |  41.1 |
| Runoff    | 100.0 |  96.2 |  89.6 |  83.4 |  72.3 |  60.3 |
| Hare      | 100.0 |  96.2 |  92.5 |  89.1 |  83.7 |  77.0 |
| Approval  | 100.0 |  75.6 |  70.0 |  67.4 |  63.8 |  61.2 |
| Borda     | 100.0 |  90.9 |  87.4 |  85.9 |  84.6 |  83.9 |
| Coombs    | 100.0 |  96.9 |  93.4 |  91.0 |  86.4 |  81.7 |
| Black     | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| SU max    | 100.0 |  84.1 |  79.6 |  78.4 |  77.3 |  77.5 |
| CW        | 100.0 |  91.7 |  83.1 |  75.6 |  64.3 |  52.9 |
"""
import time

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from elsim.methods import black, borda, coombs, fptp, irv, runoff
from elsim.studies import (Study, approval_at_optimal, grid, random_society,
                           utility_winner_at)

n_elections = 10_000  # Roughly 15 seconds on a 2019 6-core i7-9750H
n_voters = 25
n_cands_list = (2, 3, 4, 5, 7, 10)

ranked_methods = {'Plurality': fptp, 'Runoff': runoff, 'Hare': irv,
                  'Borda': borda, 'Coombs': coombs, 'Black': black}

rated_methods = {'SU max': utility_winner_at,
                 'Approval': approval_at_optimal}

start_time = time.monotonic()

study = Study(
    scenarios=grid(n_cands=n_cands_list),
    electorate=random_society(n_voters=n_voters),
    n_elections=n_elections,
    ranked_methods=ranked_methods,
    rated_methods=rated_methods,
    metric='condorcet',
    normalization='range',
    random_state=1984,
    n_jobs=1,
)
result = study.run()

elapsed_time = time.monotonic() - start_time
print('Elapsed:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)), '\n')

# Plot Merrill's results as dotted lines for comparison
merrill_table_1 = {
    'Plurality': {2: 100.0, 3:  79.1, 4:  69.4, 5:  62.1, 7:  52.0, 10:  42.6},
    'Runoff':    {2: 100.0, 3:  96.2, 4:  90.1, 5:  83.6, 7:  73.5, 10:  61.3},
    'Hare':      {2: 100.0, 3:  96.2, 4:  92.7, 5:  89.1, 7:  84.8, 10:  77.9},
    'Approval':  {2: 100.0, 3:  76.0, 4:  69.8, 5:  67.1, 7:  63.7, 10:  61.3},
    'Borda':     {2: 100.0, 3:  90.8, 4:  87.3, 5:  86.2, 7:  85.3, 10:  84.3},
    'Coombs':    {2: 100.0, 3:  96.3, 4:  93.4, 5:  90.2, 7:  86.1, 10:  81.1},
    'Black':     {2: 100.0, 3: 100.0, 4: 100.0, 5: 100.0, 7: 100.0, 10: 100.0},
    'SU max':    {2: 100.0, 3:  84.4, 4:  80.2, 5:  77.9, 7:  77.2, 10:  77.8},
    'CW':        {2: 100.0, 3:  91.6, 4:  83.4, 5:  75.8, 7:  64.3, 10:  52.5},
}

x = np.array([scenario['n_cands'] for scenario in result.scenarios])
rows = {row[0]: np.array(row[1:])
        for row in result.table(
            methods=['Plurality', 'Runoff', 'Hare', 'Approval', 'Borda',
                     'Coombs', 'Black', 'SU max', 'CW'])}

plt.figure(f'Figure 1. {n_voters} voters, {n_elections} elections')
plt.title('Figure 1: Condorcet Efficiencies for a Random Society')
for method in ('Plurality', 'Runoff', 'Hare', 'Approval', 'Borda', 'Coombs',
               'Black'):
    mx, my = zip(*sorted(merrill_table_1[method].items()),
                strict=True)
    plt.plot(mx, my, ':', lw=0.8)

# Restart color cycle, so result colors match
plt.gca().set_prop_cycle(None)

table = []
for method in ('Plurality', 'Runoff', 'Hare', 'Approval', 'Borda', 'Coombs',
               'Black'):
    plt.plot(x, rows[method], '-', label=method)
    table.append([method, *rows[method]])
table.append(['SU max', *rows['SU max']])
table.append(['CW', *rows['CW']])

print(tabulate(table, ["Method", *x], tablefmt="pipe", floatfmt='.1f'))

plt.plot([], [], 'k:', lw=0.8, label='Merrill')  # Dummy plot for label
plt.legend()
plt.grid(True, color='0.7', linestyle='-', which='major', axis='both')
plt.grid(True, color='0.9', linestyle='-', which='minor', axis='both')
plt.ylim(40, 102)
plt.xlim(1.8, 10.2)
plt.show()
