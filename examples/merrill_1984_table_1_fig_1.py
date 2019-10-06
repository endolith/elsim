"""
Reproduce Table 1 and Figure 1

Condorcet Efficiencies for a Random Society
(25 voters)

from

S. Merrill III, "A Comparison of Efficiency of Multicandidate
Electoral Systems", American Journal of Political Science, vol. 28,
no. 1, pp. 23-48, 1984.  :doi:`10.2307/2110786`

Typical result:

          2      3       4       5       7       10
Plurality 100.0   79.1    68.4    61.8    51.4    41.1
Runoff    100.0   96.2    89.6    83.4    72.3    60.3
Hare      100.0   96.2    92.5    89.1    83.7    77.0
Approval  100.0   75.6    70.0    67.4    63.8    61.2
Borda     100.0   90.9    87.4    85.9    84.6    83.9
Coombs    100.0   96.9    93.4    91.0    86.4    81.7
Black     100.0  100.0   100.0   100.0   100.0   100.0
SU max    100.0   84.1    79.6    78.4    77.3    77.5
CW        100.0   91.7    83.1    75.6    64.3    52.9
"""
import time
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from elsim.methods import (fptp, runoff, irv, approval, borda, coombs,
                           black, utility_winner, condorcet)
from elsim.elections import random_utilities
from elsim.strategies import honest_rankings, approval_optimal

n = 10_000
n_voters = 25
n_cands_list = (2, 3, 4, 5, 7, 10)

ranked_methods = {'Plurality': fptp, 'Runoff': runoff, 'Hare': irv,
                  'Borda': borda, 'Coombs': coombs, 'Black': black}

rated_methods = {'SU max': utility_winner,
                 'Approval': lambda utilities, tiebreaker:
                     approval(approval_optimal(utilities), tiebreaker)}

count = {key: Counter() for key in (ranked_methods.keys() |
                                    rated_methods.keys() | {'CW'})}

start_time = time.monotonic()

for iteration in range(n):
    for n_cands in n_cands_list:
        utilities = random_utilities(n_voters, n_cands)

        """
        "Simulated utilities were normalized by range, that is, each voter's
        set of utilities were linearly expanded so that the highest and lowest
        utilities for each voter were 1 and 0, respectively."

        This is necessary for the SU Maximizer results to match Merrill's.
        """
        utilities -= utilities.min(1)[:, np.newaxis]
        utilities /= utilities.max(1)[:, np.newaxis]

        rankings = honest_rankings(utilities)

        # If there is a Condorcet winner, analyze election, otherwise skip it
        CW = condorcet(rankings)
        if CW is not None:
            count['CW'][n_cands] += 1

            for name, func in ranked_methods.items():
                if func(rankings, tiebreaker='random') == CW:
                    count[name][n_cands] += 1

            for name, func in rated_methods.items():
                if func(utilities, tiebreaker='random') == CW:
                    count[name][n_cands] += 1

elapsed_time = time.monotonic() - start_time
print('Elapsed:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)), '\n')

# Plot Merrill's results as dots for comparison
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

plt.figure(f'Figure 1. {n_voters} voters, {n} iterations')
for method in ('Plurality', 'Runoff', 'Hare', 'Approval', 'Borda', 'Coombs',
               'Black'):
    x, y = zip(*sorted(merrill_table_1[method].items()))
    plt.plot(x, y, '.')

# Restart color cycle, so result colors match
plt.gca().set_prop_cycle(None)

# Number of candidates
print('', *n_cands_list, sep='\t')

# Of those elections with CW, likelihood that method chooses CW
x_cw, y_cw = zip(*sorted(count['CW'].items()))
for method in ('Plurality', 'Runoff', 'Hare', 'Approval', 'Borda', 'Coombs',
               'Black'):
    x, y = zip(*sorted(count[method].items()))
    plt.plot(x, np.array(y)/y_cw*100, '-', label=method)
    print(method + '\t', '\t'.join(f'{v:.1f}' for v in np.array(y)/y_cw*100))

# Likelihood that social utility maximizer is Condorcet Winner
x, y = zip(*sorted(count['SU max'].items()))
print('SU max\t', '\t'.join(f'{v:.1f}' for v in np.array(y)/y_cw*100))

# Likelihood of Condorcet Winner (normalized by n iterations)
print('CW\t', '\t'.join(f'{v:.1f}' for v in np.asarray(y_cw)/n*100))

plt.legend()
plt.grid(True, color='0.7', linestyle='-', which='major', axis='both')
plt.grid(True, color='0.9', linestyle='-', which='minor', axis='both')
plt.ylim(40, 102)
plt.xlim(1.8, 10.2)
