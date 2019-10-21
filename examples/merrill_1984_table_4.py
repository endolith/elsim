"""
Reproduce Table 4

Social Utility Efficiencies under Random Society and
Spatial Model Assumptions
(201 voters, 5 candidates)

from

S. Merrill III, "A Comparison of Efficiency of Multicandidate
Electoral Systems", American Journal of Political Science, vol. 28,
no. 1, pp. 23-48, 1984.  :doi:`10.2307/2110786`

Typical result:
Disp:       1.0   1.0     1.0     1.0     0.5     0.5     0.5     0.5
Corr:       0.5   0.5     0.0     0.0     0.5     0.5     0.0     0.0
D:          2.0   4.0     2.0     4.0     2.0     4.0     2.0     4.0
---------------------------------------------------------------------
Plurality  75.9  81.2    81.5    92.7    14.9    18.9    30.2    54.4
Runoff     91.7  94.7    92.2    97.6    43.6    51.3    56.1    75.9
Hare       92.8  95.2    94.6    98.4    52.5    62.9    61.2    84.1
Approval   96.7  97.4    97.0    98.6    96.2    97.2    96.0    98.0
Borda      98.1  98.7    98.5    99.5    97.1    98.0    97.5    99.0
Coombs     97.3  97.6    97.9    98.9    94.6    94.8    95.5    96.7
Black      97.6  97.9    98.2    99.1    95.9    96.5    96.7    98.0
"""
import time
from collections import Counter
import numpy as np
from elsim.methods import (fptp, runoff, irv, approval, borda, coombs,
                           black, utility_winner)
from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.strategies import honest_rankings, approval_optimal

n = 10_000
n_voters = 201
n_cands = 5

ranked_methods = {'Plurality': fptp, 'Runoff': runoff, 'Hare': irv,
                  'Borda': borda, 'Coombs': coombs, 'Black': black}

rated_methods = {'SU max': utility_winner,
                 'Approval': lambda utilities, tiebreaker:
                     approval(approval_optimal(utilities), tiebreaker)}

start_time = time.monotonic()

#             disp, corr, D
conditions = ((1.0, 0.5, 2),
              (1.0, 0.5, 4),
              (1.0, 0.0, 2),
              (1.0, 0.0, 4),
              (0.5, 0.5, 2),
              (0.5, 0.5, 4),
              (0.5, 0.0, 2),
              (0.5, 0.0, 4),
              )

results = []

for disp, corr, D in conditions:
    print(disp, corr, D)

    count = Counter()

    for iteration in range(n):
        v, c = normal_electorate(n_voters, n_cands, dims=D, corr=corr,
                                 disp=disp)

        """
        "Simulated utilities were normalized by range, that is, each voter's
        set of utilities were linearly expanded so that the highest and lowest
        utilities for each voter were 1 and 0, respectively."

        TODO: standard scores vs normalized don't matter for the ranked systems
        and don't affect approval much

        but This is necessary for the SU Maximizer results to match Merrill's.
        """
        utilities = normed_dist_utilities(v, c)
        rankings = honest_rankings(utilities)

        # Find the social utility winner and accumulate utilities
        UW = utility_winner(utilities)
        count['UW'] += utilities.sum(0)[UW]

        for name, func in rated_methods.items():
            winner = func(utilities, tiebreaker='random')
            count[name] += utilities.sum(0)[winner]

        for name, func in ranked_methods.items():
            winner = func(rankings, tiebreaker='random')
            count[name] += utilities.sum(0)[winner]

    results.append(count)

elapsed_time = time.monotonic() - start_time
print('Elapsed:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)), '\n')

print('Disp:    ', '\t'.join(f'{v: >5.1f}' for v in np.array(conditions).T[0]))
print('Corr:    ', '\t'.join(f'{v: >5.1f}' for v in np.array(conditions).T[1]))
print('D:       ', '\t'.join(f'{v: >5.1f}' for v in np.array(conditions).T[2]))
print('-'*69)

# Calculate Social Utility Efficiency from summed utilities
y_uw = np.array([c['SU max'] for c in results])

for method in ('Plurality', 'Runoff', 'Hare', 'Approval', 'Borda', 'Coombs',
               'Black'):
    y = np.array([c[method] for c in results])
    SUE = (np.array(y) - n_voters * n / 2)/(np.array(y_uw) - n_voters * n / 2)
    print(f'{method: <9}', '\t'.join(f'{v: >5.1f}' for v in SUE*100))
