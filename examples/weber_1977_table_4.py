"""
Reproduce Table 4 from p. 17 of Comparison of Voting Systems.

The expected social utility of the elected candidate, under three voting
systems.

from

Weber, Robert J. (1978). "Comparison of Public Choice Systems".
Cowles Foundation Discussion Papers. Cowles Foundation for Research in
Economics. No. 498. https://cowles.yale.edu/publications/cfdp/cfdp-498

Typical result with n = 1_000_000:

|    |   Standard |   Borda |   Approval |
|---:|-----------:|--------:|-----------:|
|  2 |     1.2505 |  1.2922 |     1.2920 |
|  3 |     1.8330 |  1.8748 |     1.8644 |
|  4 |     2.3889 |  2.4236 |     2.4210 |
|  5 |     2.9171 |  2.9768 |     2.9727 |
| 10 |     5.5970 |  5.6701 |     5.6714 |
| 15 |     8.2248 |  8.3207 |     8.3246 |
| 20 |    10.8325 | 10.9470 |    10.9537 |
| 25 |    13.4320 | 13.5583 |    13.5659 |
| 30 |    16.0177 | 16.1577 |    16.1669 |

"""
import time
from collections import Counter
import numpy as np
from tabulate import tabulate
from elsim.methods import fptp, borda, approval
from elsim.elections import random_utilities
from elsim.strategies import honest_rankings, approval_optimal

n = 30_000  # Roughly 30 seconds
n_voters_list = (2, 3, 4, 5, 10, 15, 20, 25, 30)
n_cands = 3

ranked_methods = {'Standard': fptp, 'Borda': borda}
rated_methods = {'Approval': lambda utilities, tiebreaker:
                 approval(approval_optimal(utilities), tiebreaker)}

count = {key: Counter() for key in (ranked_methods.keys() |
                                    rated_methods.keys())}

start_time = time.monotonic()

for iteration in range(n):
    for n_voters in n_voters_list:
        utilities = random_utilities(n_voters, n_cands)

        for name, method in rated_methods.items():
            winner = method(utilities, tiebreaker='random')
            count[name][n_voters] += utilities.sum(axis=0)[winner]

        rankings = honest_rankings(utilities)
        for name, method in ranked_methods.items():
            winner = method(rankings, tiebreaker='random')
            count[name][n_voters] += utilities.sum(axis=0)[winner]

elapsed_time = time.monotonic() - start_time
print('Elapsed:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)), '\n')

table = {}

# Calculate the expected utilities over all iterations
for method in ('Standard', 'Borda', 'Approval'):
    x, y = zip(*sorted(count[method].items()))
    table.update({method: np.array(y) / n})

print(tabulate(table, 'keys', showindex=n_voters_list,
               tablefmt="pipe", floatfmt='.4f'))
