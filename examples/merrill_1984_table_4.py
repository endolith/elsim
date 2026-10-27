"""
Reproduce Table 4

Social Utility Efficiencies under […] Spatial Model Assumptions
(201 voters, 5 candidates)

from

S. Merrill III, "A Comparison of Efficiency of Multicandidate
Electoral Systems", American Journal of Political Science, vol. 28,
no. 1, pp. 23-48, 1984.  :doi:`10.2307/2110786`

(Not including random society column)

Typical result with 100_000 elections:

| Disp      |   1.0 |   1.0 |   1.0 |   1.0 |   0.5 |   0.5 |   0.5 |   0.5 |
| Corr      |   0.5 |   0.5 |   0.0 |   0.0 |   0.5 |   0.5 |   0.0 |   0.0 |
| Dims      |     2 |     4 |     2 |     4 |     2 |     4 |     2 |     4 |
|:----------|------:|------:|------:|------:|------:|------:|------:|------:|
| Plurality |  67.1 |  75.1 |  76.6 |  90.3 |   3.0 |   7.1 |  23.2 |  49.6 |
| Runoff    |  88.3 |  92.6 |  89.6 |  96.3 |  35.0 |  43.4 |  51.4 |  73.3 |
| Hare      |  89.8 |  93.0 |  92.7 |  97.5 |  42.5 |  55.9 |  55.0 |  81.4 |
| Approval  |  95.7 |  97.0 |  96.6 |  98.3 |  95.3 |  96.4 |  95.1 |  97.5 |
| Borda     |  97.6 |  98.2 |  98.0 |  99.3 |  96.1 |  97.4 |  96.9 |  98.6 |
| Coombs    |  96.0 |  96.7 |  97.1 |  98.1 |  92.8 |  93.3 |  94.1 |  95.6 |
| Black     |  96.6 |  97.2 |  97.6 |  98.7 |  94.9 |  95.9 |  96.1 |  97.6 |

The paper describes 200 candidate sets, each combined with five electorates,
but it does not state whether the reported SUE is the mean of the 200 per-set
SUEs or a single ratio pooled over all 1,000 elections.  Our simulations are
closer to Table 4 when we average per set, so that is likely the estimator
Merrill used.  This script reports that per-set mean; pooling all elections
instead gives a ratio of expectations.  Using 2,000 sets reduces our sampling
error.  The paper's 200-set cells carry an estimated 3-4 percentage points of
standard error at low dispersion, so agreement should be judged against that
larger uncertainty rather than exact equality; some remaining differences are
still a few standard errors.
"""
import random
import time

import numpy as np
from tabulate import tabulate

from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.methods import (approval, black, borda, coombs, fptp, irv, runoff)
from elsim.strategies import approval_optimal, honest_rankings

n_elections = 10_000  # Roughly 60 seconds on a 2019 6-core i7-9750H
n_voters = 201
n_cands = 5
n_electorates_per_set = 5
n_candidate_sets = n_elections // n_electorates_per_set
assert n_candidate_sets * n_electorates_per_set == n_elections

seed = 1984
random.seed(seed)
rng = np.random.default_rng(seed)

ranked_methods = {'Plurality': fptp, 'Runoff': runoff, 'Hare': irv,
                  'Borda': borda, 'Coombs': coombs, 'Black': black}

rated_methods = {'Approval': lambda utilities, tiebreaker:
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

    per_set_sue = {name: [] for name in (*rated_methods, *ranked_methods)}

    for _set in range(n_candidate_sets):
        # Merrill combines each candidate set with five electorates; this
        # script draws a new electorate sample for each set.
        _, c = normal_electorate(0, n_cands, dims=D, corr=corr, disp=disp,
                                 random_state=rng)
        utility_sums = {name: 0.0 for name in per_set_sue}
        max_utility_sum = 0.0
        random_utility_sum = 0.0

        for _ in range(n_electorates_per_set):
            v, _ = normal_electorate(n_voters, n_cands, dims=D, corr=corr,
                                     disp=disp, random_state=rng)

            # Simulated utilities were normalized by range, i.e. each
            # voter's utilities were linearly expanded so that their highest
            # and lowest utilities were 1 and 0.
            utilities = normed_dist_utilities(v, c)
            rankings = honest_rankings(utilities)
            utility_sum = utilities.sum(axis=0)

            max_utility_sum += utility_sum.max()
            random_utility_sum += utility_sum.mean()

            for name, method in rated_methods.items():
                winner = method(utilities, tiebreaker='random')
                utility_sums[name] += utility_sum[winner]

            for name, method in ranked_methods.items():
                winner = method(rankings, tiebreaker='random')
                utility_sums[name] += utility_sum[winner]

        # This script computes social-utility efficiency for each candidate
        # set (its five electorates combined) and averages over sets. The
        # paper does not state whether it averaged or pooled; per-set
        # averaging is closer to Table 4.
        denominator = max_utility_sum - random_utility_sum
        for name in per_set_sue:
            per_set_sue[name].append(
                (utility_sums[name] - random_utility_sum)
                / denominator * 100)

    results.append({name: np.mean(values)
                    for name, values in per_set_sue.items()})

elapsed_time = time.monotonic() - start_time
print('Elapsed:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)), '\n')

# Neither Tabulate nor Markdown support column span or multiple headers, but
# at least this prints to plain text in a readable way.
header = ['Disp\nCorr\nDims'] + [f'{x}\n{y}\n{z}' for x, y, z in conditions]

table = []
for method in ('Plurality', 'Runoff', 'Hare', 'Approval', 'Borda', 'Coombs',
               'Black'):
    y = np.array([c[method] for c in results])
    table.append([method, *y])
print(tabulate(table, header, tablefmt="pipe", floatfmt='.1f'))
