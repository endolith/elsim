"""
Reproduce Table 1

Limiting Values of Probabilities that a Given Alternative Wins, and that There
is No Majority Winner, for Equally Likely Rank Orders

from

Niemi, R. G.; Weisberg, H. (1968). "A mathematical solution for the probability
of the paradox of voting". Behavioral Science. 13 (4): 317-323.
:doi:`10.1002/bs.3830130406` PMID 5663898.

Example output with n_elections = 1_000, n_voters = 100_000 (16 minutes):

|       |     2 |     3 |     4 |     5 |     6 |    10 |    23 |    49 |
|:------|------:|------:|------:|------:|------:|------:|------:|------:|
| Niemi | 0.000 | 0.088 | 0.175 | 0.251 | 0.315 | 0.489 | 0.712 | 0.841 |
| Sim   | 0.003 | 0.091 | 0.188 | 0.255 | 0.328 | 0.511 | 0.695 | 0.855 |
| Diff  | 0.003 | 0.003 | 0.013 | 0.004 | 0.013 | 0.022 | 0.017 | 0.014 |

More accuracy with n_elections = 10_000, n_voters = 100_000 (3 minutes):

|       |     2 |     3 |     4 |     5 |     6 |     7 |
|:------|------:|------:|------:|------:|------:|------:|
| Niemi | 0.000 | 0.088 | 0.175 | 0.251 | 0.315 | 0.369 |
| Sim   | 0.002 | 0.095 | 0.179 | 0.253 | 0.321 | 0.372 |
| Diff  | 0.002 | 0.008 | 0.004 | 0.002 | 0.006 | 0.003 |

Many candidates, with n_elections = 10_000, n_voters = 100_000 (3.5 hours)

|       |    10 |    23 |    49 |
|:------|------:|------:|------:|
| Niemi | 0.489 | 0.712 | 0.841 |
| Sim   | 0.488 | 0.728 | 0.843 |
| Diff  | 0.001 | 0.015 | 0.003 |

"""
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tabulate import tabulate

from elsim.elections import impartial_culture
from elsim.methods import condorcet

# Probability That There Is No Majority Winner
niemi_table = [.0000, .0000, .0877, .1755, .2513, .3152, .3692, .4151, .4545,
               .4887, .5187, .5452, .5687, .5898, .6087, .6259, .6416, .6559,
               .6690, .6811, .6923, .7027, .7123, .7213, .7297, .7376, .7451,
               .7520, .7586, .7648, .7707, .7763, .7816, .7866, .7911, .7960,
               .8004, .8045, .8085, .8123, .8160, .8195, .8228, .8261, .8292,
               .8322, .8351, .8379, .8405]
niemi_table = dict(enumerate(niemi_table[1:], start=2))

# It needs many simulations to get similar accuracy as the analytical results
n_elections = 2_000  # Roughly 30 seconds
n_voters = 100_000  # m = infinity
n_cands_list = (2, 3, 4, 5, 6, 7)  # 49 takes many minutes

# Simulate more than just one election per worker to improve efficiency
batch_size = 100
n_batches = n_elections // batch_size
assert n_batches * batch_size == n_elections


def simulate_batch(n_cands):
    condorcet_paradox_count = Counter()
    # Reuse the same chunk of memory to save time
    election = np.empty((n_voters, n_cands), dtype=np.uint8)
    for iteration in range(batch_size):
        election[:] = impartial_culture(n_voters, n_cands)
        CW = condorcet(election)
        if CW is None:
            condorcet_paradox_count[n_cands] += 1
    return condorcet_paradox_count


jobs = []
for n_cands in n_cands_list:
    jobs.extend([delayed(simulate_batch)(n_cands)] * n_batches)

print(f'{len(jobs)} tasks total:')
results = Parallel(n_jobs=-3, verbose=5)(jobs)
condorcet_paradox_counts = sum(results, Counter())

x, y = zip(*niemi_table.items())
plt.plot(x, y, label='Niemi')

x, y = zip(*sorted(condorcet_paradox_counts.items()))
y = np.asarray(y) / n_elections  # Percent likelihood of paradox
plt.plot(x, y, '.', label='Simulation')

plt.legend()
plt.grid(True, color='0.7', linestyle='-', which='major', axis='both')
plt.grid(True, color='0.9', linestyle='-', which='minor', axis='both')
plt.show()

table = [{k: v for k, v in niemi_table.items()
          if k in condorcet_paradox_counts},
         {k: v/n_elections for k, v in condorcet_paradox_counts.items()},
         {k: abs(niemi_table[k]-v/n_elections)
          for k, v in condorcet_paradox_counts.items()}]
print(tabulate(table, "keys", showindex=['Niemi', 'Sim', 'Diff'],
               tablefmt="pipe", floatfmt='.3f'))
