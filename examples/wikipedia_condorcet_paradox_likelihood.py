"""
Reproduce table from Wikipedia: Condorcet paradox

https://en.wikipedia.org/wiki/Condorcet_paradox#Impartial_culture_model

Example output with n_elections = 1_000_000:

|      |     3 |   101 |   201 |   301 |   401 |   501 |   601 |
|:-----|------:|------:|------:|------:|------:|------:|------:|
| WP   | 5.556 | 8.690 | 8.732 | 8.746 | 8.753 | 8.757 | 8.760 |
| Sim  | 5.558 | 8.684 | 8.729 | 8.755 | 8.740 | 8.761 | 8.757 |
| Diff | 0.002 | 0.006 | 0.003 | 0.009 | 0.013 | 0.004 | 0.003 |

With n_elections = 10_000_000 (which takes forever):

|      |     3 |   101 |   201 |   301 |   401 |   501 |   601 |
|:-----|------:|------:|------:|------:|------:|------:|------:|
| WP   | 5.556 | 8.690 | 8.732 | 8.746 | 8.753 | 8.757 | 8.760 |
| Sim  | 5.553 | 8.682 | 8.748 | 8.743 | 8.747 | 8.759 | 8.752 |
| Diff | 0.003 | 0.008 | 0.016 | 0.003 | 0.006 | 0.002 | 0.008 |

"""
from collections import Counter
import numpy as np
from scipy.stats import binomtest
import matplotlib.pyplot as plt
from tabulate import tabulate
from joblib import Parallel, delayed
from elsim.methods import condorcet
from elsim.elections import impartial_culture

# Number of voters vs percent of elections with Condorcet paradox.
WP_table = {3:   5.556,
            101: 8.690,
            201: 8.732,
            301: 8.746,
            401: 8.753,
            501: 8.757,
            601: 8.760}

# It needs many simulations to get similar accuracy as the analytical results
n_elections = 300_000  # Roughly 30 seconds
n_cands = 3

# Simulate more than just one election per worker to improve efficiency
batch_size = 100
n_batches = n_elections // batch_size
assert n_batches * batch_size == n_elections


def simulate_batch(n_voters):
    condorcet_paradox_count = Counter()
    # Reuse the same chunk of memory to save time
    election = np.empty((n_voters, n_cands), dtype=np.uint8)
    for iteration in range(batch_size):
        election[:] = impartial_culture(n_voters, n_cands)
        CW = condorcet(election)
        if CW is None:
            condorcet_paradox_count[n_voters] += 1
    return condorcet_paradox_count


jobs = []
for n_voters in WP_table:
    jobs.extend([delayed(simulate_batch)(n_voters)] * n_batches)

print(f'{len(jobs)} tasks total:')
results = Parallel(n_jobs=-3, verbose=5)(jobs)
condorcet_paradox_counts = sum(results, Counter())

x, y = zip(*WP_table.items())
plt.plot(x, y, label='WP')

x, y = zip(*sorted(condorcet_paradox_counts.items()))
CP = np.asarray(y) / n_elections  # Likelihood of paradox

# Add 95% confidence interval error bars (Clopper-Pearson exact method)
ci = np.empty((2, len(y)))
for i in range(len(y)):
    ci[:, i] = binomtest(y[i], n_elections).proportion_ci()
yerr = ci - CP
yerr[0] = -yerr[0]
plt.errorbar(x, CP*100, yerr*100, fmt='-', label='Simulation')

plt.legend()
plt.grid(True, color='0.7', linestyle='-', which='major', axis='both')
plt.grid(True, color='0.9', linestyle='-', which='minor', axis='both')
plt.show()

table = [WP_table,
         {k: v/n_elections*100 for k, v in condorcet_paradox_counts.items()},
         {k: abs(WP_table[k]-v/n_elections*100)
          for k, v in condorcet_paradox_counts.items()}]
print(tabulate(table, "keys", showindex=['WP', 'Sim', 'Diff'], tablefmt="pipe",
               floatfmt='.3f'))
