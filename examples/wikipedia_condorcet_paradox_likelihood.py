"""
Reproduce table from Wikipedia: Condorcet paradox

https://en.wikipedia.org/wiki/Condorcet_paradox#Impartial_culture_model

Example output with iterations = 1_000_000:

|      |     3 |   101 |   201 |   301 |   401 |   501 |   601 |
|:-----|------:|------:|------:|------:|------:|------:|------:|
| WP   | 5.556 | 8.690 | 8.732 | 8.746 | 8.753 | 8.757 | 8.760 |
| Sim  | 5.558 | 8.684 | 8.729 | 8.755 | 8.740 | 8.761 | 8.757 |
| Diff | 0.002 | 0.006 | 0.003 | 0.009 | 0.013 | 0.004 | 0.003 |

With iterations = 10_000_000 (which takes forever):

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

# It needs many iterations to get similar accuracy as the analytical results
iterations = 300_000  # Roughly 30 seconds
n_cands = 3

# Do more than just one iteration per worker to improve efficiency
batch = 100
n = iterations // batch
assert n * batch == iterations


def func():
    is_CP = Counter()  # Is there a Condorcet paradox?
    for n_voters in WP_table:
        # Reuse the same chunk of memory to save time
        election = np.empty((n_voters, n_cands), dtype=np.uint8)
        for _ in range(batch):
            election[:] = impartial_culture(n_voters, n_cands)
            CW = condorcet(election)
            if CW is None:
                is_CP[n_voters] += 1
    return is_CP


p = Parallel(n_jobs=-3, verbose=5)(delayed(func)() for _ in range(n))
is_CP = sum(p, Counter())

x, y = zip(*WP_table.items())
plt.plot(x, y, label='WP')

x, y = zip(*sorted(is_CP.items()))
CP = np.asarray(y) / iterations  # Likelihood of paradox

# Add 95% confidence interval error bars (Clopper-Pearson exact method)
ci = np.empty((2, len(y)))
for i in range(len(y)):
    ci[:, i] = binomtest(y[i], iterations).proportion_ci()
yerr = ci - CP
yerr[0] = -yerr[0]
plt.errorbar(x, CP*100, yerr*100, fmt='-', label='Simulation')

plt.legend()
plt.grid(True, color='0.7', linestyle='-', which='major', axis='both')
plt.grid(True, color='0.9', linestyle='-', which='minor', axis='both')

table = [WP_table,
         {k: v/iterations*100 for k, v in is_CP.items()},
         {k: abs(WP_table[k]-v/iterations*100) for k, v in is_CP.items()}]
print(tabulate(table, "keys", showindex=['WP', 'Sim', 'Diff'], tablefmt="pipe",
               floatfmt='.3f'))
