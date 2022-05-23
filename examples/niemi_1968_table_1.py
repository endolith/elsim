"""
Reproduce Table 1

Limiting Values of Probabilities that a Given Alternative Wins, and that There
is No Majority Winner, for Equally Likely Rank Orders

from

Niemi, R. G.; Weisberg, H. (1968). "A mathematical solution for the probability
of the paradox of voting". Behavioral Science. 13 (4): 317–323.
:doi:`10.1002/bs.3830130406` PMID 5663898.

Example output with iterations = 1_000, n_voters = 100_000 (16 minutes):

|       |     2 |     3 |     4 |     5 |     6 |    10 |    23 |    49 |
|:------|------:|------:|------:|------:|------:|------:|------:|------:|
| Niemi | 0.000 | 0.088 | 0.175 | 0.251 | 0.315 | 0.489 | 0.712 | 0.841 |
| Sim   | 0.003 | 0.091 | 0.188 | 0.255 | 0.328 | 0.511 | 0.695 | 0.855 |
| Diff  | 0.003 | 0.003 | 0.013 | 0.004 | 0.013 | 0.022 | 0.017 | 0.014 |

More accuracy with iterations = 10_000, n_voters = 100_000 (3 minutes):

|       |     2 |     3 |     4 |     5 |     6 |     7 |
|:------|------:|------:|------:|------:|------:|------:|
| Niemi | 0.000 | 0.088 | 0.175 | 0.251 | 0.315 | 0.369 |
| Sim   | 0.002 | 0.095 | 0.179 | 0.253 | 0.321 | 0.372 |
| Diff  | 0.002 | 0.008 | 0.004 | 0.002 | 0.006 | 0.003 |

"""
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from joblib import Parallel, delayed
from elsim.methods import condorcet
from elsim.elections import impartial_culture

# Probability That There Is No Majority Winner
niemi_table = [.0000, .0000, .0877, .1755, .2513, .3152, .3692, .4151, .4545,
               .4887, .5187, .5452, .5687, .5898, .6087, .6259, .6416, .6559,
               .6690, .6811, .6923, .7027, .7123, .7213, .7297, .7376, .7451,
               .7520, .7586, .7648, .7707, .7763, .7816, .7866, .7911, .7960,
               .8004, .8045, .8085, .8123, .8160, .8195, .8228, .8261, .8292,
               .8322, .8351, .8379, .8405]
niemi_table = dict(enumerate(niemi_table[1:], start=2))

# It needs many iterations to get similar accuracy as the analytical results
iterations = 2_000  # Roughly 30 seconds
n_voters = 100_000  # m = infinity
n_cands_list = (2, 3, 4, 5, 6, 7)  # 49 takes many minutes

# Do more than just one iteration per worker to improve efficiency
batch = 10
n = iterations // batch
assert n * batch == iterations


def func():
    is_CP = Counter()  # Is there a Condorcet paradox?
    for n_cands in n_cands_list:
        # Reuse the same chunk of memory to save time
        election = np.empty((n_voters, n_cands), dtype=np.uint8)
        for iteration in range(batch):
            election[:] = impartial_culture(n_voters, n_cands)
            CW = condorcet(election)
            if CW is None:
                is_CP[n_cands] += 1
    return is_CP


p = Parallel(n_jobs=-3, verbose=5)(delayed(func)() for i in range(n))
is_CP = sum(p, Counter())

x, y = zip(*niemi_table.items())
plt.plot(x, y, label='Niemi')

x, y = zip(*sorted(is_CP.items()))
y = np.asarray(y) / iterations  # Percent likelihood of paradox
plt.plot(x, y, '.', label='Simulation')

plt.legend()
plt.grid(True, color='0.7', linestyle='-', which='major', axis='both')
plt.grid(True, color='0.9', linestyle='-', which='minor', axis='both')

table = [{k: v for k, v in niemi_table.items() if k in is_CP},
         {k: v/iterations for k, v in is_CP.items()},
         {k: abs(niemi_table[k]-v/iterations) for k, v in is_CP.items()}]
print(tabulate(table, "keys", showindex=['Niemi', 'Sim', 'Diff'],
               tablefmt="pipe", floatfmt='.3f'))
