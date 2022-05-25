"""
Reproduce Table 2

Probabilities of No Majority Winner, P(m, n), for Equally Likely Rank Orders

from

Niemi, R. G.; Weisberg, H. (1968). "A mathematical solution for the probability
of the paradox of voting". Behavioral Science. 13 (4): 317–323.
:doi:`10.1002/bs.3830130406` PMID 5663898.

(Not including m = ∞ case which is covered by niemi_1968_table_1.py)

Example output with iterations = 1_000_000:

|    |      3 |      5 |      7 |      9 |     11 |     13 |     15 |     17 |
|---:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
|  3 | 0.0558 | 0.0698 | 0.0755 | 0.0779 | 0.0799 | 0.0812 | 0.0819 | 0.0826 |
|  4 | 0.1110 | 0.1388 | 0.1506 | 0.1556 | 0.1599 | 0.1625 | 0.1641 | 0.1653 |
|  5 | 0.1599 | 0.1996 | 0.2146 | 0.2230 | 0.2288 | 0.2331 | 0.2347 | 0.2367 |
|  6 | 0.2023 | 0.2518 | 0.2707 | 0.2814 | 0.2884 | 0.2925 | 0.2948 | 0.2974 |

"""
from collections import Counter
import numpy as np
from tabulate import tabulate
from joblib import Parallel, delayed
from elsim.methods import condorcet
from elsim.elections import impartial_culture

# It needs many iterations to get similar accuracy as the analytical results
iterations = 100_000  # Roughly 30 seconds
n_voters_list = (3, 5, 7, 9, 11, 13, 15, 17)  # , 19, 21, 23, 25, 27, 29, 59)
n_cands_list = (3, 4, 5, 6)

# Do more than just one iteration per worker to improve efficiency
batch = 10
n = iterations // batch
assert n * batch == iterations


def func():
    is_CP = Counter()  # Is there a Condorcet paradox?
    for n_voters in n_voters_list:
        for n_cands in n_cands_list:
            # Reuse the same chunk of memory to save time
            election = np.empty((n_voters, n_cands), dtype=np.uint8)
            for iteration in range(batch):
                election[:] = impartial_culture(n_voters, n_cands)
                CW = condorcet(election)
                if CW is None:
                    is_CP[n_cands, n_voters] += 1
    return is_CP


p = Parallel(n_jobs=-3, verbose=5)(delayed(func)() for i in range(n))
is_CP = sum(p, Counter())

nm, P = zip(*sorted(is_CP.items()))
P = np.asarray(P) / iterations  # Percent likelihood of paradox

table = []
for n in n_cands_list:
    row = [q / iterations for (x, y), q in sorted(is_CP.items()) if x == n]
    table.append(row)

print(tabulate(table, n_voters_list, tablefmt="pipe", showindex=n_cands_list,
               floatfmt='.4f'))
