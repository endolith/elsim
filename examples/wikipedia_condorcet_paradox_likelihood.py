"""
Reproduce table from Wikipedia: Condorcet paradox

https://en.wikipedia.org/wiki/Condorcet_paradox#Impartial_culture_model

Example output with iterations = 1_000_000:

            3    101     201     301     401     501     601
    WP  5.556  8.690   8.732   8.746   8.753   8.757   8.760
    Sim 5.558  8.684   8.729   8.755   8.740   8.761   8.757

With iterations = 10_000_000 (which takes forever):

            3    101     201     301     401     501     601
    WP  5.556  8.690   8.732   8.746   8.753   8.757   8.760
    Sim 5.553  8.682   8.748   8.743   8.747   8.759   8.752
"""
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
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
iterations = 50_000
n_cands = 3
is_CP = Counter()  # Is there a Condorcet paradox?
for n_voters in WP_table:
    for iteration in range(iterations):
        election = impartial_culture(n_voters, n_cands)
        CW = condorcet(election)
        if CW is None:
            is_CP[n_voters] += 1

x = list(WP_table.keys())
y = list(WP_table.values())
plt.plot(x, y, label='WP')

x = list(is_CP.keys())
y = list(is_CP.values())
y = np.asarray(y) / iterations * 100  # Percent
plt.plot(x, y, '-', label='Simulation')

plt.legend()
plt.grid(True, color='0.7', linestyle='-', which='major', axis='both')
plt.grid(True, color='0.9', linestyle='-', which='minor', axis='both')

# Number of candidates
print('   ', *(f'{v: >5}' for v in x), sep='\t')

# Likelihood of Condorcet Winner
print('WP ', *(f'{t:.3f}' for t in WP_table.values()), sep='\t')
print('Sim', *(f'{t:.3f}' for t in y), sep='\t')
