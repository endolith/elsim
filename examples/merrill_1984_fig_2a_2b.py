"""
Reproduce Figures 2.a and 2.b

Simulated Scatter Plot of 201 Voters and Five Candidates from Bivariate Normal
Distributions
(relative dispersion = 1.0 or .5)

from

S. Merrill III, "A Comparison of Efficiency of Multicandidate
Electoral Systems", American Journal of Political Science, vol. 28,
no. 1, pp. 23-48, 1984.  :doi:`10.2307/2110786`
"""
import matplotlib.pyplot as plt

from elsim.elections import normal_electorate

n_voters = 201
n_cands = 5

# "(The correlation for this scatter plot is .5; the distribution has been
# rotated to principal axes.)"
corr = 0.5

for disp in (0.5, 1.0):
    voters, cands = normal_electorate(n_voters, n_cands, dims=2, corr=corr,
                                      disp=disp)

    plt.figure(f'Dispersion {disp}')
    plt.scatter(voters[:, 0], voters[:, 1], marker='.', label='Voters')
    plt.scatter(cands[:, 0], cands[:, 1], marker='o', label='Candidates')

    plt.grid(True, color='0.7', linestyle='-', which='major')
    plt.grid(True, color='0.9', linestyle='-', which='minor')
    plt.legend()
    plt.axis('square')
    plt.axis([-5, 5, -5, 5])
    plt.show()
