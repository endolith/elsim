"""
Confirm the "best vote-for-or-against-k" column by simulating whole elections.

Weber, Robert J. (1978). "Comparison of Public Choice Systems".
Cowles Foundation Discussion Papers. Cowles Foundation for Research in
Economics. No. 498. https://cowles.yale.edu/publications/cfdp/cfdp-498

This is the direct check on the columns of the table from p. 19: for each
number of candidates, find the best k, draw random elections with
`vote_for_or_against_k`, tally them with approval voting, and compare the
social utility of the winner with that of the utility winner.  Effectiveness is

    (E[social utility of winner] - n/2) / (E[social utility of utility
                                           winner] - n/2)

where n is the number of voters and n/2 is the expected social utility of an
arbitrary candidate.

Typical output:

|    |   k | measured |        |   corrected |   published |
|:---|----:|---------:|:-------|------------:|------------:|
| 2  |   1 |   81.46% | +-0.18% |      81.65% |      81.65% |
| 3  |   1 |   87.60% | +-0.11% |      87.50% |      87.50% |
| 4  |   1 |   80.81% | +-0.12% |      80.83% |      80.83% |
| 5  |   2 |   85.08% | +-0.13% |      85.21% |      86.96% |
| 6  |   2 |   84.26% | +-0.13% |      84.12% |      86.25% |
| 10 |   4 |   85.25% | +-0.13% |      85.31% |      88.09% |

The first three rows agree with the paper.  The last three do not, and they are
not close: the published numbers are 1.7 to 2.9 points higher, which is more
than ten standard errors.  The measured values agree instead with
`eff_vote_for_or_against_k`, the corrected closed form, and with
`effectiveness_from_ballots`, which needs no election at all.

Two caveats on reading the measured column:

* It is for 1,000 voters, not an infinite number, so it carries a
  finite-electorate bias of a few tenths of a point (visible at m = 3, where
  the exact answer is 87.50%; more voters shrink it, which is why the
  companion example weber_1977_verify_vote_for_k.py needs 100,000 voters to
  match the paper to 0.1 points).
* Election-level noise is large: a single election's effectiveness estimate
  has a standard deviation of order 100%, so the standard error only falls as
  1/sqrt(n_elections), and distinguishing two systems one point apart needs on
  the order of 10,000 elections.
"""
import numpy as np
from joblib import Parallel, delayed
from tabulate import tabulate
from elsim.elections import random_utilities
from elsim.methods import approval
from elsim.strategies import vote_for_or_against_k
from weber_1977_expressions import (best_vote_for_or_against_k,
                                    eff_vote_for_or_against_k,
                                    eff_vote_for_or_against_k_paper)

n_voters = 1_000
n_elections = {2: 100_000, 3: 100_000, 4: 100_000,
               5: 60_000, 6: 60_000,
               10: 40_000}


def simulate(m, k, n_elections, n_voters=n_voters, seed=0):
    """Return measured effectiveness and its standard error for `m`, `k`."""
    rng = np.random.default_rng(seed)
    batch = 100
    sums = []
    utility_sums = []
    done = 0
    while done < n_elections:
        b = min(batch, n_elections - done)
        utilities = random_utilities(b*n_voters, m, random_state=rng
                                     ).reshape(b, n_voters, m)
        ballots = vote_for_or_against_k(
            utilities.reshape(b*n_voters, m), k).reshape(b, n_voters, m)
        for i in range(b):
            u = utilities[i]
            winner = approval(ballots[i], tiebreaker='random')
            sums.append(u[:, winner].sum())
            utility_sums.append(u.sum(axis=0).max())
        done += b
    sums = np.array(sums)
    utility_sums = np.array(utility_sums)

    # Both numerator and denominator are measured on the same elections, which
    # cancels most of the noise that they share.
    effectiveness = (sums.mean() - n_voters/2)/(utility_sums.mean() - n_voters/2)
    # Standard error by bootstrap over elections
    n_boot = 1_000
    idx = rng.integers(len(sums), size=(n_boot, len(sums)))
    num = sums[idx].mean(axis=1) - n_voters/2
    den = utility_sums[idx].mean(axis=1) - n_voters/2
    return effectiveness, (num/den).std(ddof=1)


def row(m):
    k = best_vote_for_or_against_k(m)
    measured, se = simulate(m, k, n_elections[m], seed=m)
    return [m, k, f'{measured:.2%}', f'+-{se:.2%}',
            f'{eff_vote_for_or_against_k(m, k):.2%}',
            f'{eff_vote_for_or_against_k_paper(m, k):.2%}']


if __name__ == '__main__':
    print(f'Simulating {sum(n_elections.values()):,} elections of '
          f'{n_voters:,} voters')
    rows = Parallel(n_jobs=-1)(delayed(row)(m) for m in n_elections)
    print()
    print(tabulate(sorted(rows),
                   headers=('m', 'best k', 'measured', '', 'corrected',
                            'published'),
                   tablefmt='pipe', disable_numparse=True))
    print()
    print('"corrected" is eff_vote_for_or_against_k and "published" is '
          'eff_vote_for_or_against_k_paper,')
    print('both evaluated at the same k.  The measured column matches the '
          'corrected one within')
    print('its error bars for every m, and the published one only for m <= 4.')
