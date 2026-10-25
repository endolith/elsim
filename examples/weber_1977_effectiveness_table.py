"""
Reproduce table: "The Effectiveness of Several Voting Systems".

Weber, Robert J. (1978). "Comparison of Public Choice Systems".
Cowles Foundation Discussion Papers. Cowles Foundation for Research in
Economics. No. 498. https://cowles.yale.edu/publications/cfdp/cfdp-498

This reproduces the table from p. 19 without simulating any elections.  For a
random society the effectiveness of any system in which the candidate with the
largest total weight wins is

    eff = Corr(u_A - u_B, w_A - w_B)

the correlation, over a random voter and a random pair of candidates A and B,
between how much more the voter likes A than B and how much more weight they
give A than B (see `weber_1977_expressions.effectiveness_from_ballots`).  So a
million random voters and the ballot rules in `elsim.strategies` are enough,
and there is no noise from tallying: the results below are good to about 0.02
points.

|    |   Standard |   Vote-for-half |   Best VFOAK |   Borda |
|:---|-----------:|----------------:|-------------:|--------:|
| 2  |     81.63% |          81.63% |       81.63% |  81.63% |
| 3  |     75.00% |          75.00% |       87.49% |  86.60% |
| 4  |     69.26% |          79.98% |       80.83% |  89.43% |
| 5  |     64.55% |          79.07% |       85.22% |  91.30% |
| 6  |     60.60% |          81.30% |       84.10% |  92.56% |
| 10 |     49.80% |          83.00% |       85.32% |  95.35% |

(Compare weber_1977_expressions.py, which computes the closed forms instead
and gets 81.65% / 75.00% / 69.28% / 64.55% / 60.61% / 49.79% for Standard, and
the paper's published 81.65% / 87.50% / 80.83% / 86.96% / 86.25% / 88.09% for
"best vote-for-or-against-k".)

The "best vote-for-or-against-k" column is where the paper is wrong: for
m <= 4 it agrees with the paper, and from m = 5 on it does not, because the
paper's closed form for that system is wrong (see
weber_1977_expressions.eff_vote_for_or_against_k for where and why).  The
ballot rule is the one the paper describes, and it is implemented in
`elsim.strategies.vote_for_or_against_k`; the discrepancy is in the paper's
arithmetic, not in the simulation.  weber_1977_verify_vote_for_or_against_k.py
runs whole elections and reaches the same conclusion.

The other three columns reproduce the paper to within 0.03 points.
"""
import time
import numpy as np
from tabulate import tabulate
from elsim.strategies import vote_for_k, vote_for_or_against_k
from weber_1977_expressions import (best_vote_for_or_against_k, eff_borda,
                                    eff_standard, eff_vote_for_half,
                                    eff_vote_for_or_against_k_paper,
                                    eff_vote_for_k,
                                    effectiveness_from_ballots)

m_cands_list = (2, 3, 4, 5, 6, 10)
n_voters = 1_000_000

start_time = time.monotonic()


def plump(utilities):
    """Standard (plurality) voting: approve only your favorite."""
    return vote_for_k(utilities, 1)


def vote_for_half(utilities):
    return vote_for_k(utilities, utilities.shape[1]//2)


def borda(utilities):
    m = utilities.shape[1]
    return m - np.argsort(np.argsort(-utilities, axis=1), axis=1)


print(f'Measuring each column from {n_voters:,} random voters '
      f'({time.monotonic() - start_time:.1f}s)')
best_k = [best_vote_for_or_against_k(m) for m in m_cands_list]
measured = {
    'Standard': [effectiveness_from_ballots(plump, m, n_voters)
                 for m in m_cands_list],
    'Vote-for-half': [effectiveness_from_ballots(vote_for_half, m, n_voters)
                      for m in m_cands_list],
    'Best VFOAK': [effectiveness_from_ballots(
        lambda u, k=k: vote_for_or_against_k(u, k), m, n_voters)
        for m, k in zip(m_cands_list, best_k)],
    'Borda': [effectiveness_from_ballots(borda, m, n_voters)
              for m in m_cands_list],
}

print(f'\nThe Effectiveness of Several Voting Systems '
      f'({time.monotonic() - start_time:.1f}s)\n')
print(tabulate(measured, 'keys', showindex=m_cands_list, tablefmt='pipe',
               floatfmt='.2%'))

print('\nThe closed forms, and the published "best VFOAK" values:\n')
closed = {
    'Standard': [eff_standard(m) for m in m_cands_list],
    'Vote-for-half': [eff_vote_for_half(m) for m in m_cands_list],
    'Best VFOAK (published)': [
        eff_vote_for_or_against_k_paper(m, k)
        for m, k in zip(m_cands_list, best_k)],
    'Borda': [eff_borda(m) for m in m_cands_list],
}
print(tabulate(closed, 'keys', showindex=m_cands_list, tablefmt='pipe',
               floatfmt='.2%'))

print(f'\nBest k, and measured value as a fraction of the same system\'s '
      f'vote-for-k value:')
for m, k in zip(m_cands_list, best_k):
    if k == m/2:
        ratio = 'same system'
    else:
        ratio = f'{measured["Best VFOAK"][m_cands_list.index(m)]/eff_vote_for_k(m, k):.3f}'
    print(f'  m = {m:2d}: k = {k}, {measured["Best VFOAK"][m_cands_list.index(m)]:.2%}'
          f'  (vs vote-for-k {eff_vote_for_k(m, k):.2%}, ratio {ratio})')

print(f'\nTotal time: {time.monotonic() - start_time:.1f}s')
