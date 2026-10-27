"""
Weber's effectiveness expressions, for comparison.

Weber, Robert J. (1978). "Comparison of Public Choice Systems".
Cowles Foundation Discussion Papers. Cowles Foundation for Research in
Economics. No. 498. https://cowles.yale.edu/publications/cfdp/cfdp-498

These all use the impartial culture / random society model and an infinite
number of voters.

When run, this reproduces the "The Effectiveness of Several Voting Systems"
table from page 19:

|    |   Standard |   Vote-for-half |   Best Vote-for-or-against-k |   Borda |
|:---|-----------:|----------------:|-----------------------------:|--------:|
| 2  |     81.65% |          81.65% |                       81.65% |  81.65% |
| 3  |     75.00% |          75.00% |                       87.50% |  86.60% |
| 4  |     69.28% |          80.00% |                       80.83% |  89.44% |
| 5  |     64.55% |          79.06% |                       86.96% |  91.29% |
| 6  |     60.61% |          81.32% |                       86.25% |  92.58% |
| 10 |     49.79% |          82.99% |                       88.09% |  95.35% |
| ∞  |      0.00% |          86.60% |                       92.25% | 100.00% |

The "Best Vote-for-or-against-k" column is wrong for m >= 5: its derivation
treats the feasible region of utility profiles as a simplex, but for m >= 5
the region has additional vertices, so the published centroid is wrong.
The corrected values are:

| m  |  published* | corrected |
|:---|------------:|----------:|
| 2  |      81.65% |    81.65% |
| 3  |      87.50% |    87.50% |
| 4  |      80.83% |    80.83% |
| 5  | ~~86.96%~~  |    85.21% |
| 6  | ~~86.25%~~  |    84.12% |
| 10 | ~~88.09%~~  |    85.31% |
| ∞  | ~~92.25%~~  |    86.60% |

* The published formula overstates the effectiveness for m >= 5.  See
  :func:`eff_vote_for_or_against_k`.
"""
from fractions import Fraction
from functools import cache
from math import comb

from numpy import round, sqrt
from numpy.testing import assert_, assert_almost_equal


def eff_standard(m):
    """
    Calculate effectiveness of Standard FPTP voting system.

    Parameters
    ----------
    m : int
        Total number of candidates.

    Returns
    -------
    eff : float
        Effectiveness of FPTP with `m` candidates and infinite number of
        random voters.
    """
    return sqrt(3*m)/(m+1)


def eff_vote_for_k(m, k):
    """
    Calculate effectiveness of the "vote for `k`" approval voting variant.

    Parameters
    ----------
    m : int
        Total number of candidates.
    k : int
        Number of candidates that each voter approves of.

    Returns
    -------
    eff : float
        Effectiveness of "vote-for-`k`", with `m` candidates and infinite
        number of random voters.
    """
    return 1/(m+1) * sqrt(3*m*k*(m-k) / (m-1))


def eff_vote_for_half(m):
    """
    Calculate effectiveness of the "vote-for-half" approval voting variant.

    Parameters
    ----------
    m : int
        Total number of candidates.

    Returns
    -------
    eff : float
        Effectiveness of "vote-for-half", with `m` candidates and infinite
        number of random voters.
    """
    k = m // 2
    return eff_vote_for_k(m, k)


@cache
def _expected_max_centered_sum(m, k):
    """
    Calculate ``E[max(A, B)]`` exactly, as a Fraction.

    ``A`` is the total utility of the top `k` candidates above the voter's
    mean utility, and ``B`` is minus the total utility of the bottom `k`
    candidates above that mean.  A voter approves of the top `k` when
    ``A >= B`` and of the top ``m - k`` otherwise, so ``E[max(A, B)]`` is the
    numerator of the effectiveness.

    ``A - B`` is a fixed linear combination of the ordered utilities, which
    are partial sums of ``m + 1`` Dirichlet(1, ..., 1) spacings.
    ``E[(sum a_j G_j)^+]`` is a confluent divided difference of the truncated
    power ``x_+**(m + 1)``, so the result is a finite rational number.

    Parameters
    ----------
    m : int
        Total number of candidates.
    k : int
        Number of candidates that each voter approves or disapproves of.

    Returns
    -------
    gain : Fraction
        ``E[max(A, B)]``.
    """
    if not 0 < k <= m // 2:
        raise ValueError(f'k={k} must be between 1 and m//2={m//2}')

    # A - B, in ascending order of the utilities.
    coefficients = [Fraction(int(i < k or i >= m - k)) - Fraction(2*k, m)
                    for i in range(m)]
    # The ordered utilities are partial sums of m + 1 Dirichlet spacings.
    knots = sorted(sum(coefficients[j:], Fraction())
                   for j in range(m + 1))

    # Confluent divided differences of x_+**(m + 1) at the knots.
    power = m + 1
    differences = [max(x, Fraction())**power for x in knots]
    for order in range(1, m + 1):
        next_differences = []
        for i in range(m + 1 - order):
            left, right = knots[i], knots[i + order]
            if left == right:
                value = (comb(power, order)
                         * max(left, Fraction())**(power - order))
            else:
                value = ((differences[i + 1] - differences[i])
                         / (right - left))
            next_differences.append(value)
        differences = next_differences

    return Fraction(k*(m - k), 2*(m + 1)) + differences[0]/(m + 1)


def eff_vote_for_or_against_k(m, k):
    """
    Calculate effectiveness of the "vote-for-or-against-k" voting system.

    This is a variant of combined approval voting (CAV) in which every voter
    approves of `k` candidates or disapproves of `k` candidates (equivalently,
    approves of the top ``m - k``), whichever is better for them.

    Parameters
    ----------
    m : int
        Total number of candidates.
    k : int
        Number of candidates that each voter approves or disapproves of.

    Returns
    -------
    eff : float
        Effectiveness of "vote-for-or-against-`k`", with `m` candidates and
        infinite number of random voters.

    Notes
    -----
    The published formula :func:`eff_vote_for_or_against_k_paper` is correct
    for ``m <= 4`` and for ``k = m/2``, but overstates the effectiveness for
    every larger `m`.  This function uses the exact expectation computed by
    :func:`_expected_max_centered_sum`.
    """
    if not 0 < k <= m // 2:
        raise ValueError(f'k={k} must be between 1 and m//2={m//2}')
    return (sqrt(12*m / ((m - 1) * k * (m - k)))
            * float(_expected_max_centered_sum(m, k)))


def eff_vote_for_or_against_k_paper(m, k):
    """
    Calculate the published effectiveness of "vote-for-or-against-k".

    This is the expression printed in the paper.  It is correct for
    ``m <= 4`` and for ``k = m/2``, but overstates the effectiveness for
    larger `m`; see :func:`eff_vote_for_or_against_k`.

    Parameters
    ----------
    m : int
        Total number of candidates.
    k : int
        Number of candidates that each voter approves or disapproves of.

    Returns
    -------
    eff : float
        Effectiveness of "vote-for-or-against-`k`", with `m` candidates and
        infinite number of random voters.
    """
    # m % 2 handles the +1 if odd, +0 if even condition
    return (5*m**2 + m % 2 - 6*m*k) / (4*(m+1)) * sqrt(3*k/(m * (m-1) * (m-k)))


def best_vote_for_or_against_k(m):
    """
    Find `k` that maximizes effectiveness of "vote-for-or-against-k" system.

    This is the value that maximizes the corrected effectiveness for a given
    `m`, when every voter uses it.  The effectiveness is unimodal in `k`, so
    this uses a ternary search.

    Parameters
    ----------
    m : int
        Total number of candidates.

    Returns
    -------
    k : int
        Number of candidates for every voter to approve or disapprove.

    Notes
    -----
    For ``m <= 8`` this returns the same `k` as the published
    :func:`best_vote_for_or_against_k_paper`.  The two diverge from ``m = 9``,
    where the corrected best `k` is larger.
    """
    lo, hi = 1, m // 2
    while hi - lo > 3:
        a = lo + (hi - lo)//3
        b = hi - (hi - lo)//3
        if eff_vote_for_or_against_k(m, a) < eff_vote_for_or_against_k(m, b):
            lo = a
        else:
            hi = b
    return max(range(lo, hi + 1),
               key=lambda k: eff_vote_for_or_against_k(m, k))


def best_vote_for_or_against_k_paper(m):
    """
    Find `k` that maximizes the published effectiveness expression.

    This is the `k` used by the published table, which is not always the best
    `k` for the true effectiveness; see :func:`best_vote_for_or_against_k`.

    Parameters
    ----------
    m : int
        Total number of candidates.

    Returns
    -------
    k : float
        Number of candidates for every voter to approve or disapprove.
    """
    alpha = (9 - sqrt(21))/12
    return round(alpha * m)


def eff_best_vote_for_or_against_k(m):
    """
    Calculate effectiveness of the best "vote-for-or-against-k" system.

    This is a variant of combined approval voting (CAV) in which every voter
    approves or disapproves of `k` candidates.  In this "best" case, `k` is
    chosen to maximize the corrected effectiveness for a given `m`.

    Parameters
    ----------
    m : int
        Total number of candidates.

    Returns
    -------
    eff : float
        Effectiveness of "vote-for-or-against-`k`", with `m` candidates and
        infinite number of random voters.
    """
    k = best_vote_for_or_against_k(m)
    return eff_vote_for_or_against_k(m, k)


def eff_best_vote_for_or_against_k_paper(m):
    """
    Calculate the published effectiveness of the best VFOAK system.

    This is the value in the published table: `k` maximizes the published
    expression, which overstates the effectiveness for ``m >= 5``.

    Parameters
    ----------
    m : int
        Total number of candidates.

    Returns
    -------
    eff : float
        Effectiveness of "vote-for-or-against-`k`", with `m` candidates and
        infinite number of random voters.
    """
    k = best_vote_for_or_against_k_paper(m)
    return eff_vote_for_or_against_k_paper(m, k)


def eff_borda(m):
    """
    Calculate effectiveness of the Borda count voting system.

    Parameters
    ----------
    m : int
        Total number of candidates.

    Returns
    -------
    eff : float
        Effectiveness of Borda count with `m` candidates and infinite number
        of random voters.
    """
    return sqrt(m / (m+1))


def test_cases():
    # Verify various statements from the paper
    """
    When `m` is even and `k = m/2`, the vote-for-or-against-k system coincides
    with the vote-for-k system.
    """
    for m in range(2, 100, 2):
        k = m // 2
        assert_almost_equal(eff_vote_for_or_against_k(m, k),
                            eff_vote_for_k(m, k))

    """
    For k < m/2, it is easily verified that the vote-for-or-against-k voting
    system is strictly more effective than the vote-for-k system.
    """
    for m in range(2, 15):
        for k in range(1, (m+1)//2):
            assert_(eff_vote_for_or_against_k(m, k) > eff_vote_for_k(m, k))

    """
    For three-candidate elections we see that the standard voting system
    (75% effective) is appreciably less effective than the approval voting
    system (87.5%)
    """
    assert_almost_equal(eff_standard(3), 0.75)
    assert_almost_equal(eff_vote_for_or_against_k(3, 1), 0.875)

    """
    A few values from the table
    """
    assert_almost_equal(eff_standard(10), 49.79/100, decimal=4)
    assert_almost_equal(eff_vote_for_half(5), 79.06/100, decimal=4)
    assert_almost_equal(eff_best_vote_for_or_against_k(4), 80.83/100, 4)
    assert_almost_equal(eff_borda(6), 92.58/100, decimal=4)

    """
    The exact numerator is a rational number.
    """
    assert _expected_max_centered_sum(5, 2) == Fraction(97, 180)
    assert _expected_max_centered_sum(3, 1) == Fraction(7, 24)

    """
    The corrected effectiveness, and the published values it replaces.
    """
    assert_almost_equal(eff_vote_for_or_against_k(5, 2), 0.852058, 6)
    assert_almost_equal(eff_vote_for_or_against_k(6, 2), 0.841187, 6)
    assert_almost_equal(eff_vote_for_or_against_k(10, 4), 0.853099, 6)
    assert_almost_equal(eff_vote_for_or_against_k_paper(5, 2), 0.869626, 6)
    assert_almost_equal(eff_vote_for_or_against_k_paper(6, 2), 0.862483, 6)
    assert_almost_equal(eff_vote_for_or_against_k_paper(10, 4), 0.880875, 6)

    """
    The two formulas agree for m <= 4 and when k = m/2.
    """
    for m in range(2, 5):
        for k in range(1, m//2 + 1):
            assert_almost_equal(eff_vote_for_or_against_k(m, k),
                                eff_vote_for_or_against_k_paper(m, k))
    for m in range(2, 20, 2):
        k = m // 2
        assert_almost_equal(eff_vote_for_or_against_k(m, k),
                            eff_vote_for_or_against_k_paper(m, k))

    """
    The corrected best k agrees with the paper through m = 8, then diverges.
    """
    for m in range(2, 9):
        assert (best_vote_for_or_against_k(m)
                == best_vote_for_or_against_k_paper(m))
    assert (best_vote_for_or_against_k(9)
            != best_vote_for_or_against_k_paper(9))


if __name__ == '__main__':
    test_cases()

    from tabulate import tabulate

    m_cands_list = (2, 3, 4, 5, 6, 10, 255)
    methods = (
        ('Standard', eff_standard),
        ('Vote-for-half', eff_vote_for_half),
        ('Best Vote-for-or-against-k (published)*',
         eff_best_vote_for_or_against_k_paper),
        ('Best Vote-for-or-against-k (corrected)',
         eff_best_vote_for_or_against_k),
        ('Borda', eff_borda),
    )
    table = {name: [method(m) for m in m_cands_list]
             for name, method in methods}
    limits = {'Standard': 0,
              'Vote-for-half': sqrt(3)/2,
              'Best Vote-for-or-against-k (published)*': 92.25/100,
              'Best Vote-for-or-against-k (corrected)': sqrt(3)/2,
              'Borda': 1}
    for name in table:
        table[name].append(limits[name])

    print(tabulate(table, 'keys', showindex=m_cands_list + ('∞',),
                   tablefmt="pipe", floatfmt='.2%'))
    print('\n* The published formula overstates the effectiveness for '
          'm >= 5 (see the module docstring).')
