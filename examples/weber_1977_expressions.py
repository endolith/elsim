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
"""
from numpy import sqrt, round
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


def eff_vote_for_or_against_k(m, k):
    """
    Calculate effectiveness of the "vote-for-or-against-k" voting system.

    This is a variant of combined approval voting (CAV) in which every voter
    approves or disapproves of `k` candidates.

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

    This `k` is the value that maximizes the effectiveness of this system for
    a given `m`, when every voter uses it.

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
    Calculate effectiveness of the best "vote-for-or-against-k" voting system.

    This is a variant of combined approval voting (CAV) in which every voter
    approves or disapproves of `k` candidates.  In this "best" case, `k` is the
    chosen to maximize the effectiveness for a given `m`.

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


if __name__ == '__main__':
    test_cases()

    from tabulate import tabulate
    from numpy import array

    table = {}
    m_cands_list = (2, 3, 4, 5, 6, 10, 1e30)
    for m in m_cands_list:
        for name, method in (('Standard', eff_standard),
                             ('Vote-for-half', eff_vote_for_half),
                             ('Best Vote-for-or-against-k',
                              eff_best_vote_for_or_against_k),
                             ('Borda', eff_borda)):
            table[name] = method(array(m_cands_list))

    print(tabulate(table, 'keys', showindex=m_cands_list[:-1] + ('∞',),
                   tablefmt="pipe", floatfmt='.2%'))
