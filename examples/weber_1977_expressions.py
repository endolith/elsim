"""
Weber's effectiveness expressions, with one correction.

Weber, Robert J. (1978). "Comparison of Public Choice Systems".
Cowles Foundation Discussion Papers. Cowles Foundation for Research in
Economics. No. 498. https://cowles.yale.edu/publications/cfdp/cfdp-498

These all use the impartial culture / random society model, in which every
voter's utility for every candidate is an independent uniform random number,
and the number of voters is infinite.  Running this file reproduces the table
from page 19, "The Effectiveness of Several Voting Systems":

|    |   Standard |   Vote-for-half |   Best VFOAK (published) |   Best VFOAK (correct) |   Borda |
|:---|-----------:|----------------:|-------------------------:|-----------------------:|--------:|
| 2  |     81.65% |          81.65% |                   81.65% |                 81.65% |  81.65% |
| 3  |     75.00% |          75.00% |                   87.50% |                 87.50% |  86.60% |
| 4  |     69.28% |          80.00% |                   80.83% |                 80.83% |  89.44% |
| 5  |     64.55% |          79.06% |                   86.96% |                 85.21% |  91.29% |
| 6  |     60.61% |          81.32% |                   86.25% |                 84.12% |  92.58% |
| 10 |     49.79% |          82.99% |                   88.09% |                 85.31% |  95.35% |
| ∞  |      0.00% |          86.60% |                   92.25% |                 86.60% | 100.00% |

Three of the four columns are reproduced exactly (to the 0.01 points that the
paper prints).  The fourth, "Best vote-for-or-against-k", is not, and cannot
be: the paper's closed form for that system is correct only for ``m <= 4`` and
for ``k = m/2``, and overstates the effectiveness by 1.8 to 2.8 points at
m = 5, 6, 10, by 5.2 points at m = 10, k = 2 (the worst case in the table), and
by 5.7 points in the large-m limit.

What the system is
------------------
Each voter may approve of `k` candidates or disapprove of `k` candidates (the
two weight sets are (1,...,1,0,...,0) with k ones and (0,...,0,-1,...,-1) with
k minus-ones).  Adding 1 to every weight of the second option does not change
the winner, so the voter's choice is really "approve my top k" or "approve my
top m-k".  Weber's optimal-strategy rule (maximize the sum of
``w_c * (u_c - ubar)``) makes the voter compare the average utility of the
``m - 2k`` *central* candidates with their own average utility, and approve
``m - k`` candidates if the central ones are above average.  That rule is what
`elsim.strategies.vote_for_or_against_k` implements, and it is correct.

Where the paper goes wrong
--------------------------
The effectiveness of "vote-for-k" is ``sqrt(24) E[G_k] /
((m-1) sqrt(p))`` with ``E[G_k] = k(m-k)/(2(m+1))`` and
``p = 2k(m-k)/(m(m-1))``; "vote for or against k" changes only the numerator,
to ``E[max(G_k, G_{m-k})] = E[G_k] + (m-2k) E[(c - ubar)^+]``.  So everything
hinges on the central gap ``E[(c - ubar)^+]``, the expected amount by which the
central candidates beat the voter's average.  The paper uses
``k/(8(m+1))`` for it (plus a parity correction for odd m).  That value is
exactly right at m = 3 and m = 4 -- where the paper worked the system out case
by case -- and it is right by construction at ``k = m/2``, where the
correction term is multiplied by zero.  Everywhere else it is too large, and
it does not shrink as fast as the true value:

* for fixed k, the true central gap falls off like ``m**-3/2``, faster than
  the paper's ``k/(8(m+1))``.  The switching term ``(m-2k) * gap`` is then
  ``O(1/sqrt(m))`` against the order-m main term ``E[G_k] ~ k/2``, so the
  advantage of the system over "vote-for-k" vanishes; but with the paper's gap
  the switching term tends to the constant ``k/8``, a fixed 25% on top of
  ``E[G_k]``, so the published formula overstates effectiveness by a factor
  approaching 5/4 (for k = 2: 1.19 at m = 10, 1.22 at m = 20, 1.25 by
  m = 100).
* for ``k = alpha*m`` the central average does not concentrate, so the true
  switching term is ``O(sqrt(m))`` against the order-m main term: it vanishes,
  but slowly.  The corrected effectiveness therefore tends to
  ``sqrt(3 alpha (1 - alpha))``, the "vote-for-k" value, and its best case is
  ``sqrt(3)/2 = 86.60%`` at ``alpha -> 1/2``.

Consequently the paper's large-m limit of 92.25%, and the ``k/m = 0.368``
that produces it, are artifacts.  The correct limit is 86.60%, which is the
*same* limit as "vote-for-half".  "Vote for or against k" does beat
"vote-for-half" for small m (85.21% vs 79.06% at m = 5), but the margin
shrinks with m (86.50% vs 86.18% at m = 100) instead of growing to 5.7
points.

The paper is self-consistent in its *description* of the system; it is the
closed form, and hence the column, that is wrong.  The 3-candidate row
(87.50%) -- the paper's headline, and still the most effective system in the
table -- is unaffected, as are the 2- and 4-candidate rows.

How the corrected column is computed
------------------------------------
`eff_vote_for_or_against_k` evaluates the formula above, with the central gap
obtained by numerical integration (`central_gap`); there is no elementary
closed form for it.  `eff_vote_for_or_against_k_paper` is the published
expression, kept so the published table can be reproduced.

How it is checked
-----------------
Two independent checks:

* `effectiveness_from_ballots` computes the same quantity a completely
  different way.  For a random society, the effectiveness of any system is
  ``Corr(u_A - u_B, w_A - w_B)`` over a random voter and a random pair of
  candidates -- the correlation between how much more a voter likes A than B
  and how much more weight they give A than B.  Fed with the ballots that
  `elsim.strategies.vote_for_or_against_k` actually produces, it reproduces
  every closed form here to a few hundredths of a point.
* examples/weber_1977_verify_vote_for_or_against_k.py runs whole elections
  instead: it draws 1,000-voter electorates, tallies the ballots with approval
  voting, and measures the social utility of the winner.  It agrees with the
  corrected column for every m, and with the published column only for m <= 4.
  (That check is much noisier -- an election-level effectiveness estimate has
  a standard deviation of order 100% -- which is why the other two exist.)
"""
from functools import lru_cache
import numpy as np
from numpy import sqrt, round
from scipy.stats import qmc
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


def _central_sum_coefficients(m, k):
    """
    Integer weights `l` such that ``c - ubar = sum(l * W) / (m * (m - 2k))``.

    ``W`` is a Dirichlet(1, ..., 1) vector of length ``m + 1`` (the spacings of
    `m` independent uniform random variables), ``c`` is the average utility of
    the ``m - 2k`` middle-ranked candidates, and ``ubar`` is the average utility
    of all `m` candidates.
    """
    l = np.zeros(m + 1)
    for j in range(1, m + 2):
        top = m*max(0, (m - k) - max(j, k + 1) + 1)
        total = (m - 2*k)*((m - j + 1) if j <= m else 0)
        l[j-1] = top - total
    return l


@lru_cache(maxsize=None)
def central_gap(m, k, n_samples=2**20, seed=0):
    """
    Calculate ``E[max(0, c - ubar)]``, the expected "central gap".

    `c` is the average utility of the ``m - 2k`` middle-ranked candidates, and
    ``ubar`` is the average utility of all `m` candidates, for a single voter
    in a random society.

    This is the correction term that "vote-for-or-against-k" adds on top of
    "vote-for-k": a voter votes for their top ``m - k`` candidates instead of
    their top `k` candidates exactly when ``c > ubar``, which gains them
    ``(m - 2k) * (c - ubar)``.  (Both options give the same total number of
    approvals only when ``k = m/2``, which is why the term vanishes there.)

    There appears to be no elementary closed form, so it is evaluated as an
    expectation over a uniform Dirichlet vector, i.e. over the spacings of `m`
    independent uniform random variables.  The integrand is piecewise linear
    in that vector, so a scrambled Sobol sequence converges much faster than
    random sampling: with the default of 2**20 points the error is well under
    0.01%.  The result is deterministic.

    Parameters
    ----------
    m : int
        Total number of candidates.
    k : int
        Number of candidates that each voter approves or disapproves of.
    n_samples : int, optional
        Number of quadrature points.
    seed : int, optional
        Seed for the Sobol scrambling.

    Returns
    -------
    gap : float
        ``E[max(0, c - ubar)]``.
    """
    if m == 2*k:
        return 0.0
    l = _central_sum_coefficients(m, k)
    points = qmc.Sobol(d=m + 1, scramble=True, seed=seed)
    total = 0.0
    drawn = 0
    batch = 2**18
    while drawn < n_samples:
        b = min(batch, n_samples - drawn)
        # Map uniforms to a uniform point on the simplex: exponential spacings
        W = -np.log1p(-points.random(b))
        W /= W.sum(axis=1, keepdims=True)
        total += np.maximum(W @ l, 0).sum()
        drawn += b
    return total/n_samples/(m*(m - 2*k))


def eff_vote_for_or_against_k(m, k, n_samples=2**20, seed=0):
    r"""
    Calculate effectiveness of the "vote-for-or-against-k" voting system.

    This is a variant of combined approval voting (CAV) in which every voter
    approves of `k` candidates or disapproves of `k` candidates (equivalently,
    approves of their top ``m - k``), whichever they prefer.[1]_

    With ``J`` the number of candidates a voter approves of (``k`` or
    ``m - k``) and ``G(J)`` the total utility the voter's approved candidates
    have above their average utility, the effectiveness works out to

    .. math::

        \mathrm{eff} = \frac{\sqrt{24}\,\mathbb{E}[\max(G_k, G_{m-k})]}
                             {(m - 1)\sqrt{p}}, \qquad
        p = \frac{2k(m-k)}{m(m-1)}

    where ``E[G_k] = k(m-k) / (2(m+1))`` and, because a voter switches to
    approving ``m - k`` candidates exactly when the *central* candidates beat
    their average utility,

    .. math::

        \mathbb{E}[\max(G_k, G_{m-k})] =
            \mathbb{E}[G_k] + (m - 2k)\,\mathbb{E}[(c - \bar u)^+].

    The second term has no elementary closed form and is evaluated
    numerically (see `central_gap`), so this is exact up to that quadrature.

    Parameters
    ----------
    m : int
        Total number of candidates.
    k : int
        Number of candidates that each voter approves or disapproves of.
    n_samples : int, optional
        Number of quadrature points used for the central-gap expectation.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    eff : float
        Effectiveness of "vote-for-or-against-`k`", with `m` candidates and
        infinite number of random voters.

    See Also
    --------
    eff_vote_for_or_against_k_paper : The formula printed in the published
        paper, which is what the published table was computed from.

    Notes
    -----
    The published formula :func:`eff_vote_for_or_against_k_paper` is correct
    for ``m <= 4`` and for ``k = m/2`` (where this system coincides with
    "vote-for-k"), but it *overstates* the effectiveness for every larger `m`:

    =====  =====  =========  =========  ======
      m      k     correct    published   diff
    =====  =====  =========  =========  ======
      3      1     87.50%      87.50%   0.00
      4      1     80.83%      80.83%   0.00
      5      1     75.04%      77.46%  -2.42
      5      2     85.21%      86.96%  -1.76
      6      2     84.12%      86.25%  -2.13
      7      3     84.85%      86.80%  -1.95
      8      3     84.97%      87.65%  -2.68
     10      4     85.31%      88.09%  -2.78
    =====  =====  =========  =========  ======

    The two agree for ``m <= 4`` because the paper's expression for the central
    gap, ``k / (8(m+1))`` (plus a parity correction for odd `m`), happens to
    equal the true value there.  For ``m >= 5`` the true central gap is much
    smaller, and it shrinks as `m` grows, so the paper's formula drifts away.
    The same error produces the published claim that "best vote-for-or-against
    k" is 92.25% effective for large `m`: in fact it tends to
    ``sqrt(3)/2 = 86.60%``, the same limit as "vote-for-half", because the
    switching term is only order ``sqrt(m)`` against the order-`m` main term.
    The Monte Carlo value of this column therefore cannot reproduce the
    published table for ``m >= 5`` (and nothing is wrong with the ballot rule,
    which ``elsim.strategies.vote_for_or_against_k`` implements correctly).

    References
    ----------
    .. [1] Weber, Robert J. (1978). "Comparison of Public Choice Systems".
       Cowles Foundation Discussion Papers. Cowles Foundation for Research in
       Economics. No. 498. https://cowles.yale.edu/publications/cfdp/cfdp-498
    """
    if not 0 < k <= m//2:
        raise ValueError(f'k={k} must be between 1 and m//2={m//2}')
    expected_gain = k*(m-k)/(2*(m+1))
    expected_gain += (m - 2*k)*central_gap(m, k, n_samples, seed)
    p = 2*k*(m-k)/(m*(m-1))
    return sqrt(24)*expected_gain/((m-1)*sqrt(p))


def eff_vote_for_or_against_k_paper(m, k):
    """
    Calculate effectiveness using the formula printed in Weber (1978).

    This is the expression that the published "Best vote-for-or-against-k"
    column was computed from.  It agrees with the correct value
    (:func:`eff_vote_for_or_against_k`) only for ``m <= 4`` and for
    ``k = m/2``; for ``m >= 5`` it overstates the effectiveness by up to ~5
    percentage points.  It is kept here so that the published table can be
    reproduced exactly.

    Parameters
    ----------
    m : int
        Total number of candidates.
    k : int
        Number of candidates that each voter approves or disapproves of.

    Returns
    -------
    eff : float
        The published effectiveness of "vote-for-or-against-`k`".

    See Also
    --------
    eff_vote_for_or_against_k : The correct expression.
    """
    # m % 2 handles the +1 if odd, +0 if even condition
    return (5*m**2 + m % 2 - 6*m*k) / (4*(m+1)) * sqrt(3*k/(m * (m-1) * (m-k)))


def best_vote_for_or_against_k(m, n_samples=2**20, seed=0):
    """
    Find `k` that maximizes effectiveness of "vote-for-or-against-k" system.

    This `k` is the value that maximizes the effectiveness of this system for
    a given `m`, when every voter uses it.

    The paper instead uses ``k = round(0.368 * m)``, from the maximizing
    `alpha = (9 - sqrt(21))/12 ~ 0.368` of its (incorrect) closed form.  The
    correct optimum is closer to ``m/2`` for large `m`, and the two choices
    give different `k` from about ``m = 9`` on.

    Parameters
    ----------
    m : int
        Total number of candidates.
    n_samples : int, optional
        Number of quadrature points used for each candidate `k`.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    k : int
        Number of candidates for every voter to approve or disapprove of.
    """
    ks = range(1, m//2 + 1)
    effs = [eff_vote_for_or_against_k(m, k, n_samples, seed) for k in ks]
    return int(ks[int(np.argmax(effs))])


def eff_best_vote_for_or_against_k(m, n_samples=2**20, seed=0):
    """
    Calculate effectiveness of the best "vote-for-or-against-k" voting system.

    This is a variant of combined approval voting (CAV) in which every voter
    approves or disapproves of `k` candidates.  In this "best" case, `k` is the
    chosen to maximize the effectiveness for a given `m`.

    Parameters
    ----------
    m : int
        Total number of candidates.
    n_samples : int, optional
        Number of quadrature points used for each candidate `k`.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    eff : float
        Effectiveness of "vote-for-or-against-`k`", with `m` candidates and
        infinite number of random voters.
    """
    k = best_vote_for_or_against_k(m, n_samples, seed)
    return eff_vote_for_or_against_k(m, k, n_samples, seed)


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


def effectiveness_from_ballots(ballot_fn, m, n_voters=1_000_000, seed=0):
    r"""
    Calculate the effectiveness of a voting system from its ballots.

    For a random society of voters with independent, uniformly distributed
    utilities, the effectiveness of any system in which the candidate with the
    largest total weight wins is[1]_

    .. math::

        \mathrm{eff} = \mathrm{Corr}(u_A - u_B, w_A - w_B)

    the correlation, over a random voter and a random pair of candidates,
    between how much more the voter likes candidate A than candidate B and how
    much more weight the voter gives to A than to B.  (A voter's own ballot
    only affects the outcome at order ``1/sqrt(n)``, and it is exactly that
    order that sets the effectiveness, so the whole calculation collapses to
    this single-voter correlation.)  Equivalently, with
    ``kappa = Cov(u_c, w_c)`` and ``rho = -Cov(u_c, w_d)`` for ``d != c``,

    .. math::

        \mathrm{eff} = \frac{\sqrt{12}(\kappa + \rho)}
                             {\sqrt{\sigma^2 - \gamma}}

    where ``sigma^2 = Var(w_c)`` and ``gamma = Cov(w_c, w_d)``.

    This is a cheap and accurate way to check effectiveness numbers: it needs
    no election at all, and a million voters give about 0.02% accuracy.  A
    direct election simulation, by contrast, is dominated by ``O(1/sqrt(n))``
    finite-electorate bias, and needs on the order of ``10^8`` voter-elections
    before its noise is small enough to distinguish two similar systems.

    Parameters
    ----------
    ballot_fn : callable
        Function taking a 2D array of utilities (voters x candidates) and
        returning a 2D array of the weights each voter assigns to each
        candidate.  This is the same interface as the functions in
        `elsim.strategies`, and it does not have to be approval-like: Borda
        weights work too.
    m : int
        Total number of candidates.
    n_voters : int, optional
        Number of random voters to average over.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    eff : float
        Effectiveness with `m` candidates and infinite number of random
        voters.

    References
    ----------
    .. [1] Weber, Robert J. (1978). "Comparison of Public Choice Systems".
       Cowles Foundation Discussion Papers. Cowles Foundation for Research in
       Economics. No. 498. https://cowles.yale.edu/publications/cfdp/cfdp-498

    Examples
    --------
    >>> from elsim.strategies import vote_for_k
    >>> round(effectiveness_from_ballots(
    ...     lambda u: vote_for_k(u, 2), 10, 200_000), 3)
    0.813
    """
    rng = np.random.default_rng(seed)
    total_s = np.zeros((m, m))      # sum of (u_c - 1/2) * w_d
    total_ww = np.zeros((m, m))     # sum of w_c * w_d
    total_w = np.zeros(m)
    batch = 100_000
    done = 0
    while done < n_voters:
        b = min(batch, n_voters - done)
        utilities = rng.random((b, m))
        weights = np.asarray(ballot_fn(utilities), dtype=float)
        du = utilities - 0.5
        total_s += du.T @ weights
        total_ww += weights.T @ weights
        total_w += weights.sum(axis=0)
        done += b

    total_s /= n_voters
    total_ww /= n_voters
    mu = total_w.sum()/n_voters/m

    # Average kappa and rho over all ordered pairs of distinct candidates
    diagonal = np.diag_indices(m)
    off = ~np.eye(m, dtype=bool)
    kappa = total_s[diagonal].mean()
    rho = -total_s[off].mean()
    var_w = (total_ww[diagonal].mean() - mu**2)
    cov_w = (total_ww[off].mean() - mu**2)
    return sqrt(12)*(kappa + rho)/sqrt(var_w - cov_w)


def test_cases():
    # Verify various statements from the paper
    """
    When `m` is even and `k = m/2`, the vote-for-or-against-k system coincides
    with the vote-for-k system.
    """
    for m in range(2, 100, 2):
        k = m // 2
        assert_almost_equal(eff_vote_for_or_against_k(m, k, 2**19),
                            eff_vote_for_k(m, k))
        assert_almost_equal(eff_vote_for_or_against_k_paper(m, k),
                            eff_vote_for_k(m, k))

    """
    For k < m/2, it is easily verified that the vote-for-or-against-k voting
    system is strictly more effective than the vote-for-k system.
    """
    for m in range(3, 15):
        for k in range(1, (m+1)//2):
            assert_(eff_vote_for_or_against_k(m, k, 2**19)
                    > eff_vote_for_k(m, k))

    """
    For three-candidate elections we see that the standard voting system
    (75% effective) is appreciably less effective than the approval voting
    system (87.5%)
    """
    assert_almost_equal(eff_standard(3), 0.75)
    assert_almost_equal(eff_vote_for_or_against_k(3, 1), 0.875, decimal=3)

    """
    A few values from the table
    """
    assert_almost_equal(eff_standard(10), 49.79/100, decimal=4)
    assert_almost_equal(eff_vote_for_half(5), 79.06/100, decimal=4)
    assert_almost_equal(eff_borda(6), 92.58/100, decimal=4)

    """
    The published formula for vote-for-or-against-k is right for m <= 4, and
    overstates the effectiveness for m >= 5.
    """
    for m in (2, 3, 4):
        for k in range(1, m//2 + 1):
            assert_almost_equal(eff_vote_for_or_against_k(m, k, 2**19),
                                eff_vote_for_or_against_k_paper(m, k),
                                decimal=3)
    for m, k in ((5, 1), (5, 2), (6, 2), (10, 4)):
        assert_(eff_vote_for_or_against_k(m, k, 2**19)
                < eff_vote_for_or_against_k_paper(m, k) - 1/100)

    """
    The best vote-for-or-against-k effectiveness approaches, but never beats,
    vote-for-half's large-m limit of sqrt(3)/2, and never reaches the published
    92.25%.
    """
    for m in (5, 6, 10, 20):
        eff = eff_best_vote_for_or_against_k(m, n_samples=2**19)
        assert_(eff <= sqrt(3)/2)
        assert_(eff < 0.9225)
    assert_almost_equal(eff_vote_for_half(10**6), sqrt(3)/2, decimal=5)

    """
    The covariance formula agrees with the closed forms it is checked against.
    """
    from elsim.strategies import vote_for_k, vote_for_or_against_k

    def borda(u):
        m = u.shape[1]
        ranks = np.argsort(np.argsort(-u, axis=1), axis=1)
        return m - ranks

    def plump(u):
        return vote_for_k(u, 1)

    for m in (2, 3, 4, 5):
        assert_almost_equal(effectiveness_from_ballots(plump, m, 200_000),
                            eff_standard(m), decimal=3)
        for k in range(1, m//2 + 1):
            assert_almost_equal(
                effectiveness_from_ballots(lambda u: vote_for_k(u, k), m,
                                           200_000),
                eff_vote_for_k(m, k), decimal=3)
            assert_almost_equal(
                effectiveness_from_ballots(
                    lambda u: vote_for_or_against_k(u, k), m, 200_000),
                eff_vote_for_or_against_k(m, k, 2**19), decimal=3)
        assert_almost_equal(effectiveness_from_ballots(borda, m, 200_000),
                            eff_borda(m), decimal=3)


if __name__ == '__main__':
    test_cases()

    from tabulate import tabulate

    m_cands_list = (2, 3, 4, 5, 6, 10)
    alpha = (9 - sqrt(21))/12          # the paper's k/m, from its closed form
    published_k = [max(round(alpha*m), 1) for m in m_cands_list]
    correct_k = [best_vote_for_or_against_k(m) for m in m_cands_list]
    published_inf = (5 - 6*alpha)/4*sqrt(3*alpha/(1 - alpha))
    table = {
        'Standard': [eff_standard(m) for m in m_cands_list] + [0.0],
        'Vote-for-half': [eff_vote_for_half(m) for m in m_cands_list]
                         + [sqrt(3)/2],
        'Best VFOAK (published)': [
            eff_vote_for_or_against_k_paper(m, k)
            for m, k in zip(m_cands_list, published_k)] + [published_inf],
        'Best VFOAK (correct)': [
            eff_vote_for_or_against_k(m, k)
            for m, k in zip(m_cands_list, correct_k)] + [sqrt(3)/2],
        'Borda': [eff_borda(m) for m in m_cands_list] + [1.0],
    }

    print(tabulate(table, 'keys', showindex=tuple(m_cands_list) + ('∞',),
                   tablefmt='pipe', floatfmt='.2%'))
    print()
    print(f"Both VFOAK columns use k = {dict(zip(m_cands_list, correct_k))}"
          f" (the paper's k = round({alpha:.3f} m) agrees for every row).")
    print("The published formula is exact for m <= 4 and for k = m/2, and too")
    print("high otherwise:")
    for m in (5, 6, 10):
        k = correct_k[m_cands_list.index(m)]
        print(f"  m = {m:2d}, k = {k}: "
              f"{eff_vote_for_or_against_k_paper(m, k):.2%} published, "
              f"{eff_vote_for_or_against_k(m, k):.2%} correct "
              f"({100*(eff_vote_for_or_against_k_paper(m, k) - eff_vote_for_or_against_k(m, k)):+.1f} points)")
    print(f"  m = ∞:    {published_inf:.2%} published, "
          f"{sqrt(3)/2:.2%} correct "
          f"({100*(published_inf - sqrt(3)/2):+.1f} points)")
