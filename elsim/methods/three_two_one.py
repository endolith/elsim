"""
3-2-1 voting (Jameson Quinn).

Ballots use integer ratings 0 = Bad, 1 = OK, 2 = Good per candidate.
"""
import numpy as np

from elsim.methods._common import (_get_tiebreak, _no_tiebreak,
                                   _order_tiebreak_keep, _random_tiebreak)

_tiebreak_map = {'order': _order_tiebreak_keep,
                 'random': _random_tiebreak,
                 None: _no_tiebreak}


def _pairwise_rating_preference(election, a, b, tiebreaker):
    """
    Winner between candidates ``a`` and ``b`` where higher rating wins each ballot.

    Ties on the same ballot count toward neither side. Overall ties use total
    rating points (Good=2, OK=1, Bad=0), then ``tiebreaker``.
    """
    ea = election[:, a]
    eb = election[:, b]
    a_strict = (ea > eb).sum()
    b_strict = (eb > ea).sum()
    if a_strict > b_strict:
        return a
    if b_strict > a_strict:
        return b
    sa = ea.sum()
    sb = eb.sum()
    if sa > sb:
        return a
    if sb > sa:
        return b
    tiebreak = _get_tiebreak(tiebreaker, _tiebreak_map)
    return tiebreak([a, b])[0]


def three_two_one(election, tiebreaker=None):
    """
    Find the winner using 3-2-1 voting.

    Semifinalists are the three candidates with the most Good ratings; ties are
    broken by total score (Good=2, OK=1). Among those three, the one with the
    most Bad ratings is eliminated (ties broken by lower total score, then lower
    candidate ID). The remaining two are compared pairwise by higher rating on
    each ballot. [1]_

    Party-specific rules for the third semifinalist and blank ballots from the
    full specification are omitted here (they are optional for one-off
    simulations).

    Parameters
    ----------
    election : array_like
        Shape ``(n_voters, n_candidates)``. Ratings must be 0, 1, or 2.

    tiebreaker : {'random', 'order', None}, optional
        Used whenever the rules leave a tie after score comparisons.

    Returns
    -------
    winner : {int, None}

    References
    ----------
    .. [1] https://electowiki.org/wiki/3-2-1_voting

    """
    election = np.asarray(election)

    if election.ndim != 2:
        raise ValueError('Election must be a 2D array')
    if election.min() < 0 or election.max() > 2:
        raise ValueError('3-2-1 ballots must use ratings 0, 1, and 2 only')

    n_voters, n_cands = election.shape
    tiebreak = _get_tiebreak(tiebreaker, _tiebreak_map)

    good = (election == 2).sum(axis=0)
    score = election.sum(axis=0)

    if n_cands == 1:
        return 0

    if n_cands == 2:
        return _pairwise_rating_preference(election, 0, 1, tiebreaker)

    semifinalists = sorted(range(n_cands),
                           key=lambda i: (-int(good[i]), -int(score[i]), i))[:3]

    sf = semifinalists
    bad_sf = [(election[:, j] == 0).sum() for j in sf]

    mx_bad = max(bad_sf)
    tied_max = [r for r in range(3) if bad_sf[r] == mx_bad]
    elim_rel = sorted(tied_max, key=lambda r: (score[sf[r]], sf[r]))[0]

    finalists = [sf[i] for i in range(3) if i != elim_rel]
    if len(finalists) != 2:
        raise RuntimeError('expected two finalists')

    a, b = finalists[0], finalists[1]
    return _pairwise_rating_preference(election, a, b, tiebreaker)
