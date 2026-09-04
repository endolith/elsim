"""Two-party primary-style tallies for spatial simulations."""
import numpy as np

from elsim.methods._common import (_get_tiebreak, _no_tiebreak,
                                   _order_tiebreak_keep, _random_tiebreak)
from elsim.methods.fptp import sntv

_tiebreak_map = {'order': _order_tiebreak_keep,
                 'random': _random_tiebreak,
                 None: _no_tiebreak}


def nominee_restricted_plurality(rankings, voter_indices, allowed_candidates,
                                 tiebreaker=None):
    """
    Plurality winner restricted to candidates in ``allowed_candidates``.

    Each voter contributes one vote to their highest-ranked candidate among the
    allowed set (same order as on their full ranking).

    Parameters
    ----------
    rankings : array_like
        Full ranked ballots, shape ``(n_voters, n_cands)``.
    voter_indices : array_like
        Indices of voters participating in this round (e.g. primary electorate).
    allowed_candidates : array_like
        Candidate IDs allowed on this ballot.

    tiebreaker : {'random', 'order', None}, optional
        Breaks ties for first place.

    Returns
    -------
    winner : int
        Candidate ID of the nominee.
    """
    rankings = np.asarray(rankings)
    allowed_candidates = np.asarray(allowed_candidates, dtype=np.int64)
    tallies = np.zeros(rankings.shape[1], dtype=np.int64)
    allowed_set = frozenset(int(x) for x in allowed_candidates.flat)

    for i in voter_indices:
        for c in rankings[i]:
            cc = int(c)
            if cc in allowed_set:
                tallies[cc] += 1
                break

    sub = tallies[allowed_candidates]
    winners_rel = np.flatnonzero(sub == np.max(sub))
    winners = allowed_candidates[winners_rel].tolist()
    tiebreak = _get_tiebreak(tiebreaker, _tiebreak_map)
    return tiebreak(winners)[0]


def pairwise_majority_from_rankings(rankings, voter_indices, cand_a, cand_b,
                                    tiebreaker=None):
    """
    Majority winner between two candidates using relative ranking order.

    Parameters
    ----------
    rankings : array_like
        Full rankings.
    voter_indices : array_like
        Voters participating in this comparison (e.g. general-election turnout).
    cand_a, cand_b : int
        Candidate IDs.

    tiebreaker : {'random', 'order', None}, optional
        Used when exactly half prefer each candidate.

    Returns
    -------
    winner : int
    """
    rankings = np.asarray(rankings)
    tiebreak = _get_tiebreak(tiebreaker, _tiebreak_map)
    tally_a = 0
    tally_b = 0

    for i in voter_indices:
        ballot = rankings[i]
        pos_a = np.argmax(ballot == cand_a)
        pos_b = np.argmax(ballot == cand_b)
        if pos_a < pos_b:
            tally_a += 1
        elif pos_b < pos_a:
            tally_b += 1

    if tally_a > tally_b:
        return cand_a
    if tally_b > tally_a:
        return cand_b
    return tiebreak([cand_a, cand_b])[0]


def open_partisan_primary(rankings, n_cands_left_cluster,
                          primary_left_voters, primary_right_voters,
                          general_voters, tiebreaker=None):
    """
    Open partisan primary then general election (both plurality).

    Each party holds a primary among its own candidates using only that party's
    primary electorate; the winners face each other in a general election using
    ``general_voters``.

    Candidate IDs ``0 .. n_cands_left_cluster - 1`` are the left party;
    ``n_cands_left_cluster .. n_cands - 1`` are the right party.

    Parameters
    ----------
    rankings : array_like
        Honest full rankings for all voters.
    n_cands_left_cluster : int
        Number of candidates in the left party.
    primary_left_voters, primary_right_voters : array_like
        Voter indices participating in each party's primary (often subsets).
    general_voters : array_like
        Voter indices in the general election.

    tiebreaker : {'random', 'order', None}, optional

    Returns
    -------
    winner : int
        Winning candidate ID.
    """
    rankings = np.asarray(rankings)
    n_cands = rankings.shape[1]
    left_c = np.arange(0, n_cands_left_cluster)
    right_c = np.arange(n_cands_left_cluster, n_cands)

    nom_l = nominee_restricted_plurality(rankings, primary_left_voters,
                                         left_c, tiebreaker)
    nom_r = nominee_restricted_plurality(rankings, primary_right_voters,
                                         right_c, tiebreaker)

    return pairwise_majority_from_rankings(rankings, general_voters,
                                           nom_l, nom_r, tiebreaker)


def closed_partisan_primary_runoff(rankings, n_cands_left_cluster,
                                   primary_left_voters, primary_right_voters,
                                   runoff_voters, tiebreaker=None):
    """
    Closed partisan primaries followed by a top-two runoff.

    Same primaries as :func:`open_partisan_primary`, but the contest between
    nominees uses only ``runoff_voters`` (can be a subset of all voters).

    Parameters
    ----------
    rankings : array_like
    n_cands_left_cluster : int
    primary_left_voters, primary_right_voters : array_like
    runoff_voters : array_like
        Voters participating in the general/runoff between nominees.

    tiebreaker : {'random', 'order', None}, optional

    Returns
    -------
    winner : int
    """
    rankings = np.asarray(rankings)
    n_cands = rankings.shape[1]
    left_c = np.arange(0, n_cands_left_cluster)
    right_c = np.arange(n_cands_left_cluster, n_cands)

    nom_l = nominee_restricted_plurality(rankings, primary_left_voters,
                                         left_c, tiebreaker)
    nom_r = nominee_restricted_plurality(rankings, primary_right_voters,
                                         right_c, tiebreaker)

    return pairwise_majority_from_rankings(rankings, runoff_voters,
                                           nom_l, nom_r, tiebreaker)


def top_two_runoff_reduced_turnout(rankings, first_round_voters,
                                   runoff_voters, tiebreaker=None):
    """
    Top-two runoff where the pairwise round uses a subset of voters.

    The top two candidates by first-preference plurality among
    ``first_round_voters`` advance; the winner is decided by pairwise majority
    among ``runoff_voters``.

    Parameters
    ----------
    rankings : array_like
    first_round_voters : array_like
        Voters counted for establishing the top two (general-election cohort).
    runoff_voters : array_like
        Usually a subset of voters for the second round.

    tiebreaker : {'random', 'order', None}, optional

    Returns
    -------
    winner : int
    """
    rankings = np.asarray(rankings)
    first_prefs = rankings[first_round_voters, 0]
    top_two = sntv(first_prefs, n=2, tiebreaker=tiebreaker)
    if top_two is None:
        return None

    finalists = sorted(top_two)
    if len(finalists) == 1:
        return finalists[0]
    if len(finalists) != 2:
        return None

    a, b = finalists[0], finalists[1]
    return pairwise_majority_from_rankings(rankings, runoff_voters,
                                           a, b, tiebreaker)
