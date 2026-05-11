"""
Two-round voting methods that combine a primary (narrowing the field) with a
general election.

These implement nonpartisan blanket primaries and related reforms: approval
plus runoff (unified primary), pick-one top-four or top-five with an IRV
general (Final Four / Final Five), the same pick-one primary with a
contingent general among finalists, a ranked sequential primary that leaves
``n`` survivors (see ``irv(..., n_survivors=n)``) followed by that contingent
general, and Condorcet in the general.
"""
import numpy as np

from elsim.methods._common import (_get_tiebreak, _no_tiebreak,
                                   _order_tiebreak_keep, _random_tiebreak)
from elsim.methods.condorcet import condorcet
from elsim.methods.fptp import sntv
from elsim.methods.irv import irv
from elsim.methods.runoff import runoff

_tiebreak_map_keep = {'order': _order_tiebreak_keep,
                      'random': _random_tiebreak,
                      None: _no_tiebreak}


def _top_n_from_plurality_tallies(tallies, n, tiebreaker):
    """
    Candidate indices with the top ``n`` first-preference counts, breaking
    boundary ties the same way as ``sntv``.
    """
    tallies = np.asarray(tallies)
    n_cands = len(tallies)
    if n < 1:
        raise ValueError('n must be at least 1')
    if n >= n_cands:
        return set(range(n_cands))

    sorted_indices = np.argsort(tallies)
    top_candidates = sorted_indices[-n:]
    worst_top = top_candidates[0]
    best_not_top = sorted_indices[-n - 1]

    if tallies[worst_top] == tallies[best_not_top]:
        untied_winners = sorted_indices[tallies[sorted_indices] >
                                        tallies[worst_top]]
        tied_candidates = np.where(tallies == tallies[worst_top])[0]
        tiebreak = _get_tiebreak(tiebreaker, _tiebreak_map_keep)
        n_needed = n - len(untied_winners)
        tie_winners = tiebreak(list(tied_candidates), n_needed)
        if tie_winners == [None]:
            return None
        winners = set(untied_winners) | set(tie_winners)
    else:
        winners = set(top_candidates)

    return set(int(w) for w in winners)


def _primary_top_n_approval(approval_election, n, tiebreaker):
    approval_election = np.asarray(approval_election, dtype=np.uint8)
    tallies = approval_election.sum(axis=0)
    n_cands = approval_election.shape[1]
    if n >= n_cands:
        return set(range(n_cands))
    return _top_n_from_plurality_tallies(tallies, n, tiebreaker)


def _restrict_ballots(election, allowed):
    """Drop non-finalists from each ballot, preserving order (local IDs 0..k-1)."""
    election = np.asarray(election)
    allowed_sorted = sorted(allowed)
    old_to_new = {cand: j for j, cand in enumerate(allowed_sorted)}
    new_to_old = {j: cand for j, cand in enumerate(allowed_sorted)}
    rows = []
    for ballot in election:
        row = [old_to_new[c] for c in ballot.tolist() if c in old_to_new]
        rows.append(row)
    return np.asarray(rows, dtype=election.dtype), new_to_old


def _head_to_head_two(finalist_0, finalist_1, election, tiebreaker):
    """Pairwise winner between two finalists using full rankings (contingent vote)."""
    tiebreak = _get_tiebreak(tiebreaker, _tiebreak_map_keep)
    n_voters = election.shape[0]
    finalist_0_tally = 0
    finalist_1_tally = 0
    for ballot in election:
        ballot_list = ballot.tolist()
        if ballot_list.index(finalist_0) < ballot_list.index(finalist_1):
            finalist_0_tally += 1
        else:
            finalist_1_tally += 1
    assert finalist_0_tally + finalist_1_tally == n_voters
    if finalist_0_tally == finalist_1_tally:
        return tiebreak([finalist_0, finalist_1])[0]
    if finalist_0_tally > finalist_1_tally:
        return finalist_0
    return finalist_1


def approval_runoff(approval_election, ranked_election, tiebreaker=None):
    """
    Find the winner of an election using a top-two approval primary and a
    pairwise runoff on ranked ballots.

    The two candidates with the most approvals advance.  The general election
    is modeled as a pairwise majority vote between those two on the ranked
    ballots (the contingent vote, same ranked-ballot abstraction as ``runoff``
    uses between its finalists). [1]_ [2]_  Delemazure et al. define and study
    the approval-with-runoff family in the computational social choice literature. [3]_

    Parameters
    ----------
    approval_election : array_like
        A 2D collection of approval ballots.  See `approval` for format.
    ranked_election : array_like
        A collection of ranked ballots for the same voters as
        ``approval_election`` (same number of rows).  See `borda` for format.
    tiebreaker : {'random', 'order', None}, optional
        If there is a tie, and `tiebreaker` is ``'random'``, a random finalist
        is returned in the primary or in a tied general.
        If 'order', the lowest-ID tied candidate is returned.
        By default, ``None`` is returned for ties.

    Returns
    -------
    winner : {int, None}
        The ID number of the winner, or ``None`` for an unbroken tie.

    References
    ----------
    .. [1] `Unified primary <https://en.wikipedia.org/wiki/Unified_primary>`__
    .. [2] `Contingent vote <https://en.wikipedia.org/wiki/Contingent_vote>`__
    .. [3] `Delemazure et al., "Approval with Runoff" (IJCAI 2022) <https://doi.org/10.24963/ijcai.2022/33>`__

    Examples
    --------
    >>> A, B, C = 0, 1, 2
    >>> approvals = [[1, 1, 0], [1, 1, 0], [0, 1, 1]]
    >>> ranked = [[B, A, C], [B, C, A], [C, B, A]]
    >>> approval_runoff(approvals, ranked)
    1
    """
    approval_election = np.asarray(approval_election, dtype=np.uint8)
    ranked_election = np.asarray(ranked_election)
    if approval_election.shape[0] != ranked_election.shape[0]:
        raise ValueError('approval_election and ranked_election must have the '
                         'same number of rows (voters)')
    if approval_election.max() > 1:
        raise ValueError('Approval ballots must contain only 0 and 1')

    finalists = _primary_top_n_approval(approval_election, 2, tiebreaker)
    if finalists is None:
        return None
    if len(finalists) == 1:
        (only,) = tuple(finalists)
        return only
    finalist_0, finalist_1 = sorted(finalists)
    out = _head_to_head_two(finalist_0, finalist_1, ranked_election,
                            tiebreaker)
    return int(out) if out is not None else None


def top_n_irv(election, n, tiebreaker=None):
    """
    Find the winner of an election using a pick-one top-``n`` primary and an
    IRV general.

    The ``n`` candidates with the most first-preference votes advance (same
    rule as ``sntv``).  The winner is then chosen by ``irv`` on the same
    rankings restricted to those finalists.  ``top_n_irv(..., n=4)`` matches
    Alaska's nonpartisan top-four primary with an IRV general; ``n`` = 5 is
    the ranked general used in Final Five-style proposals. [1]_ [2]_ [3]_

    Parameters
    ----------
    election : array_like
        A collection of ranked ballots.  See `borda` for election format.
        Currently, this must include full rankings for each voter.
    n : int
        Number of candidates who advance from the primary (``n`` = 4 is Final
        Four; ``n`` = 5 is the slate size in Final Five together with this IRV
        general).
    tiebreaker : {'random', 'order', None}, optional
        If there is a tie, and `tiebreaker` is ``'random'``, a random finalist
        is returned or tied candidates are eliminated at random, according to
        the underlying ``sntv`` / ``irv`` steps.
        If 'order', the lowest-ID tied candidate is preferred in each tie.
        By default, ``None`` is returned for ties.

    Returns
    -------
    winner : {int, None}
        The ID number of the winner, or ``None`` for an unbroken tie.

    References
    ----------
    .. [1] `Top-four primary <https://en.wikipedia.org/wiki/Top-four_primary>`__
    .. [2] `Alaska Division of Elections, "Ranked Choice Voting" (top-four primary and RCV general) <https://www.elections.alaska.gov/Core/RCV.php>`__
    .. [3] `FairVote, "Top Four" policy guide (PDF, 2013) <https://archive3.fairvote.org/assets/Top-Four-Policy-Guide.pdf>`__

    Examples
    --------
    >>> A, B, C = 0, 1, 2
    >>> election = [*6*[[A, B, C]], *3*[[B, A, C]], *1*[[C, B, A]]]
    >>> top_n_irv(election, 2)
    0
    """
    finalists = sntv(election, n, tiebreaker)
    if finalists is None:
        return None
    sub, new_to_old = _restrict_ballots(election, finalists)
    w = irv(sub, tiebreaker)
    if w is None:
        return None
    return int(new_to_old[w])


def top_n_runoff(election, n, tiebreaker=None):
    """
    Find the winner of an election using a pick-one top-``n`` primary and a
    contingent general among those finalists.

    The primary is ``sntv``: the ``n`` candidates with the most first-preference
    votes in the full field advance.  The general is ``runoff`` applied to the
    sub-election that keeps only those finalists on each ballot.  That is not
    the same as calling ``runoff`` on the original ballots, because ``runoff``
    here first ranks first preferences **among the finalists only** to pick
    two of them, then breaks a deadlock with a pairwise vote on the same
    restricted rankings (the contingent vote, as in ``runoff``). [1]_ [2]_

    Parameters
    ----------
    election : array_like
        A collection of ranked ballots.  See `borda` for election format.
        Currently, this must include full rankings for each voter.
    n : int
        Number of candidates who advance from the primary.
    tiebreaker : {'random', 'order', None}, optional
        If there is a tie, and `tiebreaker` is ``'random'``, a random finalist
        is returned.
        If 'order', the lowest-ID tied candidate is returned.
        By default, ``None`` is returned for ties.

    Returns
    -------
    winner : {int, None}
        The ID number of the winner, or ``None`` for an unbroken tie.

    References
    ----------
    .. [1] `Top-four primary: Variations <https://en.wikipedia.org/wiki/Top-four_primary#Variations>`__
    .. [2] `Electoral Reform Society, "What is the Supplementary Vote?" <https://www.electoral-reform.org.uk/voting-systems/types-of-voting-systems/supplementary-vote-sv/>`__

    Examples
    --------
    >>> A, B, C, D = 0, 1, 2, 3
    >>> election = [*3*[[A, B, C, D]], *2*[[B, A, C, D]], *1*[[C, D, A, B]]]
    >>> isinstance(top_n_runoff(election, 4), int)
    True
    """
    finalists = sntv(election, n, tiebreaker)
    if finalists is None:
        return None
    sub, new_to_old = _restrict_ballots(election, finalists)
    w = runoff(sub, tiebreaker)
    if w is None:
        return None
    return int(new_to_old[w])


def top_n_condorcet(election, n, tiebreaker=None):
    """
    Find the winner of an election using a pick-one top-``n`` primary and a
    Condorcet pairwise general.

    The primary uses the same top-``n`` rule as ``sntv``.  The general election
    applies ``condorcet`` to the restricted rankings (no tiebreaker in the
    general, matching ``condorcet`` itself). [1]_  Condorcet's 1785 essay
    defines pairwise majority comparisons among candidates. [2]_  A modern
    overview is in [3]_.

    Parameters
    ----------
    election : array_like
        A collection of ranked ballots.  See `borda` for election format.
        Currently, this must include full rankings for each voter.
    n : int
        Number of candidates who advance from the primary.
    tiebreaker : {'random', 'order', None}, optional
        Used only for the primary.  If there is a tie, and `tiebreaker` is
        ``'random'``, random tied candidates are returned.
        If 'order', the lowest-ID tied candidates are returned.
        By default, ``None`` is returned for ties in the primary.

    Returns
    -------
    winner : {int, None}
        The ID number of the winner, or ``None`` for an unbroken tie in the
        primary or no Condorcet winner in the general.

    References
    ----------
    .. [1] `Top-four primary: Variations <https://en.wikipedia.org/wiki/Top-four_primary#Variations>`__
    .. [2] `Condorcet (1785), *Essai sur l'application de l'analyse à la probabilité des décisions rendues à la pluralité des voix* (BnF Gallica) <https://gallica.bnf.fr/ark:/12148/bpt6k417181>`__
    .. [3] `Condorcet method <https://en.wikipedia.org/wiki/Condorcet_method>`__

    Examples
    --------
    >>> A, B, C = 0, 1, 2
    >>> election = [[A, B, C], [A, B, C], [B, A, C]]
    >>> top_n_condorcet(election, 2)
    0
    """
    finalists = sntv(election, n, tiebreaker)
    if finalists is None:
        return None
    sub, new_to_old = _restrict_ballots(election, finalists)
    w = condorcet(sub)
    if w is None:
        return None
    return int(new_to_old[w])


def irv_primary_top_n_runoff(election, n, tiebreaker=None):
    """
    Find the winner of an election using a ranked sequential primary to a
    slate of ``n``, then a contingent general among those finalists.

    The primary is ``irv(..., n_survivors=n)``: the same last-place elimination
    and transfers as ``irv``, but without stopping when someone reaches an
    overall majority, until ``n`` candidates remain.  The general is ``runoff``
    on ballots restricted to that slate (first preferences among finalists
    only to pick two of them, then the pairwise stage as in ``runoff``). [1]_ [2]_

    Parameters
    ----------
    election : array_like
        A collection of ranked ballots.  See `borda` for election format.
        Currently, this must include full rankings for each voter.
    n : int
        Target number of finalists after the primary.
    tiebreaker : {'random', 'order', None}, optional
        If there is a tie, and `tiebreaker` is ``'random'``, a random finalist
        is returned.
        If 'order', the lowest-ID tied candidate is returned.
        By default, ``None`` is returned for ties.

    Returns
    -------
    winner : {int, None}
        The ID number of the winner, or ``None`` for an unbroken tie.

    References
    ----------
    .. [1] `Top-four primary: Variations <https://en.wikipedia.org/wiki/Top-four_primary#Variations>`__
    .. [2] `FairVote, "Top Four" policy guide (PDF, 2013) <https://archive3.fairvote.org/assets/Top-Four-Policy-Guide.pdf>`__

    Examples
    --------
    >>> A, B, C = 0, 1, 2
    >>> election = [*6*[[A, B, C]], *3*[[B, A, C]], *1*[[C, B, A]]]
    >>> irv_primary_top_n_runoff(election, 2, tiebreaker='order')
    0
    """
    finalists = irv(election, tiebreaker, n_survivors=n)
    if finalists is None:
        return None
    sub, new_to_old = _restrict_ballots(election, finalists)
    w = runoff(sub, tiebreaker)
    if w is None:
        return None
    return int(new_to_old[w])


def top_four_irv(election, tiebreaker=None):
    """
    Find the winner of an election using a pick-one top-four primary and an
    IRV general.

    Same as ``top_n_irv(election, 4, tiebreaker)``. [1]_ [2]_ [3]_

    Parameters
    ----------
    election : array_like
        Passed to `top_n_irv`.
    tiebreaker : {'random', 'order', None}, optional
        Passed to `top_n_irv`.

    Returns
    -------
    winner : {int, None}
        The ID number of the winner, or ``None`` for an unbroken tie.

    References
    ----------
    .. [1] `Top-four primary <https://en.wikipedia.org/wiki/Top-four_primary>`__
    .. [2] `Alaska Division of Elections, "Ranked Choice Voting" (top-four primary and RCV general) <https://www.elections.alaska.gov/Core/RCV.php>`__
    .. [3] `FairVote, "Top Four" policy guide (PDF, 2013) <https://archive3.fairvote.org/assets/Top-Four-Policy-Guide.pdf>`__
    """
    return top_n_irv(election, 4, tiebreaker)


def top_five_irv(election, tiebreaker=None):
    """
    Find the winner of an election using a pick-one top-five primary and an
    IRV general.

    Same as ``top_n_irv(election, 5, tiebreaker)``. [1]_ [2]_  Gehl & Porter
    (2017) introduce the Final Five reform framing. [3]_

    Parameters
    ----------
    election : array_like
        Passed to `top_n_irv`.
    tiebreaker : {'random', 'order', None}, optional
        Passed to `top_n_irv`.

    Returns
    -------
    winner : {int, None}
        The ID number of the winner, or ``None`` for an unbroken tie.

    References
    ----------
    .. [1] `Top-four primary <https://en.wikipedia.org/wiki/Top-four_primary>`__
    .. [2] `Final Five Voting <https://en.wikipedia.org/wiki/Top-four_primary#Final_Five_Voting>`__
    .. [3] `Gehl & Porter (2017), "Why Competition in the Politics Industry is Failing America" (PDF) <https://www.hbs.edu/competitiveness/Documents/why-competition-in-the-politics-industry-is-failing-america.pdf>`__
    """
    return top_n_irv(election, 5, tiebreaker)


def top_four_runoff(election, tiebreaker=None):
    """
    Find the winner of an election using a pick-one top-four primary and a
    contingent general among finalists.

    Same as ``top_n_runoff(election, 4, tiebreaker)``. [1]_ [2]_

    Parameters
    ----------
    election : array_like
        Passed to `top_n_runoff`.
    tiebreaker : {'random', 'order', None}, optional
        Passed to `top_n_runoff`.

    Returns
    -------
    winner : {int, None}
        The ID number of the winner, or ``None`` for an unbroken tie.

    References
    ----------
    .. [1] `Top-four primary: Variations <https://en.wikipedia.org/wiki/Top-four_primary#Variations>`__
    .. [2] `Electoral Reform Society, "What is the Supplementary Vote?" <https://www.electoral-reform.org.uk/voting-systems/types-of-voting-systems/supplementary-vote-sv/>`__
    """
    return top_n_runoff(election, 4, tiebreaker)


def top_five_runoff(election, tiebreaker=None):
    """
    Find the winner of an election using a pick-one top-five primary and a
    contingent general among finalists.

    Same as ``top_n_runoff(election, 5, tiebreaker)``. [1]_ [2]_ [3]_ [4]_

    Parameters
    ----------
    election : array_like
        Passed to `top_n_runoff`.
    tiebreaker : {'random', 'order', None}, optional
        Passed to `top_n_runoff`.

    Returns
    -------
    winner : {int, None}
        The ID number of the winner, or ``None`` for an unbroken tie.

    References
    ----------
    .. [1] `Top-four primary: Variations <https://en.wikipedia.org/wiki/Top-four_primary#Variations>`__
    .. [2] `Final Five Voting <https://en.wikipedia.org/wiki/Top-four_primary#Final_Five_Voting>`__
    .. [3] `Electoral Reform Society, "What is the Supplementary Vote?" <https://www.electoral-reform.org.uk/voting-systems/types-of-voting-systems/supplementary-vote-sv/>`__
    .. [4] `Gehl & Porter (2017), "Why Competition in the Politics Industry is Failing America" (PDF) <https://www.hbs.edu/competitiveness/Documents/why-competition-in-the-politics-industry-is-failing-america.pdf>`__
    """
    return top_n_runoff(election, 5, tiebreaker)


def top_four_condorcet(election, tiebreaker=None):
    """
    Find the winner of an election using a pick-one top-four primary and a
    Condorcet pairwise general.

    Same as ``top_n_condorcet(election, 4, tiebreaker)``. [1]_ [2]_ [3]_

    Parameters
    ----------
    election : array_like
        Passed to `top_n_condorcet`.
    tiebreaker : {'random', 'order', None}, optional
        Passed to `top_n_condorcet`.

    Returns
    -------
    winner : {int, None}
        The ID number of the winner, or ``None`` for an unbroken tie.

    References
    ----------
    .. [1] `Top-four primary: Variations <https://en.wikipedia.org/wiki/Top-four_primary#Variations>`__
    .. [2] `Condorcet (1785), *Essai sur l'application de l'analyse à la probabilité des décisions rendues à la pluralité des voix* (BnF Gallica) <https://gallica.bnf.fr/ark:/12148/bpt6k417181>`__
    .. [3] `Condorcet method <https://en.wikipedia.org/wiki/Condorcet_method>`__
    """
    return top_n_condorcet(election, 4, tiebreaker)


def top_five_condorcet(election, tiebreaker=None):
    """
    Find the winner of an election using a pick-one top-five primary and a
    Condorcet pairwise general.

    Same as ``top_n_condorcet(election, 5, tiebreaker)``. [1]_ [2]_ [3]_ [4]_ [5]_

    Parameters
    ----------
    election : array_like
        Passed to `top_n_condorcet`.
    tiebreaker : {'random', 'order', None}, optional
        Passed to `top_n_condorcet`.

    Returns
    -------
    winner : {int, None}
        The ID number of the winner, or ``None`` for an unbroken tie.

    References
    ----------
    .. [1] `Top-four primary: Variations <https://en.wikipedia.org/wiki/Top-four_primary#Variations>`__
    .. [2] `Final Five Voting <https://en.wikipedia.org/wiki/Top-four_primary#Final_Five_Voting>`__
    .. [3] `Condorcet (1785), *Essai sur l'application de l'analyse à la probabilité des décisions rendues à la pluralité des voix* (BnF Gallica) <https://gallica.bnf.fr/ark:/12148/bpt6k417181>`__
    .. [4] `Condorcet method <https://en.wikipedia.org/wiki/Condorcet_method>`__
    .. [5] `Gehl & Porter (2017), "Why Competition in the Politics Industry is Failing America" (PDF) <https://www.hbs.edu/competitiveness/Documents/why-competition-in-the-politics-industry-is-failing-america.pdf>`__
    """
    return top_n_condorcet(election, 5, tiebreaker)
