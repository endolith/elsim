import json
import random
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import integers, lists, permutations

from elsim.methods import irv, irv_rounds

_FIXTURES = Path(__file__).resolve().parent / 'fixtures'


def collect_random_results(method, election):
    """
    Run multiple elections with tiebreaker='random' and collect the set of all
    winners.
    """
    random.seed(47)  # Deterministic test
    winners = set()
    for _trial in range(10):
        winner = method(election, tiebreaker='random')
        assert isinstance(winner, int)
        winners.add(winner)
    return winners


def _irish_1990_election():
    """
    1990 Irish presidential election (three candidates).

    First preferences and Currie transfers are scaled 1:1000 from the official
    counts on Wikipedia; the 206/37 split among Currie voters reproduces the
    published second-round totals (Robinson 817,830; Lenihan 731,273).

    https://en.wikipedia.org/wiki/Instant-runoff_voting#1990_Irish_presidential_election
    """
    Robinson, Lenihan, Currie = 0, 1, 2
    return [*694 * [[Lenihan, Robinson, Currie]],
            *612 * [[Robinson, Lenihan, Currie]],
            *206 * [[Currie, Robinson, Lenihan]],
            * 37 * [[Currie, Lenihan, Robinson]]]


def _prahran_2014_election():
    """
    2014 Prahran (Victoria) state election, eight candidates.

    Built from first-preference counts and VEC preference-flow percentages
    (Parliamentary Library / Wikipedia Examples table).  Remaining candidates
    after each flow are filled in ballot-paper ID order below the stated
    second preference.

    https://en.wikipedia.org/wiki/Instant-runoff_voting#2014_Prahran_election_(Victoria)
    """
    Hibbins, Walker, Pharaoh, Goldsmith, Stefanopoulos, \
        NewtonBrown, Gullone, Menadue = range(8)

    first_prefs = {
        Menadue: 82,
        Stefanopoulos: 227,
        Walker: 282,
        Goldsmith: 247,
        Gullone: 837,
        Pharaoh: 9586,
        Hibbins: 9160,
        NewtonBrown: 16582,
    }
    flows = [
        (Menadue, {NewtonBrown: 0.122, Pharaoh: 0.085, Hibbins: 0.134}),
        (Stefanopoulos, {NewtonBrown: 0.216, Pharaoh: 0.191, Hibbins: 0.195}),
        (Walker, {NewtonBrown: 0.278, Pharaoh: 0.173, Hibbins: 0.134}),
        (Goldsmith, {NewtonBrown: 0.335, Pharaoh: 0.195, Hibbins: 0.266}),
        (Gullone, {NewtonBrown: 0.233, Pharaoh: 0.19, Hibbins: 0.577}),
        (Pharaoh, {NewtonBrown: 0.129, Hibbins: 0.871}),
    ]

    def fill_tail(first, second):
        order = [first]
        if second is not None:
            order.append(second)
        for cand in range(8):
            if cand not in order:
                order.append(cand)
        return order

    def split_votes(n, fracs):
        keys = list(fracs.keys())
        raw = {k: fracs[k] * n for k in keys}
        base = {k: int(raw[k]) for k in keys}
        rem = n - sum(base.values())
        if rem:
            for k in sorted(keys, key=lambda k: -(raw[k] - base[k]))[:rem]:
                base[k] += 1
        return base

    election = []
    fp = dict(first_prefs)
    for elim, fracs in flows:
        n = fp.pop(elim)
        second_counts = split_votes(n, fracs)
        exhausted = n - sum(second_counts.values())
        for second, cnt in second_counts.items():
            election.extend(fill_tail(elim, second) for _ in range(cnt))
        election.extend(fill_tail(elim, None) for _ in range(exhausted))
    for cand, n in fp.items():
        election.extend(fill_tail(cand, None) for _ in range(n))
    return election


def _burlington_2009_election():
    """
    2009 Burlington mayoral election (six candidates on the ballot).

    Rankings are derived from Electowiki inline ballot counts; candidates not
    scored on a ballot are appended in ascending ID order so every voter has a
    full ranking (required by ``irv``).  This matches the usual reconstruction
    used for Burlington IRV analyses.

    https://en.wikipedia.org/wiki/Instant-runoff_voting#2009_Burlington_mayoral_election
    https://electowiki.org/wiki/2009_Burlington,_Vermont_Mayoral_Election_data
    """
    name_to_id = {'Kiss': 0, 'Montroll': 1, 'Wright': 2, 'Smith': 3,
                  'Simpson': 4, 'Write-in': 5}
    all_names = list(name_to_id)
    entries = json.loads(
        (_FIXTURES / 'burlington_2009_inline_ballots.json').read_text())
    election = []
    for entry in entries:
        vote = entry['vote']
        ranked = sorted(vote.keys(), key=lambda k: -vote[k])
        for name in all_names:
            if name not in ranked:
                ranked.append(name)
        ballot = [name_to_id[name] for name in ranked]
        election.extend([ballot] * entry['qty'])
    return election


@pytest.mark.parametrize("tiebreaker", [None, 'random', 'order'])
def test_strict_majority(tiebreaker):
    A, B, C = 0, 1, 2
    election = [[A, B, C],
                [B, C, A],
                [A, C, B],
                ]
    assert irv(election, tiebreaker) == A

    election = [*3*[[A, B, C]],
                *7*[[B, C, A]],
                *2*[[C, B, A]],
                ]
    assert irv(election, tiebreaker) == B


def test_no_tiebreak_tied_losers():
    A, B, C = 0, 1, 2
    election = [[A, B, C],
                [B, C, A],
                [A, C, B],
                [B, C, A],
                [C, B, A],
                [C, A, B],
                [C, B, A],
                ]
    assert irv(election) is None


def test_one_round():
    # 60% majority, tie between others
    election = np.array([[2, 0, 1],
                         [0, 1, 2],
                         [2, 0, 1],
                         [1, 2, 0],
                         [2, 1, 0],
                         [2, 0, 1],
                         [1, 0, 2],
                         [2, 0, 1],
                         [2, 1, 0],
                         [0, 2, 1]])
    assert irv(election) == 2
    assert irv(election, tiebreaker='order') == 2
    assert irv(election, tiebreaker='random') == 2

    # 50% winner, 30%, 20% for others
    # In this case, Candidate 2 picks up an additional vote in the runoff,
    # making it unambiguous.
    election = np.array([[2, 0, 1],
                         [0, 1, 2],
                         [1, 0, 2],
                         [2, 0, 1],
                         [2, 1, 0],
                         [0, 1, 2],
                         [2, 0, 1],
                         [1, 2, 0],
                         [2, 1, 0],
                         [0, 2, 1]])
    assert irv(election) == 2
    assert irv(election, tiebreaker='order') == 2
    assert irv(election, tiebreaker='random') == 2

    # 50%, 30%, 20%
    # This is ambiguous. It would make sense for the 50% candidate to win
    # outright, but technically they don't have a majority, so we have to
    # eliminate another, so there's now a 50/50 split, and then tiebreak
    # between the two, which might pick a different candidate, even though they
    # got fewer first-preference votes.
    election = np.array([[2, 0, 1],
                         [0, 1, 2],
                         [1, 0, 2],
                         [2, 0, 1],
                         [2, 1, 0],
                         [0, 1, 2],
                         [2, 0, 1],
                         [1, 0, 2],
                         [2, 1, 0],
                         [0, 2, 1]])
    assert irv(election) is None
    assert irv(election, tiebreaker='order') == 0
    assert collect_random_results(irv, election) == {0, 2}

    # 50%, 25%, 25%
    # If 0 eliminated by tiebreak, another transfers to each and 2 wins
    # If 1 eliminated by tiebreak, 2 transfer to 0 and a second tiebreak
    # So either 0 or 2 wins.
    election = np.array([[2, 0, 1],
                         [0, 1, 2],
                         [1, 0, 2],
                         [2, 0, 1],
                         [2, 0, 1],
                         [1, 0, 2],
                         [2, 1, 0],
                         [0, 2, 1]])
    assert irv(election) is None
    assert irv(election, tiebreaker='order') == 0
    assert collect_random_results(irv, election) == {0, 2}

    # 50%, 25%, 25%
    # 0 or 1 is eliminated and transfers votes to the other, making it a tie.
    # So any candidate can win.
    election = np.array([[2, 0, 1],
                         [0, 1, 2],
                         [1, 0, 2],
                         [2, 0, 1],
                         [2, 0, 1],
                         [1, 0, 2],
                         [2, 1, 0],
                         [0, 1, 2]])
    assert irv(election) is None
    assert irv(election, tiebreaker='order') == 0
    assert collect_random_results(irv, election) == {0, 1, 2}

    # 50% exact tie
    election = np.array([[2, 0, 1],
                         [1, 0, 2],
                         [1, 2, 0],
                         [2, 1, 0]])
    assert irv(election) is None
    assert irv(election, tiebreaker='order') == 1
    assert collect_random_results(irv, election) == {1, 2}

    # Complete cycle, anyone can win
    election = np.array([[0, 1, 2],
                         [1, 2, 0],
                         [2, 0, 1]])
    assert irv(election) is None
    assert irv(election, tiebreaker='order') == 0
    assert collect_random_results(irv, election) == {0, 1, 2}


@pytest.mark.parametrize("tiebreaker", [None, 'random', 'order'])
def test_examples(tiebreaker):
    # Standard Tennessee example (three round)
    # https://en.wikipedia.org/wiki/Template:Tenn_voting_example
    Memphis, Nashville, Chattanooga, Knoxville = 0, 1, 2, 3
    election = [*42*[[Memphis, Nashville, Chattanooga, Knoxville]],
                *26*[[Nashville, Chattanooga, Knoxville, Memphis]],
                *15*[[Chattanooga, Knoxville, Nashville, Memphis]],
                *17*[[Knoxville, Chattanooga, Nashville, Memphis]],
                ]
    assert irv(election, tiebreaker) == Knoxville

    # Three-round example from Ques 9
    # http://www.yorku.ca/bucovets/4380/exercises/exercises_1_a.pdf
    v, w, x, y, z = 0, 1, 2, 3, 4
    election = [*11*[[v, w, x, y, z]],
                *12*[[w, x, y, z, v]],
                *13*[[x, v, w, y, z]],
                *14*[[y, w, v, z, x]],
                *15*[[z, v, x, w, y]],
                ]
    assert irv(election, tiebreaker) == w

    # Two-round example from
    # https://en.wikipedia.org/wiki/Instant-runoff_voting#Five_voters,_three_candidates
    Bob, Bill, Sue = 0, 1, 2
    election = np.array([[Bob, Bill, Sue],  # a
                         [Sue, Bob, Bill],  # b
                         [Bill, Sue, Bob],  # c
                         [Bob, Bill, Sue],  # d
                         [Sue, Bob, Bill],  # e
                         ])
    assert irv(election, tiebreaker) == Sue

    # Two-round example from
    # https://en.wikipedia.org/wiki/Condorcet_method#Comparison_with_instant_runoff_and_first-past-the-post_(plurality)
    A, B, C = 0, 1, 2
    election = [*499*[[A, B, C]],
                *  3*[[B, C, A]],
                *498*[[C, B, A]],
                ]
    assert irv(election, tiebreaker) == C  # "IRV elects C"

    # Two-round example from
    # http://pi.math.cornell.edu/~ismythe/Lec_04_web.pdf#page=16
    election = [[A, C, B],
                [A, C, B],
                [B, C, A],
                [B, C, A],
                [C, A, B],
                ]
    assert irv(election, tiebreaker) == A  # "A wins under IRV"

    # Examples from http://pi.math.cornell.edu/~ismythe/Lec_05_web.pdf#page=19
    # Two-round
    election = [*6*[[A, B, C]],
                *5*[[C, A, B]],
                *4*[[B, C, A]],
                *2*[[B, A, C]],
                ]
    assert irv(election, tiebreaker) == A  # A wins IRV

    # Two-round
    election = [*6*[[A, B, C]],
                *5*[[C, A, B]],
                *4*[[B, C, A]],
                *2*[[A, B, C]],
                ]
    assert irv(election, tiebreaker) == C  # C wins IRV

    # Four-round example from
    # https://medium.com/@t2ee6ydscv/how-ranked-choice-voting-elects-extremists-fa101b7ffb8e
    r, b, g, o, y = 0, 1, 2, 3, 4
    election = [*31*[[r, b, g, o, y]],
                * 5*[[b, r, g, o, y]],
                * 8*[[b, g, r, o, y]],
                * 1*[[b, g, o, r, y]],
                * 6*[[g, b, o, r, y]],
                * 1*[[g, b, o, y, r]],
                * 6*[[g, o, b, y, r]],
                * 2*[[o, g, b, y, r]],
                * 5*[[o, g, y, b, r]],
                * 7*[[o, y, g, b, r]],
                *28*[[y, o, g, b, r]],
                ]
    assert irv(election) == r


@pytest.mark.parametrize("tiebreaker", [None, 'random', 'order'])
def test_wikipedia_examples(tiebreaker):
    """
    Real elections from the Examples section of the Wikipedia IRV article.

    https://en.wikipedia.org/wiki/Instant-runoff_voting#Examples
    """
    assert irv(_irish_1990_election(), tiebreaker) == 0  # Mary Robinson

    assert irv(_prahran_2014_election(), tiebreaker) == 0  # Sam Hibbins

    assert irv(_burlington_2009_election(), tiebreaker) == 0  # Bob Kiss


def test_wikipedia_examples_elimination_order():
    """
    Elimination sequence matches published IRV counts for Wikipedia examples.
    """
    irish = irv_rounds(_irish_1990_election(), 'order', record_rounds=True)
    assert [r['loser'] for r in irish['rounds']] == [2]  # Austin Currie
    assert irish['winner'] == 0  # Mary Robinson

    prahran = irv_rounds(_prahran_2014_election(), 'order', record_rounds=True)
    prahran_losers = [r['loser'] for r in prahran['rounds']]
    assert prahran_losers[0] == 7  # Menadue first
    assert prahran_losers[-1] == 2  # Pharaoh last before the final two
    assert set(prahran_losers) == {7, 4, 1, 3, 6, 2}
    assert prahran['winner'] == 0  # Sam Hibbins

    burlington = irv_rounds(
        _burlington_2009_election(), 'order', record_rounds=True)
    assert [r['loser'] for r in burlington['rounds']] == [
        4, 5, 3, 1,  # Simpson, Write-in, Smith, Montroll
    ]
    assert burlington['winner'] == 0  # Bob Kiss


def test_irv_rounds_matches_irv():
    election = np.array([[2, 0, 1],
                         [0, 1, 2],
                         [2, 0, 1],
                         [1, 2, 0],
                         [2, 1, 0],
                         [2, 0, 1],
                         [1, 0, 2],
                         [2, 0, 1],
                         [2, 1, 0],
                         [0, 2, 1]])
    assert irv(election, 'order') == irv_rounds(election, 'order')['winner']
    traced = irv_rounds(election, 'order', record_rounds=True)
    assert traced['final_ballots'].shape == (len(election),)
    assert len(traced['final_tallies']) == 3


def test_irv_rounds_min_remaining():
    A, B, C = 0, 1, 2
    election = [[A, C, B],
                [A, C, B],
                [B, C, A],
                [B, C, A],
                [C, A, B]]
    result = irv_rounds(election, 'order', min_remaining=2, record_rounds=True)
    assert sorted(np.flatnonzero(~result['eliminated_mask'])) == [0, 1]
    assert len(result['rounds']) == 1
    assert result['rounds'][0]['loser'] == 2


def test_eliminate_no_votes():
    # First, 0 is eliminated for getting no votes.
    # Then 1 and 2 are tied.
    election = [[1, 0, 2],
                [2, 0, 1]]

    # With no tiebreaker, None is returned because of tie between 1 and 2.
    assert irv(election) is None

    # With order tiebreaker, 2 is eliminated, because lower IDs preferred.
    # Then 1 wins unanimously.
    # If 0 had not been eliminated, 0 would win the tie between 0 and 1.
    assert irv(election, 'order') == 1

    # With random tiebreaker, 0 should never win.
    assert collect_random_results(irv, election) == {1, 2}


def complete_ranked_ballots(min_cands=3, max_cands=256, min_voters=1,
                            max_voters=1000):
    n_cands = integers(min_value=min_cands, max_value=max_cands)
    return n_cands.flatmap(lambda n: lists(permutations(range(n)),
                                           min_size=min_voters,
                                           max_size=max_voters))


@pytest.mark.parametrize("tiebreaker", ['random', 'order'])
@given(election=complete_ranked_ballots(min_cands=1, max_cands=25,
                                        min_voters=1, max_voters=100))
def test_legit_winner(election, tiebreaker):
    n_cands = np.shape(election)[1]
    assert irv(election, tiebreaker) in range(n_cands)


@given(election=complete_ranked_ballots(min_cands=1, max_cands=25,
                                        min_voters=1, max_voters=100))
def test_legit_winner_none(election):
    n_cands = np.shape(election)[1]
    assert irv(election) in {None} | set(range(n_cands))


if __name__ == "__main__":
    # Run unit tests, in separate process to avoid warnings about cached
    # modules, printing output line by line in realtime
    from subprocess import PIPE, Popen
    with Popen(['pytest',
                '--tb=short',  # shorter traceback format
                '--hypothesis-show-statistics',
                str(__file__)], stdout=PIPE, bufsize=1,
               universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')
