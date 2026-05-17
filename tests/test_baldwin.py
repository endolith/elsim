import random

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import integers, lists, permutations

from elsim.methods import baldwin, baldwin_rounds


def collect_random_results(method, election):
    """
    Run multiple elections with tiebreaker='random' and collect the set of all
    winners.
    """
    random.seed(47)  # Deterministic test
    winners = set()
    for trial in range(10):
        winner = method(election, tiebreaker='random')
        assert isinstance(winner, int)
        winners.add(winner)
    return winners


@pytest.mark.parametrize("tiebreaker", [None, 'random', 'order'])
def test_strict_majority(tiebreaker):
    A, B, C = 0, 1, 2
    election = [[A, B, C],
                [B, C, A],
                [A, C, B],
                ]
    assert baldwin(election, tiebreaker) == A

    election = [*3*[[A, B, C]],
                *7*[[B, C, A]],
                *2*[[C, B, A]],
                ]
    assert baldwin(election, tiebreaker) == B


def test_different_from_irv():
    # Classic 5-voter example where Baldwin (like Coombs) elects C but IRV elects A.
    # B has the lowest Borda score and is eliminated first; then C wins with majority.
    # Under IRV, C is eliminated first (fewest first-choice votes) and A wins.
    A, B, C = 0, 1, 2
    election = [[A, C, B],
                [A, C, B],
                [B, C, A],
                [B, C, A],
                [C, A, B],
                ]
    assert baldwin(election) == C

    from elsim.methods import irv
    assert irv(election) == A


@pytest.mark.parametrize("tiebreaker", [None, 'random', 'order'])
def test_examples(tiebreaker):
    # Standard Tennessee example
    # https://en.wikipedia.org/wiki/Template:Tenn_voting_example
    Memphis, Nashville, Chattanooga, Knoxville = 0, 1, 2, 3
    election = [*42*[[Memphis, Nashville, Chattanooga, Knoxville]],
                *26*[[Nashville, Chattanooga, Knoxville, Memphis]],
                *15*[[Chattanooga, Knoxville, Nashville, Memphis]],
                *17*[[Knoxville, Chattanooga, Nashville, Memphis]],
                ]
    assert baldwin(election, tiebreaker) == Nashville


def test_no_tiebreak_tied_losers():
    A, B, C = 0, 1, 2
    # Perfect cycle — every candidate has the same Borda score
    election = [[A, B, C],
                [B, C, A],
                [C, A, B]]
    assert baldwin(election) is None


def test_baldwin_rounds_matches_baldwin():
    A, B, C = 0, 1, 2
    election = [[A, C, B],
                [A, C, B],
                [B, C, A],
                [B, C, A],
                [C, A, B]]
    assert baldwin(election, 'order') == baldwin_rounds(election, 'order')['winner']
    traced = baldwin_rounds(election, 'order', record_rounds=True)
    assert traced['final_ballots'].shape == (len(election),)
    assert len(traced['final_tallies']) == 3
    assert len(traced['rounds']) == 1
    assert traced['rounds'][0]['loser'] == B
    # Borda snapshots present when record_rounds=True
    assert 'borda_before' in traced['rounds'][0]
    assert 'promoted_per_voter' in traced['rounds'][0]


def test_baldwin_rounds_min_remaining():
    # 65 voters, 5 candidates; no candidate reaches majority in either of the
    # first two rounds, so min_remaining=3 stops after exactly 2 eliminations.
    # Round 1: z (lowest Borda) eliminated; round 2: y (next lowest) eliminated.
    v, w, x, y, z = 0, 1, 2, 3, 4
    election = [*11*[[v, w, x, y, z]],
                *12*[[w, x, y, z, v]],
                *13*[[x, v, w, y, z]],
                *14*[[y, w, v, z, x]],
                *15*[[z, v, x, w, y]],
                ]
    result = baldwin_rounds(election, 'order', min_remaining=3, record_rounds=True)
    assert np.sum(~result['eliminated_mask']) == 3
    assert len(result['rounds']) == 2
    assert result['rounds'][0]['loser'] == z
    assert result['rounds'][1]['loser'] == y


def complete_ranked_ballots(min_cands=3, max_cands=256, min_voters=1,
                            max_voters=1000):
    n_cands = integers(min_value=min_cands, max_value=max_cands)
    return n_cands.flatmap(lambda n: lists(permutations(range(n)),
                                           min_size=min_voters,
                                           max_size=max_voters))


@pytest.mark.parametrize("tiebreaker", ['random', 'order'])
@given(election=complete_ranked_ballots(min_cands=1, max_cands=25,
                                        min_voters=1, max_voters=100))
def test_legit_winner_order(election, tiebreaker):
    n_cands = np.shape(election)[1]
    assert baldwin(election, tiebreaker) in range(n_cands)


@given(election=complete_ranked_ballots(min_cands=1, max_cands=25,
                                        min_voters=1, max_voters=100))
def test_legit_winner_none(election):
    n_cands = np.shape(election)[1]
    assert baldwin(election) in {None} | set(range(n_cands))


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
