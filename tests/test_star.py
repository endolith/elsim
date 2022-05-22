import random
import numpy as np
import pytest
from hypothesis import given
from elsim.methods import star
import test_score
score_ballots = test_score.score_ballots


def collect_random_results(method, election):
    """
    Run multiple elections with tiebreaker='random' and collect the set of all
    winners.
    """
    random.seed(2014)  # Deterministic test
    winners = set()
    for trial in range(10):
        winner = method(election, tiebreaker='random')
        assert isinstance(winner, int)
        winners.add(winner)
    return winners


@pytest.mark.parametrize("method", [star])
@pytest.mark.parametrize("tiebreaker", [None, 'random', 'order'])
def test_basic(tiebreaker, method):
    # Standard Tennessee example
    # https://en.wikipedia.org/wiki/STAR_voting#Example
    # https://electowiki.org/wiki/STAR_voting#Example
    #                 Memphis Nashville Chattanooga Knoxville
    election = [*42*[[5,      2,        1,          0]],
                *26*[[0,      5,        2,          1]],
                *15*[[0,      3,        5,          3]],
                *17*[[0,      2,        4,          5]],
                ]

    assert method(election, tiebreaker) == 1  # Nashville

    # Every example from https://rangevoting.org/StarVoting.html
    # (https://web.archive.org/web/20201105181703/ in case of changes)
    # '51% vote "A5, B0, C4," while 49% vote "A0, B5, C4."'
    #                 A  B  C
    election = [*51*[[5, 0, 4]],
                *49*[[0, 5, 4]],
                ]

    assert method(election, tiebreaker) == 0  # "A wins that runoff"

    # '51% vote "A5, B0, C4, D4" while 49% vote "A0, B5, C4, D3."'
    #                 A  B  C  D
    election = [*51*[[5, 0, 4, 4]],
                *49*[[0, 5, 4, 3]],
                ]

    assert method(election, tiebreaker) == 2  # "whereupon C wins"

    # District-partitioning paradox
    #                A  B  C  D  E
    election = [*5*[[3, 3, 4, 0, 5]],
                *3*[[5, 5, 3, 0, 0]],
                *1*[[5, 3, 3, 0, 0]],
                ]

    assert method(election, tiebreaker) == 0  # "which A wins"

    #                A  B  C  D  E
    election = [*5*[[3, 0, 4, 3, 5]],
                *3*[[5, 0, 3, 5, 0]],
                *1*[[5, 0, 3, 3, 0]],
                ]

    assert method(election, tiebreaker) == 0  # "Also A wins the East"

    #                A  B  C  D  E
    election = [*5*[[3, 3, 4, 0, 5]],
                *3*[[5, 5, 3, 0, 0]],
                *1*[[5, 3, 3, 0, 0]],
                *5*[[3, 0, 4, 3, 5]],
                *3*[[5, 0, 3, 5, 0]],
                *1*[[5, 0, 3, 3, 0]],
                ]

    assert method(election, tiebreaker) == 2  # "C wins the countrywide"

    # "No-show" paradox:
    #                A  B  C  D  E
    election = [*4*[[3, 2, 4, 5, 0]],
                *3*[[3, 5, 0, 0, 4]],
                *2*[[3, 1, 4, 0, 5]],
                ]

    assert method(election, tiebreaker) == 0  # "A wins"

    # "one new voter comes"
    #                A  B  C  D  E
    election = [*4*[[3, 2, 4, 5, 0]],
                *3*[[3, 5, 0, 0, 4]],
                *2*[[3, 1, 4, 0, 5]],
                *1*[[5, 0, 2, 0, 0]],
                ]

    assert method(election, tiebreaker) == 2  # "C wins"

    # "loser B was a convicted multiple murderer"
    #                A  C  D  E
    election = [*4*[[3, 4, 5, 0]],
                *3*[[3, 0, 0, 4]],
                *2*[[3, 4, 0, 5]],
                ]

    assert method(election, tiebreaker) == 1  # "C wins"

    # "mucho goofiness"
    #                 A  B  C
    election = [* 9*[[5, 1, 0]],
                *12*[[0, 5, 1]],
                * 8*[[1, 0, 5]],
                ]

    assert method(election, tiebreaker) == 0  # "A wins"

    # "LESS IS MORE"
    #                 A  B  C
    # election = [* 9*[[5, 1, 0]],
    #             *10*[[0, 5, 1]],
    #             * 2*[[3, 0, 5]],
    #             * 8*[[1, 0, 5]],
    #             ]

    # assert method(election, tiebreaker) == 1  # "That makes B win with STAR"

    # Not correct.  Tallies are [59, 59, 60].  A and B are tied for 2nd.
    # A was preferred on more ballots than B, so finalists are A and C.
    # B cannot win.

    # "NO, MORE IS LESS"
    #                A  B  C
    election = [*9*[[5, 1, 0]],
                *7*[[0, 5, 1]],
                *2*[[5, 1, 0]],
                *3*[[5, 0, 0]],
                *8*[[1, 0, 5]],
                ]

    assert method(election, tiebreaker) == 2  # "causes C … to win"

    # "If we delete 5-10 BCA voters"
    for x in range(5, 11):
        v1 = 12 - x
        #                 A  B  C
        election = [* 9*[[5, 1, 0]],
                    *v1*[[0, 5, 1]],
                    * 8*[[1, 0, 5]],
                    ]

        assert method(election, tiebreaker) == 2  # "C wins"

    # "If 1-12 more CAB voters came"
    for x in range(1, 13):
        v2 = 8 + x
        #                    A  B  C
        election = [* 9*[[5, 1, 0]],
                    *12*[[0, 5, 1]],
                    *v2*[[1, 0, 5]],
                    ]

        assert method(election, tiebreaker) == 1  # "B win"

    # "1-12 BCA voters could have"
    for x in range(1, 13):
        v1 = 12 - x
        v2 = x
        #                     A  B  C
        election = [* 9*[[5, 1, 0]],
                    *v1*[[0, 5, 1]],
                    *v2*[[0, 1, 5]],
                    * 8*[[1, 0, 5]],
                    ]

        assert method(election, tiebreaker) != 0  # prevented … A from winning"

    # "just erase B from all ballots"
    #                 A  C
    election = [* 9*[[5, 0]],
                *12*[[0, 1]],
                * 8*[[1, 5]],
                ]

    assert method(election, tiebreaker) == 1  # "now C wins"

    # "REVERSE all ballots"
    #                 A  B  C
    # election = [* 9*[[5, 1, 0]],
    #             *12*[[0, 5, 1]],
    #             * 8*[[1, 0, 5]],
    #             ]

    # # "i.e. score X becomes 5-X."
    # election = 5 - np.asarray(election)

    # assert method(election, tiebreaker) == 2  # "now C wins"

    # Not correct.  Tallies are [92, 76, 93], A and C are finalists.
    # A is preferred over C by 20 voters, C is preferred over A by 9.  A wins.

    # "PARTITION INTO DISTRICTS"
    # "West"
    #                A  B  C
    election = [*3*[[5, 1, 0]],
                *4*[[0, 5, 1]],
                *4*[[1, 0, 5]],
                ]

    assert method(election, tiebreaker) == 1  # "B is the STAR winner"

    # "North"
    #                A  B  C
    election = [*2*[[5, 1, 0]],
                *3*[[0, 5, 1]],  # "B1" typo in original
                *4*[[1, 0, 5]],
                ]

    assert method(election, tiebreaker) == 1  # "B is the STAR winner"

    # "East"
    #                A  B  C
    election = [*4*[[5, 1, 0]],
                *5*[[0, 5, 1]],
                ]

    assert method(election, tiebreaker) == 1  # "B is the STAR winner"

    # "whole country"
    #                A  B  C
    election = [*3*[[5, 1, 0]],
                *4*[[0, 5, 1]],
                *4*[[1, 0, 5]],
                *2*[[5, 1, 0]],
                *3*[[0, 5, 1]],  # "B1" typo in original
                *4*[[1, 0, 5]],
                *4*[[5, 1, 0]],
                *5*[[0, 5, 1]],
                ]

    assert method(election, tiebreaker) == 0  # "A wins"

    # "thwarted beats-all winner"
    #                 A  B  C
    election = [*13*[[5, 0, 1]],
                *12*[[2, 0, 5]],
                *12*[[0, 5, 1]],
                * 6*[[1, 5, 0]],
                ]

    assert method(election, tiebreaker) == 0  # "A wins"

    # "3-candidate Nanson-Baldwin or STAR election"
    #                 A  B  C
    election = [*23*[[0, 2, 1]],
                *16*[[2, 0, 1]],
                *11*[[1, 0, 2]],
                *10*[[2, 1, 0]],
                ]

    assert method(election, tiebreaker) == 2  # "C wins"

    # "suppose we change the 16 ACB ballots in the pink row to CAB"
    #                 A  B  C
    election = [*23*[[0, 2, 1]],
                *16*[[1, 0, 2]],
                *11*[[1, 0, 2]],
                *10*[[2, 1, 0]],
                ]

    assert method(election, tiebreaker) == 1  # "B wins"

    # "17-voter 3-candidate IRV or STAR election"
    #                A  B  C
    election = [*5*[[0, 1, 5]],
                *4*[[5, 0, 1]],
                *8*[[1, 5, 0]],
                ]

    assert method(election, tiebreaker) == 2  # "C wins"

    # "suppose two of the BAC voters"
    #                A  B  C
    election = [*5*[[0, 1, 5]],
                *4*[[5, 0, 1]],
                *6*[[1, 5, 0]],
                *2*[[5, 1, 0]],
                ]

    assert method(election, tiebreaker) == 1  # "B wins"

    # Example from https://www.youtube.com/watch?v=3-mOeUXAkV0
    #            Abby Ben Carmen DeAndre
    election = [[3,   4,  0,     5],
                [1,   0,  5,     1],
                [0,   2,  5,     5],
                [5,   0,  2,     3],
                [1,   0,  4,     5],
                ]

    assert method(election, tiebreaker) == 3  # "DeAndre wins"


@pytest.mark.parametrize("method", [star])
def test_ties(method):
    # Two-way tie between candidates 1 and 2
    election = np.array([[0, 3, 2],
                         [3, 0, 1],
                         [0, 1, 3],
                         [0, 2, 0],
                         [0, 3, 3],
                         ])
    # No tiebreaker:
    assert method(election, tiebreaker=None) is None

    # Mode 'order' should always prefer lowest candidate ID
    assert method(election, tiebreaker='order') == 1

    # Mode 'random' should choose all tied candidates at random
    assert collect_random_results(method, election) == {1, 2}

    # Three-way tie between 0, 1, and 2
    election = np.array([[0, 1, 1],
                         [0, 1, 2],
                         [0, 2, 1],
                         [2, 4, 0],
                         [4, 1, 0],
                         [1, 3, 0],
                         [1, 0, 3],
                         [3, 0, 1],
                         [1, 0, 4],
                         ])

    # No tiebreaker:
    assert method(election, tiebreaker=None) is None

    # TODO: Add every type of tie at every stage.

#     # Mode 'order' should always prefer lowest candidate ID
#     assert method(election, tiebreaker='order') == 0

#     # Mode 'random' should choose all tied candidates at random
#     assert collect_random_results(method, election) == {0, 1, 2}


@pytest.mark.parametrize("method", [star])
@pytest.mark.parametrize("tiebreaker", ['random', 'order'])
@given(election=score_ballots(min_cands=1, max_cands=25,
                              min_voters=1, max_voters=100))
def test_legit_winner(election, tiebreaker, method):
    n_cands = election.shape[1]
    winner = method(election, tiebreaker)
    assert isinstance(winner, int)
    assert winner in range(n_cands)


@pytest.mark.parametrize("method", [star])
@given(election=score_ballots(min_cands=1, max_cands=25,
                              min_voters=1, max_voters=100))
def test_legit_winner_none(election, method):
    n_cands = election.shape[1]
    winner = method(election)
    assert isinstance(winner, (int, type(None)))
    assert winner in set(range(n_cands)) | {None}


@pytest.mark.parametrize("method", [star])
def test_invalid(method):
    with pytest.raises(ValueError):
        method([[-2, -2, -2],
                [0, +1, -1]])


if __name__ == "__main__":
    # Run unit tests, in separate process to avoid warnings about cached
    # modules, printing output line by line in realtime
    from subprocess import Popen, PIPE
    with Popen(['pytest',
                '--tb=short',  # shorter traceback format
                '--hypothesis-show-statistics',
                str(__file__)], stdout=PIPE, bufsize=1,
               universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')
