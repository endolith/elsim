import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import integers, lists, permutations
from black import black


@pytest.mark.parametrize("tiebreaker", [None, 'random', 'order'])
def test_condorcet_winner(tiebreaker):
    # Standard Tennessee example
    # https://en.wikipedia.org/wiki/Template:Tenn_voting_example
    Memphis, Nashville, Chattanooga, Knoxville = 0, 1, 2, 3
    election = [*42*[[Memphis, Nashville, Chattanooga, Knoxville]],
                *26*[[Nashville, Chattanooga, Knoxville, Memphis]],
                *15*[[Chattanooga, Knoxville, Nashville, Memphis]],
                *17*[[Knoxville, Chattanooga, Nashville, Memphis]],
                ]

    assert black(election, tiebreaker) == Nashville

    # Example from Ques 9
    # http://www.yorku.ca/bucovets/4380/exercises/exercises_1_a.pdf
    v, w, x, y, z = 0, 1, 2, 3, 4
    election = [*11*[[v, w, x, y, z]],
                *12*[[w, x, y, z, v]],
                *13*[[x, v, w, y, z]],
                *14*[[y, w, v, z, x]],
                *15*[[z, v, x, w, y]],
                ]

    assert black(election, tiebreaker) == v

    # Example from
    # https://en.wikipedia.org/wiki/Condorcet_method#Pairwise_counting_and_matrices
    election = np.array([[1, 2, 0, 3],
                         [3, 0, 2, 1],
                         [0, 2, 1, 3],
                         ])
    assert black(election, tiebreaker) == 0

    # Example from https://electowiki.org/wiki/Condorcet_Criterion
    election = np.concatenate((np.tile([0, 1, 2], (499, 1)),
                               np.tile([2, 1, 0], (498, 1)),
                               np.tile([1, 2, 0], (3, 1)),
                               ))
    assert black(election, tiebreaker) == 1

    # Example from
    # https://www3.nd.edu/~apilking/Math10170/Information/Lectures/Lecture_3.Head%20To%20Head%20Comparisons.pdf
    Colley, Henry, Taylor = 0, 1, 2
    election = np.array([[Colley,  Henry, Taylor],
                         [ Henry, Colley, Taylor],
                         [ Henry, Colley, Taylor],
                         [Taylor, Colley,  Henry],
                         [Taylor,  Henry, Colley],
                         ])
    assert black(election, tiebreaker) == 1

    # Example from https://www.whydomath.org/node/voting/impossible.html
    election = np.array([[0, 2, 3, 1],
                         [1, 2, 3, 0],
                         [3, 0, 2, 1],
                         [0, 1, 3, 2],
                         [3, 0, 2, 1],
                         ])
    assert black(election, tiebreaker) == 3

    # Table 3.1 from Mackie - Democracy Defended
    # (Borda and Condorcet results differ)
    A, B, C, D, E = 0, 1, 2, 3, 4
    election = [*4*[[A, E, D, C, B]],
                *3*[[B, C, E, D, A]],
                *2*[[C, D, E, B, A]],
                ]

    assert black(election, tiebreaker) == C  # "and C is the Condorcet winner"

    # Table 3 from On_the_Relevance_of_Theoretical_Results_to_Voting_.pdf
    election = [[D, E, A, B, C],
                [E, A, C, B, D],
                [C, D, E, A, B],
                [D, E, B, C, A],
                [E, B, A, D, C],
                ]

    assert black(election, tiebreaker) == D

    # Example from
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
    assert black(election) == g


@pytest.mark.parametrize("tiebreaker", [None, 'random', 'order'])
def test_condorcet_cycle(tiebreaker):
    # Examples from https://rangevoting.org/CondBurial.html
    A, B, C = 0, 1, 2
    election = [*46*[[A, B, C]],
                *44*[[B, C, A]],
                * 5*[[C, A, B]],
                * 5*[[C, B, A]],
                ]
    assert black(election, tiebreaker) == B  # "Black awards the victory to B"

    election = [*49*[[C, B, A]],
                *31*[[A, C, B]],
                *17*[[B, A, C]],
                * 3*[[A, B, C]],
                ]
    assert black(election, tiebreaker) == C


def complete_ranked_ballots(min_cands=3, max_cands=25, min_voters=1,
                            max_voters=100):
    n_cands = integers(min_value=min_cands, max_value=max_cands)
    return n_cands.flatmap(lambda n: lists(permutations(range(n)),
                                           min_size=min_voters,
                                           max_size=max_voters))


@pytest.mark.parametrize("tiebreaker", ['random', 'order'])
@given(election=complete_ranked_ballots(min_cands=1, max_cands=25,
                                        min_voters=1, max_voters=100))
def test_legit_winner(election, tiebreaker):
    election = np.asarray(election)
    n_cands = election.shape[1]
    winner = black(election, tiebreaker)
    assert isinstance(winner, int)
    assert winner in range(n_cands)


@given(election=complete_ranked_ballots(min_cands=1, max_cands=25,
                                        min_voters=1, max_voters=100))
def test_legit_winner_none(election):
    election = np.asarray(election)
    n_cands = election.shape[1]
    winner = black(election)
    assert isinstance(winner, (int, type(None)))
    assert winner in set(range(n_cands)) | {None}


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
