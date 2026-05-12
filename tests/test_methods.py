import pytest

from elsim.methods import (approval, approval_runoff, black, borda,
                           combined_approval, coombs, fptp, irv,
                           irv_primary_top_n_runoff, runoff, score,
                           top_n_condorcet, top_n_irv, top_n_runoff,
                           utility_winner)

_blanket_ranked = [
    lambda e, tb=None: top_n_irv(e, 4, tb),
    lambda e, tb=None: top_n_irv(e, 5, tb),
    lambda e, tb=None: top_n_runoff(e, 4, tb),
    lambda e, tb=None: top_n_runoff(e, 5, tb),
    lambda e, tb=None: irv_primary_top_n_runoff(e, 4, tb),
    lambda e, tb=None: top_n_irv(e, 3, tb),
    lambda e, tb=None: top_n_runoff(e, 3, tb),
    lambda e, tb=None: top_n_condorcet(e, 3, tb),
    lambda e, tb=None: top_n_condorcet(e, 4, tb),
]


@pytest.mark.parametrize("method", [black, borda, fptp, runoff, irv, coombs,
                                    approval, combined_approval,
                                    utility_winner, score])
def test_invalid_tiebreaker(method):
    with pytest.raises(ValueError):
        election = [[0, 1],
                    [1, 0]]
        method(election, tiebreaker='duel')


def test_invalid_tiebreaker_approval_runoff():
    with pytest.raises(ValueError):
        approval_runoff([[1, 0]], [[0, 1]], tiebreaker='duel')


@pytest.mark.parametrize("method", [black, borda, fptp, runoff, irv, coombs,
                                    *_blanket_ranked])
def test_ranked_method_degenerate_case(method):
    election = [[0]]
    assert method(election) == 0
    assert method(election, 'random') == 0
    assert method(election, 'order') == 0

    election = [[0], [0], [0]]
    assert method(election) == 0
    assert method(election, 'random') == 0
    assert method(election, 'order') == 0


@pytest.mark.parametrize("method", [black, borda, fptp, runoff, irv, coombs,
                                    *_blanket_ranked])
def test_ranked_method_unanimity(method):
    election = [[3, 0, 1, 2], [3, 0, 2, 1], [3, 2, 1, 0]]
    assert method(election) == 3
    assert method(election, 'random') == 3
    assert method(election, 'order') == 3


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
