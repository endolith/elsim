import pytest
from black import black
from borda import borda
from fptp import fptp
from runoff import runoff
from irv import irv
from condorcet import condorcet


@pytest.mark.parametrize("method", [black, borda, fptp, runoff, irv])
def test_invalid_tiebreaker(method):
    with pytest.raises(ValueError):
        election = [[0, 1],
                    [1, 0]]
        method(election, tiebreaker='dictator')


@pytest.mark.parametrize("method", [black, borda, fptp, runoff, irv])
def test_degenerate_case(method):
    election = [[0]]
    assert method(election) == 0
    assert method(election, 'random') == 0
    assert method(election, 'order') == 0

    election = [[0], [0], [0]]
    assert method(election) == 0
    assert method(election, 'random') == 0
    assert method(election, 'order') == 0


# No tiebreaker parameter
def test_degenerate_condorcet_case():
    election = [[0]]
    assert condorcet(election) == 0

    election = [[0], [0], [0]]
    assert condorcet(election) == 0


@pytest.mark.parametrize("method", [black, borda, fptp, runoff, irv])
def test_unanimity(method):
    election = [[3, 0, 1, 2], [3, 0, 2, 1], [3, 2, 1, 0]]
    assert method(election) == 3
    assert method(election, 'random') == 3
    assert method(election, 'order') == 3


# No tiebreaker parameter
def test_unanimity_condorcet():
    election = [[3, 0, 1, 2], [3, 0, 2, 1], [3, 2, 1, 0]]
    assert condorcet(election) == 3


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
