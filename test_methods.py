import pytest
from black import black
from borda import borda
from fptp import fptp
from runoff import runoff
from irv import irv
from coombs import coombs
from approval import approval
from utility_winner import utility_winner


@pytest.mark.parametrize("method", [black, borda, fptp, runoff, irv, coombs,
                                    approval, utility_winner])
def test_invalid_tiebreaker(method):
    with pytest.raises(ValueError):
        election = [[0, 1],
                    [1, 0]]
        method(election, tiebreaker='dictator')


@pytest.mark.parametrize("method", [black, borda, fptp, runoff, irv, coombs])
def test_ranked_method_degenerate_case(method):
    election = [[0]]
    assert method(election) == 0
    assert method(election, 'random') == 0
    assert method(election, 'order') == 0

    election = [[0], [0], [0]]
    assert method(election) == 0
    assert method(election, 'random') == 0
    assert method(election, 'order') == 0


@pytest.mark.parametrize("method", [black, borda, fptp, runoff, irv, coombs])
def test_ranked_method_unanimity(method):
    election = [[3, 0, 1, 2], [3, 0, 2, 1], [3, 2, 1, 0]]
    assert method(election) == 3
    assert method(election, 'random') == 3
    assert method(election, 'order') == 3


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
