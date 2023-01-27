"""
Use Hypothesis to find simple elections that violate Condorcet compliance.

This depends on Hypothesis' "shrinking" algorithm, which is not guaranteed to
find the absolute simplest case (or any at all), but typically works well.
https://hypothesis.readthedocs.io/en/latest/data.html#shrinking

Typical result:

    Falsifying example: test_condorcet_compliance(
        election=[[0, 1, 2], [1, 0, 2], [1, 0, 2], [2, 0, 1], [2, 0, 1]],
    )

IRV:
    Candidate 0 is eliminated first.
    That voter's ballot is transferred to Candidate 1.
    Candidate 1 now has 3 first-choice votes out of 5 and is the IRV winner.

Condorcet:
    Candidate 0 is preferred over Candidate 1 by 3 of 5 voters.
    Candidate 0 is preferred over Candidate 2 by 3 of 5 voters.
    Candidate 0 is therefore the Condorcet winner.
"""
from hypothesis import given, assume, settings
from hypothesis.strategies import lists, permutations, integers
from elsim.methods import condorcet, irv


def complete_ranked_ballots(min_cands=3, max_cands=25, min_voters=1,
                            max_voters=100):
    n_cands = integers(min_value=min_cands, max_value=max_cands)
    return n_cands.flatmap(lambda n: lists(permutations(range(n)),
                                           min_size=min_voters,
                                           max_size=max_voters))


@settings(max_examples=5000, database=None)
@given(election=complete_ranked_ballots(min_cands=1, max_cands=3,
                                        min_voters=1, max_voters=100))
def test_condorcet_compliance(election):
    """
    Find simplest election in which IRV does not choose the Condorcet winner.
    """
    assume(condorcet(election) is not None)  # Not a cycle
    assume(irv(election) is not None)  # Not a tie
    assert irv(election) == condorcet(election)


try:
    test_condorcet_compliance()
except AssertionError:
    # Just print the Falsifying example, don't raise anything.
    pass
