"""
Implements various election methods.

These take collections of ballots (elections) as inputs and return the winner
according to the rules of that method.
"""
from .approval import approval, combined_approval
from .black import black
from .borda import borda
from .condorcet import (condorcet, condorcet_from_matrix,
                        ranked_election_to_matrix)
from .coombs import coombs
from .fptp import fptp, sntv
from .irv import irv
from .runoff import runoff
from .score import score
from .star import matrix_from_scores, star
from .three_two_one import three_two_one
from .utility_winner import utility_winner
from .partisan_primaries import (
    closed_partisan_primary_runoff,
    nominee_restricted_plurality,
    open_partisan_primary,
    pairwise_majority_from_rankings,
    top_two_runoff_reduced_turnout,
)
