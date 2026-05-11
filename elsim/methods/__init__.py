"""
Implements various election methods.

These take collections of ballots (elections) as inputs and return the winner
according to the rules of that method.
"""
from elsim.methods.approval import approval, combined_approval
from elsim.methods.black import black
from elsim.methods.borda import borda
from elsim.methods.condorcet import (condorcet, condorcet_from_matrix,
                                     ranked_election_to_matrix)
from elsim.methods.coombs import coombs
from elsim.methods.fptp import fptp, sntv
from elsim.methods.irv import irv
from elsim.methods.partisan_primaries import (
    closed_partisan_primary_runoff,
    nominee_restricted_plurality,
    open_partisan_primary,
    pairwise_majority_from_rankings,
    top_two_runoff_reduced_turnout,
)
from elsim.methods.runoff import runoff
from elsim.methods.score import score
from elsim.methods.star import matrix_from_scores, star
from elsim.methods.three_two_one import three_two_one
from elsim.methods.utility_winner import utility_winner

__all__ = [
    'approval',
    'black',
    'borda',
    'closed_partisan_primary_runoff',
    'combined_approval',
    'condorcet',
    'condorcet_from_matrix',
    'coombs',
    'fptp',
    'irv',
    'matrix_from_scores',
    'nominee_restricted_plurality',
    'open_partisan_primary',
    'pairwise_majority_from_rankings',
    'ranked_election_to_matrix',
    'runoff',
    'sntv',
    'score',
    'star',
    'three_two_one',
    'top_two_runoff_reduced_turnout',
    'utility_winner',
]
