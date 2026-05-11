"""
Implements various election methods.

These take collections of ballots (elections) as inputs and return the winner
according to the rules of that method.
"""
from elsim.methods.approval import approval, combined_approval
from elsim.methods.black import black
from elsim.methods.borda import borda
from elsim.methods.condorcet import (
    condorcet,
    condorcet_from_matrix,
    ranked_election_to_matrix,
)
from elsim.methods.coombs import coombs
from elsim.methods.fptp import fptp, sntv
from elsim.methods.irv import irv
from elsim.methods.runoff import runoff
from elsim.methods.score import score
from elsim.methods.star import matrix_from_scores, star
from elsim.methods.utility_winner import utility_winner

__all__ = [
    'approval',
    'black',
    'borda',
    'combined_approval',
    'condorcet',
    'condorcet_from_matrix',
    'coombs',
    'fptp',
    'irv',
    'matrix_from_scores',
    'ranked_election_to_matrix',
    'runoff',
    'sntv',
    'score',
    'star',
    'utility_winner',
]
