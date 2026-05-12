"""
Implements various election methods.

These take collections of ballots (elections) as inputs and return the winner
according to the rules of that method.
"""
from .approval import approval, combined_approval
from .black import black
from .blanket_primary import (approval_runoff, irv_primary_top_n_runoff,
                             top_n_condorcet, top_n_irv, top_n_runoff)
from .borda import borda
from .condorcet import (condorcet, condorcet_from_matrix,
                        ranked_election_to_matrix)
from .coombs import coombs
from .fptp import fptp, sntv
from .irv import irv
from .runoff import runoff
from .score import score
from .star import matrix_from_scores, star
from .utility_winner import utility_winner

__all__ = [
    'approval',
    'approval_runoff',
    'black',
    'borda',
    'combined_approval',
    'condorcet',
    'condorcet_from_matrix',
    'coombs',
    'fptp',
    'irv',
    'irv_primary_top_n_runoff',
    'matrix_from_scores',
    'ranked_election_to_matrix',
    'runoff',
    'sntv',
    'score',
    'star',
    'top_n_condorcet',
    'top_n_irv',
    'top_n_runoff',
    'utility_winner',
]
