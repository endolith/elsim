"""
Implements various election methods.

These take collections of ballots (elections) as inputs and return the winner
according to the rules of that method.
"""
from elsim.methods.approval import approval, combined_approval
from elsim.methods.black import black
from elsim.methods.blanket_primary import (approval_runoff, irv_eliminate_to_n,
                                           irv_primary_top_n_irv,
                                           irv_primary_top_n_runoff,
                                           top_five_condorcet, top_five_irv,
                                           top_five_runoff, top_four_condorcet,
                                           top_four_irv, top_four_runoff,
                                           top_n_condorcet, top_n_irv,
                                           top_n_runoff)
from elsim.methods.borda import borda
from elsim.methods.condorcet import (condorcet, condorcet_from_matrix,
                                     ranked_election_to_matrix)
from elsim.methods.coombs import coombs
from elsim.methods.fptp import fptp, sntv
from elsim.methods.irv import irv
from elsim.methods.runoff import runoff
from elsim.methods.score import score
from elsim.methods.star import matrix_from_scores, star
from elsim.methods.utility_winner import utility_winner

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
    'irv_eliminate_to_n',
    'irv_primary_top_n_irv',
    'irv_primary_top_n_runoff',
    'matrix_from_scores',
    'ranked_election_to_matrix',
    'runoff',
    'sntv',
    'score',
    'star',
    'top_five_condorcet',
    'top_five_irv',
    'top_five_runoff',
    'top_four_condorcet',
    'top_four_irv',
    'top_four_runoff',
    'top_n_condorcet',
    'top_n_irv',
    'top_n_runoff',
    'utility_winner',
]
