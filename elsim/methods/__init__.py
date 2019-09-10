"""
Implements various election methods, which take collections of
ballots (elections) as inputs and return the winner according
to the rules of that method.
"""
from .approval import approval, approval_optimal
from .black import black
from .borda import borda
from .condorcet import (condorcet, condorcet_from_matrix,
                        ranked_election_to_matrix)
from .coombs import coombs
from .fptp import fptp
from .irv import irv
from .runoff import runoff
from .utility_winner import utility_winner
