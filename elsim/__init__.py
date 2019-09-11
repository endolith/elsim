"""
Election Simulator 3000
-----------------------

This package provides three sub-modules, which are used to simulate one or more
elections by combining their functions in the following order:

`elections`
    Functions that generate (random) collections of voter-candidate utilities
    for different types of electorates.
`strategies`
    Functions that convert utilities into ballots, following various strategic
    rules (including honesty).
`methods`
   Functions that implement voting methods, which calculate a winner from a
   collection of ballots.
"""
from . import elections, methods
