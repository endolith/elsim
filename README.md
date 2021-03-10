# Election Simulator 3000
[![CircleCI](https://circleci.com/gh/endolith/elsim.svg?style=shield)](https://circleci.com/gh/endolith/elsim)
[![Actions Status](https://github.com/endolith/elsim/workflows/Python%20package/badge.svg)](https://github.com/endolith/elsim/actions)

This is a library of functions for running many simulations of different elections ([Impartial Culture](https://en.wikipedia.org/wiki/Impartial_culture), spatial model, ...) held using different [voting methods](https://en.wikipedia.org/wiki/Electoral_system) ([Borda count](https://en.wikipedia.org/wiki/Borda_count), [Approval voting](https://en.wikipedia.org/wiki/Approval_voting), ...).

Goals:

- Fast (~25,000 elections per second on Core i7-9750H)
- Flexible
- Well-documented, easily-used and improved upon by other people
- Well-tested and bug-free
- Able to reproduce peer-reviewed research

# Requirements
See `requirements.txt`.  As of this README, it includes  [`numpy`](https://numpy.org/) and [`scipy`](https://www.scipy.org/)for the simulations, `tabulate` for printing example tables,  and  [`pytest`](https://docs.pytest.org/en/latest/), [`hypothesis`](https://hypothesis.readthedocs.io/en/latest/), and `pytest-cov` for running the tests.  All should be installable through `conda`.

Optionally, `elsim` can use [`numba`](http://numba.pydata.org/) for speed.  If not available, the code will still run, just more slowly.

# Installation
One possibility is to install with pip:

    pip install git+https://github.com/endolith/elsim.git

# Documentation
Currently just the docstrings of the submodules and functions themselves, in [`numpydoc` format](https://numpydoc.readthedocs.io/en/latest/format.html).

# Usage
Specify an election with three candidates (0, 1, 2), where two voters rank candidates 0 > 2 > 1, two voters rank candidates 1 > 2 > 0, and one ranks candidates 2 > 0 > 1:

```python
>>> election = [[0, 2, 1],
                [0, 2, 1],
                [1, 2, 0],
                [1, 2, 0],
                [2, 0, 1]]
```

Calculate the winner using Black's method:

```python
>>> from elsim.methods import black
>>> black(election)
2
```

Candidate 2 is the Condorcet winner, and wins under Black's method.

See [/examples](/examples) folder for more on what it can do, such as reproductions of previous research.

# Tests
Tests can be run by installing the testing dependencies and then running `pytest` in the project folder.

# Bugs / Requests
File issues on the [Github issue tracker](https://github.com/endolith/elsim/issues).

# Similar projects

## Election simulators

- 1D:
  - http://zesty.ca/voting/voteline/ (Flash, 5 candidates, normal/uniform/bimodal distribution)
  - https://demonstrations.wolfram.com/ComparingVotingSystemsForANormalDistributionOfVoters/
- 2D:
  - http://bolson.org/voting/sim_one_seat/www/spacegraph.html
  - http://zesty.ca/voting/sim/
  - http://rangevoting.org/IEVS/Pictures.html
- ND:
  - https://github.com/electology/vse-sim

## Voting system implementations

* [See Electowiki](https://electowiki.org/wiki/Voting_links#Election_calculators)
