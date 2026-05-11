"""Tests for :mod:`docrep`-backed shared docstring anchors (see ``elsim._docstrings``)."""


def test_utilities_2d_doc_anchor_executes():
    from elsim._docstrings import _utilities_2d_param_doc

    assert _utilities_2d_param_doc() is None


def test_election_common_doc_anchor_executes():
    from elsim.elections.elections import _election_common_param_doc

    assert _election_common_param_doc() is None


def test_docrep_substitutions_present():
    from elsim.elections import random_utilities
    from elsim.strategies import honest_rankings

    assert 'n_voters : int' in (random_utilities.__doc__ or '')
    assert 'utilities : array_like' in (honest_rankings.__doc__ or '')
