"""Tests for :mod:`docrep`-backed shared docstring anchors."""


def test_utilities_2d_doc_anchor_executes():
    from elsim._docstrings import _utilities_2d_param_doc

    assert _utilities_2d_param_doc() is None


def test_election_common_doc_anchor_executes():
    from elsim.elections import _election_common_param_doc

    assert _election_common_param_doc() is None


def test_docrep_substitutions_present():
    from elsim.elections import random_utilities
    from elsim.strategies import honest_rankings

    assert 'n_voters : int' in (random_utilities.__doc__ or '')
    assert 'utilities : array_like' in (honest_rankings.__doc__ or '')


def test_doc_helpers_tolerate_missing_docstrings():
    """docrep processing must no-op when Python runs with -OO."""
    from elsim._docstrings import docstrings

    def undocumented():
        pass

    undocumented.__doc__ = None
    assert docstrings.dedent(undocumented) is undocumented
    decorated = docstrings.get_sections(base='undocumented')(undocumented)
    assert decorated is undocumented
