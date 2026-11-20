"""Shared :mod:`docrep` snippets for recurring NumPy-style parameter docs."""

from docrep import DocstringProcessor

docstrings = DocstringProcessor()

_dedent = docstrings.dedent
def _safe_dedent(func):
    """Skip docstring processing when Python was run with -OO."""
    if func.__doc__ is None:
        return func
    return _dedent(func)
docstrings.dedent = _safe_dedent

_get_sections = docstrings.get_sections
def _safe_get_sections(*args, **kwargs):
    """Skip section extraction when the docstring is unavailable."""
    decorator = _get_sections(*args, **kwargs)
    def apply(func):
        if func.__doc__ is None:
            return func
        return decorator(func)
    return apply
docstrings.get_sections = _safe_get_sections


@docstrings.get_sections(base='utilities_2d', sections=['Parameters'])
@docstrings.dedent
def _utilities_2d_param_doc():
    """
    Shared documentation for the ``utilities`` parameter.

    Parameters
    ----------
    utilities : array_like
        A 2D collection of utilities.

        Rows represent voters, and columns represent candidate IDs.
        Higher utility numbers mean greater approval of that candidate by that
        voter.
    """
    pass


if _utilities_2d_param_doc.__doc__ is not None:
    docstrings.keep_params('utilities_2d.parameters', 'utilities')
