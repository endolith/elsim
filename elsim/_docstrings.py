"""Shared :mod:`docrep` snippets for recurring NumPy-style parameter docs."""

from docrep import DocstringProcessor

docstrings = DocstringProcessor()


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


docstrings.keep_params('utilities_2d.parameters', 'utilities')
