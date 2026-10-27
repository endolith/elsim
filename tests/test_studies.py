import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from elsim.methods import borda, fptp, irv
from elsim.studies import (Study, approval_at_optimal, conditions, grid,
                           random_society, spatial_normal)


def test_grid_fixed_and_product():
    """Scalars stay fixed while iterables expand, in product order."""
    assert grid(n_cands=[2, 3], n_voters=25) == [
        {'n_cands': 2, 'n_voters': 25},
        {'n_cands': 3, 'n_voters': 25},
    ]
    assert grid(n_voters=[3, 5], n_cands=[3, 4]) == [
        {'n_voters': 3, 'n_cands': 3},
        {'n_voters': 3, 'n_cands': 4},
        {'n_voters': 5, 'n_cands': 3},
        {'n_voters': 5, 'n_cands': 4},
    ]


def test_conditions_rows():
    """Explicit rows become dictionaries, with either mappings or keys."""
    expected = [{'disp': 1.0, 'corr': 0.5, 'D': 2},
                {'disp': 0.5, 'corr': 0.0, 'D': 4}]
    assert conditions([(1.0, 0.5, 2), (0.5, 0.0, 4)],
                      ('disp', 'corr', 'D')) == expected
    assert conditions(expected) == expected


def test_conditions_rejects_mismatched_row():
    """A row with the wrong number of values is rejected clearly."""
    with pytest.raises(ValueError, match='row 0 has 2 values'):
        conditions([(1.0, 0.5)], ('disp', 'corr', 'D'))


def test_study_condorcet_counts():
    """Condorcet counts never exceed the number of elections with a CW."""
    study = Study(
        grid(n_cands=3, n_voters=25),
        random_society(),
        200,
        ranked_methods={'Plurality': fptp, 'Borda': borda, 'Hare': irv},
        rated_methods={'Approval': approval_at_optimal},
        metric='condorcet',
        random_state=7,
    )
    result = study.run()
    values = result.values[0]
    assert 0 <= values['CW'] <= study.n_elections
    for name in ('Plurality', 'Borda', 'Hare', 'Approval'):
        assert 0 <= values[name] <= values['CW']
    assert result.table(percent=False)[0][0] == 'Plurality'


def test_study_sue_per_candidate_set():
    """SUE groups electorates into candidate sets and averages per set."""
    study = Study(
        conditions([(1.0, 0.5, 2), (0.5, 0.0, 2)], ('disp', 'corr', 'D')),
        spatial_normal(n_voters=31, n_cands=3),
        20,
        ranked_methods={'Plurality': fptp},
        metric='sue',
        electorates_per_set=5,
        normalization='standard',
        random_state=11,
    )
    result = study.run()
    assert len(result.values) == 2
    for value in result.by_method()['Plurality']:
        assert 0 <= value <= 100


def test_study_requires_divisible_grouping():
    """n_elections must be a whole number of candidate sets."""
    with pytest.raises(ValueError, match='divisible'):
        Study(grid(n_cands=3), random_society(), 7,
              ranked_methods={'Plurality': fptp}, electorates_per_set=5)


def test_spatial_draw_set_reuses_candidates():
    """Every electorate in a set shares the candidate positions."""
    model = spatial_normal(n_voters=21, n_cands=4, dims=2, corr=0.5,
                           disp=1.0)
    elections = model.draw_set({'n_voters': 21, 'n_cands': 4}, 5,
                               np.random.default_rng(3))
    assert len(elections) == 5
    for election in elections[1:]:
        assert_array_equal(election.candidates, elections[0].candidates)
    assert not np.array_equal(elections[1].utilities,
                              elections[0].utilities)


def test_utility_normalization_values():
    """Range and standard normalization produce the expected row moments."""
    from elsim.studies import normalize_utilities
    utilities = np.array([[1.0, 2.0, 4.0], [0.0, 5.0, 5.0]])
    ranged = normalize_utilities(utilities, 'range')
    assert_allclose(ranged.min(axis=1), [0, 0])
    assert_allclose(ranged.max(axis=1), [1, 1])
    standard = normalize_utilities(utilities, 'standard')
    assert_allclose(standard.mean(axis=1), [0, 0], atol=1e-12)
    assert_allclose(standard.std(axis=1), [1, 1])
