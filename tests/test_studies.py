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

    constant = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]])
    for mode in ('range', 'standard'):
        normalized = normalize_utilities(constant, mode)
        assert_array_equal(normalized[0], np.zeros(3))


def test_study_sue_handles_none_winner_and_zero_denominator():
    """Ties credit mean utility; zero-spread sets report NaN."""
    from elsim.studies.models import Election

    class FixedElectorate:
        def draw(self, scenario, rng):
            return Election(
                utilities=np.array([[10., 0.], [10., 0.],
                                    [0., 1.], [0., 1.]]),
                rankings=np.array([[0, 1], [0, 1], [1, 0], [1, 0]]),
            )

        def draw_set(self, scenario, n, rng):
            return [self.draw(scenario, rng) for _ in range(n)]

    tied = Study([{}], FixedElectorate(), 1,
                 ranked_methods={'Plurality': fptp},
                 metric='sue', normalization=None, tiebreaker=None)
    assert tied.run().values[0]['Plurality'] == 0

    one_candidate = Study(grid(n_cands=1, n_voters=2), random_society(), 1,
                          ranked_methods={'Plurality': fptp},
                          metric='sue', normalization=None,
                          tiebreaker='order', random_state=1)
    assert np.isnan(one_candidate.run().values[0]['Plurality'])


def test_grid_no_axes_and_bad_mapping():
    """No axes gives one empty scenario; mappings are rejected."""
    assert grid() == [{}]
    with pytest.raises(TypeError, match='not mappings'):
        grid(n_cands={'a': 1})


def test_conditions_rejects_mapping_with_keys_and_missing_keys():
    """Mixing mapping rows with keys, or sequences without keys, is invalid."""
    with pytest.raises(ValueError, match='row is already a mapping'):
        conditions([{'n_cands': 3}], ('n_cands',))
    with pytest.raises(ValueError, match='keys are required'):
        conditions([(3,)])


def test_metric_helpers():
    """The small metric helpers delegate to the right underlying methods."""
    from elsim.methods import utility_winner
    from elsim.studies import normalize_utilities, utility_winner_at
    from elsim.studies.metrics import rankings_from
    utilities = np.array([[0.1, 0.9], [0.8, 0.2]])
    assert utility_winner_at(utilities, 'order') == utility_winner(
        utilities, tiebreaker='order')
    assert normalize_utilities(utilities, None) is utilities
    with pytest.raises(ValueError, match='unknown utility normalization'):
        normalize_utilities(utilities, 'bogus')
    assert_array_equal(rankings_from(utilities), [[1, 0], [0, 1]])


def test_models_draw_and_draw_set():
    """Each model's draw helpers return the expected ballot data."""
    from elsim.studies import (impartial_culture, random_society,
                               spatial_normal)
    rng = np.random.default_rng(5)
    scenario = {'n_voters': 7, 'n_cands': 3}

    spatial = spatial_normal(dims=2, corr=0.5, disp=1.0)
    one = spatial.draw(scenario, rng)
    assert one.utilities.shape == (7, 3)
    assert spatial.draw_set(scenario, 2, rng)[0].utilities.shape == (7, 3)

    society = random_society()
    one = society.draw(scenario, rng)
    assert one.utilities.shape == (7, 3)
    assert len(society.draw_set(scenario, 2, rng)) == 2

    culture = impartial_culture()
    one = culture.draw(scenario, rng)
    assert one.rankings.shape == (7, 3)
    assert len(culture.draw_set(scenario, 2, rng)) == 2


def test_study_scored_methods():
    """Scored methods receive honest score ballots from the score rule."""
    from elsim.methods import score
    study = Study(
        grid(n_cands=3, n_voters=25),
        random_society(),
        50,
        scored_methods={'Score': score},
        metric='condorcet',
        random_state=13,
    )
    values = study.run().values[0]
    assert 0 <= values['Score'] <= values['CW']


def test_study_tiebreaker_none_and_run_shortcut():
    """A None tiebreaker passes through, and run() is a shortcut."""
    from elsim.studies import run
    result = run(
        grid(n_cands=2, n_voters=7),
        random_society(),
        10,
        ranked_methods={'Plurality': fptp},
        metric='condorcet',
        tiebreaker=None,
        random_state=3,
    )
    assert result.values[0]['Plurality'] <= result.values[0]['CW']


def test_study_impartial_culture_rankings_only():
    """Impartial culture can run ranked methods without utilities."""
    from elsim.studies import impartial_culture
    result = Study(
        grid(n_cands=3, n_voters=25),
        impartial_culture(),
        50,
        ranked_methods={'Plurality': fptp},
        metric='condorcet',
        random_state=17,
    ).run()
    assert 0 <= result.values[0]['Plurality'] <= result.values[0]['CW']


def test_study_impartial_culture_rejects_rated_methods():
    """Rated methods need utilities, which impartial culture does not have."""
    from elsim.studies import impartial_culture
    study = Study(
        grid(n_cands=3, n_voters=25),
        impartial_culture(),
        10,
        rated_methods={'Approval': approval_at_optimal},
        random_state=19,
    )
    with pytest.raises(ValueError, match='needs utilities'):
        study.run()


def test_study_validation_and_chunking():
    """Bad options are rejected; chunked task generation is deterministic."""
    with pytest.raises(ValueError, match="metric must be"):
        Study(grid(), random_society(), 1, metric='bogus')
    with pytest.raises(ValueError, match='must be positive'):
        Study(grid(), random_society(), 1, electorates_per_set=0)
    with pytest.raises(ValueError, match='divisible'):
        Study(grid(), random_society(), 5, electorates_per_set=2)

    study = Study(
        grid(n_cands=3, n_voters=7),
        random_society(),
        10,
        ranked_methods={'Plurality': fptp},
        n_jobs=2,
        random_state=np.random.default_rng(5),
    )
    tasks = study._tasks()
    assert len(tasks) == 2
    assert [task[0] for task in tasks] == [0, 0]


def test_study_result_table_percent_and_by_method():
    """Percent tables divide by CW; by_method is aligned with scenarios."""
    study = Study(
        grid(n_cands=[2, 3], n_voters=7),
        random_society(),
        20,
        ranked_methods={'Plurality': fptp},
        metric='condorcet',
        random_state=23,
    )
    result = study.run()
    table = result.table()
    cw_row = next(row for row in table if row[0] == 'CW')
    assert 0 <= cw_row[1] <= 100
    assert len(result.by_method()['Plurality']) == 2


def test_grid_bytes_is_scalar():
    """Bytes are treated as a scalar, not an iterable of integers."""
    assert grid(x=b'abc') == [{'x': b'abc'}]


def test_parallel_requires_joblib(monkeypatch):
    """n_jobs > 1 gives an actionable error when joblib is unavailable."""
    import sys
    monkeypatch.setitem(sys.modules, 'joblib', None)
    study = Study(
        grid(n_cands=3, n_voters=7),
        random_society(),
        10,
        ranked_methods={'Plurality': fptp},
        n_jobs=2,
        random_state=29,
    )
    with pytest.raises(ImportError, match=r'elsim\[parallel\]'):
        study.run()
