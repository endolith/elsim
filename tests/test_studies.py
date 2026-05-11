"""Tests for elsim.studies (Monte Carlo helpers, parameter expansion)."""

from collections import Counter
from functools import partial

import numpy as np
import pytest

from elsim.methods import approval, black, borda, condorcet, coombs, fptp, irv, runoff, utility_winner
from elsim.strategies import approval_optimal
from elsim.studies import (
    SerialBackend,
    accumulate_spatial_condorcet_by_ncands,
    accumulate_spatial_sue_by_ncands,
    approval_at_optimal,
    expand_product,
    expand_rows,
    expand_zip,
    merge_counters,
    random_society_utility_updates,
    ranked_rated_utility_updates,
    run_batched,
    spatial_random_reference_utility_updates,
    tally_condorcet_agreement,
)


def test_expand_product_scalar_and_list():
    got = expand_product(n_voters=[10, 20], n_cands=3)
    assert got == [{"n_voters": 10, "n_cands": 3}, {"n_voters": 20, "n_cands": 3}]


def test_expand_product_bytes_scalar():
    assert expand_product(blob=b"ab") == [{"blob": b"ab"}]


def test_expand_rows_empty():
    assert expand_rows((), ("a",)) == []


def test_run_batched_uses_implicit_serial_backend():
    out = run_batched(lambda k: k, n_trials=5, batch_size=2)
    assert out == [2, 2, 1]


def test_expand_zip_basic():
    assert expand_zip(a=[1, 2], b=[3, 4]) == [{"a": 1, "b": 3}, {"a": 2, "b": 4}]


def test_expand_zip_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        expand_zip(a=[1, 2], b=[3])


def test_expand_rows_merrill_style():
    rows = ((1.0, 0.5, 2), (0.5, 0.0, 4))
    keys = ("disp", "corr", "D")
    assert expand_rows(rows, keys) == [
        {"disp": 1.0, "corr": 0.5, "D": 2},
        {"disp": 0.5, "corr": 0.0, "D": 4},
    ]


def test_expand_rows_width_mismatch():
    with pytest.raises(ValueError, match="row 0"):
        expand_rows([(1, 2, 3)], ("a", "b"))


def test_merge_counters():
    assert merge_counters([Counter({"a": 1}), Counter({"a": 2, "b": 1})]) == Counter({"a": 3, "b": 1})


def test_run_batched_serial():
    sizes = []

    def batch_fn(k):
        sizes.append(k)
        return k

    out = run_batched(batch_fn, n_trials=25, batch_size=10, backend=SerialBackend())
    assert out == [10, 10, 5]
    assert sizes == [10, 10, 5]


def test_run_batched_exact_batches():
    out = run_batched(lambda k: k, n_trials=30, batch_size=10, backend=SerialBackend())
    assert out == [10, 10, 10]


def test_tally_condorcet_agreement_no_cw():
    rankings = np.array(
        [
            [0, 1, 2],
            [1, 2, 0],
            [2, 0, 1],
        ],
        dtype=np.uint8,
    )
    utilities = np.zeros_like(rankings, dtype=float)
    ranked = {"Plurality": fptp}
    rated: dict = {}
    assert tally_condorcet_agreement(rankings, utilities, ranked, rated) == Counter()


def test_tally_condorcet_agreement_with_cw():
    # Candidate 0 beats everyone pairwise
    rankings = np.array(
        [
            [0, 1, 2],
            [0, 2, 1],
            [0, 1, 2],
        ],
        dtype=np.uint8,
    )
    utilities = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    assert condorcet(rankings) == 0
    assert fptp(rankings, tiebreaker="random") == 0

    ranked = {"Plurality": fptp}
    rated: dict = {}
    c = tally_condorcet_agreement(rankings, utilities, ranked, rated, tiebreaker="random")
    assert c["CW"] == 1
    assert c["Plurality"] == 1


def test_serial_backend_map_each():
    out = SerialBackend().map_each([lambda: 1, lambda: 2])
    assert out == [1, 2]


def test_parallel_delayed_map_each():
    pytest.importorskip("joblib")
    from joblib import Parallel, delayed

    def f(x):
        return x

    out = list(Parallel(n_jobs=2, verbose=0)(delayed(fn)() for fn in [partial(f, 1), partial(f, 2)]))
    assert out == [1, 2]


def test_ranked_rated_utility_updates():
    utilities = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    rankings = np.array([[0, 1], [0, 1], [0, 1]], dtype=np.uint8)

    delta = ranked_rated_utility_updates(
        utilities, rankings, {'Plurality': fptp}, {}, tiebreaker='random',
    )
    assert set(delta) == {'Plurality'}
    assert delta['Plurality'] == float(utilities.sum(axis=0)[0])


def test_spatial_random_reference_includes_rw():
    np.random.seed(0)
    utilities = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]])
    rankings = np.array([[0, 1, 2], [1, 0, 2]], dtype=np.uint8)

    delta = spatial_random_reference_utility_updates(
        utilities, rankings, {'Plurality': fptp}, {}, tiebreaker='random',
    )
    assert 'RW' in delta
    assert 'Plurality' in delta


def test_expand_zip_empty():
    assert expand_zip() == []


def test_expand_product_mapping_value_raises():
    with pytest.raises(TypeError, match="Mappings"):
        expand_product(x={"a": 1})


def test_run_batched_zero_trials():
    assert run_batched(lambda k: k, n_trials=0, batch_size=10) == []


def test_run_batched_negative_trials_raises():
    with pytest.raises(ValueError, match="non-negative"):
        run_batched(lambda k: k, n_trials=-1, batch_size=10)


def test_run_batched_invalid_batch_size_raises():
    with pytest.raises(ValueError, match="positive"):
        run_batched(lambda k: k, n_trials=10, batch_size=0)


def test_serial_backend_map_repeat_negative_raises():
    with pytest.raises(ValueError, match="non-negative"):
        SerialBackend().map_repeat(lambda: 1, n=-1)


def test_merge_counters_empty():
    assert merge_counters([]) == Counter()


def test_random_society_utility_updates_tiebreaker_none():
    utilities = np.array([[0.9, 0.1], [0.8, 0.2]])
    rankings = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    delta = random_society_utility_updates(
        utilities,
        rankings,
        {"Plurality": fptp},
        {},
        tiebreaker="random",
        uw_key="UW",
        utility_winner_tiebreaker=None,
    )
    assert "UW" in delta
    assert "Plurality" in delta


def test_random_society_utility_updates_custom_uw_key_and_rated():
    utilities = np.array([[1.0, 0.0], [1.0, 0.0]])
    rankings = np.array([[0, 1], [0, 1]], dtype=np.uint8)

    rated = {
        "Approval": lambda u, tiebreaker: approval(
            approval_optimal(u), tiebreaker,
        ),
    }
    delta = random_society_utility_updates(
        utilities,
        rankings,
        {"Plurality": fptp},
        rated,
        tiebreaker="random",
        uw_key="XX",
        utility_winner_tiebreaker="random",
    )
    assert "XX" in delta
    assert "Approval" in delta
    assert "Plurality" in delta


def test_spatial_random_reference_with_rated():
    utilities = np.array([[1.0, 0.0], [1.0, 0.0]])
    rankings = np.array([[0, 1], [0, 1]], dtype=np.uint8)

    rated = {
        "Approval": lambda u, tiebreaker: approval(
            approval_optimal(u), tiebreaker,
        ),
    }
    delta = spatial_random_reference_utility_updates(
        utilities, rankings, {"Plurality": fptp}, rated, tiebreaker="random",
    )
    assert set(delta) >= {"RW", "Plurality", "Approval"}


def test_ranked_rated_with_both_method_kinds():
    utilities = np.array([[1.0, 0.0], [1.0, 0.0]])
    rankings = np.array([[0, 1], [0, 1]], dtype=np.uint8)

    rated = {
        "Approval": lambda u, tiebreaker: approval(
            approval_optimal(u), tiebreaker,
        ),
    }
    delta = ranked_rated_utility_updates(
        utilities, rankings, {"Plurality": fptp}, rated, tiebreaker="random",
    )
    assert "Plurality" in delta
    assert "Approval" in delta


def test_approval_at_optimal_matches_explicit():
    utilities = np.array([[0.2, 0.8], [0.1, 0.9]])
    tb = "random"
    assert approval_at_optimal(utilities, tb) == approval(approval_optimal(utilities), tb)


def test_tally_condorcet_agreement_rated_branch():
    rankings = np.array([[0, 1], [0, 1], [0, 1]], dtype=np.uint8)
    utilities = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
    )
    assert condorcet(rankings) == 0
    ranked = {
        "Plurality": fptp,
        "Runoff": runoff,
        "Hare": irv,
        "Borda": borda,
        "Coombs": coombs,
        "Black": black,
    }
    rated = {"SU max": utility_winner, "Approval": approval_at_optimal}
    c = tally_condorcet_agreement(rankings, utilities, ranked, rated, tiebreaker="random")
    assert c["CW"] == 1
    assert c["SU max"] == 1
    assert c["Plurality"] == 1


def test_accumulate_spatial_condorcet_by_ncands_keys():
    np.random.seed(0)
    ranked = {"Plurality": fptp, "Hare": irv}
    rated = {"SU max": utility_winner, "Approval": approval_at_optimal}
    out = accumulate_spatial_condorcet_by_ncands(
        4,
        n_voters=15,
        n_cands_list=(3, 4),
        dims=2,
        corr=0.5,
        disp=1.0,
        ranked_methods=ranked,
        rated_methods=rated,
        tiebreaker="random",
    )
    assert set(out.keys()) == {"CW", "Plurality", "Hare", "SU max", "Approval"}
    assert sum(out["CW"].values()) <= 4 * 2


def test_accumulate_spatial_sue_by_ncands_keys():
    np.random.seed(1)
    ranked = {"Plurality": fptp}
    rated = {"SU max": utility_winner, "Approval": approval_at_optimal}
    out = accumulate_spatial_sue_by_ncands(
        3,
        n_voters=12,
        n_cands_list=(3,),
        dims=2,
        corr=0.5,
        disp=1.0,
        ranked_methods=ranked,
        rated_methods=rated,
        tiebreaker="random",
    )
    assert set(out.keys()) == {"SU max", "RW", "Plurality", "Approval"}
    assert out["RW"][3] > 0
