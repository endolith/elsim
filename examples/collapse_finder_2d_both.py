"""Find one election and render both its IRV and Baldwin collapse animations."""

from datetime import datetime
from pathlib import Path

import numpy as np

from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.strategies import honest_rankings

from examples.collapse_2d_shared import RESULTS_DIR, sort_candidates_bell_curve
from examples.collapse_finder_2d_irv import (
    run_irv_animation,
    simulate_irv_rounds,
)
from examples.collapse_finder_2d_tvr import (
    run_tvr_animation,
    simulate_tvr_rounds,
)


def election_to_traces(voters, candidates):
    """Compute honest rankings and both method traces for one election."""
    utilities = normed_dist_utilities(voters, candidates)
    rankings = np.asarray(honest_rankings(utilities))
    irv_trace = simulate_irv_rounds(rankings, candidates)
    tvr_trace = simulate_tvr_rounds(rankings, candidates)
    return rankings, irv_trace, tvr_trace


def find_both_election(n_voters, n_cands, max_trials, disp=1.0):
    """Sample one election satisfying both geometric illustration criteria."""
    for trial in range(1, max_trials + 1):
        voters, candidates = normal_electorate(
            n_voters,
            n_cands,
            dims=2,
            disp=disp,
        )
        candidates[0] = 0.0
        candidates = sort_candidates_bell_curve(candidates)
        rankings, irv_trace, tvr_trace = election_to_traces(voters, candidates)
        if irv_trace is not None and tvr_trace is not None:
            return trial, voters, candidates, rankings, irv_trace, tvr_trace
    return None


if __name__ == '__main__':
    n_voters = 5000
    n_cands = 9
    max_trials = 100_000
    frames_per_transfer = 60
    disp = 0.5
    palette_name = 'Bold_10'
    dark_background = True

    result = find_both_election(
        n_voters,
        n_cands,
        max_trials,
        disp=disp,
    )
    if result is None:
        raise RuntimeError(
            'No election found satisfying both IRV center-outward and '
            'Baldwin center-convergence criteria.'
        )

    trial, voters, candidates, rankings, irv_trace, tvr_trace = result
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = RESULTS_DIR / (
        f'collapse_2d_both_{timestamp}_nc{n_cands}_nv{n_voters}'
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / 'positions.npz', voters=voters, candidates=candidates)

    print(f'Found shared election on trial {trial}.')
    print('Rendering IRV animation...')
    run_irv_animation(
        voters,
        candidates,
        rankings,
        irv_trace,
        output_dir / 'irv',
        palette_name=palette_name,
        frames_per_transfer=frames_per_transfer,
        dark_background=dark_background,
    )
    print('Rendering Baldwin animation...')
    run_tvr_animation(
        voters,
        candidates,
        rankings,
        tvr_trace,
        output_dir / 'tvr',
        palette_name=palette_name,
        frames_per_transfer=frames_per_transfer,
        dark_background=dark_background,
    )
    print(f'Saved both animations to {output_dir.resolve()}')
