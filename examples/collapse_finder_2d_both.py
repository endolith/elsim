"""
Run both IRV and TVR (Baldwin) collapse animations from a single election.

Finds one 2D election that satisfies both:
- IRV: strict center-outward elimination order (closest to origin eliminated each round).
- TVR: converges to the center (Condorcet) candidate.

Then renders the IRV animation to Images/collapse_2d_both_*_nc*_nv*/irv/ and the
TVR animation to .../tvr/.  Same ballots, same candidates and voters; two GIFs.
"""

from datetime import datetime
from pathlib import Path

import numpy as np

from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.strategies import honest_rankings

from collapse_2d_shared import (
    sort_candidates_bell_curve,
    palette_name,
    n_voters,
    n_cands,
    max_trials,
    frames_per_transfer,
    disp,
    dark_background,
)
from collapse_finder_2d_irv import simulate_irv_rounds, run_irv_animation
from collapse_finder_2d_tvr import simulate_tvr_rounds, run_tvr_animation


# Path to positions.npz from a previous run, or None to search for a new election.
INPUT_POSITIONS = Path('Images/collapse_2d_both_20260308_141324_nc9_nv5000 great/positions.npz')
# INPUT_POSITIONS = None


def find_both_election(n_voters, n_cands, max_trials, disp=1.0):
    """
    Sample random 2D elections until one satisfies both IRV center-outward
    and TVR center-winner.

    Returns (trial, voters, candidates, rankings, irv_trace, tvr_trace) or None.
    """
    for trial in range(1, max_trials + 1):
        voters, candidates = normal_electorate(n_voters, n_cands, dims=2, disp=disp)
        candidates[0] = 0.0
        candidates = sort_candidates_bell_curve(candidates)
        utilities = normed_dist_utilities(voters, candidates)
        rankings = np.asarray(honest_rankings(utilities))

        irv_trace = simulate_irv_rounds(rankings, candidates)
        if irv_trace is None:
            continue

        tvr_trace = simulate_tvr_rounds(rankings, candidates)
        if tvr_trace is None:
            continue

        return trial, voters, candidates, rankings, irv_trace, tvr_trace
    return None


if __name__ == '__main__':
    if INPUT_POSITIONS is not None:
        data = np.load(INPUT_POSITIONS)
        voters = data['voters']
        candidates = data['candidates']
        utilities = normed_dist_utilities(voters, candidates)
        rankings = np.asarray(honest_rankings(utilities))
        irv_trace = simulate_irv_rounds(rankings, candidates)
        tvr_trace = simulate_tvr_rounds(rankings, candidates)
        if irv_trace is None:
            raise RuntimeError(
                f'IRV did not yield center-outward order for {INPUT_POSITIONS}.'
            )
        if tvr_trace is None:
            raise RuntimeError(
                f'TVR did not converge to center for {INPUT_POSITIONS}.'
            )
        trial = None
        print(f'Loaded election from {INPUT_POSITIONS}.')
    else:
        result = find_both_election(n_voters, n_cands, max_trials, disp=disp)
        if result is None:
            raise RuntimeError(
                'No election found that satisfies both IRV center-outward and TVR '
                'center-winner. Increase max_trials or reduce n_cands.'
            )
        trial, voters, candidates, rankings, irv_trace, tvr_trace = result
        print(f'Found election on trial {trial} (IRV center-outward + TVR center winner).')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('Images') / f'collapse_2d_both_{timestamp}_nc{n_cands}_nv{n_voters}'
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(output_dir / 'positions.npz', voters=voters, candidates=candidates)

    print('Rendering IRV animation...')
    run_irv_animation(
        voters, candidates, rankings, irv_trace, output_dir / 'irv',
        palette_name=palette_name,
        n_cands=n_cands,
        n_voters=n_voters,
        frames_per_transfer=frames_per_transfer,
        dark_background=dark_background,
    )
    print('Rendering TVR animation...')
    run_tvr_animation(
        voters, candidates, rankings, tvr_trace, output_dir / 'tvr',
        palette_name=palette_name,
        n_cands=n_cands,
        n_voters=n_voters,
        frames_per_transfer=frames_per_transfer,
        dark_background=dark_background,
    )
    print(f'Saved both animations to {output_dir.resolve()}')
    print(f'  IRV: {output_dir / "irv" / "collapse_2d_irv.gif"}')
    print(f'  TVR: {output_dir / "tvr" / "collapse_2d_tvr.gif"}')
