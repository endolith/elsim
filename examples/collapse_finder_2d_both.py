"""
Run both IRV and TVR (Baldwin) collapse animations from a single election.

Finds one 2D election that satisfies both:
- IRV: strict center-outward elimination order
  (closest to origin eliminated each round).
- TVR: converges to the center (Condorcet) candidate.

Then renders the IRV animation to
examples/results/collapse_2d_both_*_nc*_nv*/irv/
and the TVR animation to .../tvr/.
Same ballots, same candidates and voters; two GIFs.

Pass ``--positions path/to/positions.npz`` to reuse a saved election
instead of searching for a new one.
"""

from datetime import datetime

from collapse_2d_shared import (RESULTS_DIR, find_election, load_election,
                                parse_args)
from collapse_finder_2d_irv import run_irv_animation, simulate_irv_rounds
from collapse_finder_2d_tvr import run_tvr_animation, simulate_tvr_rounds


def simulate_both(election):
    """Return ``(irv_trace, tvr_trace)`` or ``None`` if either test fails."""
    irv_trace = simulate_irv_rounds(election)
    if irv_trace is None:
        return None
    tvr_trace = simulate_tvr_rounds(election)
    if tvr_trace is None:
        return None
    return irv_trace, tvr_trace


def load_or_find_election(config, positions):
    """
    Load the election from ``positions`` or search for a new one.

    Returns ``(election, irv_trace, tvr_trace, message)`` where ``message``
    is a one-line description of where the election came from.
    """
    if positions is not None:
        election = load_election(positions)
        irv_trace = simulate_irv_rounds(election)
        if irv_trace is None:
            raise SystemExit('IRV did not yield a center-outward order '
                             f'for {positions}.')
        tvr_trace = simulate_tvr_rounds(election)
        if tvr_trace is None:
            raise SystemExit(f'TVR did not converge to center for {positions}.')
        return election, irv_trace, tvr_trace, \
            f'Loaded election from {positions}.'

    found = find_election(config, simulate_both)
    if found is None:
        raise SystemExit('No election found that satisfies both IRV '
                         'center-outward and TVR center-winner. '
                         'Increase --max-trials or reduce --n-cands.')
    trial, election, (irv_trace, tvr_trace) = found
    return election, irv_trace, tvr_trace, \
        f'Found election on trial {trial} ' \
        '(IRV center-outward + TVR center winner).'


def main():
    config, positions = parse_args(__doc__, allow_positions=True)
    election, irv_trace, tvr_trace, message = load_or_find_election(
        config, positions)
    print(message)
    print('IRV elimination order:', irv_trace.elimination_order)
    print('TVR elimination order:', tvr_trace.elimination_order)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = (RESULTS_DIR / f'collapse_2d_both_{timestamp}'
                  f'_nc{config.n_cands}_nv{config.n_voters}')
    output_dir.mkdir(parents=True, exist_ok=True)
    election.save_positions(output_dir / 'positions.npz')

    print('Rendering IRV animation...')
    irv_gif = run_irv_animation(election, irv_trace, output_dir / 'irv',
                                config)
    print('Rendering TVR animation...')
    tvr_gif = run_tvr_animation(election, tvr_trace, output_dir / 'tvr',
                                config)
    print(f'Saved both animations to {output_dir.resolve()}')
    print(f'  IRV: {irv_gif}')
    print(f'  TVR: {tvr_gif}')


if __name__ == '__main__':
    main()
