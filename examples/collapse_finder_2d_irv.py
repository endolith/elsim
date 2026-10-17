"""
Find a 2D IRV center-outward collapse and animate transfers by round.

This mirrors the transfer animation style used in elsim2k T2R examples, but
for full IRV: one candidate eliminated per round until two remain.
"""

from datetime import datetime

import numpy as np

from elsim.methods import irv_rounds

from collapse_2d_animation import (ScorePanel, run_animation,
                                   trace_to_final_two)
from collapse_2d_shared import (RESULTS_DIR, annotate_bars, candidate_name,
                                find_election, parse_args)


# ── Analysis ──────────────────────────────────────────────────────────

def simulate_irv_rounds(election):
    """
    Run IRV until two candidates remain; keep only center-outward collapses.

    Returns a `RunoffTrace`, or ``None`` if IRV hits an elimination tie or
    the elimination order is not strictly center-outward (closest to the
    origin eliminated each round, farthest two at the end).
    """
    result = irv_rounds(election.rankings, tiebreaker=None,
                        min_remaining=2, record_rounds=True)
    if result is None:
        return None
    trace = trace_to_final_two(result)
    if trace is None or not _is_center_outward(trace, election):
        return None
    return trace


def _is_center_outward(trace, election):
    """True if each round removes the remaining candidate nearest the origin."""
    dists = election.distances_to_origin()
    remaining = set(range(election.n_cands))
    for round_data in trace.rounds:
        nearest = min(remaining, key=lambda c: (dists[c], c))
        if round_data['loser'] != nearest:
            return False
        remaining.remove(nearest)
    return sorted(remaining) == sorted(int(c) for c in np.argsort(dists)[-2:])


# ── Presentation ──────────────────────────────────────────────────────

class VoteSharePanel(ScorePanel):
    """Top panel for IRV: first-preference vote share, 0–100%."""

    def start_state(self, trace, ctx):
        return {'tallies': trace.rounds[0]['tallies_before']}

    def round_state(self, round_data, n_eliminated_before, ctx):
        return {'tallies': round_data['tallies_before'].copy()}

    def transfer_voters(self, round_data, ctx):
        return round_data['affected_voters'].copy()

    def advance(self, state, round_data, voter_batch, ballots):
        state['tallies'] = np.bincount(ballots, minlength=len(state['tallies']))

    def final_state(self, trace, ctx):
        return {'tallies': trace.final_tallies}

    def draw(self, ax, state, frame, ctx):
        theme = ctx.theme
        share = state['tallies'] / ctx.n_voters * 100
        bars = ax.bar(range(ctx.n_cands), share, tick_label=list(ctx.labels),
                      color=ctx.active_colors(frame.eliminated))
        annotate_bars(ax, bars, color=theme.fg)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Votes [%]')
        ax.grid(True, alpha=0.25, axis='y', color=theme.grid)
        ax.set_axisbelow(True)


def run_irv_animation(election, trace, output_dir, config):
    """Render the IRV animation; returns the GIF path."""
    return run_animation(election, trace, VoteSharePanel(), output_dir,
                         config, method_name='IRV',
                         gif_name='collapse_2d_irv.gif')


# ── Script ────────────────────────────────────────────────────────────

def main():
    config = parse_args(__doc__)
    found = find_election(config, simulate_irv_rounds)
    if found is None:
        raise SystemExit('No strict center-outward collapse found. '
                         'Increase --max-trials or reduce --n-cands.')
    trial, election, trace = found

    print(f'Found strict center-outward IRV collapse on trial {trial}.')
    print('Elimination order:', trace.elimination_order)
    print('Final two:', *(candidate_name(c) for c in trace.final_two))

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = (RESULTS_DIR / f'collapse_2d_irv_{timestamp}'
                  f'_nc{config.n_cands}_nv{config.n_voters}')
    gif = run_irv_animation(election, trace, output_dir, config)
    print(f'Saved frames and GIF to {gif.parent.resolve()}')


if __name__ == '__main__':
    main()
