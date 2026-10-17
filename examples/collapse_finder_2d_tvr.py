"""
Find a 2D election where TVR (Total Vote Runoff = Baldwin's method)
converges to the center candidate, and animate the Borda-score updates.

TVR eliminates the candidate with the lowest Borda score each round, where
Borda scores are re-tallied among remaining candidates only.  Unlike IRV, TVR
satisfies the Condorcet criterion: the center candidate (Condorcet winner) is
never eliminated and always wins.  The same election that shows IRV's center
collapse shows TVR's center convergence.

Pass ``--positions path/to/positions.npz`` from a collapse_finder_2d_irv.py
run to reuse that election, or omit it to search for a fresh one.
"""

from datetime import datetime

import numpy as np

from elsim.methods.baldwin import baldwin_rounds

from collapse_2d_animation import (ScorePanel, run_animation,
                                   trace_to_final_two)
from collapse_2d_shared import (RESULTS_DIR, candidate_name, find_election,
                                load_election, parse_args)


# ── Analysis ──────────────────────────────────────────────────────────

def simulate_tvr_rounds(election):
    """
    Run Baldwin's method down to two candidates; keep only elections where
    TVR's winner is the center (nearest-to-origin) candidate.

    Returns a `RunoffTrace`, or ``None`` on an elimination tie, an early
    majority (which would leave rounds un-animated), or a non-center winner.
    """
    result = baldwin_rounds(election.rankings, tiebreaker=None,
                            min_remaining=2, record_rounds=True)
    if result is None:
        return None
    trace = trace_to_final_two(result)
    if trace is None or trace.winner != election.center_candidate():
        return None
    return trace


# ── Presentation ──────────────────────────────────────────────────────

class AverageRankPanel(ScorePanel):
    """
    Top panel for TVR: average rank (1 = best) among remaining candidates.

    A candidate's TVR score divided by ``n_voters`` is the mean number of
    remaining candidates ranked *below* them, so ``avg_rank = n_active -
    score / n_voters``.  Bars therefore have height ``score / n_voters``
    and sit on a "dead zone" of ``n_cands - n_active`` rank slots that no
    longer exist; the top of each bar lands at ``n_cands - avg_rank`` on a
    fixed axis, which keeps the chart stable as candidates disappear.

    Panel state: ``{'scores': ndarray, 'n_active': int}`` where ``n_active``
    is the number of candidates the scores were tallied over.

    During a transfer the scores are kept in the *pre-elimination* scale
    (``n_active`` unchanged): promoted candidates gain one point per voter
    and the loser drains to zero.  At the end of the transfer every survivor
    holds ``borda_after + n_voters``, which is exactly the same average rank
    the next key frame shows with ``n_active - 1`` -- so the chart never
    jumps.
    """

    def start_state(self, trace, ctx):
        return {'scores': trace.rounds[0]['borda_before'].copy(),
                'n_active': ctx.n_cands}

    def round_state(self, round_data, n_eliminated_before, ctx):
        return {'scores': round_data['borda_before'].copy(),
                'n_active': ctx.n_cands - n_eliminated_before}

    def transfer_voters(self, round_data, ctx):
        # Every voter's ballot loses the loser, so every voter changes scores.
        return np.arange(ctx.n_voters)

    def advance(self, state, round_data, voter_batch, ballots):
        loser = round_data['loser']
        scores = state['scores']
        for v in voter_batch:
            promoted = round_data['promoted_per_voter'][v]
            scores[loser] -= len(promoted)   # loser's points drain to zero
            scores[promoted] += 1.0          # each promoted cand moves up one

    def final_state(self, trace, ctx):
        return {'scores': trace.rounds[-1]['borda_after'].copy(),
                'n_active': 2}

    def draw(self, ax, state, frame, ctx):
        theme = ctx.theme
        n_cands, n_voters = ctx.n_cands, ctx.n_voters
        n_active = state['n_active']
        dead_height = n_cands - n_active

        heights = state['scores'] / n_voters
        avg_ranks = n_active - heights
        bars = ax.bar(range(n_cands), heights, bottom=dead_height,
                      tick_label=list(ctx.labels),
                      color=ctx.active_colors(frame.eliminated))

        for c, rect in enumerate(bars):
            # Annotate survivors only (not the loser being faded out).
            if heights[c] <= 0 or c in frame.eliminated:
                continue
            top = rect.get_y() + rect.get_height()
            # Near the axis top a label above the bar would overflow.
            near_top = top >= n_cands - 2
            ax.annotate(f'{avg_ranks[c]:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, top),
                        xytext=(0, -4 if near_top else 3),
                        textcoords='offset points', ha='center',
                        va='top' if near_top else 'bottom',
                        color=theme.bg if near_top else theme.fg)

        ax.set_ylim(0, n_cands - 1)
        if dead_height > 0:
            ax.axhspan(0, dead_height, color=theme.dead_zone_color, zorder=0)
        # y ticks read as ranks: 1 (best) at the top, blank in the dead zone.
        tick_vals = list(range(n_cands))
        ax.set_yticks(tick_vals)
        ax.set_yticklabels(['' if v < dead_height else str(n_cands - v)
                            for v in tick_vals])
        ax.set_ylabel('Avg. rank (1=best)')
        ax.grid(True, alpha=0.25, axis='y', color=theme.grid)
        ax.set_axisbelow(True)


def run_tvr_animation(election, trace, output_dir, config):
    """Render the TVR (Baldwin) animation; returns the GIF path."""
    return run_animation(election, trace, AverageRankPanel(), output_dir,
                         config, method_name='TVR',
                         gif_name='collapse_2d_tvr.gif')


# ── Script ────────────────────────────────────────────────────────────

def main():
    config, positions = parse_args(__doc__, allow_positions=True)

    if positions is not None:
        election = load_election(positions)
        trace = simulate_tvr_rounds(election)
        if trace is None:
            raise SystemExit(f'TVR did not converge to the center candidate '
                             f'for {positions}. Try a different file.')
        print(f'Loaded election from {positions}.')
    else:
        found = find_election(config, simulate_tvr_rounds)
        if found is None:
            raise SystemExit('No TVR-convergent center election found. '
                             'Increase --max-trials or reduce --n-cands.')
        trial, election, trace = found
        print(f'Found TVR-convergent election on trial {trial}.')

    print('TVR elimination order:', trace.elimination_order)
    print('TVR winner:', candidate_name(trace.winner))

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = (RESULTS_DIR / f'collapse_2d_tvr_{timestamp}'
                  f'_nc{config.n_cands}_nv{config.n_voters}')
    gif = run_tvr_animation(election, trace, output_dir, config)
    print(f'Saved frames and GIF to {gif.parent.resolve()}')


if __name__ == '__main__':
    main()
