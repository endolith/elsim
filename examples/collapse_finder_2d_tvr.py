"""
Find a 2D center-outward IRV collapse and animate the same election under TVR
(Total Vote Runoff = Baldwin's method).

TVR eliminates the candidate with the lowest Borda score each round, where
Borda scores are re-tallied among remaining candidates only.  Unlike IRV, TVR
satisfies the Condorcet criterion: the center candidate (Condorcet winner) is
never eliminated and always wins.  The same election that shows IRV's center
collapse shows TVR's center convergence.

Set INPUT_POSITIONS to the positions.npz path saved by collapse_finder_2d_irv.py
to reuse that election, or leave None to search for a fresh one.
"""

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.methods import ranked_election_to_matrix
from elsim.methods._common import _all_indices, _inc_rank_idx, _tally_at_rank_idx
from elsim.strategies import honest_rankings

from collapse_2d_shared import (
    KEY_FRAME_MS,
    transition_step_ms,
    candidate_name,
    ceildiv,
    plot_approval_bar,
    plot_wins_with_title,
    prepare_palette_and_labels,
    get_theme,
    setup_scatter_axis_sigma,
    sort_candidates_bell_curve,
    voronoi_plot_2d_axes,
    palette_name,
    n_voters,
    n_cands,
    max_trials,
    frames_per_transfer,
    disp,
    dark_background,
)
from collapse_utils import count_wins


# Path to positions.npz from a previous run, or None to search for a fresh election.
INPUT_POSITIONS = None


def compute_borda_scores(election, eliminated_mask):
    """
    Borda scores for non-eliminated candidates as if eliminated were never on the ballot.

    election : ndarray
        Ranked ballots (n_voters × n_cands), same format as :func:`elsim.methods.borda.borda`.
    eliminated_mask : ndarray of bool, length n_cands
        ``eliminated_mask[i]`` is True if candidate ``i`` is eliminated.
    Returns ndarray of length n_cands; eliminated candidates have score 0.
    Scoring among remainders: n_remaining-1 points for 1st place, 0 for last place.
    """
    n_remaining = int(np.sum(~eliminated_mask))
    scores = np.zeros(election.shape[1], dtype=float)
    for ballot in election:
        pos = 0
        for cand_id in ballot:
            if eliminated_mask[cand_id]:
                continue
            scores[cand_id] += (n_remaining - 1 - pos)
            pos += 1
    return scores


def simulate_tvr_rounds(election, candidates):
    """
    Simulate TVR (Baldwin's method) with round-by-round trace data.

    Each round: if any remaining candidate has a majority of first-preference
    votes, they win; otherwise eliminate the lowest Borda score among remaining
    candidates (re-tallied as if eliminated candidates were never on the ballot).
    Returns None on any tie so only clean runs are kept.

    Per-round trace includes enough data to animate the ballot updates voter
    by voter: for each voter, which active candidates move up in rank when
    the loser is removed (i.e. which candidates are ranked below the loser
    in that voter's ballot).
    """
    election = np.asarray(election)
    n_voters, n_cands = election.shape
    voter_top_rank_idx = np.zeros(n_voters, dtype=np.uint8)
    cand_top_tallies = np.empty(n_cands, dtype=np.uint)
    eliminated_mask = np.zeros(n_cands, dtype=bool)
    rounds = []
    winner = None

    while np.sum(~eliminated_mask) > 1:
        _tally_at_rank_idx(cand_top_tallies, election, voter_top_rank_idx)
        cand_top_tallies_list = cand_top_tallies.tolist()

        max_cand_top_tally = max(cand_top_tallies_list)
        if max_cand_top_tally > n_voters / 2:
            winner = cand_top_tallies_list.index(max_cand_top_tally)
            break

        borda_before = compute_borda_scores(election, eliminated_mask)
        borda_list = borda_before.tolist()

        min_score = min(borda_list[c] for c in range(n_cands) if not eliminated_mask[c])
        low_scorers = _all_indices(borda_list, min_score)
        low_scorers = [c for c in low_scorers if not eliminated_mask[c]]
        if len(low_scorers) != 1:
            return None

        loser = low_scorers[0]

        # For each voter: the active candidates ranked below the loser.
        # When loser is removed, each of these gains +1 Borda point.
        promoted_per_voter = []
        for ballot in election:
            promoted = []
            found_loser = False
            for cand_id in ballot:
                if eliminated_mask[cand_id]:
                    continue
                if cand_id == loser:
                    found_loser = True
                    continue
                if found_loser:
                    promoted.append(cand_id)
            promoted_per_voter.append(promoted)

        eliminated_mask[loser] = True
        _inc_rank_idx(election, voter_top_rank_idx, eliminated_mask)
        borda_after = compute_borda_scores(election, eliminated_mask)

        rounds.append({
            'loser': loser,
            'borda_before': borda_before,
            'borda_after': borda_after,
            'promoted_per_voter': promoted_per_voter,
        })

    if winner is None:
        remaining = np.flatnonzero(~eliminated_mask)
        if len(remaining) != 1:
            return None
        winner = int(remaining[0])
    # Verify winner is nearest to origin (the Condorcet/center candidate).
    dists = np.linalg.norm(candidates, axis=1)
    center = int(np.argmin(dists))
    if winner != center:
        return None  # TVR didn't converge to center candidate; skip this election

    # final_two: the loser of the last elimination round and the overall winner.
    final_two = [rounds[-1]['loser'], winner]

    return {
        'rounds': rounds,
        'winner': winner,
        'final_two': final_two,
    }


def find_center_convergent_election(n_voters, n_cands, max_trials, disp=1.0):
    """
    Sample random 2D elections until TVR converges to the center candidate.

    Returns (trial, voters, candidates, rankings, trace).
    """
    for trial in range(1, max_trials + 1):
        voters, candidates = normal_electorate(n_voters, n_cands, dims=2, disp=disp)
        candidates[0] = 0.0
        candidates = sort_candidates_bell_curve(candidates)
        utilities = normed_dist_utilities(voters, candidates)
        rankings = np.asarray(honest_rankings(utilities))
        trace = simulate_tvr_rounds(rankings, candidates)
        if trace is not None:
            return trial, voters, candidates, rankings, trace
    return None


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_frame(
    voters,
    candidates,
    ballots,
    borda_scores,
    n_borda_active,
    colors,
    labels,
    frame_title,
    output_path,
    approval_pct,
    wins,
    eliminated=None,
    dark_background=True,
):
    """Render one TVR animation frame.

    ballots : ndarray of shape (n_voters,)
        Per-voter current first-choice candidate index.  Voters whose first
        choice is in `eliminated` are shown gray; others are colored by their
        first choice.  Updated incrementally during the transfer animation,
        exactly like the IRV script.
    borda_scores : ndarray of shape (n_cands,)
        Current Borda scores.  For candidates still active (including the loser
        currently being animated out), their scores may be non-zero.  For
        candidates eliminated in a previous round, scores are 0.
    n_borda_active : int
        The number of candidates whose Borda scores are non-trivially included in
        borda_scores (i.e. the active-set size when borda_scores was tallied).
        Used to convert Borda scores to avg_rank values for annotations.
    """
    if eliminated is None:
        eliminated = set()

    n_cands = len(candidates)
    # Candidates visible on scatter and coloured in bar charts (no loser, no old eliminated).
    remaining = [c for c in range(n_cands) if c not in eliminated]
    active_colors = [colors[n] if n not in eliminated else [0.5, 0.5, 0.5]
                     for n in range(n_cands)]

    bg, fg, grid, stroke_fg, legend_bg, legend_fg, voronoi_color, dead_zone_color = get_theme(dark_background)

    fig = plt.figure(figsize=(9, 7.5), facecolor=bg)
    ax_sc = plt.subplot2grid(shape=(6, 3), loc=(0, 0), colspan=2, rowspan=6)
    ax_bar = plt.subplot2grid(shape=(6, 3), loc=(0, 2), rowspan=2)
    ax_score = plt.subplot2grid(shape=(6, 3), loc=(2, 2), rowspan=2)
    ax_wins = plt.subplot2grid(shape=(6, 3), loc=(4, 2), rowspan=2)

    for ax in (ax_sc, ax_bar, ax_score, ax_wins):
        ax.set_facecolor(bg)
        ax.tick_params(colors=fg)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        for spine in ax.spines.values():
            spine.set_color(fg)

    # ── Scatter plot ──────────────────────────────────────────────────────────
    voters_kwargs = {'marker': '.', 'alpha': 0.25, 's': 12}
    cands_kwargs = {'marker': 'o', 's': 30, 'edgecolors': fg}

    ax_sc.scatter([], [], color=fg, **voters_kwargs, label='Voters')
    ax_sc.scatter([], [], color=fg, **cands_kwargs, label='Candidates')
    ax_sc.legend(loc='lower right', numpoints=1, fontsize='small',
                 labelcolor=legend_fg, facecolor=legend_bg, edgecolor=legend_fg)
    setup_scatter_axis_sigma(ax_sc, voters)

    voronoi_plot_2d_axes(ax_sc, candidates[remaining], line_color=voronoi_color,
                         line_alpha=0.45)

    path_effects = [PathEffects.withStroke(linewidth=3, foreground=stroke_fg)]

    # Color voters by their current first-choice candidate (the `ballots` array),
    # exactly like the IRV animation.  Voters whose first choice is in `eliminated`
    # are shown gray (active_colors handles this since eliminated candidates get gray).
    for cand in range(len(candidates)):
        cand_voters = voters[ballots == cand]
        if len(cand_voters):
            ax_sc.scatter(cand_voters[:, 0], cand_voters[:, 1],
                          color=active_colors[cand], **voters_kwargs)

    ax_sc.scatter(candidates[remaining, 0], candidates[remaining, 1],
                  color=[active_colors[c] for c in remaining], **cands_kwargs)
    for cand in remaining:
        ax_sc.annotate(labels[cand], xy=candidates[cand], xytext=(0, -15),
                       textcoords='offset points', path_effects=path_effects,
                       color=fg)

    # ── Borda/average-rank bar chart ─────────────────────────────────────────
    # y-axis is fixed at [0, n_cands-1] with tick labels: 1 (best) at top, n_cands (worst) at bottom.
    # dead_height = the number of rank slots that no longer exist (= n_cands - n_borda_active).
    # All bars sit on top of the dead zone: bottom=dead_height, so bars appear to rest on the band.
    # bar_segment = n_cands - avg_rank - dead_height  (the visible portion above the dead zone).
    # avg_rank = n_borda_active - borda/n_voters for candidates with active Borda scores.
    dead_height = n_cands - n_borda_active  # 0 in round 0, grows by 1 per round
    n_total_voters = len(voters)
    bar_segments = np.zeros(n_cands)
    avg_ranks = np.zeros(n_cands)
    for c in range(n_cands):
        if c in eliminated and borda_scores[c] == 0:
            bar_segments[c] = 0.0
        else:
            avg_ranks[c] = n_borda_active - borda_scores[c] / n_total_voters
            bar_segments[c] = max(0.0, n_cands - avg_ranks[c] - dead_height)

    bars = ax_bar.bar(range(n_cands), bar_segments, bottom=dead_height,
                      tick_label=list(labels), color=active_colors)
    for c, rect in enumerate(bars):
        top = rect.get_y() + rect.get_height()  # = dead_height + bar_segments[c]
        # Only annotate surviving candidates (not the loser being faded out).
        if bar_segments[c] > 0 and c in remaining:
            # When the bar top is within 2 units of the axis top, the label above
            # would overflow; place it inside the bar instead.
            near_top = top >= n_cands - 2
            ax_bar.annotate(
                f'{avg_ranks[c]:.1f}',
                xy=(rect.get_x() + rect.get_width() / 2, top),
                xytext=(0, -4 if near_top else 3),
                textcoords='offset points',
                ha='center',
                va='top' if near_top else 'bottom',
                color=bg if near_top else fg,
            )

    ax_bar.set_ylim(0, n_cands - 1)
    # Dead zone band: dark shading over the unreachable rank slots.
    if dead_height > 0:
        ax_bar.axhspan(0, dead_height, color=dead_zone_color, zorder=0)
    # Custom y-ticks: rank labels (1 at top = n_cands-1, n_cands at bottom = 0).
    # Suppress labels inside the dead zone.
    tick_vals = list(range(n_cands))
    tick_labels = ['' if v < dead_height else str(n_cands - v) for v in tick_vals]
    ax_bar.set_yticks(tick_vals)
    ax_bar.set_yticklabels(tick_labels)
    ax_bar.set_ylabel('Avg. rank (1=best)')
    ax_bar.grid(True, alpha=0.25, axis='y', color=grid)
    ax_bar.set_axisbelow(True)
    ax_bar.text(0.5, 1.04, frame_title, transform=ax_bar.transAxes,
                ha='center', va='center', color=fg)

    # ── Approval rating bar chart (identical to IRV) ─────────────────────────
    plot_approval_bar(ax_score, approval_pct, labels, active_colors, fg, grid)

    # ── Head-to-head wins (identical to IRV) ─────────────────────────────────
    plot_wins_with_title(ax_wins, wins, active_colors, labels, fg, gap=0.1)

    plt.tight_layout()
    plt.savefig(output_path, facecolor=bg, edgecolor='none')
    plt.close(fig)


def run_tvr_animation(
    voters,
    candidates,
    rankings,
    trace,
    output_dir,
    *,
    palette_name='Bold_10',
    n_cands=None,
    n_voters=None,
    frames_per_transfer=60,
    dark_background=True,
):
    """Render TVR (Baldwin) animation and save frames + GIF to output_dir."""
    n_cands = len(candidates) if n_cands is None else n_cands
    n_voters = len(voters) if n_voters is None else n_voters
    if len(candidates) != n_cands or len(voters) != n_voters:
        raise ValueError(
            f'Election shape does not match n_cands/n_voters: '
            f'got {len(candidates)} candidates, {len(voters)} voters; '
            f'expected n_cands={n_cands}, n_voters={n_voters}.'
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    colors, labels = prepare_palette_and_labels(palette_name, n_cands, dark_background)

    rounds = trace['rounds']
    winner = trace['winner']
    final_two = trace['final_two']

    np.savez(output_dir / 'positions.npz', voters=voters, candidates=candidates)

    utilities = normed_dist_utilities(voters, candidates)
    approval_pct = utilities.mean(axis=0) * 100
    wins = count_wins(ranked_election_to_matrix(rankings))

    n_transfer = frames_per_transfer
    transfer_step_ms = transition_step_ms(n_transfer)
    durations = []

    rng = np.random.default_rng()
    frame = 0
    eliminated = set()
    ballots = rankings[:, 0].copy()

    initial_borda = rounds[0]['borda_before']
    render_frame(
        voters=voters,
        candidates=candidates,
        ballots=ballots,
        borda_scores=initial_borda,
        n_borda_active=n_cands,
        colors=colors,
        labels=labels,
        frame_title='TVR start',
        output_path=output_dir / f'{frame:04d}.png',
        approval_pct=approval_pct,
        wins=wins,
        eliminated=set(),
        dark_background=dark_background,
    )
    durations.append(KEY_FRAME_MS)
    frame += 1

    for round_index, round_data in enumerate(rounds[:-1], start=1):
        loser = round_data['loser']
        eliminated_now = set(eliminated) | {loser}
        n_borda_active_this_round = n_cands - len(eliminated)

        title = f'Round {round_index}: eliminate {candidate_name(loser)}'
        render_frame(
            voters=voters,
            candidates=candidates,
            ballots=ballots,
            borda_scores=round_data['borda_before'],
            n_borda_active=n_borda_active_this_round,
            colors=colors,
            labels=labels,
            frame_title=title,
            output_path=output_dir / f'{frame:04d}.png',
            approval_pct=approval_pct,
            wins=wins,
            eliminated=eliminated_now,
            dark_background=dark_background,
        )
        durations.append(KEY_FRAME_MS)
        frame += 1

        all_voters = np.arange(len(voters))
        rng.shuffle(all_voters)
        per_frame = max(1, ceildiv(len(all_voters), frames_per_transfer))

        running_borda = round_data['borda_before'].copy()
        promoted_per_voter = round_data['promoted_per_voter']

        for step in range(frames_per_transfer):
            lo = step * per_frame
            hi = lo + per_frame
            batch = all_voters[lo:hi]

            for v in batch:
                running_borda[loser] -= len(promoted_per_voter[v])
                for c in promoted_per_voter[v]:
                    running_borda[c] += 1.0

                if ballots[v] == loser:
                    for cand in rankings[v]:
                        if cand not in eliminated_now:
                            ballots[v] = cand
                            break

            render_frame(
                voters=voters,
                candidates=candidates,
                ballots=ballots,
                borda_scores=running_borda,
                n_borda_active=n_borda_active_this_round,
                colors=colors,
                labels=labels,
                frame_title=title,
                output_path=output_dir / f'{frame:04d}.png',
                approval_pct=approval_pct,
                wins=wins,
                eliminated=eliminated_now,
                dark_background=dark_background,
            )
            is_last_transfer = (step == frames_per_transfer - 1)
            durations.append(KEY_FRAME_MS if is_last_transfer else transfer_step_ms)
            frame += 1

        eliminated.add(loser)

    render_frame(
        voters=voters,
        candidates=candidates,
        ballots=ballots,
        borda_scores=rounds[-1]['borda_before'],
        n_borda_active=2,
        colors=colors,
        labels=labels,
        frame_title=f'TVR winner: {candidate_name(winner)}',
        output_path=output_dir / f'{frame:04d}.png',
        approval_pct=approval_pct,
        wins=wins,
        eliminated=set(range(n_cands)) - set(final_two),
        dark_background=dark_background,
    )
    durations.append(KEY_FRAME_MS)

    frame_paths = sorted(output_dir.glob('*.png'))
    images = [Image.open(p) for p in frame_paths]
    gif_path = output_dir / 'collapse_2d_tvr.gif'
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
    )
    for im in images:
        im.close()

    return output_dir


if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('Images') / f'collapse_2d_tvr_{timestamp}_nc{n_cands}_nv{n_voters}'

    if INPUT_POSITIONS is not None:
        data = np.load(INPUT_POSITIONS)
        voters = data['voters']
        candidates = data['candidates']
        utilities = normed_dist_utilities(voters, candidates)
        rankings = np.asarray(honest_rankings(utilities))
        trace = simulate_tvr_rounds(rankings, candidates)
        if trace is None:
            raise RuntimeError(
                f'TVR did not converge to center candidate for the election in '
                f'{INPUT_POSITIONS}.  Try a different positions.npz.'
            )
        print(f'Loaded election from {INPUT_POSITIONS}.')
    else:
        result = find_center_convergent_election(n_voters, n_cands, max_trials, disp=disp)
        if result is None:
            raise RuntimeError(
                'No TVR-convergent center election found. '
                'Increase max_trials or reduce n_cands.'
            )
        trial, voters, candidates, rankings, trace = result
        print(f'Found TVR-convergent election on trial {trial}.')

    print('TVR elimination order:', ' -> '.join(candidate_name(r['loser']) for r in trace['rounds']))
    print('TVR winner:', candidate_name(trace['winner']))

    run_tvr_animation(
        voters, candidates, rankings, trace, output_dir,
        palette_name=palette_name,
        n_cands=n_cands,
        n_voters=n_voters,
        frames_per_transfer=frames_per_transfer,
        dark_background=dark_background,
    )
    print(f'Saved frames and GIF to {output_dir.resolve()}')
