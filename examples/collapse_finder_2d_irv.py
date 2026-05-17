"""
Find a 2D IRV center-outward collapse and animate transfers by round.

This mirrors the transfer animation style used in elsim2k T2R examples, but
for full IRV: one candidate eliminated per round until two remain.
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
from elsim.methods import irv, ranked_election_to_matrix
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


def simulate_irv_rounds(election, candidates):
    """
    Run IRV until two candidates remain; keep only center-outward collapses.

    Uses :func:`elsim.methods.irv.irv` with ``record_rounds=True``.  Returns None if
    IRV hits an elimination tie or the elimination order is not strict
    center-outward (closest to the origin eliminated each round, farthest two
    at the end).
    """
    dists_to_origin = np.linalg.norm(candidates, axis=1)
    n_cands = len(candidates)

    result = irv(
        election, tiebreaker=None, min_remaining=2, record_rounds=True,
    )
    if result is None:
        return None

    eliminated = set()
    for round_data in result['rounds']:
        remaining = [c for c in range(n_cands) if c not in eliminated]
        expected_loser = min(remaining, key=lambda c: (dists_to_origin[c], c))
        if round_data['loser'] != expected_loser:
            return None
        eliminated.add(round_data['loser'])

    final_two = sorted(np.flatnonzero(~result['eliminated_mask']))
    farthest_two = sorted(np.argsort(dists_to_origin)[-2:])
    if final_two != farthest_two:
        return None

    return {
        'rounds': result['rounds'],
        'final_two': final_two,
        'final_ballots': result['final_ballots'],
        'final_tallies': result['final_tallies'],
        'dists_to_origin': dists_to_origin,
    }


def find_center_outward_election(n_voters, n_cands, max_trials, disp=1.0):
    """Sample random 2D elections until the strict center-outward pattern appears.

    Returns (trial, voters, candidates, rankings, trace).
    rankings is the full honest-rankings ballot matrix (n_voters × n_cands),
    returned so callers don't need to recompute it.
    """
    for trial in range(1, max_trials + 1):
        voters, candidates = normal_electorate(n_voters, n_cands, dims=2, disp=disp)
        candidates[0] = 0.0  # Always one candidate exactly at the center
        candidates = sort_candidates_bell_curve(candidates)
        utilities = normed_dist_utilities(voters, candidates)
        rankings = np.asarray(honest_rankings(utilities))
        trace = simulate_irv_rounds(rankings, candidates)
        if trace is not None:
            return trial, voters, candidates, rankings, trace
    return None


def render_frame(
    voters,
    candidates,
    ballots,
    tallies,
    colors,
    labels,
    frame_title,
    output_path,
    approval_pct,
    wins,
    eliminated=None,
    dark_background=True,
):
    """Render one animation frame in the same visual style as T2R examples.

    approval_pct : ndarray of shape (n_cands,)
        Per-candidate approval rating 0–100, precomputed from utilities.
    wins : list of int
        Head-to-head win count per candidate, precomputed from full rankings.
    """
    if eliminated is None:
        eliminated = set()

    n_cands = len(candidates)
    n_voters = len(voters)
    active_colors = [colors[n] if n not in eliminated else [0.5, 0.5, 0.5] for n in range(n_cands)]

    bg, fg, grid, stroke_fg, legend_bg, legend_fg, voronoi_color, _ = get_theme(dark_background)

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

    voters_kwargs = {'marker': '.', 'alpha': 0.25, 's': 12}
    cands_kwargs = {'marker': 'o', 's': 30, 'edgecolors': fg}

    ax_sc.scatter([], [], color=fg, **voters_kwargs, label='Voters')
    ax_sc.scatter([], [], color=fg, **cands_kwargs, label='Candidates')
    ax_sc.legend(loc='lower right', numpoints=1, fontsize='small', labelcolor=legend_fg,
                 facecolor=legend_bg, edgecolor=legend_fg)
    setup_scatter_axis_sigma(ax_sc, voters)

    remaining = [c for c in range(n_cands) if c not in eliminated]
    voronoi_plot_2d_axes(ax_sc, candidates[remaining], line_color=voronoi_color, line_alpha=0.45)

    path_effects = [PathEffects.withStroke(linewidth=3, foreground=stroke_fg)]

    for cand in range(n_cands):
        cand_voters = voters[ballots == cand]
        if len(cand_voters):
            ax_sc.scatter(cand_voters[:, 0], cand_voters[:, 1], color=active_colors[cand], **voters_kwargs)

    ax_sc.scatter(candidates[remaining, 0], candidates[remaining, 1],
                  color=[active_colors[c] for c in remaining], **cands_kwargs)
    for cand in remaining:
        ax_sc.annotate(labels[cand], xy=candidates[cand], xytext=(0, -15),
                       textcoords='offset points', path_effects=path_effects, color=fg)

    bars = ax_bar.bar(range(n_cands), tallies / n_voters * 100, tick_label=list(labels), color=active_colors)
    for rect in bars:
        height = rect.get_height()
        if height > 0:
            ax_bar.annotate(
                f'{height:.0f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center',
                va='bottom',
                color=fg,
            )

    ax_bar.set_ylim(0, 100)
    ax_bar.set_ylabel('Votes [%]')
    ax_bar.grid(True, alpha=0.25, axis='y', color=grid)
    ax_bar.set_axisbelow(True)
    ax_bar.text(0.5, 1.04, frame_title, transform=ax_bar.transAxes, ha='center', va='center', color=fg)

    plot_approval_bar(ax_score, approval_pct, labels, active_colors, fg, grid)
    plot_wins_with_title(ax_wins, wins, active_colors, labels, fg, gap=0.1)

    plt.tight_layout()
    plt.savefig(output_path, facecolor=bg, edgecolor='none')
    plt.close(fig)


def run_irv_animation(
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
    """Render IRV center-outward animation and save frames + GIF to output_dir.

    output_dir is created if needed. Caller can pass n_cands/n_voters for labels
    (defaults from candidates/voters shape).
    """
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
    final_two = trace['final_two']

    np.savez(output_dir / 'positions.npz', voters=voters, candidates=candidates)

    utilities = normed_dist_utilities(voters, candidates)
    approval_pct = utilities.mean(axis=0) * 100
    wins = count_wins(ranked_election_to_matrix(rankings))

    n_transfer = frames_per_transfer
    transfer_step_ms = transition_step_ms(n_transfer)
    durations = []

    frame = 0
    initial = rounds[0]
    render_frame(
        voters=voters,
        candidates=candidates,
        ballots=initial['ballots_before'],
        tallies=initial['tallies_before'],
        colors=colors,
        labels=labels,
        frame_title='IRV start',
        output_path=output_dir / f'{frame:04d}.png',
        approval_pct=approval_pct,
        wins=wins,
        eliminated=set(),
        dark_background=dark_background,
    )
    durations.append(KEY_FRAME_MS)
    frame += 1

    rng = np.random.default_rng()
    eliminated = set()

    for round_index, round_data in enumerate(rounds, start=1):
        loser = round_data['loser']
        eliminated_now = set(eliminated)
        eliminated_now.add(loser)

        title = f'Round {round_index}: eliminate {candidate_name(loser)}'
        render_frame(
            voters=voters,
            candidates=candidates,
            ballots=round_data['ballots_before'],
            tallies=round_data['tallies_before'],
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

        ballots = round_data['ballots_before'].copy()
        affected = round_data['affected_voters'].copy()
        rng.shuffle(affected)
        per_frame = max(1, ceildiv(len(affected), frames_per_transfer))

        for step in range(frames_per_transfer):
            lo = step * per_frame
            hi = lo + per_frame
            changing = affected[lo:hi]
            ballots[changing] = round_data['ballots_after'][changing]
            tallies = np.bincount(ballots, minlength=n_cands)

            render_frame(
                voters=voters,
                candidates=candidates,
                ballots=ballots,
                tallies=tallies,
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
        ballots=trace['final_ballots'],
        tallies=trace['final_tallies'],
        colors=colors,
        labels=labels,
        frame_title=f'Final two: {candidate_name(final_two[0])} vs {candidate_name(final_two[1])}',
        output_path=output_dir / f'{frame:04d}.png',
        approval_pct=approval_pct,
        wins=wins,
        eliminated=set(range(n_cands)) - set(final_two),
        dark_background=dark_background,
    )
    durations.append(KEY_FRAME_MS)

    frame_paths = sorted(output_dir.glob('*.png'))
    images = [Image.open(p) for p in frame_paths]
    gif_path = output_dir / 'collapse_2d_irv.gif'
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
    output_dir = Path('Images') / f'collapse_2d_irv_{timestamp}_nc{n_cands}_nv{n_voters}'

    result = find_center_outward_election(n_voters, n_cands, max_trials, disp=disp)
    if result is None:
        raise RuntimeError('No strict center-outward collapse found. Increase max_trials or reduce n_cands.')

    trial, voters, candidates, rankings, trace = result
    print(f'Found strict center-outward IRV collapse on trial {trial}.')
    print('Elimination order:', ' -> '.join(candidate_name(r['loser']) for r in trace['rounds']))
    print('Final two:', candidate_name(trace['final_two'][0]), candidate_name(trace['final_two'][1]))

    run_irv_animation(
        voters, candidates, rankings, trace, output_dir,
        palette_name=palette_name,
        n_cands=n_cands,
        n_voters=n_voters,
        frames_per_transfer=frames_per_transfer,
        dark_background=dark_background,
    )
    print(f'Saved frames and GIF to {output_dir.resolve()}')
