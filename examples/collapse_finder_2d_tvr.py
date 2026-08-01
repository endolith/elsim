"""Find and animate a 2D Baldwin/TVR Condorcet-winner election.

Baldwin satisfies the Condorcet criterion: a Condorcet winner has an
above-average Borda score and is never eliminated. Total Vote Runoff uses the
same elimination rule and shares this Condorcet consistency.
"""

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.methods import condorcet_from_matrix, ranked_election_to_matrix
from elsim.methods.baldwin import BaldwinResult, baldwin_rounds
from elsim.strategies import honest_rankings

from examples.collapse_2d_shared import (
    KEY_FRAME_MS,
    RESULTS_DIR,
    ceildiv,
    candidate_name,
    create_frame_scaffold,
    count_wins,
    prepare_palette_and_labels,
    sort_candidates_bell_curve,
    transition_step_ms,
)


def validate_condorcet_winner(result, candidates, election):
    """Keep traces where Baldwin elects the verified center Condorcet winner."""
    if result is None:
        return None
    candidates = np.asarray(candidates)
    condorcet_winner = condorcet_from_matrix(
        ranked_election_to_matrix(election)
    )
    if condorcet_winner is None:
        return None
    center_candidate = int(np.argmin(np.linalg.norm(candidates, axis=1)))
    if condorcet_winner != center_candidate or result.winner != condorcet_winner:
        return None
    return result


def simulate_tvr_rounds(election, candidates):
    """Run Baldwin and verify it elects the center Condorcet winner."""
    return validate_condorcet_winner(
        baldwin_rounds(election, tiebreaker=None),
        candidates,
        election,
    )


def find_center_convergent_election(n_voters, n_cands, max_trials, disp=1.0):
    """Sample elections whose center candidate is the Condorcet winner."""
    for trial in range(1, max_trials + 1):
        voters, candidates = normal_electorate(
            n_voters,
            n_cands,
            dims=2,
            disp=disp,
        )
        candidates[0] = 0.0
        candidates = sort_candidates_bell_curve(candidates)
        utilities = normed_dist_utilities(voters, candidates)
        rankings = np.asarray(honest_rankings(utilities))
        trace = simulate_tvr_rounds(rankings, candidates)
        if trace is not None:
            return trial, voters, candidates, rankings, trace
    return None


def average_ranks_from_borda(borda_scores, n_active, n_voters):
    """Convert one-based Borda scores to one-based average ranks."""
    return (n_active + 1) - np.asarray(borda_scores) / n_voters


def _plot_borda_panel(
    axis,
    borda_scores,
    n_active,
    n_cands,
    n_voters,
    labels,
    colors,
    fg,
    grid,
    dead_zone_color,
    title,
):
    """Plot average ranks with a dead zone for eliminated rank slots."""
    dead_height = n_cands - n_active
    average_scores = np.asarray(borda_scores, dtype=float) / n_voters
    average_ranks = average_ranks_from_borda(
        borda_scores,
        n_active,
        n_voters,
    )
    bar_segments = np.maximum(average_scores, 0)
    for candidate in range(n_cands):
        if borda_scores[candidate] <= 0:
            bar_segments[candidate] = 0

    bars = axis.bar(
        range(n_cands),
        bar_segments,
        bottom=dead_height,
        tick_label=list(labels),
        color=colors,
    )
    for candidate, rect in enumerate(bars):
        if bar_segments[candidate] > 0 and borda_scores[candidate] > 0:
            axis.annotate(
                f'{average_ranks[candidate]:.1f}',
                xy=(rect.get_x() + rect.get_width() / 2, rect.get_y() + rect.get_height()),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center',
                va='bottom',
                color=fg,
            )

    axis.set_ylim(0, n_cands)
    if dead_height:
        axis.axhspan(0, dead_height, color=dead_zone_color, zorder=0)
    tick_values = list(range(n_cands + 1))
    tick_labels = [
        '' if value <= dead_height else str(n_cands + 1 - value)
        for value in tick_values
    ]
    axis.set_yticks(tick_values)
    axis.set_yticklabels(tick_labels)
    axis.set_ylabel('Avg. rank (1=best)')
    axis.grid(True, alpha=0.25, axis='y', color=grid)
    axis.set_axisbelow(True)
    axis.text(
        0.5,
        1.04,
        title,
        transform=axis.transAxes,
        ha='center',
        va='center',
        color=fg,
    )


def render_frame(
    voters,
    candidates,
    ballots,
    borda_scores,
    n_active,
    favorability_pct,
    wins,
    colors,
    labels,
    frame_title,
    output_path,
    eliminated=None,
    dark_background=True,
):
    """Render one Baldwin frame using the shared panel scaffold."""
    fig, axes, active_colors, theme = create_frame_scaffold(
        voters,
        candidates,
        ballots,
        favorability_pct,
        wins,
        colors,
        labels,
        eliminated=eliminated,
        dark_background=dark_background,
    )
    _, fg, grid, dead_zone_color = theme
    _plot_borda_panel(
        axes['middle'],
        borda_scores,
        n_active,
        len(candidates),
        len(voters),
        labels,
        active_colors,
        fg,
        grid,
        dead_zone_color,
        frame_title,
    )
    fig.tight_layout()
    fig.savefig(output_path, facecolor=theme[0], edgecolor='none')
    plt.close(fig)


def _clear_png_frames(output_dir):
    """Remove only numbered PNG frames from a reusable animation directory."""
    for path in Path(output_dir).glob('[0-9][0-9][0-9][0-9].png'):
        path.unlink()


def run_tvr_animation(
    voters,
    candidates,
    rankings,
    trace: BaldwinResult,
    output_dir,
    *,
    palette_name='Bold_10',
    frames_per_transfer=60,
    dark_background=True,
    seed=0,
):
    """Render a full Baldwin elimination trace and save its GIF."""
    voters = np.asarray(voters)
    candidates = np.asarray(candidates)
    rankings = np.asarray(rankings)
    n_voters = len(voters)
    n_cands = len(candidates)
    if rankings.shape != (n_voters, n_cands):
        raise ValueError(
            f'Rankings shape {rankings.shape} does not match '
            f'{n_voters} voters and {n_cands} candidates.'
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_png_frames(output_dir)
    colors, labels = prepare_palette_and_labels(
        palette_name,
        n_cands,
        dark_background,
    )
    utilities = normed_dist_utilities(voters, candidates)
    favorability_pct = utilities.mean(axis=0) * 100
    wins = count_wins(ranked_election_to_matrix(rankings))
    choices = rankings[:, 0].copy()
    eliminated = set()
    durations = []
    frame = 0
    first_round = trace.rounds[0]
    render_frame(
        voters,
        candidates,
        choices,
        first_round.borda_before,
        n_cands,
        favorability_pct,
        wins,
        colors,
        labels,
        'Baldwin start',
        output_dir / f'{frame:04d}.png',
        eliminated=eliminated,
        dark_background=dark_background,
    )
    durations.append(KEY_FRAME_MS)
    frame += 1

    rng = np.random.default_rng(seed)
    for round_index, round_ in enumerate(trace.rounds, start=1):
        loser = int(round_.eliminated)
        eliminated_now = eliminated | {loser}
        n_active = n_cands - len(eliminated)
        render_frame(
            voters,
            candidates,
            choices,
            round_.borda_before,
            n_active,
            favorability_pct,
            wins,
            colors,
            labels,
            f'Round {round_index}: eliminate {candidate_name(loser)}',
            output_dir / f'{frame:04d}.png',
            eliminated=eliminated_now,
            dark_background=dark_background,
        )
        durations.append(KEY_FRAME_MS)
        frame += 1

        higher_ranked = round_.higher_ranked_candidates
        transferred_voters = round_.transferred_voters
        transferred_to = round_.transferred_to
        order = rng.permutation(n_voters)
        per_frame = max(1, ceildiv(n_voters, frames_per_transfer))
        running_borda = round_.borda_before.copy()
        running_choices = choices.copy()
        for step in range(frames_per_transfer):
            start = step * per_frame
            stop = min(start + per_frame, n_voters)
            for voter in order[start:stop]:
                higher = higher_ranked[voter]
                running_borda[higher] -= 1
                running_borda[loser] -= n_active - len(higher)
                transferred = np.flatnonzero(transferred_voters == voter)
                if len(transferred):
                    running_choices[voter] = transferred_to[transferred[0]]
            if step == frames_per_transfer - 1:
                if not np.array_equal(running_borda, round_.borda_after):
                    raise AssertionError('Baldwin Borda transition did not match trace.')
                running_borda = round_.borda_after.copy()
                running_choices[transferred_voters] = transferred_to
            render_frame(
                voters,
                candidates,
                running_choices,
                running_borda,
                n_active,
                favorability_pct,
                wins,
                colors,
                labels,
                f'Round {round_index}: eliminate {candidate_name(loser)}',
                output_dir / f'{frame:04d}.png',
                eliminated=eliminated_now,
                dark_background=dark_background,
            )
            durations.append(
                KEY_FRAME_MS
                if step == frames_per_transfer - 1
                else transition_step_ms(frames_per_transfer)
            )
            frame += 1
        choices = running_choices
        eliminated.add(loser)

    render_frame(
        voters,
        candidates,
        trace.final_choices,
        trace.rounds[-1].borda_after,
        1,
        favorability_pct,
        wins,
        colors,
        labels,
        f'Baldwin winner: {candidate_name(trace.winner)}',
        output_dir / f'{frame:04d}.png',
        eliminated=set(range(n_cands)) - {int(trace.winner)},
        dark_background=dark_background,
    )
    durations.append(KEY_FRAME_MS)

    frame_paths = sorted(output_dir.glob('[0-9][0-9][0-9][0-9].png'))
    images = [Image.open(path) for path in frame_paths]
    gif_path = output_dir / 'collapse_2d_tvr.gif'
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
    )
    for image in images:
        image.close()
    return output_dir


if __name__ == '__main__':
    n_voters = 5000
    n_cands = 9
    max_trials = 100_000
    frames_per_transfer = 60
    disp = 0.5
    palette_name = 'Bold_10'
    dark_background = True
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = RESULTS_DIR / (
        f'collapse_2d_tvr_{timestamp}_nc{n_cands}_nv{n_voters}'
    )

    result = find_center_convergent_election(
        n_voters,
        n_cands,
        max_trials,
        disp=disp,
    )
    if result is None:
        raise RuntimeError(
            'No Baldwin election with a center Condorcet winner found. '
            'Increase max_trials or reduce n_cands.'
        )

    trial, voters, candidates, rankings, trace = result
    print(f'Found Baldwin election with a center Condorcet winner on trial {trial}.')
    print(
        'Elimination order:',
        ' -> '.join(candidate_name(round_.eliminated) for round_ in trace.rounds),
    )
    print('Baldwin winner:', candidate_name(trace.winner))
    run_tvr_animation(
        voters,
        candidates,
        rankings,
        trace,
        output_dir,
        palette_name=palette_name,
        frames_per_transfer=frames_per_transfer,
        dark_background=dark_background,
    )
    print(f'Saved frames and GIF to {output_dir.resolve()}')
