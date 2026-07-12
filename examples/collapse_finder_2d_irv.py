"""Find and animate a two-dimensional IRV center-outward collapse."""

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.methods import ranked_election_to_matrix
from elsim.methods.irv import IRVResult, irv_rounds
from elsim.strategies import honest_rankings

from examples.collapse_2d_shared import (
    KEY_FRAME_MS,
    RESULTS_DIR,
    ceildiv,
    candidate_name,
    create_frame_scaffold,
    get_palette_colors,
    prepare_palette_and_labels,
    sort_candidates_bell_curve,
    transition_step_ms,
)
from examples.collapse_2d_shared import count_wins


def validate_center_outward(result, candidates):
    """Return ``result`` only when it has a strict center-outward collapse."""
    if result is None:
        return None
    if len(result.initially_eliminated):
        return None

    candidates = np.asarray(candidates)
    distances = np.linalg.norm(candidates, axis=1)
    active = set(range(len(candidates)))
    for round_ in result.rounds:
        if round_.eliminated != min(
            active, key=lambda candidate: (distances[candidate], candidate)
        ):
            return None
        active.remove(round_.eliminated)

    expected_final = set(np.argsort(distances)[-2:])
    if set(result.active_candidates) != expected_final:
        return None
    if len(result.active_candidates) != 2:
        return None
    return result


def simulate_irv_rounds(election, candidates):
    """Run the traced IRV count and validate its center-outward elimination."""
    return validate_center_outward(
        irv_rounds(election, tiebreaker=None, stop_at=2),
        candidates,
    )


def find_center_outward_election(n_voters, n_cands, max_trials, disp=1.0):
    """Sample elections until a strict center-outward IRV trace is found."""
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
        trace = simulate_irv_rounds(rankings, candidates)
        if trace is not None:
            return trial, voters, candidates, rankings, trace
    return None


def _plot_votes_panel(axis, tallies, labels, colors, fg, grid, title):
    """Plot first-choice vote percentages in the scaffold's middle panel."""
    n_voters = tallies.sum()
    bars = axis.bar(
        range(len(labels)),
        tallies / n_voters * 100 if n_voters else tallies,
        tick_label=list(labels),
        color=colors,
    )
    for rect in bars:
        height = rect.get_height()
        if height > 0:
            axis.annotate(
                f'{height:.0f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center',
                va='bottom',
                color=fg,
            )
    axis.set_ylim(0, 100)
    axis.set_ylabel('Votes [%]')
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
    tallies,
    favorability_pct,
    wins,
    colors,
    labels,
    frame_title,
    output_path,
    eliminated=None,
    dark_background=True,
):
    """Render one IRV frame using the shared four-panel scaffold."""
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
    _, fg, grid, _ = theme
    _plot_votes_panel(
        axes['middle'],
        tallies,
        labels,
        active_colors,
        fg,
        grid,
        frame_title,
    )
    fig.tight_layout()
    fig.savefig(output_path, facecolor=theme[0], edgecolor='none')
    plt.close(fig)


def _clear_png_frames(output_dir):
    """Remove only numbered PNG frames from a reusable animation directory."""
    for path in Path(output_dir).glob('[0-9][0-9][0-9][0-9].png'):
        path.unlink()


def run_irv_animation(
    voters,
    candidates,
    rankings,
    trace: IRVResult,
    output_dir,
    *,
    palette_name='Bold_10',
    frames_per_transfer=60,
    dark_background=True,
    seed=0,
):
    """Render a traced IRV collapse and save its frames and GIF."""
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
    initially_eliminated = set(map(int, trace.initially_eliminated))
    eliminated = set(initially_eliminated)
    durations = []
    frame = 0

    initial_tallies = np.bincount(choices, minlength=n_cands)
    render_frame(
        voters,
        candidates,
        choices,
        initial_tallies,
        favorability_pct,
        wins,
        colors,
        labels,
        'IRV start',
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
        render_frame(
            voters,
            candidates,
            choices,
            round_.tallies_before,
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

        transferred_voters = round_.transferred_voters.copy()
        transferred_to = round_.transferred_to.copy()
        order = rng.permutation(len(transferred_voters))
        per_frame = max(1, ceildiv(len(order), frames_per_transfer))
        tallies = round_.tallies_before.copy()
        for step in range(frames_per_transfer):
            start = step * per_frame
            stop = min(start + per_frame, len(order))
            for position in order[start:stop]:
                voter = transferred_voters[position]
                target = transferred_to[position]
                tallies[loser] -= 1
                tallies[target] += 1
                choices[voter] = target
            if step == frames_per_transfer - 1:
                tallies = round_.tallies_after.copy()
                choices[transferred_voters] = transferred_to
            render_frame(
                voters,
                candidates,
                choices,
                tallies,
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
        eliminated.add(loser)

    render_frame(
        voters,
        candidates,
        trace.final_choices,
        trace.final_tallies,
        favorability_pct,
        wins,
        colors,
        labels,
        'Final two',
        output_dir / f'{frame:04d}.png',
        eliminated=set(range(n_cands)) - set(trace.active_candidates),
        dark_background=dark_background,
    )
    durations.append(KEY_FRAME_MS)

    frame_paths = sorted(output_dir.glob('[0-9][0-9][0-9][0-9].png'))
    images = [Image.open(path) for path in frame_paths]
    gif_path = output_dir / 'collapse_2d_irv.gif'
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
        f'collapse_2d_irv_{timestamp}_nc{n_cands}_nv{n_voters}'
    )

    result = find_center_outward_election(
        n_voters,
        n_cands,
        max_trials,
        disp=disp,
    )
    if result is None:
        raise RuntimeError(
            'No strict center-outward collapse found. '
            'Increase max_trials or reduce n_cands.'
        )

    trial, voters, candidates, rankings, trace = result
    print(f'Found strict center-outward IRV collapse on trial {trial}.')
    print(
        'Elimination order:',
        ' -> '.join(candidate_name(round_.eliminated) for round_ in trace.rounds),
    )
    print(
        'Final two:',
        ', '.join(candidate_name(candidate) for candidate in trace.active_candidates),
    )
    run_irv_animation(
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
