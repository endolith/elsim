"""
Shared frame rendering and GIF sequencing for the 2D collapse animations.

The IRV and TVR animations are identical apart from the top-right "score"
panel (vote share for IRV, average rank for TVR) and how that panel's
numbers evolve while voters are transferred.  Those two concerns are
captured by the `ScorePanel` strategy; everything else lives in
`run_animation`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as PathEffects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from elsim.methods import ranked_election_to_matrix  # noqa: E402

from collapse_2d_shared import (  # noqa: E402
    GRAY, candidate_name, ceildiv, get_theme, plot_approval_bar,
    plot_wins_with_title, prepare_palette, setup_scatter_axis_sigma,
    voronoi_plot_2d_axes,
)
from collapse_utils import count_wins  # noqa: E402


# ── Data carried between the finder scripts and the animator ──────────

@dataclass(frozen=True)
class RunoffTrace:
    """A run of a sequential-elimination method down to the final two."""
    rounds: list                 # per-round dicts from *_rounds(record_rounds)
    final_two: list              # the two survivors, sorted
    final_ballots: np.ndarray    # per-voter top choice among the final two
    final_tallies: np.ndarray    # first-preference tallies among final two
    winner: int

    @property
    def elimination_order(self):
        return ' -> '.join(candidate_name(r['loser']) for r in self.rounds)


@dataclass
class FrameContext:
    """Static per-animation data every frame needs."""
    election: object             # collapse_2d_shared.Election
    colors: list
    labels: str
    theme: object                # collapse_2d_shared.Theme
    approval_pct: np.ndarray
    wins: list

    @property
    def n_cands(self):
        return self.election.n_cands

    @property
    def n_voters(self):
        return self.election.n_voters

    def active_colors(self, eliminated):
        return [GRAY if c in eliminated else self.colors[c]
                for c in range(self.n_cands)]


@dataclass
class Frame:
    """Everything that changes from one frame to the next."""
    title: str
    ballots: np.ndarray          # per-voter current first choice
    eliminated: set              # candidates drawn gray / removed from map
    panel_state: object          # owned and interpreted by the ScorePanel
    duration_ms: int


# ── Strategy for the top-right panel ──────────────────────────────────

class ScorePanel(ABC):
    """
    How a method's score panel looks and how it animates a transfer.

    The animator owns the ballots and the eliminated set; the panel only
    owns its *panel state* (e.g. tallies or running Borda scores).
    """

    @abstractmethod
    def start_state(self, trace, ctx):
        """Panel state for the opening frame."""

    @abstractmethod
    def round_state(self, round_data, n_eliminated_before, ctx):
        """Mutable panel state at the key frame that announces a loser."""

    @abstractmethod
    def transfer_voters(self, round_data, ctx):
        """Voter indices whose transfer is animated this round."""

    @abstractmethod
    def advance(self, state, round_data, voter_batch, ballots):
        """Update ``state`` in place after ``voter_batch`` transferred."""

    @abstractmethod
    def final_state(self, trace, ctx):
        """Panel state for the closing frame."""

    @abstractmethod
    def draw(self, ax, state, frame, ctx):
        """Draw the panel (title is added by the animator)."""


# ── Rendering ─────────────────────────────────────────────────────────

def _draw_map(ax, frame, ctx):
    """Voter/candidate scatter with Voronoi cells of remaining candidates."""
    election, theme = ctx.election, ctx.theme
    colors = ctx.active_colors(frame.eliminated)
    remaining = [c for c in range(ctx.n_cands) if c not in frame.eliminated]

    voters_kwargs = {'marker': '.', 'alpha': 0.25, 's': 12}
    cands_kwargs = {'marker': 'o', 's': 30, 'edgecolors': theme.fg}

    ax.scatter([], [], color=theme.fg, **voters_kwargs, label='Voters')
    ax.scatter([], [], color=theme.fg, **cands_kwargs, label='Candidates')
    ax.legend(loc='lower right', numpoints=1, fontsize='small',
              labelcolor=theme.legend_fg, facecolor=theme.legend_bg,
              edgecolor=theme.legend_fg)
    setup_scatter_axis_sigma(ax, election.voters)
    voronoi_plot_2d_axes(ax, election.candidates[remaining],
                         line_color=theme.voronoi_color, line_alpha=0.45)

    for cand in range(ctx.n_cands):
        cand_voters = election.voters[frame.ballots == cand]
        if len(cand_voters):
            ax.scatter(cand_voters[:, 0], cand_voters[:, 1],
                       color=colors[cand], **voters_kwargs)

    ax.scatter(election.candidates[remaining, 0],
               election.candidates[remaining, 1],
               color=[colors[c] for c in remaining], **cands_kwargs)
    halo = [PathEffects.withStroke(linewidth=3, foreground=theme.stroke_fg)]
    for cand in remaining:
        ax.annotate(ctx.labels[cand], xy=election.candidates[cand],
                    xytext=(0, -15), textcoords='offset points',
                    path_effects=halo, color=theme.fg)


def render_frame(frame, panel, ctx, output_path):
    """Render one animation frame (map + three side panels) to a PNG."""
    theme = ctx.theme
    fig = plt.figure(figsize=(9, 7.5), facecolor=theme.bg)
    ax_map = plt.subplot2grid((6, 3), (0, 0), colspan=2, rowspan=6)
    ax_score = plt.subplot2grid((6, 3), (0, 2), rowspan=2)
    ax_approval = plt.subplot2grid((6, 3), (2, 2), rowspan=2)
    ax_wins = plt.subplot2grid((6, 3), (4, 2), rowspan=2)
    theme.style_axes(ax_map, ax_score, ax_approval, ax_wins)

    colors = ctx.active_colors(frame.eliminated)
    _draw_map(ax_map, frame, ctx)
    panel.draw(ax_score, frame.panel_state, frame, ctx)
    ax_score.text(0.5, 1.04, frame.title, transform=ax_score.transAxes,
                  ha='center', va='center', color=theme.fg)
    plot_approval_bar(ax_approval, ctx.approval_pct, ctx.labels, colors,
                      theme)
    plot_wins_with_title(ax_wins, ctx.wins, colors, ctx.labels, theme,
                         gap=0.1)

    plt.tight_layout()
    plt.savefig(output_path, facecolor=theme.bg, edgecolor='none')
    plt.close(fig)


class FrameWriter:
    """Numbers frames, remembers their durations and assembles the GIF."""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths = []
        self.durations = []

    def write(self, frame, panel, ctx):
        path = self.output_dir / f'{len(self.paths):04d}.png'
        render_frame(frame, panel, ctx, path)
        self.paths.append(path)
        self.durations.append(frame.duration_ms)

    def save_gif(self, filename):
        # Only frames written by *this* run: a reused directory may still
        # hold stale PNGs from a previous run.
        images = [Image.open(p) for p in self.paths]
        gif_path = self.output_dir / filename
        images[0].save(gif_path, save_all=True, append_images=images[1:],
                       duration=self.durations, loop=0)
        for im in images:
            im.close()
        return gif_path


# ── Driver ────────────────────────────────────────────────────────────

def run_animation(election, trace, panel, output_dir, config, *,
                  method_name, gif_name):
    """
    Render start → (eliminate, transfer…)×rounds → final frames and a GIF.

    Parameters
    ----------
    election : collapse_2d_shared.Election
    trace : RunoffTrace
    panel : ScorePanel
        Method-specific top-right panel.
    output_dir : path-like
    config : collapse_2d_shared.AnimationConfig
    method_name : str
        Used in the opening frame title, e.g. ``'IRV'``.
    gif_name : str
        File name of the GIF inside ``output_dir``.
    """
    if election.n_cands != config.n_cands or \
            election.n_voters != config.n_voters:
        raise ValueError(
            f'Election shape does not match config: got {election.n_cands} '
            f'candidates, {election.n_voters} voters; expected '
            f'n_cands={config.n_cands}, n_voters={config.n_voters}.')

    ctx = FrameContext(
        election=election,
        colors=prepare_palette(config.palette_name, config.n_cands,
                               config.dark_background),
        labels=config.labels(),
        theme=get_theme(config.dark_background),
        approval_pct=election.approval_pct(),
        wins=count_wins(ranked_election_to_matrix(election.rankings)),
    )
    writer = FrameWriter(output_dir)
    election.save_positions(writer.output_dir / 'positions.npz')

    rng = np.random.default_rng(config.seed)
    key_ms = config.key_frame_ms
    step_ms = config.transition_step_ms()
    n_steps = config.frames_per_transfer

    start_ballots = (trace.rounds[0]['ballots_before'] if trace.rounds
                     else trace.final_ballots)
    writer.write(Frame(f'{method_name} start', start_ballots, set(),
                       panel.start_state(trace, ctx), key_ms), panel, ctx)

    eliminated = set()
    for round_index, round_data in enumerate(trace.rounds, start=1):
        loser = round_data['loser']
        eliminated_now = eliminated | {loser}
        title = f'Round {round_index}: eliminate {candidate_name(loser)}'

        ballots = round_data['ballots_before'].copy()
        state = panel.round_state(round_data, len(eliminated), ctx)
        writer.write(Frame(title, ballots, eliminated_now, state, key_ms),
                     panel, ctx)

        voters = panel.transfer_voters(round_data, ctx)
        rng.shuffle(voters)
        per_step = max(1, ceildiv(len(voters), n_steps))
        for step in range(n_steps):
            batch = voters[step * per_step:(step + 1) * per_step]
            # Voters not affected by this elimination have identical
            # before/after ballots, so this is safe for every panel.
            ballots[batch] = round_data['ballots_after'][batch]
            panel.advance(state, round_data, batch, ballots)
            is_last = step == n_steps - 1
            writer.write(Frame(title, ballots, eliminated_now, state,
                               key_ms if is_last else step_ms), panel, ctx)

        eliminated.add(loser)

    a, b = (candidate_name(c) for c in trace.final_two)
    writer.write(Frame(
        f'{method_name} winner: {candidate_name(trace.winner)}  ({a} vs {b})',
        trace.final_ballots, set(range(ctx.n_cands)) - set(trace.final_two),
        panel.final_state(trace, ctx), key_ms), panel, ctx)

    return writer.save_gif(gif_name)


def trace_to_final_two(result):
    """
    Build a `RunoffTrace` from a ``*_rounds(min_remaining=2)`` result.

    Returns ``None`` if the count did not reach exactly two candidates
    (e.g. an early majority left three or more standing), because the
    animation assumes one full elimination per remaining candidate.
    """
    final_two = sorted(int(c) for c in np.flatnonzero(~result['eliminated_mask']))
    if len(final_two) != 2:
        return None
    winner = result['winner']
    if winner is None:  # head-to-head between the final two
        winner = int(np.argmax(result['final_tallies']))
    return RunoffTrace(rounds=result['rounds'], final_two=final_two,
                       final_ballots=result['final_ballots'],
                       final_tallies=result['final_tallies'], winner=winner)
