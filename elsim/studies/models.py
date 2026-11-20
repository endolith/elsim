"""
Electorate models used by :class:`elsim.studies.Study`.

A model knows how to draw one or more electorates for a scenario.  Spatial
models reuse a candidate set across electorates; random-society models draw
fresh utilities for every electorate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from elsim.elections import (impartial_culture as _impartial_culture,
                             normal_electorate, normed_dist_utilities,
                             random_utilities)


@dataclass(frozen=True)
class Election:
    """One simulated electorate."""

    voters: np.ndarray | None = None
    candidates: np.ndarray | None = None
    utilities: np.ndarray | None = None
    rankings: np.ndarray | None = None


class SpatialNormal:
    """Multivariate-normal spatial electorate, with reusable candidate sets."""

    def __init__(self, **defaults: Any) -> None:
        self.defaults = defaults

    def _params(self, scenario: dict[str, Any]) -> dict[str, Any]:
        params = {**self.defaults, **scenario}
        params.setdefault('dims', 2)
        params.setdefault('corr', 0.0)
        params.setdefault('disp', 1.0)
        return params

    def draw_set(self, scenario: dict[str, Any], n_electorates: int,
                 rng: np.random.Generator) -> list[Election]:
        p = self._params(scenario)
        _, candidates = normal_electorate(
            0, p['n_cands'], dims=p['dims'], corr=p['corr'],
            disp=p['disp'], random_state=rng)
        out = []
        for _ in range(n_electorates):
            voters, _ = normal_electorate(
                p['n_voters'], p['n_cands'], dims=p['dims'], corr=p['corr'],
                disp=p['disp'], random_state=rng)
            utilities = normed_dist_utilities(voters, candidates)
            out.append(Election(voters=voters, candidates=candidates,
                                utilities=utilities))
        return out

    def draw(self, scenario: dict[str, Any],
             rng: np.random.Generator) -> Election:
        return self.draw_set(scenario, 1, rng)[0]


class RandomSociety:
    """Independent uniform utilities for every voter-candidate pair."""

    def __init__(self, **defaults: Any) -> None:
        self.defaults = defaults

    def _params(self, scenario: dict[str, Any]) -> dict[str, Any]:
        return {**self.defaults, **scenario}

    def draw(self, scenario: dict[str, Any],
             rng: np.random.Generator) -> Election:
        p = self._params(scenario)
        utilities = random_utilities(p['n_voters'], p['n_cands'],
                                     random_state=rng)
        return Election(utilities=utilities)

    def draw_set(self, scenario: dict[str, Any], n_electorates: int,
                 rng: np.random.Generator) -> list[Election]:
        return [self.draw(scenario, rng) for _ in range(n_electorates)]


class ImpartialCulture:
    """Uniform random rankings, with no cardinal utilities."""

    def __init__(self, **defaults: Any) -> None:
        self.defaults = defaults

    def _params(self, scenario: dict[str, Any]) -> dict[str, Any]:
        return {**self.defaults, **scenario}

    def draw(self, scenario: dict[str, Any],
             rng: np.random.Generator) -> Election:
        p = self._params(scenario)
        rankings = _impartial_culture(p['n_voters'], p['n_cands'],
                                      random_state=rng)
        return Election(rankings=rankings)

    def draw_set(self, scenario: dict[str, Any], n_electorates: int,
                 rng: np.random.Generator) -> list[Election]:
        return [self.draw(scenario, rng) for _ in range(n_electorates)]


def spatial_normal(**defaults: Any) -> SpatialNormal:
    """Spatial normal model; scenario keys override ``defaults``."""
    return SpatialNormal(**defaults)


def random_society(**defaults: Any) -> RandomSociety:
    """Random-society / impartial-culture utilities."""
    return RandomSociety(**defaults)


def impartial_culture(**defaults: Any) -> ImpartialCulture:
    """Uniform random rankings without cardinal utilities."""
    return ImpartialCulture(**defaults)


__all__ = [
    'Election',
    'ImpartialCulture',
    'RandomSociety',
    'SpatialNormal',
    'impartial_culture',
    'random_society',
    'spatial_normal',
]
