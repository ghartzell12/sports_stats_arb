"""Weekly shape library."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflfm import distributions
from nflfm.distributions import ShapeLibrary, build_shape_library


def synthetic(seasons=(2023, 2024), players_per_pos=6, weeks=17, seed=1):
    """Weekly data where each player has a stable level plus noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for pos in ("QB", "RB", "WR", "TE"):
        for p in range(players_per_pos):
            for season in seasons:
                level = 8.0 + 2.0 * p
                for week in range(1, weeks + 1):
                    rows.append(
                        {
                            "player_id": f"{pos}{p:02d}",
                            "position": pos,
                            "team": f"T{p % 3}",
                            "season": season,
                            "week": week,
                            "fp": max(0.0, level + rng.normal(0, level * 0.4)),
                        }
                    )
    return pd.DataFrame(rows)


@pytest.fixture
def library():
    return build_shape_library(synthetic(), min_weeks=50)


def test_shapes_are_scale_free(library):
    """Dividing by each player's own mean should centre every pool near 1."""
    for pool in library.shapes.values():
        assert pool.mean() == pytest.approx(1.0, abs=0.05)


def test_sampling_recovers_the_requested_level(library):
    rng = np.random.default_rng(0)
    pos, tier = next(iter(library.shapes))
    draws = library.sample(pos, tier, ppg=15.0, size=(2000, 17), rng=rng,
                           include_missed_games=False)
    assert draws.mean() == pytest.approx(15.0, rel=0.05)
    assert draws.shape == (2000, 17)


def test_missed_games_lower_the_mean_and_add_zeroes():
    lib = ShapeLibrary(shapes={("WR", 1): np.array([1.0])}, played_rate={("WR", 1): 0.8})
    rng = np.random.default_rng(0)
    draws = lib.sample("WR", 1, ppg=10.0, size=(5000, 17), rng=rng)
    assert (draws == 0.0).mean() == pytest.approx(0.2, abs=0.02)
    assert draws.mean() == pytest.approx(8.0, rel=0.05)


def test_unknown_tier_falls_back_within_position(library):
    assert library.key_for("WR", 99)[0] == "WR"


def test_unknown_position_is_an_error(library):
    with pytest.raises(KeyError, match="no shapes for position"):
        library.key_for("K", 1)


def test_low_volume_seasons_are_excluded():
    """A 2-game season would produce a meaningless shape."""
    frame = synthetic()
    short = frame[(frame["player_id"] == "WR00") & (frame["week"] <= 2)]
    rest = frame[frame["player_id"] != "WR00"]
    lib = build_shape_library(pd.concat([rest, short], ignore_index=True), min_weeks=50)
    assert all(np.isfinite(pool).all() for pool in lib.shapes.values())


def test_impossible_minimums_raise_rather_than_return_empty():
    with pytest.raises(ValueError, match="no player-seasons met the minimums"):
        build_shape_library(synthetic(), min_ppg=10_000.0, min_weeks=50)


def test_summary_reports_dispersion_per_key(library):
    summary = library.summary()
    assert {"position", "tier", "cv", "p_dud", "p_spike"} <= set(summary.columns)
    assert (summary["cv"] > 0).all()
