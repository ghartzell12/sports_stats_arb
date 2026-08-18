"""Comparable-player matching and shape moments."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflfm import comparables
from nflfm.comparables import build_index, describe


def synthetic(seed=2):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(30):
        pos = ("WR", "RB", "QB", "TE")[p % 4]
        level = 6.0 + 0.5 * p
        for season in (2023, 2024):
            for week in range(1, 18):
                rows.append(
                    {
                        "player_id": f"{pos}{p:02d}",
                        "player_display_name": f"{pos} Player {p}",
                        "position": pos,
                        "team": f"T{p % 4}",
                        "season": season,
                        "week": week,
                        "fp": max(0.0, level + rng.normal(0, level * 0.4)),
                        "targets": 2.0 + p * 0.2,
                        "carries": 1.0 + p * 0.1,
                        "attempts": 20.0 if pos == "QB" else 0.0,
                        "receptions": 1.5 + p * 0.15,
                        "receiving_air_yards": 20.0 + p,
                        "passing_tds": 1.0 if pos == "QB" else 0.0,
                        "rushing_tds": 0.2,
                        "receiving_tds": 0.3,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def index():
    return build_index(synthetic())


def test_describe_reports_the_higher_moments():
    symmetric = np.random.default_rng(0).normal(1.0, 0.2, 100_000)
    stats = describe(symmetric)
    assert stats["skew"] == pytest.approx(0.0, abs=0.05)
    assert stats["kurtosis"] == pytest.approx(0.0, abs=0.05)


def test_describe_detects_right_skew():
    right_tailed = np.random.default_rng(0).lognormal(0.0, 0.6, 100_000)
    assert describe(right_tailed)["skew"] > 1.0


def test_describe_handles_a_constant_pool():
    assert describe(np.full(50, 3.0))["cv"] == 0.0


def test_neighbours_are_ordered_by_distance(index):
    query = {"ppg": 12.0, "rec_pg": 5.0, "ypr": 11.0, "td_dependence": 0.3}
    comps = index.neighbours("WR", query, k=5)
    assert len(comps) == 5
    assert comps["distance"].is_monotonic_increasing


def test_closest_comp_is_the_most_similar_level(index):
    """Holding the other features fixed, a higher query matches higher seasons."""
    fixed = {"rec_pg": 5.0, "ypr": 11.0, "td_dependence": 0.3}
    low = index.neighbours("WR", {"ppg": 7.0, **fixed}, k=3)
    high = index.neighbours("WR", {"ppg": 20.0, **fixed}, k=3)
    assert low["ppg"].mean() < high["ppg"].mean()


def test_missing_features_are_reported_by_name(index):
    with pytest.raises(KeyError, match="ypr"):
        index.neighbours("WR", {"ppg": 12.0, "rec_pg": 5.0, "td_dependence": 0.3})


def test_unknown_position_is_rejected(index):
    with pytest.raises(KeyError, match="no comparable seasons"):
        index.neighbours("K", {"ppg": 8.0})


def test_shape_pool_is_scale_free(index):
    query = {"ppg": 12.0, "rec_pg": 5.0, "ypr": 11.0, "td_dependence": 0.3}
    pool = index.shape_pool("WR", query, k=10)
    assert pool.mean() == pytest.approx(1.0, abs=0.15)
    assert pool.size > 0


def test_more_neighbours_widen_the_pool(index):
    query = {"ppg": 12.0, "rec_pg": 5.0, "ypr": 11.0, "td_dependence": 0.3}
    assert index.shape_pool("WR", query, k=10).size > index.shape_pool("WR", query, k=3).size


def test_moments_are_returned_for_a_query(index):
    query = {"ppg": 12.0, "rec_pg": 5.0, "ypr": 11.0, "td_dependence": 0.3}
    stats = index.moments("WR", query, k=10)
    assert {"cv", "skew", "kurtosis"} <= set(stats)


def test_features_exclude_short_seasons():
    frame = synthetic()
    frame = frame[~((frame.player_id == "WR00") & (frame.week > 3))]
    features = comparables.season_features(frame)
    assert "WR00" not in set(features.player_id)
