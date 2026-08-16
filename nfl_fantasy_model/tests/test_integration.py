"""End-to-end checks against the live nflverse feeds.

Skipped by default because they hit the network and download ~10MB per season.
Run with ``NFLFM_NETWORK_TESTS=1 pytest tests/test_integration.py``.
"""

from __future__ import annotations

import os

import pytest

from nflfm import scoring
from nflfm.data import load_weekly_stats
from nflfm.features import build_feature_frame
from nflfm.models import RollingMeanProjector, backtest

pytestmark = pytest.mark.skipif(
    os.environ.get("NFLFM_NETWORK_TESTS") != "1",
    reason="set NFLFM_NETWORK_TESTS=1 to run tests that download nflverse data",
)

SEASONS = [2023, 2024]


@pytest.fixture(scope="module")
def weekly():
    return load_weekly_stats(SEASONS)


def test_weekly_feed_has_the_expected_shape(weekly):
    assert set(weekly["season"]) == set(SEASONS)
    assert set(weekly["position"]) <= {"QB", "RB", "WR", "TE"}
    assert weekly["week"].between(1, 22).all()
    assert not weekly.duplicated(["player_id", "season", "week"]).any()


@pytest.mark.parametrize(
    "rules,reference",
    [(scoring.PPR, "fantasy_points_ppr"), (scoring.STANDARD, "fantasy_points")],
)
def test_our_scoring_reproduces_nflverses(weekly, rules, reference):
    """The strongest available check on the rules: nflverse computes the same
    two rulesets independently, so ours must match theirs row for row."""
    ours = scoring.score_frame(weekly, rules)
    assert (ours - weekly[reference]).abs().max() < 1e-6


def test_2019_resolves_through_the_fallback_url():
    """2019 is absent from the intermediate upstream layout — it must still load."""
    frame = load_weekly_stats([2019])
    assert len(frame) > 5000


def test_seasons_since_2025_are_available():
    """Guards against the release tag moving again and silently going stale."""
    frame = load_weekly_stats([2025])
    assert frame["week"].max() >= 17


def test_baseline_backtest_beats_a_naive_constant(weekly):
    scored = scoring.add_fantasy_points(weekly, scoring.PPR)
    features = build_feature_frame(scored)
    table = backtest(RollingMeanProjector(5), features, min_train_seasons=1)

    assert not table.empty
    # A 5-game average should land well inside 8 PPR points and rank players
    # far better than chance.
    assert (table["mae"] < 8).all()
    assert (table["spearman"] > 0.4).all()
