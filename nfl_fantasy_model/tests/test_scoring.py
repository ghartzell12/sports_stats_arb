"""Scoring rules, checked against hand-computed stat lines."""

from __future__ import annotations

import pandas as pd
import pytest

from nflfm import scoring
from nflfm.scoring import ScoringRules

# A 300/3/1 passing line with a rushing score.
QB_LINE = {
    "position": "QB",
    "passing_yards": 300,
    "passing_tds": 3,
    "passing_interceptions": 1,
    "rushing_yards": 20,
    "rushing_tds": 1,
}

# 8 catches, 100 yards, a TD, and a lost fumble.
WR_LINE = {
    "position": "WR",
    "receptions": 8,
    "receiving_yards": 100,
    "receiving_tds": 1,
    "receiving_fumbles_lost": 1,
}


def test_qb_line_is_scoring_agnostic_to_receptions():
    # 12 + 12 - 2 + 2 + 6 = 30 under every preset, since QBs don't catch passes.
    for rules in (scoring.STANDARD, scoring.HALF_PPR, scoring.PPR):
        assert rules.score_stats(QB_LINE) == pytest.approx(30.0)


@pytest.mark.parametrize(
    "rules,expected",
    [
        (scoring.STANDARD, 14.0),  # 10 + 6 - 2
        (scoring.HALF_PPR, 18.0),  # + 8 * 0.5
        (scoring.PPR, 22.0),  # + 8 * 1.0
    ],
)
def test_receptions_scale_with_preset(rules, expected):
    assert rules.score_stats(WR_LINE) == pytest.approx(expected)


def test_te_premium_applies_only_to_tight_ends():
    wr = scoring.TE_PREMIUM.score_stats(WR_LINE)
    te = scoring.TE_PREMIUM.score_stats({**WR_LINE, "position": "TE"})
    assert te - wr == pytest.approx(4.0)  # 8 receptions * 0.5


def test_missing_and_nan_stats_count_as_zero():
    assert scoring.PPR.score_stats({}) == 0.0
    assert scoring.PPR.score_stats({"receiving_yards": float("nan")}) == 0.0
    assert scoring.PPR.score_stats({"receptions": None}) == 0.0


def test_yardage_bonuses_are_thresholds():
    rules = ScoringRules(name="bonus", receiving_yard_bonuses={100: 3.0})
    assert rules.score_stats({"receiving_yards": 99}) == pytest.approx(9.9)
    assert rules.score_stats({"receiving_yards": 100}) == pytest.approx(13.0)


def test_all_fumble_columns_are_penalized():
    line = {
        "rushing_fumbles_lost": 1,
        "receiving_fumbles_lost": 1,
        "sack_fumbles_lost": 1,
    }
    assert scoring.PPR.score_stats(line) == pytest.approx(-6.0)


def test_frame_scoring_matches_row_scoring():
    frame = pd.DataFrame([QB_LINE, WR_LINE])
    scored = scoring.score_frame(frame, scoring.PPR)
    expected = [scoring.PPR.score_stats(row) for row in (QB_LINE, WR_LINE)]
    assert scored.tolist() == pytest.approx(expected)


def test_frame_scoring_tolerates_absent_columns():
    frame = pd.DataFrame({"receiving_yards": [50.0, None], "position": ["WR", "WR"]})
    assert scoring.score_frame(frame, scoring.PPR).tolist() == pytest.approx([5.0, 0.0])


def test_add_fantasy_points_appends_without_mutating():
    frame = pd.DataFrame([WR_LINE])
    result = scoring.add_fantasy_points(frame, scoring.PPR)
    assert "fp" not in frame.columns
    assert result["fp"].iloc[0] == pytest.approx(22.0)


def test_preset_lookup_rejects_unknown_names():
    assert scoring.preset("ppr") is scoring.PPR
    with pytest.raises(KeyError, match="unknown scoring preset"):
        scoring.preset("superflex")
