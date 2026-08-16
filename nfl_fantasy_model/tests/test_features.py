"""Feature builders, with leakage as the property under test."""

from __future__ import annotations

import pandas as pd
import pytest

from nflfm.features import build


def game_log(fp_by_week, player_id="00-0000001", season=2024):
    """One player's season as a weekly frame."""
    return pd.DataFrame(
        {
            "player_id": player_id,
            "season": season,
            "week": range(1, len(fp_by_week) + 1),
            "fp": fp_by_week,
        }
    )


def test_lags_shift_by_games_not_weeks():
    frame = build.add_lags(game_log([10.0, 20.0, 30.0]), columns=["fp"], lags=[1, 2])
    assert frame["fp_lag1"].tolist()[1:] == [10.0, 20.0]
    assert frame["fp_lag2"].tolist()[2:] == [10.0]
    assert pd.isna(frame["fp_lag1"].iloc[0])


def test_rolling_mean_excludes_the_current_week():
    frame = build.add_rolling_means(
        game_log([10.0, 20.0, 30.0, 40.0]), columns=["fp"], windows=[2]
    )
    # Week 3's window is weeks 1-2, not weeks 2-3.
    assert frame["fp_r2"].iloc[2] == pytest.approx(15.0)
    assert frame["fp_r2"].iloc[3] == pytest.approx(25.0)
    assert pd.isna(frame["fp_r2"].iloc[0])


def test_history_does_not_leak_across_players():
    frame = pd.concat(
        [game_log([10.0, 10.0], "A"), game_log([99.0, 99.0], "B")], ignore_index=True
    )
    result = build.add_rolling_means(frame, columns=["fp"], windows=[3])
    by_player = result.set_index("player_id")["fp_r3"]
    assert by_player.loc["A"].tolist() == pytest.approx([float("nan"), 10.0], nan_ok=True)
    assert by_player.loc["B"].tolist() == pytest.approx([float("nan"), 99.0], nan_ok=True)


def test_rolling_window_spans_seasons_for_the_same_player():
    frame = pd.concat(
        [game_log([10.0, 20.0], season=2023), game_log([30.0], season=2024)],
        ignore_index=True,
    )
    result = build.add_rolling_means(frame, columns=["fp"], windows=[3])
    assert result["fp_r3"].iloc[2] == pytest.approx(15.0)


def test_games_played_counts_prior_appearances_only():
    result = build.add_games_played(game_log([1.0, 2.0, 3.0]))
    assert result["games_played_prior"].tolist() == [0, 1, 2]


def test_absent_columns_are_skipped_silently():
    frame = game_log([10.0, 20.0])
    result = build.add_lags(frame, columns=["fp", "target_share"], lags=[1])
    assert "fp_lag1" in result.columns
    assert "target_share_lag1" not in result.columns


def test_build_feature_frame_reports_its_own_columns():
    frame = build.build_feature_frame(
        game_log([10.0, 20.0, 30.0]), columns=["fp"], windows=[3], lags=[1]
    )
    assert build.feature_columns(frame) == ["fp_lag1", "fp_r3", "games_played_prior"]


def test_stub_features_announce_themselves():
    empty = pd.DataFrame()
    with pytest.raises(NotImplementedError):
        build.add_matchup_features(empty, empty)
    with pytest.raises(NotImplementedError):
        build.add_usage_features(empty, empty)
