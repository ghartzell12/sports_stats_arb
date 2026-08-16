"""Baseline projectors and walk-forward evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflfm.features import build_feature_frame
from nflfm.models import LastGameProjector, RollingMeanProjector, backtest, metrics, season_splits


def synthetic_weekly(seasons=(2020, 2021, 2022, 2023), players=6, weeks=17, seed=0):
    """A small league whose scores are a player-level mean plus noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(players):
        skill = 8.0 + 2.0 * p
        for season in seasons:
            for week in range(1, weeks + 1):
                rows.append(
                    {
                        "player_id": f"{p:02d}",
                        "position": ["QB", "RB", "WR"][p % 3],
                        "season": season,
                        "week": week,
                        "fp": skill + rng.normal(0, 3),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def features():
    return build_feature_frame(synthetic_weekly(), columns=["fp"], windows=[3, 5], lags=[1])


def test_rolling_projector_reads_its_window(features):
    model = RollingMeanProjector(5).fit(features)
    predictions = model.predict(features)
    known = features["fp_r5"].notna()
    assert predictions[known].tolist() == pytest.approx(features.loc[known, "fp_r5"].tolist())


def test_rolling_projector_fills_cold_starts_with_position_means(features):
    model = RollingMeanProjector(5).fit(features)
    predictions = model.predict(features)
    assert predictions.notna().all()

    cold = features["fp_r5"].isna()
    assert cold.any()
    expected = features.loc[cold, "position"].map(model.position_means_)
    assert predictions[cold].tolist() == pytest.approx(expected.tolist())


def test_projector_requires_its_feature_column():
    model = RollingMeanProjector(5)
    with pytest.raises(KeyError, match="fp_r5"):
        model.predict(pd.DataFrame({"fp": [1.0]}))


def test_last_game_projector_returns_the_previous_result(features):
    predictions = LastGameProjector().fit(features).predict(features)
    known = features["fp_lag1"].notna()
    assert predictions[known].tolist() == pytest.approx(features.loc[known, "fp_lag1"].tolist())


def test_metrics_ignore_unpaired_rows():
    actual = pd.Series([10.0, 20.0, 30.0])
    predicted = pd.Series([10.0, 20.0, None])
    result = metrics(actual, predicted)
    assert result["n"] == 2
    assert result["mae"] == pytest.approx(0.0)


def test_metrics_on_empty_input_do_not_raise():
    result = metrics(pd.Series(dtype=float), pd.Series(dtype=float))
    assert result["n"] == 0


def test_season_splits_only_train_on_the_past(features):
    splits = list(season_splits(features, min_train_seasons=2))
    assert [season for _, _, season in splits] == [2022, 2023]
    for train, test, season in splits:
        assert train["season"].max() < season
        assert set(test["season"]) == {season}


def test_backtest_reports_one_row_per_held_out_season(features):
    table = backtest(RollingMeanProjector(5), features, min_train_seasons=2)
    assert table["season"].tolist() == [2022, 2023]
    assert (table["mae"] > 0).all()


def test_averaging_beats_last_game_on_noisy_data(features):
    """The point of the baselines: smoothing should beat a single sample."""
    rolling = backtest(RollingMeanProjector(5), features, min_train_seasons=2)["mae"].mean()
    last = backtest(LastGameProjector(), features, min_train_seasons=2)["mae"].mean()
    assert rolling < last
