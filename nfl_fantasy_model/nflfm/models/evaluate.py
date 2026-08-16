"""Backtesting.

Fantasy data is a time series, so a random train/test split leaks the future
into the past and flatters every model. Evaluation here is walk-forward: train
on everything through season N, score season N+1, repeat.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from .base import Projector


def season_splits(
    frame: pd.DataFrame,
    min_train_seasons: int = 3,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, int]]:
    """Yield ``(train, test, test_season)`` walking forward one season at a time."""
    seasons = sorted(frame["season"].unique())
    for i in range(min_train_seasons, len(seasons)):
        test_season = seasons[i]
        yield (
            frame[frame["season"] < test_season],
            frame[frame["season"] == test_season],
            int(test_season),
        )


def metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Error and rank-agreement metrics for one set of projections.

    Spearman matters more than MAE for lineup decisions: getting the *order* of
    players right is what wins a week, even if the point totals are off.
    """
    paired = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if paired.empty:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "spearman": float("nan")}

    error = paired["actual"] - paired["predicted"]
    return {
        "n": float(len(paired)),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "spearman": _spearman(paired["actual"], paired["predicted"]),
    }


def _spearman(actual: pd.Series, predicted: pd.Series) -> float:
    """Rank correlation, computed as Pearson on average ranks.

    Equivalent to ``Series.corr(method="spearman")`` but without pulling in
    scipy, which pandas requires for that method and which is otherwise
    unnecessary here.
    """
    if len(actual) < 2:
        return float("nan")
    return float(actual.rank().corr(predicted.rank()))


def backtest(
    model: Projector,
    frame: pd.DataFrame,
    target: str = "fp",
    min_train_seasons: int = 3,
) -> pd.DataFrame:
    """Walk-forward evaluation, one row of metrics per held-out season."""
    rows = []
    for train, test, season in season_splits(frame, min_train_seasons):
        model.fit(train, target=target)
        row = {"model": model.name, "season": season}
        row.update(metrics(test[target], model.predict(test)))
        rows.append(row)
    return pd.DataFrame(rows)


def compare(
    models: Sequence[Projector],
    frame: pd.DataFrame,
    target: str = "fp",
    min_train_seasons: int = 3,
) -> pd.DataFrame:
    """Backtest several models and report each one's average across seasons."""
    results = pd.concat(
        [backtest(m, frame, target, min_train_seasons) for m in models],
        ignore_index=True,
    )
    return (
        results.groupby("model", observed=True)[["mae", "rmse", "spearman"]]
        .mean()
        .sort_values("mae")
        .reset_index()
    )
