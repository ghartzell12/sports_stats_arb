"""Feature engineering.

The one rule everything here obeys: **a feature for week W may only use data
from weeks < W.** Fantasy modeling is trivially easy to leak into — a rolling
mean that includes the current week will look brilliant in backtest and be
useless on Sunday morning — so the shift happens inside these helpers rather
than being left to the caller.

The rolling/lag machinery below is complete. The domain features
(:func:`add_matchup_features`, :func:`add_usage_features`) are stubs.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

#: Stats whose recent history carries most of the signal for skill positions.
DEFAULT_ROLLING_STATS: tuple[str, ...] = (
    "fp",
    "targets",
    "receptions",
    "receiving_yards",
    "carries",
    "rushing_yards",
    "attempts",
    "passing_yards",
    "target_share",
    "air_yards_share",
    "wopr",
)

DEFAULT_WINDOWS: tuple[int, ...] = (3, 5, 10)


def sort_timeline(frame: pd.DataFrame) -> pd.DataFrame:
    """Order rows so that per-player history runs forward in time."""
    return frame.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def add_lags(
    frame: pd.DataFrame,
    columns: Iterable[str] = DEFAULT_ROLLING_STATS,
    lags: Sequence[int] = (1, 2, 3),
    group: str = "player_id",
) -> pd.DataFrame:
    """Append ``{column}_lag{n}`` — the player's value ``n`` games earlier.

    Lags are taken over a player's game log, not the calendar, so a bye or an
    inactive week does not shift the window.
    """
    frame = sort_timeline(frame)
    present = [c for c in columns if c in frame.columns]
    new = {
        f"{column}_lag{n}": frame.groupby(group, observed=True)[column].shift(n)
        for column in present
        for n in lags
    }
    return frame.assign(**new)


def add_rolling_means(
    frame: pd.DataFrame,
    columns: Iterable[str] = DEFAULT_ROLLING_STATS,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    group: str = "player_id",
    min_periods: int = 1,
) -> pd.DataFrame:
    """Append ``{column}_r{window}`` — the mean of the previous ``window`` games.

    The current row is shifted out before the window is taken, so the value is
    knowable before kickoff.
    """
    frame = sort_timeline(frame)
    present = [c for c in columns if c in frame.columns]
    grouped = frame.groupby(group, observed=True)

    new = {}
    for column in present:
        prior = grouped[column].shift(1)
        for window in windows:
            new[f"{column}_r{window}"] = prior.groupby(frame[group], observed=True).rolling(
                window, min_periods=min_periods
            ).mean().reset_index(level=0, drop=True)
    return frame.assign(**new)


def add_games_played(frame: pd.DataFrame, group: str = "player_id") -> pd.DataFrame:
    """Append ``games_played_prior`` — career games logged before this row."""
    frame = sort_timeline(frame)
    return frame.assign(
        games_played_prior=frame.groupby(group, observed=True).cumcount()
    )


def add_matchup_features(frame: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """TODO: opponent strength, game environment, and Vegas context.

    Planned inputs, all derivable from ``loaders.load_schedules()`` and the
    weekly feed:

    * opponent fantasy points allowed by position, trailing and shrunk toward
      the league mean (small samples early in a season are noise);
    * implied team total from the spread and over/under, which is the single
      strongest public matchup signal;
    * home/away, rest days, and outdoor/indoor.
    """
    raise NotImplementedError("matchup features not implemented yet")


def add_usage_features(frame: pd.DataFrame, snap_counts: pd.DataFrame) -> pd.DataFrame:
    """TODO: snap share, route participation, and role stability.

    Usage predicts next week better than production does — a receiver's target
    share is far more stable week to week than their yardage. Join
    ``loaders.load_snap_counts()`` and build trailing snap share, plus a
    change-point flag for players whose role just moved (injury ahead of them
    on the depth chart, trade, coaching change).
    """
    raise NotImplementedError("usage features not implemented yet")


def build_feature_frame(
    weekly: pd.DataFrame,
    columns: Iterable[str] = DEFAULT_ROLLING_STATS,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    lags: Sequence[int] = (1, 2, 3),
) -> pd.DataFrame:
    """Assemble the leak-free feature frame the models train on.

    Expects ``weekly`` to already carry an ``fp`` target column (see
    :func:`nflfm.scoring.add_fantasy_points`).
    """
    frame = add_lags(weekly, columns=columns, lags=lags)
    frame = add_rolling_means(frame, columns=columns, windows=windows)
    return add_games_played(frame)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """The generated feature columns present in ``frame``, in a stable order."""
    return sorted(c for c in frame.columns if _is_generated(c))


def _is_generated(column: str) -> bool:
    """True for the ``_lagN`` / ``_rN`` / ``_prior`` columns built above."""
    if column.endswith("_prior"):
        return True
    tail = column.rsplit("_", 1)[-1]
    for prefix in ("lag", "r"):
        if tail.startswith(prefix) and tail[len(prefix) :].isdigit():
            return True
    return False
