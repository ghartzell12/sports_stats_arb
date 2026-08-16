"""Apply :class:`~nflfm.scoring.rules.ScoringRules` to a DataFrame."""

from __future__ import annotations

import pandas as pd

from .rules import FUMBLE_COLUMNS, ScoringRules


def score_frame(frame: pd.DataFrame, rules: ScoringRules) -> pd.Series:
    """Fantasy points for every row of a weekly-stats frame.

    Columns the frame does not carry are treated as zero, so this works on the
    trimmed core-column view as well as the raw 114-column feed.
    """
    points = pd.Series(0.0, index=frame.index)

    for column, weight in rules.linear_weights().items():
        if weight and column in frame.columns:
            points += weight * _col(frame, column)

    fumbles = sum(
        (_col(frame, c) for c in FUMBLE_COLUMNS if c in frame.columns),
        start=pd.Series(0.0, index=frame.index),
    )
    points += rules.fumbles_lost * fumbles

    if rules.reception_premiums and {"position", "receptions"} <= set(frame.columns):
        premium = frame["position"].map(rules.reception_premiums).fillna(0.0)
        points += premium.astype(float) * _col(frame, "receptions")

    for column, bonuses in (
        ("passing_yards", rules.passing_yard_bonuses),
        ("rushing_yards", rules.rushing_yard_bonuses),
        ("receiving_yards", rules.receiving_yard_bonuses),
    ):
        if not bonuses or column not in frame.columns:
            continue
        yards = _col(frame, column)
        for threshold, pts in bonuses.items():
            points += pts * (yards >= threshold).astype(float)

    return points


def add_fantasy_points(
    frame: pd.DataFrame,
    rules: ScoringRules,
    column: str = "fp",
) -> pd.DataFrame:
    """Return ``frame`` with a fantasy-points column appended."""
    return frame.assign(**{column: score_frame(frame, rules)})


def _col(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(frame[name], errors="coerce").fillna(0.0)
