"""Baselines to beat.

A projection model is only interesting relative to a naive alternative, and
the naive alternatives in fantasy are strong: a player's trailing average is
hard to beat by much. Anything added later gets measured against these.
"""

from __future__ import annotations

import pandas as pd

from .base import Projector


class RollingMeanProjector(Projector):
    """Project this week as the mean of the player's previous ``window`` games.

    Reads the precomputed ``fp_r{window}`` column from
    :func:`nflfm.features.build.build_feature_frame`, falling back to the
    positional mean for players without enough history (rookies, week 1).
    """

    def __init__(self, window: int = 5) -> None:
        self.window = window
        self.name = f"rolling_mean_{window}"
        self.position_means_: pd.Series | None = None
        self.global_mean_: float = 0.0

    @property
    def source_column(self) -> str:
        return f"fp_r{self.window}"

    def fit(self, frame: pd.DataFrame, target: str = "fp") -> "RollingMeanProjector":
        self.global_mean_ = float(frame[target].mean())
        if "position" in frame.columns:
            self.position_means_ = frame.groupby("position", observed=True)[target].mean()
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if self.source_column not in frame.columns:
            raise KeyError(
                f"{self.source_column!r} missing; build features with "
                f"build_feature_frame(windows=(..., {self.window}, ...)) first"
            )
        predictions = pd.to_numeric(frame[self.source_column], errors="coerce")
        return predictions.fillna(self._fallback(frame))

    def _fallback(self, frame: pd.DataFrame) -> pd.Series:
        if self.position_means_ is not None and "position" in frame.columns:
            return frame["position"].map(self.position_means_).fillna(self.global_mean_)
        return pd.Series(self.global_mean_, index=frame.index)


class LastGameProjector(Projector):
    """Project this week as last week's result. The weakest sensible baseline."""

    name = "last_game"

    def __init__(self) -> None:
        self.global_mean_: float = 0.0

    def fit(self, frame: pd.DataFrame, target: str = "fp") -> "LastGameProjector":
        self.global_mean_ = float(frame[target].mean())
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if "fp_lag1" not in frame.columns:
            raise KeyError("'fp_lag1' missing; run build_feature_frame first")
        return pd.to_numeric(frame["fp_lag1"], errors="coerce").fillna(self.global_mean_)
