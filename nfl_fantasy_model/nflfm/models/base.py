"""The interface every projection model implements.

Deliberately sklearn-shaped (``fit``/``predict``) so that a gradient-boosted
model can be dropped in behind the same call sites as the baseline, and so
sklearn's cross-validation utilities work without an adapter.
"""

from __future__ import annotations

import abc

import pandas as pd


class Projector(abc.ABC):
    """Projects fantasy points for player-weeks."""

    #: Human-readable name used in evaluation reports.
    name: str = "projector"

    @abc.abstractmethod
    def fit(self, frame: pd.DataFrame, target: str = "fp") -> "Projector":
        """Train on a feature frame carrying the ``target`` column."""

    @abc.abstractmethod
    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """Project fantasy points for each row, indexed like ``frame``."""

    def fit_predict(self, frame: pd.DataFrame, target: str = "fp") -> pd.Series:
        return self.fit(frame, target=target).predict(frame)
