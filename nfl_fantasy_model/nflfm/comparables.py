"""Comparable-player shapes from pre-season features.

The problem with bucketing players by what they *did* is that it cannot place
a rookie, and a draft board has to price rookies. So buckets are defined by
features that are knowable before a snap is played — projected role, volume,
touchdown dependence, depth of target — all of which a projection or a prop
line supplies for a rookie just as readily as for a veteran.

Matching is nearest-neighbour in standardized feature space rather than a
fixed grid: it behaves like a very large number of buckets, degrades smoothly
where the space is thin, and yields an actual list of comparable seasons that
can be shown to a human as a sanity check ("this rookie is being priced like
2014 Mike Evans").

Where a player has a rich ladder of alternate season-long prop lines, the
market pins his distribution down directly and should be preferred over the
comps built here — see :mod:`nflfm.market`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Features used to match players, by position group. All are per-game rates
#: or ratios, so they are comparable across eras and workloads.
#: Every feature must be computable *both* from historical box scores and from
#: sportsbook component lines, or a 2026 player cannot be matched against
#: history at all. That rules out air-yards-based measures like aDOT, which no
#: book prices; yards per reception stands in for it.
FEATURES: dict[str, tuple[str, ...]] = {
    "QB": ("ppg", "pass_yards_pg", "rush_yards_pg", "td_dependence"),
    "RB": ("ppg", "rush_yards_pg", "rec_pg", "td_dependence"),
    "WR": ("ppg", "rec_pg", "ypr", "td_dependence"),
    "TE": ("ppg", "rec_pg", "ypr", "td_dependence"),
}

MIN_GAMES = 6
DEFAULT_K = 40


def season_features(weekly: pd.DataFrame, target: str = "fp") -> pd.DataFrame:
    """One row per player-season of pre-season-style features.

    Computed from realized usage, which is what history offers. When applied
    forward, the same columns come from projections instead — see
    :func:`ComparableIndex.neighbours`. That substitution assumes projected
    usage is an unbiased estimate of realized usage; projection error will make
    the matched shapes somewhat too confident, which is a known limitation
    rather than a bug.
    """
    frame = weekly.copy()
    for column in (
        "targets",
        "carries",
        "attempts",
        "passing_yards",
        "rushing_yards",
        "receiving_yards",
        "receiving_air_yards",
        "receptions",
        "passing_tds",
        "rushing_tds",
        "receiving_tds",
    ):
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    grouped = frame.groupby(["player_id", "player_display_name", "position", "season"], observed=True)
    agg = grouped.agg(
        games=(target, "size"),
        points=(target, "sum"),
        targets=("targets", "sum"),
        carries=("carries", "sum"),
        pass_att=("attempts", "sum"),
        pass_yards_total=("passing_yards", "sum"),
        rush_yards_total=("rushing_yards", "sum"),
        rec_yards_total=("receiving_yards", "sum"),
        air_yards=("receiving_air_yards", "sum"),
        receptions=("receptions", "sum"),
        pass_td=("passing_tds", "sum"),
        rush_td=("rushing_tds", "sum"),
        rec_td=("receiving_tds", "sum"),
    ).reset_index()

    agg = agg[agg["games"] >= MIN_GAMES].copy()
    agg["ppg"] = agg["points"] / agg["games"]
    agg["target_pg"] = agg["targets"] / agg["games"]
    agg["rec_pg"] = agg["receptions"] / agg["games"]
    agg["pass_yards_pg"] = agg["pass_yards_total"] / agg["games"]
    agg["rush_yards_pg"] = agg["rush_yards_total"] / agg["games"]
    agg["ypr"] = np.where(
        agg["receptions"] > 0,
        agg["rec_yards_total"] / agg["receptions"].replace(0, np.nan),
        0.0,
    )
    agg["pass_att_pg"] = agg["pass_att"] / agg["games"]
    agg["rush_att_pg"] = agg["carries"] / agg["games"]
    agg["touch_pg"] = (agg["carries"] + agg["receptions"]) / agg["games"]
    agg["adot"] = np.where(agg["targets"] > 0, agg["air_yards"] / agg["targets"].replace(0, np.nan), 0.0)
    agg["target_share_of_touches"] = np.where(
        (agg["carries"] + agg["receptions"]) > 0,
        agg["receptions"] / (agg["carries"] + agg["receptions"]),
        0.0,
    )
    # Share of fantasy points that came from touchdowns. High values mean a
    # player's scoring is hostage to a rare event, which is the main driver of
    # week-to-week spikiness once volume is accounted for.
    total_td = agg["pass_td"] * 4 + (agg["rush_td"] + agg["rec_td"]) * 6
    agg["td_dependence"] = np.where(agg["points"] > 0, total_td / agg["points"], 0.0)

    return agg.replace([np.inf, -np.inf], np.nan).fillna({"adot": 0.0, "ypr": 0.0})


def weekly_shapes(weekly: pd.DataFrame, target: str = "fp") -> pd.DataFrame:
    """Per player-season, the scale-free weekly scores used as a shape sample."""
    frame = weekly.copy()
    grouped = frame.groupby(["player_id", "season"], observed=True)[target]
    frame["_games"] = grouped.transform("size")
    frame["_mean"] = grouped.transform("mean")
    frame = frame[(frame["_games"] >= MIN_GAMES) & (frame["_mean"] > 0)].copy()
    frame["shape"] = frame[target] / frame["_mean"]
    return frame[["player_id", "season", "position", "shape"]]


@dataclass
class ComparableIndex:
    """Nearest-neighbour lookup from features to historical weekly shapes."""

    features: pd.DataFrame
    shapes: pd.DataFrame

    def __post_init__(self) -> None:
        self._scalers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._matrices: dict[str, np.ndarray] = {}
        self._index: dict[str, pd.DataFrame] = {}
        self._pools: dict[tuple[str, int], np.ndarray] = {}

        for position, columns in FEATURES.items():
            subset = self.features[self.features["position"] == position]
            if subset.empty:
                continue
            matrix = subset[list(columns)].to_numpy(dtype=float)
            mean = matrix.mean(axis=0)
            std = matrix.std(axis=0)
            std[std == 0] = 1.0
            self._scalers[position] = (mean, std)
            self._matrices[position] = (matrix - mean) / std
            self._index[position] = subset.reset_index(drop=True)

        for (player_id, season), group in self.shapes.groupby(["player_id", "season"], observed=True):
            self._pools[(player_id, season)] = group["shape"].to_numpy(dtype=float)

    def neighbours(
        self,
        position: str,
        query: dict[str, float],
        k: int = DEFAULT_K,
    ) -> pd.DataFrame:
        """The ``k`` most similar historical player-seasons, closest first.

        ``query`` supplies the same feature names as :data:`FEATURES` for this
        position, sourced from projections rather than history.
        """
        if position not in self._matrices:
            raise KeyError(f"no comparable seasons for position {position!r}")

        columns = FEATURES[position]
        missing = [c for c in columns if c not in query]
        if missing:
            raise KeyError(f"missing features for {position}: {', '.join(missing)}")

        mean, std = self._scalers[position]
        point = (np.array([query[c] for c in columns], dtype=float) - mean) / std
        distance = np.linalg.norm(self._matrices[position] - point, axis=1)

        frame = self._index[position].copy()
        frame["distance"] = distance
        return frame.nsmallest(min(k, len(frame)), "distance")

    def shape_pool(self, position: str, query: dict[str, float], k: int = DEFAULT_K) -> np.ndarray:
        """Pooled weekly shapes of the ``k`` nearest comparable seasons."""
        comps = self.neighbours(position, query, k=k)
        pools = [
            self._pools[(row.player_id, row.season)]
            for row in comps.itertuples()
            if (row.player_id, row.season) in self._pools
        ]
        if not pools:
            raise ValueError(f"no weekly shapes found for the {position} comparables")
        return np.concatenate(pools)

    def moments(self, position: str, query: dict[str, float], k: int = DEFAULT_K) -> dict[str, float]:
        """Dispersion, skew, and kurtosis of the matched shape pool."""
        return describe(self.shape_pool(position, query, k=k))


def describe(pool: np.ndarray) -> dict[str, float]:
    """Shape statistics of a scale-free weekly pool.

    Skew and kurtosis are the point of the exercise: two pools can share a
    coefficient of variation while one delivers its variance as frequent small
    misses and the other as rare enormous weeks, and only the second is worth
    paying up for in a top-heavy contest.
    """
    pool = np.asarray(pool, dtype=float)
    mean = pool.mean()
    std = pool.std()
    if std == 0:
        return {"n": float(pool.size), "cv": 0.0, "skew": 0.0, "kurtosis": 0.0, "p95": float(mean)}
    centred = (pool - mean) / std
    return {
        "n": float(pool.size),
        "cv": float(std / mean),
        "skew": float((centred**3).mean()),
        "kurtosis": float((centred**4).mean() - 3.0),
        "p95": float(np.quantile(pool, 0.95)),
    }


def build_index(weekly: pd.DataFrame, target: str = "fp") -> ComparableIndex:
    """Assemble the comparable-season index from a weekly frame."""
    return ComparableIndex(
        features=season_features(weekly, target=target),
        shapes=weekly_shapes(weekly, target=target),
    )
