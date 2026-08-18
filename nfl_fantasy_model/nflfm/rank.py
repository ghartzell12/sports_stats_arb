"""Static best ball rankings.

Combines two independent inputs:

* **level** — a median points projection per player, from the market or a
  projection source. This is the piece history cannot supply for 2026.
* **shape** — the weekly distribution around that level, from comparable
  historical seasons (:mod:`nflfm.comparables`) or, better, from a ladder of
  alternate prop lines (:mod:`nflfm.market`).

The ranking metric is **best ball value over replacement (BBVOR)**: the
expected sum, across a season, of how much a player beats a replacement-level
player at his position in the weeks where he beats him at all.

    BBVOR = E[ sum_w max(player_w - replacement_w, 0) ]

The truncation at zero is what makes this a best-ball metric rather than a
season-total one. Weeks where a player busts do not subtract, because in best
ball those weeks sit on the bench and someone else's score is used instead. A
player is therefore rewarded for his ceiling and only lightly punished for his
floor, which is the actual economics of the format.

This is an interim ranking. It prices a player in isolation; it does not yet
account for roster construction, stacking, draft capital, or the shape of the
tournament payout. Those arrive with the portfolio optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import comparables

#: Replacement level by position, as a rank among that position's starters in
#: a 12-team league with 8 starting slots.
REPLACEMENT_RANK: dict[str, int] = {"QB": 14, "RB": 32, "WR": 44, "TE": 14}

WEEKS = 17
DEFAULT_SIMS = 4000


@dataclass
class RankedPlayer:
    """One player's simulated best ball profile."""

    name: str
    position: str
    team: str
    ppg: float
    bbvor: float
    ceiling: float
    floor: float
    cv: float
    skew: float
    kurtosis: float
    comp_source: str
    adp: float


def estimate_features(
    projections: pd.DataFrame,
    index: comparables.ComparableIndex,
    level_column: str = "ppg",
) -> pd.DataFrame:
    """Fill each player's comp features, preferring observed recent usage.

    ``level_column`` names the per-game scoring level to match on. It is a
    separate argument because converting a projection between scoring systems
    can itself depend on the matched usage, so the caller may only have a
    provisional level at this point.

    Veterans carry their most recent season's usage forward. Rookies and
    players without recent data get the historical median usage of players at
    the same position and scoring level — coarser, and flagged as such, but it
    keeps a rookie in the board rather than dropping him.
    """
    history = index.features
    recent = (
        history.sort_values("season")
        .groupby(["player_display_name", "position"], observed=True)
        .tail(1)
        .set_index(["player_display_name", "position"])
    )

    rows = []
    for row in projections.itertuples():
        columns = comparables.FEATURES.get(row.position)
        if columns is None:
            continue

        level = float(getattr(row, level_column))
        key = (row.name_norm, row.position)
        observed = recent.loc[key] if key in recent.index else None

        if observed is not None:
            features = {c: float(observed[c]) for c in columns if c in observed}
            features["ppg"] = level
            source = "observed_usage"
        else:
            features = _median_features_at_level(history, row.position, level, columns)
            features["ppg"] = level
            source = "estimated_from_level"

        rows.append({**features, "position": row.position, "comp_source": source, "_idx": row.Index})

    return pd.DataFrame(rows).set_index("_idx")


def _median_features_at_level(
    history: pd.DataFrame,
    position: str,
    ppg: float,
    columns: tuple[str, ...],
    window: float = 2.0,
) -> dict[str, float]:
    """Median usage among historical seasons at a similar scoring level."""
    subset = history[
        (history["position"] == position) & (history["ppg"].sub(ppg).abs() <= window)
    ]
    if subset.empty:
        subset = history[history["position"] == position]
    return {c: float(subset[c].median()) for c in columns if c != "ppg"}


def simulate_players(
    projections: pd.DataFrame,
    features: pd.DataFrame,
    index: comparables.ComparableIndex,
    sims: int = DEFAULT_SIMS,
    weeks: int = WEEKS,
    k: int = comparables.DEFAULT_K,
    seed: int = 0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Draw ``(sims, weeks)`` scores per player from matched comparable shapes."""
    rng = np.random.default_rng(seed)
    draws = np.zeros((len(projections), sims, weeks), dtype=float)
    moments = []

    for position_rows in features.groupby("position", observed=True):
        position, group = position_rows
        columns = comparables.FEATURES[position]
        for idx, row in group.iterrows():
            query = {c: float(row[c]) for c in columns}
            pool = index.shape_pool(position, query, k=k)
            ppg = float(projections.loc[idx, "ppg"])
            draws[projections.index.get_loc(idx)] = rng.choice(pool, size=(sims, weeks)) * ppg
            stats = comparables.describe(pool)
            moments.append({"_idx": idx, **stats, "comp_source": row["comp_source"]})

    return draws, pd.DataFrame(moments).set_index("_idx")


def replacement_baselines(
    projections: pd.DataFrame,
    draws: np.ndarray,
) -> dict[str, np.ndarray]:
    """Weekly score draws for a replacement-level player at each position."""
    baselines: dict[str, np.ndarray] = {}
    for position, rank in REPLACEMENT_RANK.items():
        pool = projections[projections["position"] == position].sort_values(
            "ppg", ascending=False
        )
        if pool.empty:
            continue
        target = pool.iloc[min(rank, len(pool)) - 1]
        baselines[position] = draws[projections.index.get_loc(target.name)]
    return baselines


def best_ball_value(
    draws: np.ndarray,
    positions: pd.Series,
    baselines: dict[str, np.ndarray],
) -> np.ndarray:
    """BBVOR per player: expected season total of weekly wins over replacement."""
    values = np.zeros(draws.shape[0], dtype=float)
    for i, position in enumerate(positions):
        baseline = baselines.get(position)
        if baseline is None:
            continue
        surplus = np.maximum(draws[i] - baseline, 0.0)
        values[i] = surplus.sum(axis=-1).mean()
    return values


def historical_replacement_ppg(
    weekly: pd.DataFrame,
    ranks: dict[str, int] | None = None,
    target: str = "fp",
    min_games: int = 6,
) -> dict[str, float]:
    """Replacement-level points per game, measured across past seasons.

    Ranking *within the supplied projection set* is unsafe: a props file that
    covers 33 running backs puts "RB32" on the second-worst back in the file,
    which is nothing like the RB32 a drafter actually faces. That understates
    replacement level and inflates every running back's value over it.

    Measuring the Nth-best player at each position in each historical season
    and averaging gives a level that does not move with how many players a
    given data source happens to include.
    """
    ranks = ranks or REPLACEMENT_RANK
    frame = weekly.copy()
    per_season = (
        frame.groupby(["player_id", "position", "season"], observed=True)[target]
        .agg(games="size", points="sum")
        .reset_index()
    )
    per_season = per_season[per_season["games"] >= min_games]
    per_season["ppg"] = per_season["points"] / per_season["games"]

    levels: dict[str, float] = {}
    for position, rank_needed in ranks.items():
        subset = per_season[per_season["position"] == position]
        if subset.empty:
            continue
        by_season = []
        for _, group in subset.groupby("season", observed=True):
            ordered = group["ppg"].sort_values(ascending=False).to_numpy()
            if len(ordered) >= rank_needed:
                by_season.append(ordered[rank_needed - 1])
        if by_season:
            levels[position] = float(np.mean(by_season))
    return levels


def synthetic_replacement_draws(
    levels: dict[str, float],
    index,
    sims: int,
    weeks: int,
    seed: int = 99,
) -> dict[str, np.ndarray]:
    """Weekly draws for a replacement player at each position's measured level."""
    rng = np.random.default_rng(seed)
    baselines: dict[str, np.ndarray] = {}
    for position, ppg in levels.items():
        columns = comparables.FEATURES.get(position)
        if columns is None:
            continue
        history = index.features
        subset = history[
            (history["position"] == position) & (history["ppg"].sub(ppg).abs() <= 2.0)
        ]
        if subset.empty:
            subset = history[history["position"] == position]
        query = {c: float(subset[c].median()) for c in columns}
        query["ppg"] = ppg
        pool = index.shape_pool(position, query, k=comparables.DEFAULT_K)
        baselines[position] = rng.choice(pool, size=(sims, weeks)) * ppg
    return baselines
