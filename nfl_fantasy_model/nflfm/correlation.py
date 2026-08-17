"""Empirical correlation structure for best ball stacking.

Stacking works because teammates' fantasy weeks are not independent: when the
Bengals throw for 400 and 4, Burrow and Chase both spike together. That joint
upside is what wins a top-heavy tournament — you need the weeks where your
whole roster goes off at once, not a steady median.

Everything here is measured from historical weekly data rather than assumed.
Correlations are computed on *residuals* — each player's score minus his own
rolling baseline — because raw score correlation between two good players on
the same team is inflated by both simply being good.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Position pairs worth modeling, as (position_a, position_b).
STACK_PAIRS: tuple[tuple[str, str], ...] = (
    ("QB", "WR"),
    ("QB", "TE"),
    ("QB", "RB"),
    ("WR", "WR"),
    ("WR", "TE"),
    ("RB", "WR"),
    ("RB", "TE"),
    ("RB", "RB"),
)


@dataclass(frozen=True)
class CorrelationTable:
    """Measured correlations, keyed by relationship."""

    same_team: pd.DataFrame
    opponent: pd.DataFrame

    def lookup(self, table: str, pos_a: str, pos_b: str, rank_a: int = 1, rank_b: int = 1) -> float:
        """Correlation for a pair, falling back to the position-level mean."""
        frame = self.same_team if table == "same_team" else self.opponent
        exact = frame[
            (frame["pos_a"] == pos_a)
            & (frame["pos_b"] == pos_b)
            & (frame["rank_a"] == rank_a)
            & (frame["rank_b"] == rank_b)
        ]
        if not exact.empty:
            return float(exact["corr"].iloc[0])
        loose = frame[(frame["pos_a"] == pos_a) & (frame["pos_b"] == pos_b)]
        return float(loose["corr"].mean()) if not loose.empty else 0.0


def add_residuals(
    weekly: pd.DataFrame,
    target: str = "fp",
    min_games: int = 8,
) -> pd.DataFrame:
    """Append ``resid`` — score minus the player's own season mean, standardized.

    Removing each player's own level is what separates "these two spike
    together" from "these two are both good". Players with fewer than
    ``min_games`` in a season are dropped: their season mean is too noisy for
    the residual to mean anything.
    """
    frame = weekly.copy()
    grouped = frame.groupby(["player_id", "season"], observed=True)[target]
    counts = grouped.transform("size")
    frame = frame[counts >= min_games].copy()

    grouped = frame.groupby(["player_id", "season"], observed=True)[target]
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    frame["resid"] = (frame[target] - mean) / std.replace(0.0, np.nan)
    return frame.dropna(subset=["resid"])


def rank_within_team(
    weekly: pd.DataFrame,
    target: str = "fp",
) -> pd.DataFrame:
    """Append ``pos_rank`` — a player's rank at his position on his team.

    Ranked by season total, so WR1 means "his team's leading receiver that
    season". Stacking a QB with his WR1 is a different bet than with his WR3,
    and the correlations differ accordingly.
    """
    totals = (
        weekly.groupby(["season", "team", "position", "player_id"], observed=True)[target]
        .sum()
        .reset_index(name="season_total")
    )
    totals["pos_rank"] = (
        totals.groupby(["season", "team", "position"], observed=True)["season_total"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    return weekly.merge(
        totals[["season", "team", "position", "player_id", "pos_rank"]],
        on=["season", "team", "position", "player_id"],
        how="left",
    )


def same_team_correlations(
    weekly: pd.DataFrame,
    max_rank: int = 3,
    min_pairs: int = 50,
) -> pd.DataFrame:
    """Correlation between teammates' weekly residuals, by position and rank.

    Returns one row per (pos_a, rank_a, pos_b, rank_b) with the correlation and
    the sample size behind it.
    """
    frame = rank_within_team(add_residuals(weekly))
    frame = frame[frame["pos_rank"] <= max_rank]

    keys = ["season", "week", "team"]
    left = frame[keys + ["position", "pos_rank", "resid", "player_id"]]
    merged = left.merge(left, on=keys, suffixes=("_a", "_b"))
    # Keep each unordered pair once, and drop self-joins.
    merged = merged[merged["player_id_a"] < merged["player_id_b"]]

    return _summarize_pairs(merged)


def opponent_correlations(
    weekly: pd.DataFrame,
    max_rank: int = 3,
    min_pairs: int = 50,
) -> pd.DataFrame:
    """Correlation between players facing each other in the same game.

    Run-it-back stacks (your QB plus the opposing WR1) are bets on a shootout.
    This measures whether that effect is real and how big it is.
    """
    frame = rank_within_team(add_residuals(weekly))
    frame = frame[frame["pos_rank"] <= max_rank]

    left = frame[["season", "week", "team", "opponent_team", "position", "pos_rank", "resid", "player_id"]]
    right = left.rename(columns={"team": "opponent_team", "opponent_team": "team"})
    merged = left.merge(
        right,
        on=["season", "week", "team", "opponent_team"],
        suffixes=("_a", "_b"),
    )
    merged = merged[merged["player_id_a"] < merged["player_id_b"]]

    return _summarize_pairs(merged)


#: Canonical position ordering, so QB1-WR1 and WR1-QB1 are the same row.
POSITION_ORDER: dict[str, int] = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}


def _canonicalize(merged: pd.DataFrame) -> pd.DataFrame:
    """Orient every pair so the side ordering depends only on position/rank.

    Pairs are deduplicated by ``player_id``, which leaves the *labelling*
    arbitrary: the same QB-WR1 relationship lands in ``a`` or ``b`` depending
    on which id happened to sort first, splitting one relationship across two
    rows. Sorting each pair by (position, rank) collapses them back together.
    """
    frame = merged.copy()
    rank_a = frame["position_a"].map(POSITION_ORDER).fillna(99) * 100 + frame["pos_rank_a"]
    rank_b = frame["position_b"].map(POSITION_ORDER).fillna(99) * 100 + frame["pos_rank_b"]
    flip = (rank_a > rank_b).to_numpy()

    for left, right in (
        ("position_a", "position_b"),
        ("pos_rank_a", "pos_rank_b"),
        ("resid_a", "resid_b"),
        ("player_id_a", "player_id_b"),
    ):
        a = frame[left].to_numpy().copy()
        b = frame[right].to_numpy().copy()
        frame[left] = np.where(flip, b, a)
        frame[right] = np.where(flip, a, b)

    return frame


def _summarize_pairs(merged: pd.DataFrame, min_pairs: int = 50) -> pd.DataFrame:
    """Correlation and sample size for each position/rank combination."""
    grouped = _canonicalize(merged).groupby(
        ["position_a", "pos_rank_a", "position_b", "pos_rank_b"], observed=True
    )
    rows = []
    for (pos_a, rank_a, pos_b, rank_b), group in grouped:
        if len(group) < min_pairs:
            continue
        rows.append(
            {
                "pos_a": pos_a,
                "rank_a": int(rank_a),
                "pos_b": pos_b,
                "rank_b": int(rank_b),
                "corr": float(group["resid_a"].corr(group["resid_b"])),
                "n": int(len(group)),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("corr", ascending=False)
        .reset_index(drop=True)
    )


def build(weekly: pd.DataFrame, max_rank: int = 3) -> CorrelationTable:
    """Measure both correlation tables from a weekly frame."""
    return CorrelationTable(
        same_team=same_team_correlations(weekly, max_rank=max_rank),
        opponent=opponent_correlations(weekly, max_rank=max_rank),
    )
