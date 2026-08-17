"""Weekly scoring distributions.

A projection gives a player's expected *level*. Best ball needs his *shape* —
how his weeks scatter around that level — because the format keeps the good
weeks and discards the bad ones. Two players projected for 250 points are not
worth the same if one is steady and the other alternates duds and monsters.

Rather than assume a parametric family, the shape is taken empirically: for
each historical player-season, weekly scores are divided by that player's own
mean, producing a scale-free "shape sample". Pooling those by position and
role tier gives a library that already carries the real skew, the dud weeks,
and the injury zeroes — none of which a fitted gamma reproduces faithfully.

To project a 2026 player: pick the matching shape library, resample weeks from
it, and multiply by his projected points per game.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Role tiers, as a player's rank at his position on his own team.
#: A team's WR1 has a different weekly shape than its WR3.
TIER_EDGES: tuple[int, ...] = (1, 2, 3)

REGULAR_SEASON_WEEKS = 17
MIN_GAMES = 6
MIN_PPG = 3.0

#: Weeks a (position, tier) key needs before it is trusted as a resample pool.
MIN_WEEKS_PER_KEY = 200


@dataclass(frozen=True)
class ShapeLibrary:
    """Scale-free weekly shapes, keyed by ``(position, tier)``.

    Each value is a 1-D array of ``weekly_score / player_season_mean``, so its
    own mean is approximately 1.0 and its spread is the quantity of interest.
    """

    shapes: dict[tuple[str, int], np.ndarray]
    played_rate: dict[tuple[str, int], float] = field(default_factory=dict)

    def key_for(self, position: str, tier: int) -> tuple[str, int]:
        """Nearest available key, falling back to a coarser tier then position."""
        if (position, tier) in self.shapes:
            return (position, tier)
        for candidate in sorted(self.shapes):
            if candidate[0] == position:
                return candidate
        raise KeyError(f"no shapes for position {position!r}")

    def sample(
        self,
        position: str,
        tier: int,
        ppg: float,
        size: tuple[int, ...],
        rng: np.random.Generator,
        include_missed_games: bool = True,
    ) -> np.ndarray:
        """Draw weekly scores for a player projected at ``ppg`` points a game.

        ``size`` is typically ``(sims, weeks)``. Missed games arrive as zeros,
        drawn at the historical rate for the position and tier — availability
        is part of a player's value and best ball has no waiver wire.
        """
        key = self.key_for(position, tier)
        pool = self.shapes[key]
        draws = rng.choice(pool, size=size, replace=True) * ppg

        if include_missed_games:
            played = self.played_rate.get(key, 1.0)
            if played < 1.0:
                draws = np.where(rng.random(size) < played, draws, 0.0)
        return draws

    def summary(self) -> pd.DataFrame:
        """Dispersion statistics per key — the numbers that drive best ball."""
        rows = []
        for (position, tier), pool in sorted(self.shapes.items()):
            rows.append(
                {
                    "position": position,
                    "tier": tier,
                    "n_weeks": len(pool),
                    "cv": float(pool.std() / pool.mean()),
                    "p_dud": float((pool < 0.5).mean()),
                    "p_spike": float((pool > 2.0).mean()),
                    "p90": float(np.quantile(pool, 0.90)),
                    "played_rate": self.played_rate.get((position, tier), float("nan")),
                }
            )
        return pd.DataFrame(rows)


def _team_position_rank(weekly: pd.DataFrame, target: str) -> pd.DataFrame:
    """Rank each player at his position within his team-season by total points."""
    totals = (
        weekly.groupby(["season", "team", "position", "player_id"], observed=True)[target]
        .sum()
        .reset_index(name="_total")
    )
    totals["tier"] = (
        totals.groupby(["season", "team", "position"], observed=True)["_total"]
        .rank(ascending=False, method="first")
        .clip(upper=max(TIER_EDGES))
        .astype(int)
    )
    return weekly.merge(
        totals[["season", "team", "position", "player_id", "tier"]],
        on=["season", "team", "position", "player_id"],
        how="left",
    )


def build_shape_library(
    weekly: pd.DataFrame,
    target: str = "fp",
    min_games: int = MIN_GAMES,
    min_ppg: float = MIN_PPG,
    min_weeks: int = MIN_WEEKS_PER_KEY,
    weeks_in_season: int = REGULAR_SEASON_WEEKS,
) -> ShapeLibrary:
    """Measure weekly shapes from historical player-seasons.

    Player-seasons below ``min_games`` or ``min_ppg`` are excluded: dividing by
    a tiny or noisy mean produces shapes that are all artifact. A
    ``(position, tier)`` key needs ``min_weeks`` observations to be kept at
    all, so thin combinations fall back to a coarser key rather than being
    resampled from a handful of weeks.
    """
    frame = _team_position_rank(weekly, target)

    grouped = frame.groupby(["player_id", "season"], observed=True)[target]
    stats = grouped.agg(games="size", mean="mean").reset_index()
    keep = stats[(stats["games"] >= min_games) & (stats["mean"] >= min_ppg)]

    frame = frame.merge(keep[["player_id", "season", "mean", "games"]], on=["player_id", "season"])
    frame["shape"] = frame[target] / frame["mean"]

    shapes: dict[tuple[str, int], np.ndarray] = {}
    played: dict[tuple[str, int], float] = {}

    for (position, tier), group in frame.groupby(["position", "tier"], observed=True):
        values = group["shape"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size < min_weeks:
            continue
        shapes[(position, int(tier))] = values
        # Availability: games actually played out of a full season, averaged
        # over the player-seasons feeding this key.
        #
        # CAVEAT: this is conditional on clearing ``min_games``, so seasons lost
        # early to injury never enter the average and the rate is optimistic.
        # It is a usable within-position *comparison* (RB3s are less available
        # than WR1s) but should not be read as an absolute probability of
        # playing. Replacing it with a real injury model is a TODO.
        per_season = group.groupby(["player_id", "season"], observed=True)["games"].first()
        played[(position, int(tier))] = float(
            (per_season / weeks_in_season).clip(upper=1.0).mean()
        )

    if not shapes:
        raise ValueError("no player-seasons met the minimums; loosen min_games/min_ppg")

    return ShapeLibrary(shapes=shapes, played_rate=played)
