"""Load cached nflverse files into DataFrames.

Every loader fetches on demand, so a caller can go straight to
``load_weekly_stats([2023, 2024])`` on a cold checkout.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from ..config import SKILL_POSITIONS
from . import ingest

#: Columns worth keeping for skill-position modeling. The raw weekly file
#: carries 114 columns, most of them defensive or kicking.
WEEKLY_CORE_COLUMNS: tuple[str, ...] = (
    "player_id",
    "player_display_name",
    "position",
    "season",
    "week",
    "season_type",
    "team",
    "opponent_team",
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "sacks_suffered",
    "passing_first_downs",
    "passing_epa",
    "passing_2pt_conversions",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_fumbles_lost",
    "rushing_first_downs",
    "rushing_epa",
    "rushing_2pt_conversions",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "receiving_fumbles_lost",
    "receiving_air_yards",
    "receiving_first_downs",
    "receiving_epa",
    "receiving_2pt_conversions",
    "target_share",
    "air_yards_share",
    "wopr",
    "sack_fumbles_lost",
    "special_teams_tds",
    "fantasy_points",
    "fantasy_points_ppr",
)


def load_weekly_stats(
    seasons: Iterable[int],
    positions: Sequence[str] | None = SKILL_POSITIONS,
    season_type: str | None = "REG",
    columns: Sequence[str] | None = WEEKLY_CORE_COLUMNS,
) -> pd.DataFrame:
    """Weekly player box scores, one row per player-week.

    Pass ``positions=None`` or ``columns=None`` to opt out of the respective
    filter and get the raw feed.
    """
    paths = ingest.fetch("weekly_stats", seasons)
    frame = pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)

    if season_type is not None:
        frame = frame[frame["season_type"] == season_type]
    if positions is not None:
        frame = frame[frame["position"].isin(positions)]
    if columns is not None:
        keep = [c for c in columns if c in frame.columns]
        frame = frame[keep]

    return frame.sort_values(["season", "week", "player_id"]).reset_index(drop=True)


def load_players() -> pd.DataFrame:
    """Player master table: ids, birthdate, draft position, physicals."""
    (path,) = ingest.fetch("players")
    return pd.read_parquet(path)


def load_rosters(seasons: Iterable[int]) -> pd.DataFrame:
    """Weekly roster/status rows — the source of truth for team and depth."""
    paths = ingest.fetch("rosters", seasons)
    return pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)


def load_schedules() -> pd.DataFrame:
    """Game results and betting lines for every season, one row per game."""
    (path,) = ingest.fetch("schedules")
    return pd.read_parquet(path)


def load_snap_counts(seasons: Iterable[int]) -> pd.DataFrame:
    """Per-player offensive/defensive/special-teams snap shares."""
    paths = ingest.fetch("snap_counts", seasons)
    return pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)
