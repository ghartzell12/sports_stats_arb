"""Data acquisition and loading for public nflverse feeds."""

from .ingest import download, download_first, fetch, fetch_all
from .loaders import (
    load_players,
    load_rosters,
    load_schedules,
    load_snap_counts,
    load_weekly_stats,
)

__all__ = [
    "download",
    "download_first",
    "fetch",
    "fetch_all",
    "load_players",
    "load_rosters",
    "load_schedules",
    "load_snap_counts",
    "load_weekly_stats",
]
