"""URLs for the public nflverse data releases.

nflverse publishes tidy NFL data as parquet assets attached to GitHub
releases, one release tag per dataset. These URLs are stable and require no
API key. See https://github.com/nflverse/nflverse-data/releases

Asset names have drifted: weekly player stats moved from the ``player_stats``
release to ``stats_player``, and the file itself was renamed from
``player_stats_{season}`` to ``stats_player_week_{season}`` along the way. The
current location is listed first and the older ones are kept as fallbacks, so
a season missing from one layout still resolves (2019, for instance, is absent
from ``player_stats/stats_player_week_2019.parquet`` upstream).
"""

from __future__ import annotations

from typing import Iterable

BASE = "https://github.com/nflverse/nflverse-data/releases/download"

# Season-partitioned datasets: first pattern is canonical, rest are fallbacks.
WEEKLY_STATS_PATTERNS = (
    f"{BASE}/stats_player/stats_player_week_{{season}}.parquet",
    f"{BASE}/player_stats/stats_player_week_{{season}}.parquet",
    f"{BASE}/player_stats/player_stats_{{season}}.parquet",
)
ROSTERS_PATTERNS = (f"{BASE}/rosters/roster_{{season}}.parquet",)
SNAP_COUNTS_PATTERNS = (f"{BASE}/snap_counts/snap_counts_{{season}}.parquet",)
PBP_PATTERNS = (f"{BASE}/pbp/play_by_play_{{season}}.parquet",)

# Single-file datasets covering all seasons.
PLAYERS_PATTERNS = (f"{BASE}/players/players.parquet",)
SCHEDULES_PATTERNS = (f"{BASE}/schedules/games.parquet",)

#: Datasets the CLI knows how to fetch, mapped to their URL patterns in
#: preference order.
DATASETS: dict[str, tuple[str, ...]] = {
    "weekly_stats": WEEKLY_STATS_PATTERNS,
    "rosters": ROSTERS_PATTERNS,
    "snap_counts": SNAP_COUNTS_PATTERNS,
    "pbp": PBP_PATTERNS,
    "players": PLAYERS_PATTERNS,
    "schedules": SCHEDULES_PATTERNS,
}

#: Datasets not partitioned by season.
SEASONLESS = frozenset({"players", "schedules"})

#: Earliest and latest seasons nflverse publishes weekly stats for.
FIRST_AVAILABLE_SEASON = 1999


def is_seasonal(dataset: str) -> bool:
    """True if ``dataset`` has one file per season."""
    _require_known(dataset)
    return dataset not in SEASONLESS


def candidate_urls(dataset: str, season: int | None = None) -> list[str]:
    """Every URL that may hold ``dataset``, most current first.

    ``season`` is required for season-partitioned datasets and ignored
    otherwise.
    """
    _require_known(dataset)
    patterns = DATASETS[dataset]
    if dataset in SEASONLESS:
        return list(patterns)
    if season is None:
        raise ValueError(f"dataset {dataset!r} is season-partitioned; pass a season")
    return [p.format(season=season) for p in patterns]


def url_for(dataset: str, season: int | None = None) -> str:
    """The canonical URL for ``dataset`` — the first candidate."""
    return candidate_urls(dataset, season)[0]


def urls_for(dataset: str, seasons: Iterable[int] | None = None) -> list[str]:
    """Canonical URLs covering ``seasons`` for ``dataset``."""
    if not is_seasonal(dataset):
        return [url_for(dataset)]
    if seasons is None:
        raise ValueError(f"dataset {dataset!r} is season-partitioned; pass seasons")
    return [url_for(dataset, season) for season in seasons]


def filename_for(dataset: str, season: int | None = None) -> str:
    """Local cache filename, taken from the canonical URL.

    Deliberately independent of which fallback actually served the bytes, so
    the cache layout stays stable as upstream naming moves.
    """
    return url_for(dataset, season).rsplit("/", 1)[-1]


def _require_known(dataset: str) -> None:
    if dataset not in DATASETS:
        known = ", ".join(sorted(DATASETS))
        raise KeyError(f"unknown dataset {dataset!r}; known datasets: {known}")
