"""URL construction for the nflverse release assets (no network required)."""

from __future__ import annotations

import pytest

from nflfm.data import sources


def test_seasonal_url_embeds_the_season():
    url = sources.url_for("weekly_stats", 2024)
    assert url.endswith("/stats_player_week_2024.parquet")
    assert url.startswith(sources.BASE)


def test_seasonless_url_ignores_the_season():
    assert sources.url_for("players") == sources.url_for("players", 2024)


def test_seasonal_datasets_require_a_season():
    with pytest.raises(ValueError, match="season-partitioned"):
        sources.url_for("weekly_stats")


def test_candidates_are_distinct_and_lead_with_the_canonical_url():
    candidates = sources.candidate_urls("weekly_stats", 2019)
    assert candidates[0] == sources.url_for("weekly_stats", 2019)
    assert len(candidates) == len(set(candidates)) > 1
    assert all(str(2019) in url for url in candidates)


def test_legacy_asset_name_is_kept_as_a_fallback():
    # 2019 is missing from the intermediate layout upstream; the original
    # player_stats_{season} name still serves it.
    candidates = sources.candidate_urls("weekly_stats", 2019)
    assert any(u.endswith("/player_stats_2019.parquet") for u in candidates)


def test_single_source_datasets_have_one_candidate():
    assert len(sources.candidate_urls("rosters", 2024)) == 1
    assert sources.candidate_urls("schedules") == [sources.SCHEDULES_PATTERNS[0]]


def test_urls_for_covers_every_season():
    urls = sources.urls_for("rosters", [2022, 2023])
    assert [u.rsplit("/", 1)[-1] for u in urls] == [
        "roster_2022.parquet",
        "roster_2023.parquet",
    ]


def test_urls_for_seasonless_returns_one_url():
    assert sources.urls_for("schedules") == [sources.SCHEDULES_PATTERNS[0]]


def test_filename_mirrors_the_canonical_asset():
    assert sources.filename_for("weekly_stats", 2020) == "stats_player_week_2020.parquet"
    assert sources.filename_for("players") == "players.parquet"


def test_cache_filename_is_stable_across_fallbacks():
    # Whichever candidate serves the bytes, the cached name comes from the
    # canonical URL so the cache layout does not move with upstream naming.
    assert sources.filename_for("weekly_stats", 2019) == "stats_player_week_2019.parquet"


def test_unknown_dataset_is_rejected():
    with pytest.raises(KeyError, match="unknown dataset"):
        sources.url_for("box_scores")


def test_is_seasonal_matches_the_seasonless_set():
    assert sources.is_seasonal("weekly_stats")
    assert not sources.is_seasonal("players")
    assert not sources.is_seasonal("schedules")
