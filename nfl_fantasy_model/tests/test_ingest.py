"""Caching and fallback behaviour of the downloader (no network required)."""

from __future__ import annotations

import urllib.error

import pytest

from nflfm.data import ingest


class FakeResponse:
    """Minimal stand-in for the object ``urlopen`` returns."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self, size: int = -1) -> bytes:
        body, self._body = self._body, b""
        return body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def server(monkeypatch):
    """Serve only the URLs in ``available``, recording every attempt."""
    state = {"available": {}, "attempts": []}

    def fake_urlopen(request, *args, **kwargs):
        url = request.full_url
        state["attempts"].append(url)
        if url not in state["available"]:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return FakeResponse(state["available"][url])

    monkeypatch.setattr(ingest.urllib.request, "urlopen", fake_urlopen)
    return state


def test_download_writes_the_body(server, tmp_path):
    server["available"]["http://x/a.parquet"] = b"payload"
    dest = ingest.download("http://x/a.parquet", tmp_path / "a.parquet")
    assert dest.read_bytes() == b"payload"


def test_cached_file_is_not_refetched(server, tmp_path):
    dest = tmp_path / "a.parquet"
    dest.write_bytes(b"old")
    ingest.download("http://x/a.parquet", dest)
    assert server["attempts"] == []
    assert dest.read_bytes() == b"old"


def test_force_refetches_over_the_cache(server, tmp_path):
    server["available"]["http://x/a.parquet"] = b"new"
    dest = tmp_path / "a.parquet"
    dest.write_bytes(b"old")
    ingest.download("http://x/a.parquet", dest, force=True)
    assert dest.read_bytes() == b"new"


def test_fallback_is_used_when_the_canonical_url_is_missing(server, tmp_path):
    server["available"]["http://x/second.parquet"] = b"payload"
    dest = ingest.download_first(
        ["http://x/first.parquet", "http://x/second.parquet"],
        tmp_path / "cached.parquet",
    )
    assert dest.read_bytes() == b"payload"
    assert server["attempts"] == ["http://x/first.parquet", "http://x/second.parquet"]


def test_later_candidates_are_not_tried_once_one_succeeds(server, tmp_path):
    server["available"]["http://x/first.parquet"] = b"payload"
    ingest.download_first(
        ["http://x/first.parquet", "http://x/second.parquet"], tmp_path / "c.parquet"
    )
    assert server["attempts"] == ["http://x/first.parquet"]


def test_every_candidate_failing_reports_all_of_them(server, tmp_path):
    with pytest.raises(RuntimeError, match="404 http://x/a.parquet; 404 http://x/b.parquet"):
        ingest.download_first(
            ["http://x/a.parquet", "http://x/b.parquet"], tmp_path / "c.parquet"
        )


def test_a_failed_download_leaves_no_partial_file(server, tmp_path):
    dest = tmp_path / "c.parquet"
    with pytest.raises(RuntimeError):
        ingest.download_first(["http://x/missing.parquet"], dest)
    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []


def test_seasonal_fetch_requires_seasons(server, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="season-partitioned"):
        ingest.fetch("weekly_stats", [])
