"""Download nflverse assets into the local cache.

The cache is a plain directory of the original parquet files; nothing is
rewritten on the way in, so a cached file is byte-identical to the release
asset and can be inspected with any parquet reader.
"""

from __future__ import annotations

import logging
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence

from ..config import PATHS
from . import sources

log = logging.getLogger(__name__)

USER_AGENT = "nflfm/0.1 (+https://github.com/nflverse/nflverse-data)"
CHUNK = 1 << 20


def download(url: str, dest: Path, force: bool = False) -> Path:
    """Fetch ``url`` to ``dest``, skipping the request if already cached.

    Downloads to a ``.part`` sibling and renames on success so an interrupted
    run never leaves a truncated file that later looks cached.
    """
    return download_first([url], dest, force=force)


def download_first(urls: Sequence[str], dest: Path, force: bool = False) -> Path:
    """Try each URL in order, keeping the first that returns a body.

    Upstream asset naming has moved more than once and some seasons are only
    present under an older layout, so a 404 on the canonical URL is a reason to
    try the next candidate rather than to fail.
    """
    dest = Path(dest)
    if dest.exists() and not force:
        log.debug("cached: %s", dest.name)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    failures: list[str] = []

    for url in urls:
        log.info("downloading %s", url)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request) as response, tmp.open("wb") as fh:
                shutil.copyfileobj(response, fh, CHUNK)
        except urllib.error.HTTPError as exc:
            tmp.unlink(missing_ok=True)
            failures.append(f"{exc.code} {url}")
            continue
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

        tmp.replace(dest)
        return dest

    raise RuntimeError("could not fetch " + dest.name + "; tried: " + "; ".join(failures))


def fetch(
    dataset: str,
    seasons: Iterable[int] | None = None,
    force: bool = False,
) -> list[Path]:
    """Cache ``dataset`` locally and return the paths that now hold it."""
    paths = PATHS.ensure()
    seasons = list(seasons) if seasons is not None else None

    if not sources.is_seasonal(dataset):
        return [
            download_first(
                sources.candidate_urls(dataset),
                paths.raw / sources.filename_for(dataset),
                force=force,
            )
        ]

    if not seasons:
        raise ValueError(f"dataset {dataset!r} is season-partitioned; pass seasons")

    return [
        download_first(
            sources.candidate_urls(dataset, season),
            paths.raw / sources.filename_for(dataset, season),
            force=force,
        )
        for season in seasons
    ]


def fetch_all(seasons: Iterable[int], force: bool = False) -> dict[str, list[Path]]:
    """Cache every dataset in :data:`sources.DATASETS` for ``seasons``.

    ``pbp`` is excluded: it is roughly 100x the size of the others and is only
    needed once feature engineering moves to the play level.
    """
    seasons = list(seasons)
    return {
        name: fetch(name, seasons, force=force)
        for name in sources.DATASETS
        if name != "pbp"
    }
