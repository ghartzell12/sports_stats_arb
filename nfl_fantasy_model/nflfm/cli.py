"""Command line entry point: ``python -m nflfm``."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

from . import config, scoring
from .data import sources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nflfm",
        description="NFL fantasy football projection model.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log every download")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download nflverse data into the local cache")
    fetch.add_argument(
        "datasets",
        nargs="*",
        default=[],
        metavar="DATASET",
        help=f"datasets to fetch (default: all but pbp). Known: {', '.join(sorted(sources.DATASETS))}",
    )
    fetch.add_argument("--start", type=int, default=2015, help="first season (default: 2015)")
    fetch.add_argument("--end", type=int, default=2024, help="last season, inclusive")
    fetch.add_argument("--force", action="store_true", help="re-download cached files")
    fetch.set_defaults(func=_cmd_fetch)

    backtest = sub.add_parser("backtest", help="score the baselines on held-out seasons")
    backtest.add_argument("--start", type=int, default=2015)
    backtest.add_argument("--end", type=int, default=2024)
    backtest.add_argument(
        "--scoring",
        default="ppr",
        choices=sorted(scoring.PRESETS),
        help="league scoring preset (default: ppr)",
    )
    backtest.add_argument(
        "--min-train-seasons",
        type=int,
        default=3,
        help="seasons of history required before the first held-out season",
    )
    backtest.set_defaults(func=_cmd_backtest)

    return parser


def _cmd_fetch(args: argparse.Namespace) -> int:
    from .data import ingest

    seasons = range(args.start, args.end + 1)
    if args.datasets:
        for dataset in args.datasets:
            paths = ingest.fetch(dataset, seasons, force=args.force)
            print(f"{dataset}: {len(paths)} file(s) in {paths[0].parent}")
    else:
        fetched = ingest.fetch_all(seasons, force=args.force)
        for dataset, paths in fetched.items():
            print(f"{dataset}: {len(paths)} file(s)")
    print(f"cache: {config.PATHS.raw}")
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    from .data import load_weekly_stats
    from .features import build_feature_frame
    from .models import LastGameProjector, RollingMeanProjector, compare

    rules = scoring.preset(args.scoring)
    weekly = load_weekly_stats(range(args.start, args.end + 1))
    weekly = scoring.add_fantasy_points(weekly, rules)
    features = build_feature_frame(weekly)

    models = [LastGameProjector(), RollingMeanProjector(3), RollingMeanProjector(5)]
    table = compare(models, features, min_train_seasons=args.min_train_seasons)

    print(f"\nscoring: {rules.name}   seasons: {args.start}-{args.end}")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
