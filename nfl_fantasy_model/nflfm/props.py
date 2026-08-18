"""Ingest season-long sportsbook props and turn them into scoring inputs.

The market supplies a player's *level* — the piece history cannot provide for a
season not yet played. What it supplies is component medians (receiving yards,
rushing touchdowns, receptions), which is exactly the right granularity: a
single "projected fantasy points" figure could not be re-scored into DK rules,
because full PPR and the +3 yardage bonuses need the components separately.

Two practical problems this module solves:

* **Books quote different subsets.** Receiving yards are widely posted;
  receptions rarely are. A full-PPR model needs receptions for every pass
  catcher, so missing components are imputed from the ones that *are* quoted,
  using each player's own historical ratios where they exist.
* **Names disagree.** "Travis Etienne" and "Travis Etienne Jr." are one player;
  "Tetairoa McMilan" is a typo for "McMillan". Joining on raw names silently
  splits or drops players.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Book market name -> the component it prices.
MARKET_TO_COMPONENT: dict[str, str] = {
    "Passing Yards": "pass_yards",
    "Passing TDs": "pass_tds",
    "Rushing Yards": "rush_yards",
    "Rushing TDs": "rush_tds",
    "Receptions": "receptions",
    "Receiving Yards": "rec_yards",
    "Receiving TDs": "rec_tds",
    "Interceptions": "interceptions",
}

COMPONENTS: tuple[str, ...] = tuple(dict.fromkeys(MARKET_TO_COMPONENT.values()))

#: Components each position needs before it can be scored.
REQUIRED: dict[str, tuple[str, ...]] = {
    "QB": ("pass_yards", "pass_tds", "interceptions", "rush_yards", "rush_tds"),
    "RB": ("rush_yards", "rush_tds", "receptions", "rec_yards", "rec_tds"),
    "WR": ("receptions", "rec_yards", "rec_tds"),
    "TE": ("receptions", "rec_yards", "rec_tds"),
}

#: Fallback ratios when a player has no usable history, by position. Each maps
#: a missing component to (source component, ratio).
POSITION_RATIOS: dict[str, dict[str, tuple[str, float]]] = {
    "QB": {
        "interceptions": ("pass_yards", 0.0027),
        "rush_yards": ("pass_yards", 0.055),
        "rush_tds": ("rush_yards", 0.0090),
        "pass_tds": ("pass_yards", 0.0065),
    },
    "RB": {
        "receptions": ("rec_yards", 1 / 8.2),
        "rec_yards": ("receptions", 8.2),
        "rec_tds": ("rec_yards", 0.0035),
        "rush_tds": ("rush_yards", 0.0085),
    },
    "WR": {
        "receptions": ("rec_yards", 1 / 12.6),
        "rec_yards": ("receptions", 12.6),
        "rec_tds": ("rec_yards", 0.0060),
    },
    "TE": {
        "receptions": ("rec_yards", 1 / 10.4),
        "rec_yards": ("receptions", 10.4),
        "rec_tds": ("rec_yards", 0.0068),
    },
}

_SUFFIXES = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation and generational suffixes."""
    cleaned = re.sub(r"[^a-z ]", "", str(name).lower())
    cleaned = _SUFFIXES.sub("", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def load_props(path: str) -> pd.DataFrame:
    """Read a props CSV into a tidy frame, one row per quote."""
    frame = pd.read_csv(path).dropna(how="all")
    frame.columns = [c.strip() for c in frame.columns]
    frame = frame.rename(
        columns={
            "Player": "player",
            "Team": "team",
            "Pos": "position",
            "Market": "market",
            "Line": "line",
            "Over": "over_odds",
            "Under": "under_odds",
            "Book": "book",
        }
    )
    frame["component"] = frame["market"].map(MARKET_TO_COMPONENT)
    unknown = frame[frame["component"].isna()]["market"].unique()
    if len(unknown):
        raise ValueError(f"unmapped markets: {list(unknown)}")

    frame["name_norm"] = frame["player"].map(normalize_name)
    frame["line"] = pd.to_numeric(frame["line"], errors="coerce")
    return frame.dropna(subset=["line"])


def consensus_medians(props: pd.DataFrame) -> pd.DataFrame:
    """Average each player's lines across books into one median per component.

    Averaging is appropriate because the supplied no-vig probabilities sit at
    49.7% on average — the books' lines *are* their medians, so no
    probability-weighted adjustment is warranted. Where that stops being true
    the quotes should be re-fitted with :mod:`nflfm.market` instead.
    """
    wide = (
        props.pivot_table(
            index=["name_norm", "position"],
            columns="component",
            values="line",
            aggfunc="mean",
        )
        .reset_index()
    )
    identity = (
        props.sort_values("player")
        .groupby(["name_norm", "position"], observed=True)
        .agg(player=("player", "last"), team=("team", "last"))
        .reset_index()
    )
    merged = identity.merge(wide, on=["name_norm", "position"], how="left")
    for component in COMPONENTS:
        if component not in merged.columns:
            merged[component] = np.nan
    return merged


def historical_levels(weekly: pd.DataFrame) -> pd.DataFrame:
    """Per-player per-season component averages, for level-based imputation.

    Preferred over a ratio when the ratio's denominator is not itself quoted,
    and as a sanity anchor when it is.
    """
    frame = weekly.copy()
    for column in ("receptions", "receiving_yards", "receiving_tds", "rushing_yards",
                   "rushing_tds", "passing_yards", "passing_tds",
                   "passing_interceptions"):
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    per_season = frame.groupby(["name_norm", "season"], observed=True).agg(
        receptions=("receptions", "sum"),
        rec_yards=("receiving_yards", "sum"),
        rec_tds=("receiving_tds", "sum"),
        rush_yards=("rushing_yards", "sum"),
        rush_tds=("rushing_tds", "sum"),
        pass_yards=("passing_yards", "sum"),
        pass_tds=("passing_tds", "sum"),
        interceptions=("passing_interceptions", "sum"),
    )
    return per_season.groupby("name_norm", observed=True).mean()


def winsorize_ratios(ratios: pd.DataFrame, positions: pd.Series,
                     low: float = 0.10, high: float = 0.90) -> pd.DataFrame:
    """Clip each ratio into its position's central band.

    A ratio estimated from a player used very differently than he is about to
    be used can be wildly off — a backup quarterback's rushing yards per
    passing yard, applied to a starter's passing line, produces a rushing
    projection nothing like reality. Clipping to the position's own 10th-90th
    percentile keeps a genuine outlier's direction without its magnitude.
    """
    clipped = ratios.copy()
    joined = clipped.join(positions.rename("position"), how="left")
    for column in ratios.columns:
        bounds = joined.groupby("position", observed=True)[column].quantile([low, high]).unstack()
        lo = joined["position"].map(bounds[low])
        hi = joined["position"].map(bounds[high])
        clipped[column] = clipped[column].clip(lower=lo, upper=hi)
    return clipped


def historical_ratios(weekly: pd.DataFrame) -> pd.DataFrame:
    """Per-player component ratios from recent seasons, for imputation."""
    frame = weekly.copy()
    for column in ("receptions", "receiving_yards", "receiving_tds", "attempts",
                   "passing_yards", "passing_interceptions", "carries",
                   "rushing_yards", "rushing_tds", "passing_tds"):
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    agg = frame.groupby("name_norm", observed=True).agg(
        receptions=("receptions", "sum"),
        rec_yards=("receiving_yards", "sum"),
        rec_tds=("receiving_tds", "sum"),
        pass_yards=("passing_yards", "sum"),
        pass_tds=("passing_tds", "sum"),
        interceptions=("passing_interceptions", "sum"),
        rush_yards=("rushing_yards", "sum"),
        rush_tds=("rushing_tds", "sum"),
    )

    ratios = pd.DataFrame(index=agg.index)
    ratios["ypr"] = _safe_ratio(agg["rec_yards"], agg["receptions"])
    ratios["rec_td_per_yard"] = _safe_ratio(agg["rec_tds"], agg["rec_yards"])
    ratios["int_per_pass_yard"] = _safe_ratio(agg["interceptions"], agg["pass_yards"])
    ratios["rush_yd_per_pass_yd"] = _safe_ratio(agg["rush_yards"], agg["pass_yards"])
    ratios["rush_td_per_yard"] = _safe_ratio(agg["rush_tds"], agg["rush_yards"])
    ratios["pass_td_per_yard"] = _safe_ratio(agg["pass_tds"], agg["pass_yards"])
    return ratios


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


#: Which historical ratio backs each imputation, as
#: component -> (source component, ratio column, invert).
_RATIO_SOURCES: dict[str, tuple[str, str, bool]] = {
    "receptions": ("rec_yards", "ypr", True),
    "rec_yards": ("receptions", "ypr", False),
    "rec_tds": ("rec_yards", "rec_td_per_yard", False),
    "interceptions": ("pass_yards", "int_per_pass_yard", False),
    "rush_yards": ("pass_yards", "rush_yd_per_pass_yd", False),
    "rush_tds": ("rush_yards", "rush_td_per_yard", False),
    "pass_tds": ("pass_yards", "pass_td_per_yard", False),
}


@dataclass
class ImputationReport:
    """Which components were quoted and which had to be filled in."""

    table: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        return (
            self.table.groupby(["component", "method"], observed=True)
            .size()
            .unstack(fill_value=0)
        )


def impute_components(
    medians: pd.DataFrame,
    ratios: pd.DataFrame,
    levels: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, ImputationReport]:
    """Fill each position's required components, recording how each was obtained.

    Preference order: the book's own line, then the player's historical ratio
    against a component that *was* quoted, then the player's own historical
    level for that component, then the position-level ratio.
    """
    frame = medians.copy()
    log = []

    for row_idx, row in frame.iterrows():
        position = row["position"]
        needed = REQUIRED.get(position, ())
        player_ratios = ratios.loc[row["name_norm"]] if row["name_norm"] in ratios.index else None
        player_levels = (
            levels.loc[row["name_norm"]]
            if levels is not None and row["name_norm"] in levels.index
            else None
        )

        for component in needed:
            if pd.notna(row.get(component)):
                log.append({"player": row["player"], "component": component, "method": "quoted"})
                continue

            value, method = _impute_one(
                frame, row_idx, position, component, player_ratios, player_levels
            )
            frame.loc[row_idx, component] = value
            log.append({"player": row["player"], "component": component, "method": method})

    for component in COMPONENTS:
        frame[component] = pd.to_numeric(frame[component], errors="coerce").fillna(0.0)

    return frame, ImputationReport(pd.DataFrame(log))


def _impute_one(frame, row_idx, position, component, player_ratios, player_levels=None):
    """One component, preferring the player's own ratio over the positional one."""
    source_spec = _RATIO_SOURCES.get(component)
    if source_spec is not None:
        source, ratio_column, invert = source_spec
        source_value = frame.loc[row_idx, source] if source in frame.columns else np.nan
        if pd.notna(source_value) and player_ratios is not None:
            ratio = player_ratios.get(ratio_column, np.nan)
            if pd.notna(ratio) and ratio > 0:
                value = source_value / ratio if invert else source_value * ratio
                return float(value), "player_history"

    if player_levels is not None:
        level = player_levels.get(component, np.nan)
        if pd.notna(level) and level > 0:
            return float(level), "player_level"

    fallback = POSITION_RATIOS.get(position, {}).get(component)
    if fallback is not None:
        source, ratio = fallback
        source_value = frame.loc[row_idx, source] if source in frame.columns else np.nan
        if pd.notna(source_value):
            return float(source_value * ratio), "position_ratio"

    return 0.0, "zero"


def base_dk_points(components: pd.DataFrame) -> pd.Series:
    """Season DK points from component medians, excluding yardage bonuses.

    The +3 bonuses at 300 passing / 100 rushing / 100 receiving yards cannot be
    computed from a season total — they are weekly events, and a player's odds
    of clearing them depend on his week-to-week distribution. They are added
    during simulation instead.
    """
    return (
        0.04 * components["pass_yards"]
        + 4.0 * components["pass_tds"]
        - 1.0 * components["interceptions"]
        + 0.1 * components["rush_yards"]
        + 6.0 * components["rush_tds"]
        + 1.0 * components["receptions"]
        + 0.1 * components["rec_yards"]
        + 6.0 * components["rec_tds"]
    )
