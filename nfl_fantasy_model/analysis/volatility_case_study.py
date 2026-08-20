"""What volatility is worth, on two real players.

Finds pairs of receivers from a season who scored at nearly the same rate but
with very different week-to-week spread, then prices the difference by dropping
each into an identical roster and simulating both formats.

The 2025 pair this surfaces — Courtland Sutton and Michael Wilson — separated by
3.9 points across a whole season, is the cleanest available demonstration that a
season total does not determine a player's value in best ball.

    python analysis/volatility_case_study.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nflfm import comparables, scoring, teamsim
from nflfm.bestball.lineup import DK
from nflfm.data import load_weekly_stats

SEASON = 2025
MIN_GAMES = 16          # a full season, so the comparison is not about availability
MIN_PPG = 8.0
PPG_TOLERANCE = 0.25
SIMS = 60_000
WR_REPLACEMENT_PPG = 11.19   # measured, see rank.historical_replacement_ppg


def matched_pairs(weekly: pd.DataFrame) -> pd.DataFrame:
    """Receivers with near-identical scoring rates, ordered by spread gap."""
    wr = weekly[weekly["position"] == "WR"]
    agg = (
        wr.groupby(["player_display_name", "team"], observed=True)["fp"]
        .agg(games="size", ppg="mean", sd="std")
        .reset_index()
    )
    full = agg[(agg["games"] >= MIN_GAMES) & (agg["ppg"] >= MIN_PPG)].reset_index(drop=True)

    rows = []
    for i in range(len(full)):
        for j in range(i + 1, len(full)):
            a, b = full.loc[i], full.loc[j]
            if abs(a["ppg"] - b["ppg"]) <= PPG_TOLERANCE:
                steady, spiky = (a, b) if a["sd"] < b["sd"] else (b, a)
                rows.append(
                    {
                        "steady": steady["player_display_name"],
                        "spiky": spiky["player_display_name"],
                        "steady_ppg": steady["ppg"],
                        "spiky_ppg": spiky["ppg"],
                        "steady_sd": steady["sd"],
                        "spiky_sd": spiky["sd"],
                        "sd_gap": spiky["sd"] - steady["sd"],
                    }
                )
    return pd.DataFrame(rows).sort_values("sd_gap", ascending=False).reset_index(drop=True)


def price_pair(steady: np.ndarray, spiky: np.ndarray, index, rng) -> pd.DataFrame:
    """Drop each player into the same roster and score both formats.

    Uses one shared draw of normals so the two runs differ only by the player
    swapped in, not by sampling luck.
    """
    spec = (
        [("QB", 19.0), ("QB", 14.0)]
        + [("RB", v) for v in (17.0, 15.0, 12.5, 10.5, 9.0, 8.0)]
        + [("WR", v) for v in (17.5, 15.5, 14.0, 0.0, 11.0, 10.0, 9.0, 8.0)]
        + [("TE", v) for v in (13.0, 9.5, 8.0, 7.0)]
    )
    slot = 11  # the WR4 spot the player under test occupies
    positions = np.array([p for p, _ in spec])
    ppg = np.array([v for _, v in spec])

    def pool(position: str, level: float) -> np.ndarray:
        query = {"ppg": level}
        for column in comparables.FEATURES[position]:
            query.setdefault(
                column,
                float(index.features[index.features["position"] == position][column].median()),
            )
        return index.shape_pool(position, query, k=comparables.DEFAULT_K)

    pools = [pool(p, v) if v > 0 else None for p, v in spec]
    matrix = teamsim.correlation_matrix(list(positions), [f"T{i}" for i in range(len(spec))])
    shared = rng.standard_normal((SIMS * 17, len(spec)))

    rows = []
    for label, log in (("steady", steady), ("spiky", spiky)):
        these = list(pools)
        these[slot] = log / log.mean()          # his own shape, scale-free
        levels = ppg.copy()
        levels[slot] = log.mean()               # his own scoring rate
        weekly = teamsim.correlated_scores(
            these, levels, matrix, SIMS, 17, rng, base_normals=shared
        )
        best_ball = teamsim.best_ball_totals(weekly, positions, DK)
        managed = teamsim.managed_totals(weekly, positions, levels, DK)
        rows.append(
            {
                "player": label,
                "best_ball_mean": best_ball.mean(),
                "best_ball_p99": np.quantile(best_ball, 0.99),
                "managed_mean": managed.mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(21)
    weekly = load_weekly_stats([SEASON])
    weekly["fp"] = scoring.score_frame(weekly, scoring.DK_BEST_BALL)

    pairs = matched_pairs(weekly)
    print(f"matched pairs in {SEASON}: {len(pairs)}")
    print(pairs.head(5).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    top = pairs.iloc[0]
    logs = {
        name: weekly[weekly["player_display_name"] == name].sort_values("week")["fp"].to_numpy()
        for name in (top["steady"], top["spiky"])
    }
    print()
    for name, log in logs.items():
        print(f"{name:22s} {' '.join(f'{v:5.1f}' for v in log)}")
        print(f"{'':22s} {log.mean():.2f} ppg · {log.sum():.1f} season · sd {log.std(ddof=1):.2f}")

    history = load_weekly_stats(range(2014, SEASON + 1))
    history["fp"] = scoring.score_frame(history, scoring.DK_BEST_BALL)
    index = comparables.build_index(history)

    result = price_pair(logs[top["steady"]], logs[top["spiky"]], index, rng)
    print("\ndropped into an identical roster:")
    print(result.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    bb = result.loc[1, "best_ball_mean"] - result.loc[0, "best_ball_mean"]
    mg = result.loc[1, "managed_mean"] - result.loc[0, "managed_mean"]
    print(f"\nthe volatile player is worth {bb:+.1f} in best ball and {mg:+.1f} managed")
    print(f"their season totals differ by {logs[top['spiky']].sum() - logs[top['steady']].sum():+.1f}")


if __name__ == "__main__":
    main()
