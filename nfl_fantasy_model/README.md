# NFL Fantasy Model

A projection model for NFL fantasy football, built on the public
[nflverse](https://github.com/nflverse/nflverse-data) data releases.

This is the **skeleton**: data ingestion, league scoring, leak-free feature
construction, baseline projectors, and walk-forward backtesting all work
end to end. The interesting modeling — matchup, usage, and a learned
projector — is stubbed with the design written down and `NotImplementedError`
where the code goes.

## Quick start

```bash
pip install -e ".[dev]"     # or: pip install -r requirements.txt

python -m nflfm fetch --start 2018 --end 2025    # cache the data locally
python -m nflfm backtest --start 2018 --end 2025 --scoring ppr
pytest                                           # 49 tests, no network needed
```

The backtest above, run against live data:

```
scoring: ppr   seasons: 2018-2025
         model   mae  rmse  spearman
rolling_mean_5 4.731 6.511     0.627
rolling_mean_3 4.871 6.752     0.612
     last_game 5.649 8.029     0.537
```

That is the bar to clear. A five-game trailing average projects PPR points to
within ~4.7 per game and ranks players at 0.63 Spearman; any model added here
has to beat those numbers on held-out seasons before it is worth using.

## Layout

```
nflfm/
  config.py        paths, season constants, position scope
  data/
    sources.py     nflverse release URLs, with fallbacks for renamed assets
    ingest.py      caching downloader (atomic writes, candidate fallback)
    loaders.py     parquet -> DataFrame, fetching on demand
  scoring/
    rules.py       ScoringRules dataclass + standard/half-PPR/PPR/TE-premium
    frame.py       vectorized application to a weekly frame
  features/
    build.py       lags, rolling means, career games; matchup/usage stubs
  models/
    base.py        Projector interface (fit/predict, sklearn-shaped)
    baseline.py    RollingMeanProjector, LastGameProjector
    evaluate.py    walk-forward season splits, MAE/RMSE/Spearman
  cli.py           `nflfm fetch`, `nflfm backtest`
tests/             unit tests; test_integration.py is network-gated
data/raw/          download cache (gitignored)
```

## Design decisions worth knowing

**Scoring is data, not code.** `ScoringRules` holds per-unit point values, so
switching from PPR to a TE-premium league is a different object, not a
different code path. As a correctness check, our PPR and standard rules
reproduce nflverse's own `fantasy_points_ppr` and `fantasy_points` columns
exactly across all 11,665 player-weeks in 2023–24 (`test_integration.py`).

**Features never see the present.** A row for week W may only use data from
weeks before W. The shift lives inside `add_lags` / `add_rolling_means` rather
than being the caller's job, and `tests/test_features.py` tests the leak
directly — a rolling mean that includes the current week is the single easiest
way to build a model that backtests brilliantly and fails on Sunday.

**Evaluation walks forward.** `season_splits` trains on seasons `< N` and tests
on season `N`. A random split leaks the future into the past and flatters
everything.

**Baselines ship first.** In fantasy, the naive alternatives are strong. A
model that cannot beat a trailing average is not a model.

**Upstream naming moves.** nflverse has renamed the weekly stats asset twice,
and 2019 is missing from one of the layouts. `sources.py` lists candidates in
preference order and `ingest.download_first` falls through 404s, while the
local cache filename stays pinned to the canonical name.

## Data

Everything comes from nflverse GitHub releases — no API key, no scraping.

| Dataset | Contents | Coverage |
| --- | --- | --- |
| `weekly_stats` | player-week box scores, 150 columns | 1999– |
| `rosters` | weekly roster and status | 1999– |
| `snap_counts` | offensive/defensive/ST snap shares | 2012– |
| `schedules` | results, spreads, totals, venue | 1999– |
| `players` | ids, birthdate, draft slot, physicals | all |
| `pbp` | play-by-play (large; excluded from `fetch_all`) | 1999– |

Modeling defaults to 2006 onward (`config.FIRST_MODELING_SEASON`) — that is
when air yards, EPA, and target share become reliable.

## What is stubbed

| Piece | Where | Plan |
| --- | --- | --- |
| Matchup features | `features/build.py:add_matchup_features` | opponent points allowed by position (shrunk toward league mean), implied team total from spread + over/under, home/away, rest, venue |
| Usage features | `features/build.py:add_usage_features` | trailing snap share and route participation, plus a role-change flag — usage is more stable week to week than production |
| Learned projector | `models/` | gradient-boosted trees on the feature frame, measured against the baselines above |
| K and DST | scoring + models | need team-level aggregation and kicking distance buckets; `SKILL_POSITIONS` deliberately excludes them for now |

## Scoring presets

`standard`, `half_ppr`, `ppr`, `te_premium` are built in. Anything else is a
`ScoringRules(...)` literal:

```python
from nflfm.scoring import ScoringRules

league = ScoringRules(
    name="my_league",
    receptions=1.0,
    passing_tds=6.0,
    reception_premiums={"TE": 0.5},
    receiving_yard_bonuses={100: 3.0},
)
```
