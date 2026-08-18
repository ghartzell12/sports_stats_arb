# Supplying market data

The model needs a **median level** per player. History supplies shape; only the
market or a projection source can supply level for a season not yet played.

## How to get data in

GitHub is the only host this environment can reach, so the reliable route is:
commit a filled-in CSV to this repo and say so. Pasting the table into chat
also works for a few dozen rows.

## Schema

One row per (player, market, line). See `TEMPLATE_props.csv`.

| column | meaning |
| --- | --- |
| `player` | full name as the book lists it |
| `team`, `position` | used to join against nflverse history |
| `market` | see the list below |
| `line` | the number itself |
| `over_odds`, `under_odds` | American odds. **Both sides**, so vig can be removed properly |
| `book` | which sportsbook |
| `as_of` | date the line was pulled — lines move |

### Markets

`pass_yards`, `pass_tds`, `interceptions`, `rush_yards`, `rush_tds`,
`receptions`, `rec_yards`, `rec_tds`

## Two things that matter more than volume

**Components, not fantasy totals.** A single "projected fantasy points" number
cannot be re-scored into DK rules, because full PPR and the +3 bonuses at
300/100/100 yards need receptions and yards separately. A season-total
projection is close to useless here; component lines are what the model wants.

**Multiple lines per player, where the book offers them.** One line gives a
median. A ladder of alternate lines on the same market gives several points of
the CDF, which pins down the *shape* — the market's own view of a player's
skew. That is strictly better than inferring shape from comparable players,
and it is the only good option for rookies, who have no comparables worth
much. Even two rungs helps; three or four is plenty.

Both sides of each line matter too. A one-sided quote cannot be de-vigged, so
the fair probability has to be guessed at via an assumed overround.

## What happens next

`nflfm.market.fit_american_ladder` de-vigs each rung and fits a lognormal,
returning an implied median, sigma, and skew per player. Those feed the
distributions that drive the draft board.
