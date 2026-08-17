"""Weekly best-ball lineup selection.

Best ball scores your highest lineup automatically each week, which is the
whole reason this format rewards different players than a season-total model
would. A player who alternates 4 and 26 points is worth more than a steady
15-point player on the same total, because the 4s land on your bench and the
26s land in your lineup.

The selection itself is greedy and provably optimal for a single flex: fill
each mandatory slot with the best available at that position, then give the
flex the best remaining flex-eligible player. Moving anyone out of a mandatory
slot to free them for the flex only replaces them with someone worse, so no
swap can improve the total.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: DraftKings best ball: 20-man roster, 8 starters each week.
DK_SLOTS: dict[str, int] = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
DK_FLEX_ELIGIBLE: tuple[str, ...] = ("RB", "WR", "TE")
DK_FLEX_COUNT = 1
DK_ROSTER_SIZE = 20


@dataclass(frozen=True)
class LineupRules:
    """Starting-lineup shape for a best ball contest."""

    slots: dict[str, int]
    flex_eligible: tuple[str, ...]
    flex_count: int
    roster_size: int

    @property
    def starters(self) -> int:
        return sum(self.slots.values()) + self.flex_count


DK = LineupRules(
    slots=DK_SLOTS,
    flex_eligible=DK_FLEX_ELIGIBLE,
    flex_count=DK_FLEX_COUNT,
    roster_size=DK_ROSTER_SIZE,
)


def best_lineup_score(
    scores: np.ndarray,
    positions: np.ndarray,
    rules: LineupRules = DK,
) -> np.ndarray:
    """Total of the optimal lineup, vectorized over simulations and weeks.

    ``scores`` has shape ``(..., n_players)`` — any leading dimensions are
    treated as independent lineup decisions, so a ``(sims, weeks, players)``
    array returns ``(sims, weeks)``. ``positions`` is a length-``n_players``
    array of position strings aligned to the last axis.

    Players not eligible anywhere simply never get picked.
    """
    scores = np.asarray(scores, dtype=float)
    positions = np.asarray(positions)
    if scores.shape[-1] != positions.shape[0]:
        raise ValueError(
            f"scores last axis ({scores.shape[-1]}) must match positions "
            f"({positions.shape[0]})"
        )

    total = np.zeros(scores.shape[:-1], dtype=float)
    # Flex candidates are whatever the mandatory slots left behind.
    leftovers: list[np.ndarray] = []

    for position, count in rules.slots.items():
        column = np.flatnonzero(positions == position)
        if column.size == 0:
            continue
        # Descending sort of this position's scores for every leading index.
        ranked = -np.sort(-scores[..., column], axis=-1)
        take = min(count, ranked.shape[-1])
        total += ranked[..., :take].sum(axis=-1)
        if position in rules.flex_eligible and ranked.shape[-1] > take:
            leftovers.append(ranked[..., take:])

    if rules.flex_count and leftovers:
        bench = np.concatenate(leftovers, axis=-1)
        ranked = -np.sort(-bench, axis=-1)
        take = min(rules.flex_count, ranked.shape[-1])
        total += ranked[..., :take].sum(axis=-1)

    return total


def season_totals(
    weekly_scores: np.ndarray,
    positions: np.ndarray,
    rules: LineupRules = DK,
    weeks: slice | None = None,
) -> np.ndarray:
    """Sum the weekly optimal lineups over a set of weeks.

    ``weekly_scores`` is ``(sims, weeks, players)``; the result is ``(sims,)``.
    ``weeks`` restricts the sum, which is how the advance rounds are scored —
    each round counts only its own weeks.
    """
    weekly_scores = np.asarray(weekly_scores, dtype=float)
    if weekly_scores.ndim != 3:
        raise ValueError(f"expected (sims, weeks, players), got {weekly_scores.shape}")
    if weeks is not None:
        weekly_scores = weekly_scores[:, weeks, :]
    return best_lineup_score(weekly_scores, positions, rules).sum(axis=-1)


def lineup_usage_rate(
    weekly_scores: np.ndarray,
    positions: np.ndarray,
    rules: LineupRules = DK,
) -> np.ndarray:
    """Fraction of weeks each player's score actually reaches the lineup.

    This is the diagnostic that exposes what a season-total model misses: a
    boom/bust receiver can start in far fewer weeks than a steady one and
    still be worth more, because the weeks he does start are the big ones.
    Returns one rate per player.
    """
    weekly_scores = np.asarray(weekly_scores, dtype=float)
    positions = np.asarray(positions)
    sims, weeks, n_players = weekly_scores.shape

    started = np.zeros(n_players, dtype=float)
    flat = weekly_scores.reshape(-1, n_players)

    bench_idx: list[np.ndarray] = []
    bench_score: list[np.ndarray] = []

    for position, count in rules.slots.items():
        column = np.flatnonzero(positions == position)
        if column.size == 0:
            continue
        take = min(count, column.size)
        order = np.argsort(-flat[:, column], axis=-1)
        chosen = column[order[:, :take]]
        np.add.at(started, chosen.ravel(), 1.0)

        if position in rules.flex_eligible and column.size > take:
            remaining = column[order[:, take:]]
            bench_idx.append(remaining)
            bench_score.append(np.take_along_axis(flat, remaining, axis=-1))

    if rules.flex_count and bench_idx:
        idx = np.concatenate(bench_idx, axis=-1)
        score = np.concatenate(bench_score, axis=-1)
        take = min(rules.flex_count, idx.shape[-1])
        order = np.argsort(-score, axis=-1)[:, :take]
        np.add.at(started, np.take_along_axis(idx, order, axis=-1).ravel(), 1.0)

    return started / (sims * weeks)
