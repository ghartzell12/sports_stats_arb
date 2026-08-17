"""Best ball lineup selection."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from nflfm.bestball import DK, LineupRules, best_lineup_score, lineup_usage_rate, season_totals

# A minimal roster: 1 QB, 3 RB, 4 WR, 2 TE.
POSITIONS = np.array(["QB"] + ["RB"] * 3 + ["WR"] * 4 + ["TE"] * 2)


def brute_force(scores, positions, rules=DK):
    """Exhaustive best lineup, for checking the greedy shortcut."""
    n = len(scores)
    best = -np.inf
    for combo in itertools.permutations(range(n), rules.starters):
        qb, rb1, rb2, wr1, wr2, wr3, te, flex = combo
        if positions[qb] != "QB":
            continue
        if not all(positions[i] == "RB" for i in (rb1, rb2)):
            continue
        if not all(positions[i] == "WR" for i in (wr1, wr2, wr3)):
            continue
        if positions[te] != "TE":
            continue
        if positions[flex] not in rules.flex_eligible:
            continue
        best = max(best, sum(scores[i] for i in combo))
    return best


def test_greedy_matches_brute_force_on_random_rosters():
    rng = np.random.default_rng(7)
    for _ in range(25):
        scores = rng.gamma(2.0, 6.0, size=len(POSITIONS))
        assert best_lineup_score(scores, POSITIONS) == pytest.approx(
            brute_force(scores, POSITIONS)
        )


def test_flex_takes_the_best_leftover():
    scores = np.zeros(len(POSITIONS))
    scores[0] = 20.0  # QB
    scores[1:4] = [10.0, 9.0, 8.0]  # RBs: top 2 start, 8.0 is the leftover
    scores[4:8] = [7.0, 6.0, 5.0, 4.0]  # WRs: top 3 start, 4.0 leftover
    scores[8:10] = [30.0, 25.0]  # TEs: 30 starts, 25 is the best leftover
    # 20 + (10+9) + (7+6+5) + 30 + flex 25
    assert best_lineup_score(scores, POSITIONS) == pytest.approx(112.0)


def test_bench_points_are_discarded():
    """The core of the format: only the best lineup counts."""
    scores = np.ones(len(POSITIONS))
    with_bench_spike = scores.copy()
    with_bench_spike[3] = 0.5  # a worse RB3 cannot change the total
    assert best_lineup_score(scores, POSITIONS) == best_lineup_score(
        with_bench_spike, POSITIONS
    )


def test_spiky_player_outscores_steady_one_on_equal_totals():
    """Why weekly distributions matter more than season totals in best ball."""
    # Needs a real bench (10 players, 8 starters) or nothing can be discarded.
    positions = np.array(["QB"] + ["RB"] * 3 + ["WR"] * 4 + ["TE"] * 2)
    rules = LineupRules({"QB": 1, "RB": 2, "WR": 3, "TE": 1}, ("RB", "WR", "TE"), 1, 10)

    base = np.full((1, 4, len(positions)), 10.0)
    steady, spiky = base.copy(), base.copy()
    steady[0, :, 9] = [12.0, 12.0, 12.0, 12.0]  # TE2, 48 points, flat
    spiky[0, :, 9] = [0.0, 0.0, 0.0, 48.0]  # TE2, same 48 points, one spike

    # Steady TE2 wins the flex every week for 48. Spiky TE2 is benched three
    # times (the 10-point RB3 starts instead) and returns 48 once, for 78.
    assert season_totals(steady, positions, rules) == pytest.approx(328.0)
    assert season_totals(spiky, positions, rules) == pytest.approx(358.0)


def test_usage_rates_fill_every_slot():
    rng = np.random.default_rng(3)
    scores = rng.gamma(2.0, 6.0, size=(40, 17, len(POSITIONS)))
    rates = lineup_usage_rate(scores, POSITIONS)
    # 8 starters spread across the roster, every week.
    assert rates.sum() == pytest.approx(DK.starters)
    assert rates[0] == pytest.approx(1.0)  # the only QB always starts
    assert (rates <= 1.0).all()


def test_season_totals_can_restrict_to_a_slice_of_weeks():
    scores = np.full((2, 17, len(POSITIONS)), 1.0)
    full = season_totals(scores, POSITIONS)
    playoffs = season_totals(scores, POSITIONS, weeks=slice(14, 17))
    assert full == pytest.approx(17 * 8)
    assert playoffs == pytest.approx(3 * 8)


def test_mismatched_positions_length_is_rejected():
    with pytest.raises(ValueError, match="must match positions"):
        best_lineup_score(np.ones(5), POSITIONS)
