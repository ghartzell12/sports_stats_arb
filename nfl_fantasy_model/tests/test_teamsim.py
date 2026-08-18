"""Correlated team simulation and format differences."""

from __future__ import annotations

import numpy as np
import pytest

from nflfm import teamsim
from nflfm.bestball.lineup import DK, LineupRules

POS = np.array(["QB"] + ["RB"] * 3 + ["WR"] * 4 + ["TE"] * 2)
RULES = LineupRules({"QB": 1, "RB": 2, "WR": 3, "TE": 1}, ("RB", "WR", "TE"), 1, 10)


def test_normal_cdf_matches_the_reference():
    import math
    z = np.linspace(-4, 4, 33)
    ref = np.array([0.5 * (1 + math.erf(v / math.sqrt(2))) for v in z])
    assert np.abs(teamsim._norm_cdf(z) - ref).max() < 1e-6


def test_teammates_correlate_and_strangers_do_not():
    R = teamsim.correlation_matrix(["QB", "WR", "WR"], ["CIN", "CIN", "SEA"])
    assert R[0, 1] == pytest.approx(teamsim.SAME_TEAM[("QB", "WR")])
    assert R[0, 2] == 0.0
    assert np.allclose(R, R.T)
    assert np.diag(R) == pytest.approx(1.0)


def test_opponents_correlate_only_when_facing_each_other():
    R = teamsim.correlation_matrix(["QB", "QB"], ["CIN", "BAL"], opponents=["BAL", "CIN"])
    assert R[0, 1] == pytest.approx(teamsim.OPPONENT[("QB", "QB")])


def test_matrix_is_always_factorizable():
    """Pairwise measurements need not be consistent; the repair must hold."""
    R = teamsim.correlation_matrix(["QB"] * 3 + ["WR"] * 3, ["A"] * 6)
    np.linalg.cholesky(R)  # raises if not positive definite
    assert (np.linalg.eigvalsh(R) > 0).all()


def test_copula_preserves_each_player_marginal():
    """Correlation must couple players without distorting their own distribution."""
    rng = np.random.default_rng(0)
    pools = [np.array([1.0, 5.0, 20.0]), np.array([2.0, 4.0, 30.0])]
    R = teamsim.correlation_matrix(["QB", "WR"], ["A", "A"])
    out = teamsim.correlated_scores(pools, np.array([1.0, 1.0]), R, 4000, 5, rng)
    assert set(np.unique(out[:, :, 0])) == {1.0, 5.0, 20.0}
    assert set(np.unique(out[:, :, 1])) == {2.0, 4.0, 30.0}


def test_correlation_shows_up_in_the_draws():
    rng = np.random.default_rng(1)
    pools = [np.linspace(0.1, 3.0, 400)] * 2
    ppg = np.array([10.0, 10.0])
    linked = teamsim.correlated_scores(pools, ppg,
        teamsim.correlation_matrix(["QB", "WR"], ["A", "A"]), 3000, 6, rng)
    apart = teamsim.correlated_scores(pools, ppg,
        teamsim.correlation_matrix(["QB", "WR"], ["A", "B"]), 3000, 6, rng)
    r = lambda d: np.corrcoef(d[:, :, 0].ravel(), d[:, :, 1].ravel())[0, 1]
    assert r(linked) > 0.25
    assert abs(r(apart)) < 0.05


def test_common_random_numbers_are_actually_reused():
    rng = np.random.default_rng(2)
    pools = [np.linspace(0.5, 2.0, 50)] * 3
    ppg = np.full(3, 10.0)
    Z = rng.standard_normal((200 * 4, 3))
    R = teamsim.correlation_matrix(["QB", "WR", "WR"], ["A", "B", "C"])
    a = teamsim.correlated_scores(pools, ppg, R, 200, 4, rng, base_normals=Z)
    b = teamsim.correlated_scores(pools, ppg, R, 200, 4, rng, base_normals=Z)
    assert np.array_equal(a, b)


def test_best_ball_never_scores_below_a_fixed_lineup():
    """The formats' defining difference: one picks after, one picks before."""
    rng = np.random.default_rng(4)
    weekly = rng.gamma(2.0, 6.0, size=(300, 17, len(POS)))
    projection = weekly.mean(axis=(0, 1))
    bb = teamsim.best_ball_totals(weekly, POS, RULES)
    mg = teamsim.managed_totals(weekly, POS, projection, RULES)
    assert (bb >= mg - 1e-9).all()
    assert bb.mean() > mg.mean()


def test_stacking_moves_spread_not_mean():
    """The central result: correlation is a tail bet, never an EV gain."""
    rng = np.random.default_rng(7)
    pools = [np.linspace(0.05, 2.6, 600)] * len(POS)
    ppg = np.full(len(POS), 12.0)
    Z = rng.standard_normal((4000 * 17, len(POS)))
    apart = [f"T{i}" for i in range(len(POS))]
    together = list(apart); together[0] = together[4] = "STK"

    def totals(teams):
        R = teamsim.correlation_matrix(list(POS), teams)
        wk = teamsim.correlated_scores(pools, ppg, R, 4000, 17, rng, base_normals=Z)
        return teamsim.best_ball_totals(wk, POS, RULES)

    a, b = totals(apart), totals(together)
    assert b.mean() == pytest.approx(a.mean(), rel=0.01)   # EV unchanged
    assert b.std() > a.std()                                # spread widens


def test_projection_noise_costs_a_managed_manager_points():
    rng = np.random.default_rng(8)
    weekly = rng.gamma(2.0, 6.0, size=(400, 17, len(POS)))
    projection = weekly.mean(axis=(0, 1))
    clean = teamsim.managed_totals(weekly, POS, projection, RULES)
    noisy = teamsim.managed_totals(weekly, POS, projection, RULES, noise=0.6, rng=rng)
    assert noisy.mean() < clean.mean()


def test_format_result_summary_reports_the_tail():
    r = teamsim.FormatResult("bb", np.arange(1000.0))
    s = r.summary()
    assert s["p99"] > s["p90"] > s["p50"]
