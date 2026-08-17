"""Market-implied distributions from alternate prop ladders."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nflfm import market
from nflfm.market import ImpliedDistribution, american_to_probability, devig, fit_ladder


def test_american_odds_convert_both_directions():
    assert american_to_probability(-110) == pytest.approx(0.5238, abs=1e-4)
    assert american_to_probability(+150) == pytest.approx(0.4000, abs=1e-4)
    assert american_to_probability(+100) == pytest.approx(0.5)


def test_zero_odds_are_rejected():
    with pytest.raises(ValueError, match="cannot be zero"):
        american_to_probability(0)


def test_devig_removes_the_margin():
    # -110 both sides is a 4.8% hold; fair is 50/50.
    assert devig(-110, -110) == pytest.approx(0.5)
    assert devig(-200, +170) > 0.5
    assert devig(-110, -110) + devig(-110, -110) == pytest.approx(1.0)


def test_ladder_recovers_a_known_lognormal():
    """Price a ladder off a known distribution; the fit should invert it."""
    mu, sigma = math.log(1000.0), 0.35
    truth = ImpliedDistribution(mu=mu, sigma=sigma, n_rungs=0, residual=0.0)
    rungs = [(line, truth.probability_over(line)) for line in (700, 850, 1000, 1200, 1450)]

    fitted = fit_ladder(rungs)
    assert fitted.mu == pytest.approx(mu, abs=1e-6)
    assert fitted.sigma == pytest.approx(sigma, abs=1e-6)
    assert fitted.median == pytest.approx(1000.0, rel=1e-6)
    assert fitted.residual == pytest.approx(0.0, abs=1e-9)


def test_a_wider_ladder_implies_more_skew():
    """The whole point: the market's spread pins down the shape."""
    tight = fit_ladder([(900, 0.62), (1000, 0.50), (1100, 0.38)])
    wide = fit_ladder([(600, 0.62), (1000, 0.50), (1600, 0.38)])
    assert wide.sigma > tight.sigma
    assert wide.skew > tight.skew


def test_mean_exceeds_median_for_a_skewed_fit():
    fitted = fit_ladder([(800, 0.60), (1000, 0.50), (1300, 0.35)])
    assert fitted.mean > fitted.median


def test_quantiles_are_monotone():
    fitted = fit_ladder([(800, 0.60), (1000, 0.50), (1300, 0.35)])
    values = [fitted.quantile(p) for p in (0.1, 0.25, 0.5, 0.75, 0.9)]
    assert values == sorted(values)
    assert fitted.quantile(0.5) == pytest.approx(fitted.median)


def test_sampling_matches_the_fitted_moments():
    fitted = fit_ladder([(800, 0.60), (1000, 0.50), (1300, 0.35)])
    draws = fitted.sample(200_000, np.random.default_rng(0))
    assert draws.mean() == pytest.approx(fitted.mean, rel=0.02)
    assert np.median(draws) == pytest.approx(fitted.median, rel=0.02)


def test_probability_over_round_trips_the_input_ladder():
    rungs = [(800, 0.60), (1000, 0.50), (1300, 0.35)]
    fitted = fit_ladder(rungs)
    for line, p in rungs:
        assert fitted.probability_over(line) == pytest.approx(p, abs=0.02)


def test_american_ladder_is_devigged_before_fitting():
    fair = fit_ladder([(1000, 0.5)  , (1300, 0.35)])
    priced = market.fit_american_ladder([(1000, -110, -110), (1300, +186, -220)])
    assert priced.mu == pytest.approx(fair.mu, rel=0.05)


def test_degenerate_ladders_are_rejected():
    with pytest.raises(ValueError, match="at least two rungs"):
        fit_ladder([(1000, 0.5)])
    with pytest.raises(ValueError, match="must be in"):
        fit_ladder([(1000, 1.0), (1200, 0.4)])
    with pytest.raises(ValueError, match="distinct lines"):
        fit_ladder([(1000, 0.5), (1000, 0.4)])
