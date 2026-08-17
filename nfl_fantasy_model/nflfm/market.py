"""Market-implied season distributions from alternate prop lines.

A single season-long prop ("Over 1,050.5 receiving yards, -115") pins down one
point of a player's distribution — roughly the median. A *ladder* of alternate
lines on the same market pins down several, and several points of a CDF
determine its shape: the market's own view of how skewed a player's season is,
priced by people with money at stake.

This is strictly better than inferring shape from comparable players, because
it needs no historical analogue at all. A rookie with a full ladder is priced
as precisely as a ten-year veteran, which is exactly the case where comps are
weakest.

The fit is a two-parameter lognormal, chosen because it is closed-form from
the implied quantiles, is positive by construction, and carries the right
qualitative skew for counting stats. Its parameters come from a least-squares
line through the ladder in (z, log line) space, so every rung informs the fit
and no single rung dominates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

_NORMAL = NormalDist()


def american_to_probability(odds: float) -> float:
    """Convert American odds to an implied probability, vig included."""
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def devig(over_odds: float, under_odds: float) -> float:
    """Fair probability of the over, with the book's margin removed.

    Uses proportional (multiplicative) normalization: both sides are scaled so
    they sum to one. This assumes the book's margin is spread evenly across the
    two sides, which is the standard first-order assumption and is adequate for
    the near-even prices typical of season-long totals.
    """
    over = american_to_probability(over_odds)
    under = american_to_probability(under_odds)
    total = over + under
    if total <= 0:
        raise ValueError("implied probabilities must be positive")
    return over / total


@dataclass(frozen=True)
class ImpliedDistribution:
    """A lognormal fitted to a ladder of alternate lines."""

    mu: float
    sigma: float
    n_rungs: int
    residual: float

    @property
    def median(self) -> float:
        return math.exp(self.mu)

    @property
    def mean(self) -> float:
        return math.exp(self.mu + self.sigma**2 / 2.0)

    @property
    def skew(self) -> float:
        """Skewness of the fitted lognormal — the number comps struggle to pin down."""
        variance = math.expm1(self.sigma**2)
        return (variance + 3.0) * math.sqrt(variance)

    def quantile(self, p: float) -> float:
        """Inverse CDF."""
        if not 0.0 < p < 1.0:
            raise ValueError("p must be strictly between 0 and 1")
        return math.exp(self.mu + self.sigma * _NORMAL.inv_cdf(p))

    def probability_over(self, line: float) -> float:
        """Model probability the season total exceeds ``line``."""
        if line <= 0:
            return 1.0
        return 1.0 - _NORMAL.cdf((math.log(line) - self.mu) / self.sigma)

    def sample(self, size, rng: np.random.Generator) -> np.ndarray:
        """Draw season totals."""
        return np.exp(rng.normal(self.mu, self.sigma, size=size))


def fit_ladder(rungs: list[tuple[float, float]]) -> ImpliedDistribution:
    """Fit a lognormal to ``[(line, fair_probability_over), ...]``.

    Needs at least two rungs at distinct lines; more rungs tighten the shape.
    Probabilities must be strictly between 0 and 1 — a rung priced at certainty
    carries no information and would send the fit to infinity.
    """
    cleaned = [(float(line), float(p)) for line, p in rungs if line > 0]
    if len(cleaned) < 2:
        raise ValueError("need at least two rungs at positive lines")
    for line, p in cleaned:
        if not 0.0 < p < 1.0:
            raise ValueError(f"probability for line {line} must be in (0, 1), got {p}")

    lines = np.array([line for line, _ in cleaned], dtype=float)
    if np.unique(lines).size < 2:
        raise ValueError("rungs must sit at at least two distinct lines")

    # P(X > line) = p  =>  P(X <= line) = 1 - p  =>  z = Phi^-1(1 - p)
    z = np.array([_NORMAL.inv_cdf(1.0 - p) for _, p in cleaned], dtype=float)
    log_line = np.log(lines)

    # log(line) = mu + sigma * z, a plain least-squares line.
    design = np.vstack([np.ones_like(z), z]).T
    (mu, sigma), *_ = np.linalg.lstsq(design, log_line, rcond=None)
    if sigma <= 0:
        raise ValueError("fitted sigma is non-positive; check that the ladder is monotone")

    predicted = design @ np.array([mu, sigma])
    residual = float(np.sqrt(np.mean((log_line - predicted) ** 2)))
    return ImpliedDistribution(mu=float(mu), sigma=float(sigma), n_rungs=len(cleaned), residual=residual)


def fit_american_ladder(rungs: list[tuple[float, float, float]]) -> ImpliedDistribution:
    """Fit from raw book prices: ``[(line, over_odds, under_odds), ...]``."""
    return fit_ladder([(line, devig(over, under)) for line, over, under in rungs])


def weekly_shape_from_season(
    distribution: ImpliedDistribution,
    games: int = 17,
) -> float:
    """Per-game coefficient of variation implied by a season-total spread.

    A season total's dispersion reflects both true talent uncertainty and the
    averaging of individual weeks, so this is a floor on weekly dispersion, not
    an estimate of it: weekly variance is strictly larger than what a
    season-long line implies. Use it to *rank* players by expected volatility,
    and take the level of weekly dispersion from comparable players.
    """
    if games <= 0:
        raise ValueError("games must be positive")
    return float(math.sqrt(math.expm1(distribution.sigma**2)))
