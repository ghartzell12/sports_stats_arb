"""Correlated team simulation, and the difference between formats.

Two things a player-by-player board cannot see:

**Teammates move together.** A quarterback and his top receiver share weeks —
measured at 0.39 on residuals. Rostering both does not change either player's
own distribution, but it widens the *team's*, because their good weeks land on
top of each other. Whether that is good depends entirely on the format.

**Best ball and managed leagues bank different things.** Best ball sets your
lineup after the week is played, so a spike always counts and a bust always
sits. A managed league makes you choose first: you start players on what you
expected, and you eat whatever happens. That single difference — lineup chosen
ex post versus ex ante — is what makes volatility an asset in one format and
roughly a liability in the other.

Correlation is induced with a Gaussian copula so each player keeps the exact
empirical weekly shape measured from his comparables, and only the dependence
between players is imposed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bestball.lineup import DK, LineupRules, best_lineup_score

#: Residual correlations measured over 2014-2025 (see nflfm.correlation).
#: Keyed by (position_a, position_b) with players ranked within their team.
SAME_TEAM: dict[tuple[str, str], float] = {
    ("QB", "WR"): 0.34,   # averaged across WR1-WR3; WR1 alone is 0.387
    ("QB", "TE"): 0.29,
    ("QB", "RB"): 0.07,
    ("WR", "WR"): -0.02,
    ("WR", "TE"): -0.03,
    ("RB", "WR"): -0.04,
    ("RB", "TE"): -0.03,
    ("RB", "RB"): -0.06,
    ("TE", "TE"): -0.04,
    ("QB", "QB"): -0.18,
}

#: Players facing each other in the same game.
OPPONENT: dict[tuple[str, str], float] = {
    ("QB", "QB"): 0.20,
    ("QB", "WR"): 0.09,
    ("QB", "TE"): 0.08,
    ("WR", "WR"): 0.04,
    ("RB", "RB"): -0.06,
    ("RB", "QB"): -0.04,
}


def _lookup(table: dict[tuple[str, str], float], a: str, b: str) -> float:
    return table.get((a, b), table.get((b, a), 0.0))


def correlation_matrix(
    positions: list[str],
    teams: list[str],
    opponents: list[str] | None = None,
) -> np.ndarray:
    """Build a roster's correlation matrix from measured relationships."""
    n = len(positions)
    R = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            if teams[i] == teams[j]:
                r = _lookup(SAME_TEAM, positions[i], positions[j])
            elif opponents and (opponents[i] == teams[j] or opponents[j] == teams[i]):
                r = _lookup(OPPONENT, positions[i], positions[j])
            else:
                r = 0.0
            R[i, j] = R[j, i] = r
    return nearest_psd(R)


def nearest_psd(R: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Clip negative eigenvalues so the matrix can be factorized.

    Correlations measured pairwise need not form a consistent matrix; this is
    the standard repair, and the adjustment is tiny for realistic rosters.
    """
    vals, vecs = np.linalg.eigh((R + R.T) / 2.0)
    vals = np.clip(vals, floor, None)
    fixed = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(fixed))
    return fixed / np.outer(d, d)


def correlated_scores(
    pools: list[np.ndarray],
    ppg: np.ndarray,
    R: np.ndarray,
    sims: int,
    weeks: int,
    rng: np.random.Generator,
    base_normals: np.ndarray | None = None,
) -> np.ndarray:
    """Weekly scores of shape ``(sims, weeks, players)`` with dependence ``R``.

    Each player's marginal distribution is his empirical shape pool exactly —
    the copula only couples them.

    Pass ``base_normals`` — an uncorrelated ``(sims * weeks, players)`` draw —
    to reuse the same randomness across configurations. Comparing two rosters
    on independent draws buries a small effect under sampling noise; driving
    both from identical normals makes the *difference* between them far more
    precisely estimated, which is the only quantity of interest here.
    """
    n = len(pools)
    L = np.linalg.cholesky(R)
    raw = rng.standard_normal((sims * weeks, n)) if base_normals is None else base_normals
    z = raw @ L.T
    u = _norm_cdf(z)

    out = np.empty((sims * weeks, n), dtype=float)
    for i, pool in enumerate(pools):
        ordered = np.sort(np.asarray(pool, dtype=float))
        # Empirical quantile by direct indexing: exact for a sample, and far
        # cheaper than np.quantile over hundreds of thousands of draws.
        pos = np.clip((u[:, i] * ordered.size).astype(np.int64), 0, ordered.size - 1)
        out[:, i] = ordered[pos] * ppg[i]
    return out.reshape(sims, weeks, n)


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF, vectorized without scipy.

    Abramowitz & Stegun 7.1.26 for erf; max absolute error 1.5e-7, which is
    far below the sampling noise of any simulation this feeds.
    """
    x = z / np.sqrt(2.0)
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
           + t * (-1.453152027 + t * 1.061405429))))
    erf = sign * (1.0 - poly * np.exp(-ax * ax))
    return 0.5 * (1.0 + erf)


@dataclass(frozen=True)
class FormatResult:
    """Season totals under one format."""

    name: str
    totals: np.ndarray

    def summary(self) -> dict[str, float]:
        return {
            "mean": float(self.totals.mean()),
            "sd": float(self.totals.std()),
            "p50": float(np.quantile(self.totals, 0.50)),
            "p90": float(np.quantile(self.totals, 0.90)),
            "p99": float(np.quantile(self.totals, 0.99)),
        }


def best_ball_totals(
    weekly: np.ndarray,
    positions: np.ndarray,
    rules: LineupRules = DK,
) -> np.ndarray:
    """Lineup chosen after the week is played — the maximum, every week."""
    return best_lineup_score(weekly, positions, rules).sum(axis=-1)


def managed_totals(
    weekly: np.ndarray,
    positions: np.ndarray,
    projection: np.ndarray,
    rules: LineupRules = DK,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Lineup chosen *before* the week from projections, then scored on reality.

    This is the whole difference between the formats. The manager ranks players
    by what he expects — optionally with ``noise`` standard deviations of
    judgement error — locks the lineup, and banks whatever those players
    actually do, busts included.
    """
    sims, weeks, n = weekly.shape
    belief = np.broadcast_to(projection, (sims, weeks, n)).astype(float)
    if noise and rng is not None:
        belief = belief * np.exp(rng.normal(0.0, noise, size=(sims, weeks, n)))

    total = np.zeros((sims, weeks), dtype=float)
    bench_idx, bench_belief = [], []

    for position, count in rules.slots.items():
        column = np.flatnonzero(positions == position)
        if column.size == 0:
            continue
        take = min(count, column.size)
        order = np.argsort(-belief[:, :, column], axis=-1)
        chosen = column[order[..., :take]]
        total += np.take_along_axis(weekly, chosen, axis=-1).sum(axis=-1)
        if position in rules.flex_eligible and column.size > take:
            rest = column[order[..., take:]]
            bench_idx.append(rest)
            bench_belief.append(np.take_along_axis(belief, rest, axis=-1))

    if rules.flex_count and bench_idx:
        idx = np.concatenate(bench_idx, axis=-1)
        bel = np.concatenate(bench_belief, axis=-1)
        take = min(rules.flex_count, idx.shape[-1])
        order = np.argsort(-bel, axis=-1)[..., :take]
        picked = np.take_along_axis(idx, order, axis=-1)
        total += np.take_along_axis(weekly, picked, axis=-1).sum(axis=-1)

    return total.sum(axis=-1)
