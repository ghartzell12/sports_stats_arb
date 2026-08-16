"""League scoring rules.

Fantasy points are the model's target variable, and every league scores
differently, so scoring is kept as data (a :class:`ScoringRules` instance)
rather than baked into the feature or model code. Swap the rules and the same
pipeline projects for a different league.

The linear part of every common ruleset is a dot product of box-score columns
and per-unit point values; bonuses are the only non-linear piece.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

#: Box-score columns that contribute a fumble lost, summed before scoring.
FUMBLE_COLUMNS = ("rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost")


@dataclass(frozen=True)
class ScoringRules:
    """Per-unit point values for a league.

    Yardage values are points *per yard* (0.04 == one point per 25 yards) so
    that the whole ruleset stays a single linear map.
    """

    name: str = "custom"

    # Passing
    passing_yards: float = 0.04
    passing_tds: float = 4.0
    passing_interceptions: float = -2.0
    passing_2pt_conversions: float = 2.0

    # Rushing
    rushing_yards: float = 0.1
    rushing_tds: float = 6.0
    rushing_2pt_conversions: float = 2.0

    # Receiving
    receptions: float = 0.0
    receiving_yards: float = 0.1
    receiving_tds: float = 6.0
    receiving_2pt_conversions: float = 2.0

    # Everything else
    fumbles_lost: float = -2.0
    special_teams_tds: float = 6.0

    # Position premiums: extra points per reception, by position.
    reception_premiums: Mapping[str, float] = field(default_factory=dict)

    # Yardage bonuses: {threshold_yards: points}, awarded once when met.
    passing_yard_bonuses: Mapping[int, float] = field(default_factory=dict)
    rushing_yard_bonuses: Mapping[int, float] = field(default_factory=dict)
    receiving_yard_bonuses: Mapping[int, float] = field(default_factory=dict)

    def linear_weights(self) -> dict[str, float]:
        """The ruleset as ``{box_score_column: points_per_unit}``."""
        return {
            "passing_yards": self.passing_yards,
            "passing_tds": self.passing_tds,
            "passing_interceptions": self.passing_interceptions,
            "passing_2pt_conversions": self.passing_2pt_conversions,
            "rushing_yards": self.rushing_yards,
            "rushing_tds": self.rushing_tds,
            "rushing_2pt_conversions": self.rushing_2pt_conversions,
            "receptions": self.receptions,
            "receiving_yards": self.receiving_yards,
            "receiving_tds": self.receiving_tds,
            "receiving_2pt_conversions": self.receiving_2pt_conversions,
            "special_teams_tds": self.special_teams_tds,
        }

    def score_stats(self, stats: Mapping[str, float]) -> float:
        """Score one player-week given a mapping of box-score stats.

        Missing keys count as zero, so partial stat lines are fine.
        """
        points = sum(
            weight * _num(stats.get(column))
            for column, weight in self.linear_weights().items()
        )

        fumbles = sum(_num(stats.get(column)) for column in FUMBLE_COLUMNS)
        points += self.fumbles_lost * fumbles

        premium = self.reception_premiums.get(str(stats.get("position", "")), 0.0)
        if premium:
            points += premium * _num(stats.get("receptions"))

        for column, bonuses in (
            ("passing_yards", self.passing_yard_bonuses),
            ("rushing_yards", self.rushing_yard_bonuses),
            ("receiving_yards", self.receiving_yard_bonuses),
        ):
            yards = _num(stats.get(column))
            points += sum(pts for threshold, pts in bonuses.items() if yards >= threshold)

        return points


def _num(value) -> float:
    """Coerce a possibly-missing stat to a float."""
    if value is None:
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if result != result else result  # NaN -> 0


STANDARD = ScoringRules(name="standard", receptions=0.0)
HALF_PPR = ScoringRules(name="half_ppr", receptions=0.5)
PPR = ScoringRules(name="ppr", receptions=1.0)
TE_PREMIUM = ScoringRules(
    name="te_premium",
    receptions=1.0,
    reception_premiums={"TE": 0.5},
)

#: Rulesets addressable by name from the CLI.
PRESETS: dict[str, ScoringRules] = {
    r.name: r for r in (STANDARD, HALF_PPR, PPR, TE_PREMIUM)
}


def preset(name: str) -> ScoringRules:
    """Look up a named ruleset."""
    try:
        return PRESETS[name]
    except KeyError:
        known = ", ".join(sorted(PRESETS))
        raise KeyError(f"unknown scoring preset {name!r}; known presets: {known}") from None
