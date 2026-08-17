"""League scoring: rules as data, plus a vectorized applier."""

from .frame import add_fantasy_points, score_frame
from .rules import (
    DK_BEST_BALL,
    HALF_PPR,
    PPR,
    PRESETS,
    STANDARD,
    TE_PREMIUM,
    ScoringRules,
    preset,
)

__all__ = [
    "DK_BEST_BALL",
    "HALF_PPR",
    "PPR",
    "PRESETS",
    "STANDARD",
    "TE_PREMIUM",
    "ScoringRules",
    "add_fantasy_points",
    "preset",
    "score_frame",
]
