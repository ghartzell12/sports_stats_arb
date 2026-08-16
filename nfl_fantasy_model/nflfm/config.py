"""Paths and project-wide defaults.

Everything that a user might reasonably want to point somewhere else lives
here, so the rest of the package never hardcodes a path or a season.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repo layout: <project_root>/nflfm/config.py -> project_root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("NFLFM_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# nflverse publishes 1999-present. Modern usage/efficiency stats (air yards,
# EPA, target share) are only reliable from 2006 on, so that is the default
# floor for anything feature-related.
FIRST_SEASON = 1999
FIRST_MODELING_SEASON = 2006

# Positions this model projects. DST/K need team-level and kicking-specific
# handling that the skeleton does not implement yet.
SKILL_POSITIONS = ("QB", "RB", "WR", "TE")

REGULAR_SEASON_WEEKS = 18


@dataclass(frozen=True)
class Paths:
    """Resolved data directories, created on demand."""

    data: Path = field(default=DATA_DIR)
    raw: Path = field(default=RAW_DIR)
    processed: Path = field(default=PROCESSED_DIR)

    def ensure(self) -> "Paths":
        for p in (self.data, self.raw, self.processed):
            p.mkdir(parents=True, exist_ok=True)
        return self


PATHS = Paths()
