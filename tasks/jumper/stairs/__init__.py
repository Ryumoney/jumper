"""jumper.stairs -- climb a straight staircase, tripod gait, stair-height curriculum.

The world is generated per environment by `terrain.py` (a run-up, N steps and a
landing), with the rise growing row by row from nearly flat to `STEP_HEIGHT_RANGE`'s
top. The gait and its observation are `jumper.tripod`'s; what this task adds is the
ground, a forward-only command, and a curriculum that promotes on reaching the top.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.stairs",
    assets=JUMPER_ASSETS,
    description="jumper hexapod climbing a straight staircase; tripod gait, stair-height curriculum",
    tags=("locomotion", "jumper", "stairs", "tripod"),
)
