from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl

from arena_evaluation.processing.path.theta_star import StaticGrid, plan_route
from flat_peds import flat_peds

SEC = 1_000_000_000
PED_RADIUS_M = 0.3


def _peds(frames: Sequence[tuple[int, Sequence[Mapping[str, object]]]]) -> pl.DataFrame:
    """Flat pedestrian rows with each pedestrian's recorded body radius."""
    return flat_peds(frames).with_columns(pl.lit(PED_RADIUS_M, dtype=pl.Float64).alias("radius_m"))


def test_geometric_theta_star_obstacle_detour() -> None:
    """Theta* around a center wall must navigate around corners with line-of-sight segments."""
    lethal = np.zeros((100, 100), dtype=bool)
    # Put a vertical wall in the middle x=50, y from 20 to 80
    lethal[20:80, 48:52] = True
    start = (1.0, 2.5)  # left of wall
    goal = (4.0, 2.5)  # right of wall

    res = plan_route(StaticGrid(origin_xy=(0.0, 0.0), resolution_m=0.05, lethal=lethal), start, goal)

    assert res.null_reason is None
    assert res.length_m > np.hypot(3.0, 0.0)  # Path must be longer than straight line
    assert len(res.points) >= 3  # Start, corner waypoint(s), goal
