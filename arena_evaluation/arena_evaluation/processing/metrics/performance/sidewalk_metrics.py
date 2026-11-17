from __future__ import annotations

import re
import typing

import numpy as np
import polars as pl
import shapely
from shapely.ops import unary_union

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, Direction
from arena_evaluation.processing.metrics.common.alignment import interpolate_values
from arena_evaluation.processing.metrics.ecological.condition_metrics import (
    load_zone_geometry,
)
from arena_evaluation.storage.exceptions import MetricInputError
from arena_evaluation.storage.registry import MetricScope, define_metric
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

GOAL_RADIUS_M = 1.0
FREEZE_SPEED_MPS = 0.05
FREEZE_S = 5.0
FREEZE_GRACE_S = 2.0
EXCURSION_MERGE_S = 1.0
MIN_PATH_M = 1.0
SAMPLE_HZ = 10.0

_WALKABLE = re.compile(r"^(sidewalk|crosswalk)(_\d+)?$")

SIDEWALK_ROUTE_COMPLETION = define_metric(
    "sidewalk_route_completion",
    pl.Boolean,
    scopes=(MetricScope.ROBOT,),
    unit="",
    description="Whether the robot reached the goal on the walkable area without a collision or an off-sidewalk excursion.",
    null_reason=True,
)
SIDEWALK_OFF_SIDEWALK = define_metric(
    "sidewalk_off_sidewalk",
    pl.Boolean,
    scopes=(MetricScope.ROBOT,),
    unit="",
    description="Whether the robot left the sidewalk and crosswalk zones during the run.",
    null_reason=True,
)
SIDEWALK_FREEZES = define_metric(
    "sidewalk_freezes",
    pl.Int64,
    scopes=(MetricScope.ROBOT,),
    unit="",
    description="Number of near-zero-speed dwells longer than five seconds after the two-second start grace.",
    null_reason=True,
)
SIDEWALK_FAILURES_PER_100M = define_metric(
    "sidewalk_failures_per_100m",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="1/100m",
    description="Collisions, off-sidewalk excursions and freezes per 100 m travelled; NULL below the 1 m minimum path.",
    null_reason=True,
)
SIDEWALK_MEAN_SPEED = define_metric(
    "sidewalk_mean_speed",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="m/s",
    description="Mean speed along the travelled path up to the goal sample, m/s.",
    null_reason=True,
)


def walkable_area(
    zones: typing.Iterable[typing.Any],
) -> shapely.geometry.base.BaseGeometry | None:
    """Union of the sidewalk and crosswalk zone polygons, None when the world has none."""
    polygons = [
        shapely.make_valid(zone.polygon) for zone in zones if _WALKABLE.match(zone.name)
    ]
    return unary_union(polygons) if polygons else None


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open index ranges of the maximal True runs of mask."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def excursions(outside: np.ndarray, merge_gap: int) -> int:
    """Number of outside runs, runs closer than merge_gap samples counted once."""
    count = 0
    last_end: int | None = None
    for start, end in _runs(outside):
        if last_end is None or start - last_end >= merge_gap:
            count += 1
        last_end = end
    return count


def freezes(speed: np.ndarray, lo: int, hi: int, min_len: int) -> int:
    """Number of runs of speed under the freeze threshold lasting more than min_len samples inside [lo, hi)."""
    slow = speed < FREEZE_SPEED_MPS
    window = np.zeros(len(speed), dtype=bool)
    window[lo:hi] = True
    return sum(1 for start, end in _runs(slow & window) if end - start > min_len)


def _collisions(episode: AlignedEpisodeBundle) -> int:
    """Number of native collision-event runs of the episode."""
    frame = (episode.topics or {}).get("robot_collision_events")
    if frame is None:
        return 0
    try:
        hits = frame.sort("t_ns")["event_count"].to_numpy().astype(float) > 0
    except pl.exceptions.ColumnNotFoundError as exc:
        raise MetricInputError(
            "robot_collision_events carries no event_count column"
        ) from exc
    return len(_runs(hits))


class SidewalkMetricsCalculator(BaseMetricCalculator):
    """SidewalkBench outcomes of an episode on a world with sidewalk and crosswalk zones."""

    NAME = "sidewalk_metrics"
    CATEGORY = "performance"
    REQUIRED_TOPICS = ["robot_collision_events", "robot_tracks"]

    OUTPUTS = (
        SIDEWALK_ROUTE_COMPLETION,
        SIDEWALK_OFF_SIDEWALK,
        SIDEWALK_FREEZES,
        SIDEWALK_FAILURES_PER_100M,
        SIDEWALK_MEAN_SPEED,
    )
    DIRECTIONS = {
        "sidewalk_route_completion": Direction.HIGHER,
        "sidewalk_off_sidewalk": Direction.LOWER,
        "sidewalk_freezes": Direction.LOWER,
        "sidewalk_failures_per_100m": Direction.LOWER,
        "sidewalk_mean_speed": Direction.HIGHER,
    }
    PRIMARY_OUTPUTS = ["sidewalk_route_completion", "sidewalk_failures_per_100m"]

    def calculate(
        self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        del prior_results
        nulls = dict.fromkeys(self.output_keys())
        zones = load_zone_geometry(self.world) if self.world else None
        area = walkable_area(zones) if zones else None
        if area is None:
            return nulls
        x, y, _, time_ns = self.resolve_native_pose(episode)
        if len(time_ns) < 2:
            return nulls

        t = time_ns / 1e9
        grid = np.arange(t[0], t[-1] + 1e-9, 1.0 / SAMPLE_HZ)
        gx = interpolate_values(t, x, grid)
        gy = interpolate_values(t, y, grid)

        goal = episode.goal_pos
        reached = np.zeros(len(grid), dtype=bool)
        if goal and len(goal) >= 2:
            reached = np.hypot(gx - goal[0], gy - goal[1]) <= GOAL_RADIUS_M
        hit = np.flatnonzero(reached)
        end = int(hit[0]) + 1 if len(hit) else len(grid)

        step = np.hypot(np.diff(gx), np.diff(gy))
        speed = np.append(step, step[-1]) * SAMPLE_HZ
        path_m = float(step[: end - 1].sum())
        duration_s = (end - 1) / SAMPLE_HZ

        outside = ~shapely.intersects(area, shapely.points(gx[:end], gy[:end]))
        off_entries = excursions(outside, round(EXCURSION_MERGE_S * SAMPLE_HZ))
        n_freezes = freezes(
            speed, round(FREEZE_GRACE_S * SAMPLE_HZ), end, round(FREEZE_S * SAMPLE_HZ)
        )
        n_collisions = _collisions(episode)

        completed = bool(len(hit)) and n_collisions == 0 and not outside.any()
        return {
            "sidewalk_route_completion": completed if goal and len(goal) >= 2 else None,
            "sidewalk_off_sidewalk": bool(outside.any()),
            "sidewalk_freezes": n_freezes,
            "sidewalk_failures_per_100m": (n_collisions + off_entries + n_freezes)
            / path_m
            * 100.0
            if path_m >= MIN_PATH_M
            else None,
            "sidewalk_mean_speed": path_m / duration_s if duration_s > 0 else None,
        }
