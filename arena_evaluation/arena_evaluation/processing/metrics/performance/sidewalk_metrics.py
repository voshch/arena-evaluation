from __future__ import annotations

import re
import typing

import numpy as np
import shapely
from shapely.ops import unary_union

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.ecological.condition_metrics import load_zone_geometry
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

GOAL_RADIUS_M = 1.0
FREEZE_SPEED_MPS = 0.05
FREEZE_S = 5.0
FREEZE_GRACE_S = 2.0
EXCURSION_MERGE_S = 1.0
MIN_PATH_M = 1.0
SAMPLE_HZ = 10.0

_WALKABLE = re.compile(r"^(sidewalk|crosswalk)(_\d+)?$")


def walkable_area(zones: typing.Iterable[typing.Any]) -> shapely.geometry.base.BaseGeometry | None:
    """Union of the sidewalk and crosswalk zone polygons, None when the world has none."""
    polygons = [shapely.make_valid(zone.polygon) for zone in zones if _WALKABLE.match(zone.name)]
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
    frame = (episode.topics or {}).get("collision_events")
    if frame is None or "collision_event" not in frame.columns:
        return 0
    hits = frame.sort("time_ns")["collision_event"].to_numpy().astype(float) > 0
    return len(_runs(hits))


class SidewalkMetricsCalculator(BaseMetricCalculator):
    """SidewalkBench outcomes of an episode on a world with sidewalk and crosswalk zones."""

    NAME = "sidewalk_metrics"
    CATEGORY = "performance"
    REQUIRED_TOPICS = ["collision_events", ("tf_gt", "odom")]

    UNITS = {
        "sidewalk_route_completion": "",
        "sidewalk_off_sidewalk": "",
        "sidewalk_freezes": "",
        "sidewalk_failures_per_100m": "1/100m",
        "sidewalk_mean_speed": "m/s",
    }

    PRIMARY_OUTPUTS = ["sidewalk_route_completion", "sidewalk_failures_per_100m"]
    OUTPUT_DIRECTIONS = {
        "sidewalk_route_completion": "higher",
        "sidewalk_off_sidewalk": "lower",
        "sidewalk_freezes": "lower",
        "sidewalk_failures_per_100m": "lower",
        "sidewalk_mean_speed": "higher",
    }

    @classmethod
    def output_keys(cls) -> list[str]:
        return list(cls.UNITS)

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
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
        gx = np.interp(grid, t, x)
        gy = np.interp(grid, t, y)

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

        outside = ~shapely.intersects_xy(area, gx[:end], gy[:end])
        off_entries = excursions(outside, round(EXCURSION_MERGE_S * SAMPLE_HZ))
        n_freezes = freezes(speed, round(FREEZE_GRACE_S * SAMPLE_HZ), end, round(FREEZE_S * SAMPLE_HZ))
        n_collisions = _collisions(episode)

        completed = bool(len(hit)) and n_collisions == 0 and not outside.any()
        return {
            "sidewalk_route_completion": completed if goal and len(goal) >= 2 else None,
            "sidewalk_off_sidewalk": bool(outside.any()),
            "sidewalk_freezes": n_freezes,
            "sidewalk_failures_per_100m": (n_collisions + off_entries + n_freezes) / path_m * 100.0 if path_m >= MIN_PATH_M else None,
            "sidewalk_mean_speed": path_m / duration_s if duration_s > 0 else None,
        }
