from __future__ import annotations

import math
import typing

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, Direction
from arena_evaluation.processing.metrics.common.alignment import interpolate_values
from arena_evaluation.processing.metrics.common.routes import (
    REFERENCE_EXCEEDS_PATH,
    plan_mission,
)
from arena_evaluation.processing.path.static_environment import grid_for_map
from arena_evaluation.processing.path.theta_star import StaticGrid, plan_route
from arena_evaluation.storage.registry import MetricScope, define_metric
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

_STEPS = ((0, 1, 1.0), (1, 0, 1.0), (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)))
_OCTILE_MAX_RATIO = math.sqrt(4.0 - 2.0 * math.sqrt(2.0))

# The world asset is absent from disk: the route was never measurable, but the episode still ran.
MISSING_WORLD: typing.Final[str] = "not_recorded:world"
_ROUTE_OUTPUTS: typing.Final[tuple[str, ...]] = (
    "ne",
    "osr",
    "success_geodesic",
    "spl_geodesic",
    "ndtw",
    "sdtw",
)

NE = define_metric(
    "ne",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="m",
    description="Navigation error: geodesic distance from the final pose to the final goto goal, m.",
    null_reason=True,
)
OSR = define_metric(
    "osr",
    pl.Boolean,
    scopes=(MetricScope.ROBOT,),
    unit="",
    description="Oracle success: whether any recorded pose within the goal tolerance reaches the goal geodesically.",
    null_reason=True,
)
SUCCESS_GEODESIC = define_metric(
    "success_geodesic",
    pl.Boolean,
    scopes=(MetricScope.ROBOT,),
    unit="",
    description="Recorded success conjoined with a navigation error inside the final goto tolerance.",
    null_reason=True,
)
SPL_GEODESIC = define_metric(
    "spl_geodesic",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="1",
    description="Success weighted by the chained geodesic length through the commanded goto goals over the travelled length, 0 to 1; NULL for a successful run whose planned route is longer than the sampled path, the same reference rule as path_efficiency, and 0 for a failed one.",
    null_reason=True,
)
NDTW = define_metric(
    "ndtw",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="1",
    description="Normalised dynamic-time-warping score of the travelled path against the chained geodesic reference, 0 to 1.",
    null_reason=True,
)
SDTW = define_metric(
    "sdtw",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="1",
    description="Success weighted by the normalised dynamic-time-warping score, 0 to 1.",
    null_reason=True,
)
STUCK = define_metric(
    "stuck",
    pl.Boolean,
    scopes=(MetricScope.ROBOT,),
    unit="",
    description="Whether a failed episode dwelled within 0.2 m and 15 degrees for ten seconds or more.",
    null_reason=True,
)
GOAL_INPUTS = define_metric(
    "goal_inputs",
    pl.String,
    scopes=(MetricScope.EPISODE,),
    unit="",
    description="Goal inputs the planner was given, joined by `+`: pose, pose:oracle, instruction, instruction:route or instruction:coordinates.",
    null_reason=True,
)


def geodesic_field(
    blocked: np.ndarray, goal_cell: tuple[int, int], resolution: float
) -> np.ndarray:
    """Meters from each cell to goal_cell (gx, gy) over free 8-connected cells without corner cutting, inf if unreachable."""
    h, w = blocked.shape
    gx, gy = goal_cell
    if not (0 <= gx < w and 0 <= gy < h) or blocked[gy, gx]:
        return np.full((h, w), np.inf)

    free = ~blocked
    ys, xs = np.nonzero(free)
    rows, cols, costs = [], [], []
    for dy, dx, step in _STEPS:
        ny, nx = ys + dy, xs + dx
        inside = (ny < h) & (nx >= 0) & (nx < w)
        sy, sx, ny, nx = ys[inside], xs[inside], ny[inside], nx[inside]
        ok = free[ny, nx]
        if dy and dx:
            ok &= free[sy, nx] & free[ny, sx]
        rows.append(sy[ok] * w + sx[ok])
        cols.append(ny[ok] * w + nx[ok])
        costs.append(np.full(int(ok.sum()), step * resolution))

    graph = coo_matrix(
        (np.concatenate(costs), (np.concatenate(rows), np.concatenate(cols))),
        shape=(h * w, h * w),
    ).tocsr()
    return dijkstra(graph, directed=False, indices=gy * w + gx).reshape(h, w)


def geodesic_distances(
    field: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    resolution: float,
    origin: tuple[float, float],
) -> np.ndarray:
    """Field value at the cell containing each world point, inf outside the grid and for a non-finite point."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    finite = np.isfinite(xs) & np.isfinite(ys)
    gx = np.floor((np.where(finite, xs, origin[0]) - origin[0]) / resolution).astype(
        np.int64
    )
    gy = np.floor((np.where(finite, ys, origin[1]) - origin[1]) / resolution).astype(
        np.int64
    )
    h, w = field.shape
    inside = finite & (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
    out = np.full(len(gx), np.inf)
    out[inside] = field[gy[inside], gx[inside]]
    return out


def resample_path(points: np.ndarray, spacing: float) -> np.ndarray:
    """Points every `spacing` meters of arc length along the polyline, endpoint included."""
    points = np.asarray(points, dtype=np.float64)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], seg > 0))
    points = points[keep]
    s = np.concatenate(([0.0], np.cumsum(seg[seg > 0])))
    if s[-1] <= 0:
        return points[:1]
    stations = np.append(np.arange(0.0, s[-1], spacing), s[-1])
    return np.column_stack(
        (
            interpolate_values(s, points[:, 0], stations),
            interpolate_values(s, points[:, 1], stations),
        )
    )


def ndtw(
    reference: np.ndarray, query: np.ndarray, threshold: float, spacing: float
) -> float:
    """exp(-DTW(R, Q) / (|R| * threshold)) on both paths resampled at `spacing` (Ilharco et al. 2019)."""
    r = resample_path(reference, spacing)
    q = resample_path(query, spacing)
    d = cdist(r, q)
    n, m = d.shape
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for k in range(2, n + m + 1):
        i = np.arange(max(1, k - m), min(n, k - 1) + 1)
        j = k - i
        cost[i, j] = d[i - 1, j - 1] + np.minimum(
            np.minimum(cost[i - 1, j], cost[i, j - 1]), cost[i - 1, j - 1]
        )
    return float(np.exp(-cost[n, m] / (n * threshold)))


def stuck_window(
    xs: np.ndarray,
    ys: np.ndarray,
    yaw: np.ndarray,
    t_ns: np.ndarray,
    window_s: float,
    max_shift_m: float,
    max_turn_rad: float,
) -> bool:
    """True if some pose and the first pose `window_s` or more later differ by under both thresholds."""
    t = np.asarray(t_ns, dtype=np.float64) / 1e9
    later = np.searchsorted(t, t + window_s, side="left")
    first = np.nonzero(later < len(t))[0]
    later = later[first]
    shift = np.hypot(xs[later] - xs[first], ys[later] - ys[first])
    turn = np.abs((yaw[later] - yaw[first] + np.pi) % (2 * np.pi) - np.pi)
    return bool(np.any((shift < max_shift_m) & (turn < max_turn_rad)))


def route(phases: dict | None) -> tuple[list[tuple[float, float]], float | None]:
    """Map-frame (x, y) of every goto in the robot's EpisodeRecord.phases entry, in order, and the last one's tolerance radius."""
    if not phases:
        return [], None
    legs = [
        (phase, pose)
        for phase, pose in zip(phases["phases"], phases["map_poses"], strict=True)
        if pose is not None
    ]
    if not legs:
        return [], None
    return [(float(pose[0]), float(pose[1])) for _, pose in legs], legs[-1][0].get(
        "tolerance_radius"
    ) or None


def goal_inputs(phases: dict | None) -> str | None:
    """What the planner was given, joined by `+`: `pose`, `pose:oracle` (a pose inside a named target), `instruction`, `instruction:route` or `instruction:coordinates` (rendered from the goal pose), None when the record names no inputs."""
    if not phases or not phases.get("goal_inputs"):
        return None
    gotos = [
        (phase, told)
        for phase, pose, told in zip(
            phases["phases"], phases["map_poses"], phases["instructions"], strict=True
        )
        if pose is not None
    ]
    labels = []
    if "pose" in phases["goal_inputs"]:
        labels.append(
            "pose:oracle"
            if any(isinstance(phase.get("goto"), str) for phase, _ in gotos)
            else "pose"
        )
    if "instruction" in phases["goal_inputs"]:
        sources = {told["source"] for _, told in gotos if told}
        labels.append(
            "instruction:coordinates"
            if "coordinates" in sources
            else "instruction:route"
            if "route" in sources
            else "instruction"
        )
    return "+".join(labels)


def _within(
    grid: StaticGrid,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    threshold: float,
) -> tuple[bool, str | None]:
    """Whether the route is within the threshold, and the planner's reason when it refuses."""
    route_result = plan_route(grid, start_xy, goal_xy)
    if route_result.length_m is None:
        return False, route_result.null_reason
    return route_result.length_m <= threshold, None


class VlnMetricsCalculator(BaseMetricCalculator):
    """VLN metrics (NE, OSR, SR, SPL, nDTW, SDTW, stuck) on the static lethal grid and the chained geodesic reference."""

    NAME = "vln_metrics"
    CATEGORY = "performance"
    DEPENDS_ON = ["collision_metrics", "path_metrics"]
    REQUIRED_TOPICS = [("robot_tracks", "robot_odom")]

    OUTPUTS = (NE, OSR, SUCCESS_GEODESIC, SPL_GEODESIC, NDTW, SDTW, STUCK, GOAL_INPUTS)
    DIRECTIONS = {
        "ne": Direction.LOWER,
        "osr": Direction.HIGHER,
        "success_geodesic": Direction.HIGHER,
        "spl_geodesic": Direction.HIGHER,
        "ndtw": Direction.HIGHER,
        "sdtw": Direction.HIGHER,
        "stuck": Direction.LOWER,
    }
    PRIMARY_OUTPUTS = ["ne", "osr", "success_geodesic", "spl_geodesic", "ndtw", "sdtw"]

    NDTW_SPACING_M = 0.25
    STUCK_WINDOW_S = 10.0
    STUCK_SHIFT_M = 0.2
    STUCK_TURN_DEG = 15.0

    def calculate(
        self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        """Route metrics on the episode's world grid; a world asset that is not on disk nulls them with its reason."""
        try:
            grid = grid_for_map(
                episode.map, episode.run_dir, self.robot_params.robot_radius
            )
        except FileNotFoundError:
            results = self.calculate_on_grid(episode, prior_results, None)
            for key in _ROUTE_OUTPUTS:
                results[f"{key}_null_reason"] = MISSING_WORLD
            return results
        return self.calculate_on_grid(episode, prior_results, grid)

    def calculate_on_grid(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        grid: StaticGrid | None,
    ) -> dict[str, typing.Any]:
        """Metrics over the static grid against the final goto's tolerance, geodesic outputs None without a grid or goto, threshold outputs None without a tolerance."""
        results: dict[str, typing.Any] = {k: None for k in self.output_keys()}
        pos_x, pos_y, yaw, _, _, _ = self.resolve_robot_pose(episode)
        if len(pos_x) == 0:
            return results

        success = prior_results.get("success")
        goals, threshold = route(episode.phases)
        results["goal_inputs"] = goal_inputs(episode.phases)

        if success is not None:
            results["stuck"] = not bool(success) and stuck_window(
                pos_x,
                pos_y,
                yaw,
                episode.data["t_ns"].to_numpy(),
                self.STUCK_WINDOW_S,
                self.STUCK_SHIFT_M,
                math.radians(self.STUCK_TURN_DEG),
            )

        if grid is None or not goals:
            return results

        self._populate_geodesic_metrics(
            results,
            grid,
            pos_x,
            pos_y,
            goals,
            threshold,
            success,
            prior_results.get("path_length_m"),
        )
        return results

    def _populate_geodesic_metrics(
        self,
        results: dict[str, typing.Any],
        grid: StaticGrid,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        goals: list[tuple[float, float]],
        threshold: float | None,
        success: typing.Any,
        path_length: typing.Any,
    ) -> None:
        start_xy = (float(pos_x[0]), float(pos_y[0]))
        goal_xy = goals[-1]
        final_route = plan_route(grid, (float(pos_x[-1]), float(pos_y[-1])), goal_xy)
        results["ne"] = final_route.length_m
        if final_route.null_reason is not None:
            results["ne_null_reason"] = final_route.null_reason
        goal_cell = grid.cell_of(*goal_xy)
        if goal_cell is None:
            if threshold is not None and final_route.null_reason is not None:
                results["osr"] = None
                results["osr_null_reason"] = final_route.null_reason
            return
        field = geodesic_field(grid.lethal, goal_cell, grid.resolution_m)
        dists = geodesic_distances(
            field, pos_x, pos_y, grid.resolution_m, grid.origin_xy
        )

        finite = np.isfinite(dists)
        if threshold is None:
            return

        near = np.flatnonzero(
            finite
            & (
                dists
                <= threshold * _OCTILE_MAX_RATIO + grid.resolution_m * math.sqrt(2.0)
            )
        )
        euclid = np.hypot(pos_x[near] - goal_xy[0], pos_y[near] - goal_xy[1])
        near = near[euclid <= threshold][np.argsort(euclid[euclid <= threshold])]
        results["osr"] = False
        refusal: str | None = None
        for i in near:
            inside, reason = _within(
                grid, (float(pos_x[i]), float(pos_y[i])), goal_xy, threshold
            )
            if inside:
                results["osr"] = True
                break
            if refusal is None:
                refusal = reason
        if not results["osr"] and refusal is not None:
            results["osr"] = None
            results["osr_null_reason"] = refusal
        if results["ne"] is not None and success is not None:
            results["success_geodesic"] = bool(success) and results["ne"] <= threshold

        if not finite[0]:
            return

        mission = plan_mission(grid=grid, waypoints=(start_xy, *goals))
        if mission.points is None or mission.length_m is None:
            return
        if results["success_geodesic"] is not None and path_length is not None:
            reference_length = float(mission.length_m)
            travelled_length = float(path_length)
            ratio = (
                reference_length / travelled_length if travelled_length > 0.0 else 1.0
            )
            if ratio > 1.0:
                if results["success_geodesic"]:
                    # The sampled path came out shorter than the route: out of range, never clamped to one.
                    results["spl_geodesic"] = None
                    results["spl_geodesic_null_reason"] = REFERENCE_EXCEEDS_PATH
                else:
                    # A failed run scores zero whatever the ratio: the weight is never applied.
                    results["spl_geodesic"] = 0.0
            else:
                results["spl_geodesic"] = float(results["success_geodesic"]) * ratio

        results["ndtw"] = ndtw(
            mission.points,
            np.column_stack((pos_x, pos_y)),
            threshold,
            self.NDTW_SPACING_M,
        )
        if results["success_geodesic"] is not None:
            results["sdtw"] = float(results["success_geodesic"]) * results["ndtw"]
