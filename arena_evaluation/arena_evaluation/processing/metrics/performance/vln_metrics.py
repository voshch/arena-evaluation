from __future__ import annotations

import itertools
import math
import typing

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.path.theta_star import GeometricThetaStar, load_map_solver
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

_STEPS = ((0, 1, 1.0), (1, 0, 1.0), (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)))


def geodesic_field(blocked: np.ndarray, goal_cell: tuple[int, int], resolution: float) -> np.ndarray:
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

    graph = coo_matrix((np.concatenate(costs), (np.concatenate(rows), np.concatenate(cols))), shape=(h * w, h * w)).tocsr()
    return dijkstra(graph, directed=False, indices=gy * w + gx).reshape(h, w)


def geodesic_distances(field: np.ndarray, xs: np.ndarray, ys: np.ndarray, resolution: float, origin: tuple[float, float]) -> np.ndarray:
    """Field value at the cell of each world point, inf outside the grid."""
    gx = np.rint((np.asarray(xs, dtype=np.float64) - origin[0]) / resolution).astype(np.int64)
    gy = np.rint((np.asarray(ys, dtype=np.float64) - origin[1]) / resolution).astype(np.int64)
    h, w = field.shape
    inside = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
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
    return np.column_stack((np.interp(stations, s, points[:, 0]), np.interp(stations, s, points[:, 1])))


def ndtw(reference: np.ndarray, query: np.ndarray, threshold: float, spacing: float) -> float:
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
        cost[i, j] = d[i - 1, j - 1] + np.minimum(np.minimum(cost[i - 1, j], cost[i, j - 1]), cost[i - 1, j - 1])
    return float(np.exp(-cost[n, m] / (n * threshold)))


def stuck_window(xs: np.ndarray, ys: np.ndarray, yaw: np.ndarray, time_ns: np.ndarray, window_s: float, max_shift_m: float, max_turn_rad: float) -> bool:
    """True if some pose and the first pose `window_s` or more later differ by under both thresholds."""
    t = np.asarray(time_ns, dtype=np.float64) / 1e9
    later = np.searchsorted(t, t + window_s, side="left")
    first = np.nonzero(later < len(t))[0]
    later = later[first]
    shift = np.hypot(xs[later] - xs[first], ys[later] - ys[first])
    turn = np.abs((yaw[later] - yaw[first] + np.pi) % (2 * np.pi) - np.pi)
    return bool(np.any((shift < max_shift_m) & (turn < max_turn_rad)))


class VlnMetricsCalculator(BaseMetricCalculator):
    """VLN metrics (NE, OSR, SR, SPL, nDTW, SDTW, stuck) on the Theta* occupancy grid and reference path."""

    NAME = "vln_metrics"
    CATEGORY = "performance"
    DEPENDS_ON = ["collision_metrics", "path_metrics", "trajectory_naturalness"]
    REQUIRED_TOPICS = [("tf_gt", "odom")]

    UNITS = {
        "ne": "m",
        "osr": "",
        "success_geodesic": "",
        "spl_geodesic": "",
        "ndtw": "",
        "sdtw": "",
        "stuck": "",
    }

    PRIMARY_OUTPUTS = ["ne", "osr", "success_geodesic", "spl_geodesic", "ndtw", "sdtw"]
    OUTPUT_DIRECTIONS = {
        "ne": "lower",
        "osr": "higher",
        "success_geodesic": "higher",
        "spl_geodesic": "higher",
        "ndtw": "higher",
        "sdtw": "higher",
        "stuck": "lower",
    }

    NDTW_SPACING_M = 0.25
    STUCK_WINDOW_S = 10.0
    STUCK_SHIFT_M = 0.2
    STUCK_TURN_DEG = 15.0

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "ne",
            "osr",
            "success_geodesic",
            "spl_geodesic",
            "ndtw",
            "sdtw",
            "stuck",
        ]

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        solver = load_map_solver(episode.map or "", robot_radius=self.robot_params.robot_radius)
        return self.calculate_on_grid(episode, prior_results, solver)

    def calculate_on_grid(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any], solver: GeometricThetaStar | None) -> dict[str, typing.Any]:
        """Metrics over the solver's grid against the episode's goal tolerance, geodesic outputs None without a grid, threshold outputs None without a tolerance."""
        results: dict[str, typing.Any] = {k: None for k in self.output_keys()}
        pos_x, pos_y, yaw, _, _, _ = self.resolve_robot_pose(episode)
        if len(pos_x) == 0:
            return results

        success = prior_results.get("success")
        threshold = episode.goal_tolerance

        if success is not None and "time_ns" in episode.data.columns:
            results["stuck"] = not bool(success) and stuck_window(
                pos_x,
                pos_y,
                yaw,
                episode.data["time_ns"].to_numpy(),
                self.STUCK_WINDOW_S,
                self.STUCK_SHIFT_M,
                math.radians(self.STUCK_TURN_DEG),
            )

        if solver is None or not episode.goal_pos or len(episode.goal_pos) < 2:
            return results

        start_xy = (float(pos_x[0]), float(pos_y[0]))
        goal_xy = (float(episode.goal_pos[0]), float(episode.goal_pos[1]))
        goal_cell = solver.world_to_grid(*goal_xy)
        blocked = solver.obstacle_grid(solver.world_to_grid(*start_xy), goal_cell)
        field = geodesic_field(blocked, goal_cell, solver.resolution)
        dists = geodesic_distances(field, pos_x, pos_y, solver.resolution, solver.origin)

        finite = np.isfinite(dists)
        if finite[-1]:
            results["ne"] = float(dists[-1])
        if threshold is None:
            return results

        if finite.any():
            results["osr"] = bool(np.any(dists[finite] <= threshold))
        if results["ne"] is not None and success is not None:
            results["success_geodesic"] = bool(success) and results["ne"] <= threshold

        if not finite[0]:
            return results

        legs = [start_xy, *((float(w[0]), float(w[1])) for w in episode.waypoints), goal_xy]
        solved = [solver.solve(a, b, map_id=episode.map or "") for a, b in itertools.pairwise(legs)]
        reference = np.vstack([solved[0][0], *(pts[1:] for pts, _ in solved[1:])])
        path_length = prior_results.get("path_length")
        l0 = sum(length for _, length in solved) if episode.waypoints else prior_results.get("theta_star_length")
        if results["success_geodesic"] is not None and path_length is not None and l0 is not None:
            denom = max(float(path_length), float(l0))
            results["spl_geodesic"] = float(results["success_geodesic"]) * (float(l0) / denom if denom > 0 else 1.0)

        results["ndtw"] = ndtw(reference, np.column_stack((pos_x, pos_y)), threshold, self.NDTW_SPACING_M)
        if results["success_geodesic"] is not None:
            results["sdtw"] = float(results["success_geodesic"]) * results["ndtw"]

        return results
