"""Geometric Any-Angle Theta* Path Solver on Static Occupancy Grids.

Computes the synthetic human demonstration reference path (P_theta) and optimal
geodesic shortest distance L_0 between start and goal poses on a static 2D map.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation


@dataclass
class ThetaStarResult:
    success: bool
    path_x: np.ndarray
    path_y: np.ndarray
    geodesic_length: float
    path_points: np.ndarray


def _line_of_sight(
    grid: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> bool:
    """Supercover / Bresenham line-of-sight test on 2D boolean obstacle grid (True = obstacle)."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x = x0
    y = y0
    n = 1 + dx + dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    error = dx - dy
    dx *= 2
    dy *= 2

    h, w = grid.shape

    for _ in range(n):
        if not (0 <= x < w and 0 <= y < h):
            return False
        if grid[y, x]:
            return False
        if x == x1 and y == y1:
            break
        if error > 0:
            x += x_inc
            error -= dy
        elif error < 0:
            y += y_inc
            error += dx
        else:
            # Diagonal step: check both adjacent cells to prevent corner cutting
            if (0 <= y < h and 0 <= x + x_inc < w and grid[y, x + x_inc]) or \
               (0 <= y + y_inc < h and 0 <= x < w and grid[y + y_inc, x]):
                return False
            x += x_inc
            y += y_inc
            error += dx - dy
    return True


import ctypes
import os

_c_lib = None


def _get_c_solver():
    global _c_lib
    if _c_lib is not None:
        return _c_lib

    so_path = Path(__file__).parent / "solver.so"
    if not so_path.exists():
        cpp_path = Path(__file__).parent / "solver.cpp"
        if cpp_path.exists():
            import subprocess
            try:
                subprocess.run(
                    ["g++", "-O3", "-shared", "-fPIC", "-std=c++17", str(cpp_path), "-o", str(so_path)],
                    check=True,
                    capture_output=True,
                )
            except Exception:
                pass

    if so_path.exists():
        try:
            lib = ctypes.CDLL(str(so_path))
            lib.solve_geometric_theta_star.argtypes = [
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
            ]
            lib.solve_geometric_theta_star.restype = ctypes.c_int
            _c_lib = lib
            return _c_lib
        except Exception:
            pass
    return None


class GeometricThetaStar:
    """In-memory 2D any-angle Theta* solver on static occupancy grid maps."""

    _cache: dict[tuple, tuple[np.ndarray, float]] = {}

    def __init__(
        self,
        occupancy_grid: np.ndarray,
        resolution: float = 0.05,
        origin: tuple[float, float] = (0.0, 0.0),
        robot_radius: float = 0.3,
    ) -> None:
        """
        Args:
            occupancy_grid: 2D numpy array where True or > 50 is occupied.
            resolution: Grid resolution in meters per pixel.
            origin: (origin_x, origin_y) in world coordinates.
            robot_radius: Inflation footprint radius in meters.
        """
        self.resolution = float(resolution) if resolution > 0 else 0.05
        self.origin = origin
        self.robot_radius = float(robot_radius)

        # Build boolean obstacle mask (True = obstacle / occupied)
        if occupancy_grid.dtype == bool:
            raw_obstacle = occupancy_grid
        elif np.issubdtype(occupancy_grid.dtype, np.integer):
            raw_obstacle = (occupancy_grid > 50) | (occupancy_grid < 0)  # Occupied or unknown
        else:
            raw_obstacle = occupancy_grid > 0.5

        r_px = int(math.ceil(self.robot_radius / self.resolution))
        if r_px > 0:
            y, x = np.ogrid[-r_px:r_px + 1, -r_px:r_px + 1]
            struct = (x * x + y * y) <= (r_px * r_px)
            self.dilated_grid = binary_dilation(raw_obstacle, structure=struct)
        else:
            self.dilated_grid = raw_obstacle

        self.height, self.width = self.dilated_grid.shape

    def world_to_grid(self, wx: float, wy: float) -> tuple[int, int]:
        gx = int(round((wx - self.origin[0]) / self.resolution))
        gy = int(round((wy - self.origin[1]) / self.resolution))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> tuple[float, float]:
        wx = self.origin[0] + gx * self.resolution
        wy = self.origin[1] + gy * self.resolution
        return wx, wy

    def solve(
        self,
        start_world: tuple[float, float],
        goal_world: tuple[float, float],
        map_id: str = "",
    ) -> tuple[np.ndarray, float]:
        """Find the Euclidean shortest any-angle path between start and goal poses."""
        cache_key = (
            map_id,
            round(start_world[0], 2),
            round(start_world[1], 2),
            round(goal_world[0], 2),
            round(goal_world[1], 2),
            round(self.robot_radius, 2),
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        start_gx, start_gy = self.world_to_grid(start_world[0], start_world[1])
        goal_gx, goal_gy = self.world_to_grid(goal_world[0], goal_world[1])

        # Bounds check
        if not (0 <= start_gx < self.width and 0 <= start_gy < self.height and
                0 <= goal_gx < self.width and 0 <= goal_gy < self.height):
            dist = math.hypot(goal_world[0] - start_world[0], goal_world[1] - start_world[1])
            res = (np.array([start_world, goal_world], dtype=np.float64), dist)
            self._cache[cache_key] = res
            return res

        grid = self.dilated_grid.copy()
        for dy_i in range(-2, 3):
            for dx_i in range(-2, 3):
                if 0 <= start_gy + dy_i < self.height and 0 <= start_gx + dx_i < self.width:
                    grid[start_gy + dy_i, start_gx + dx_i] = False
                if 0 <= goal_gy + dy_i < self.height and 0 <= goal_gx + dx_i < self.width:
                    grid[goal_gy + dy_i, goal_gx + dx_i] = False

        if _line_of_sight(grid, start_gx, start_gy, goal_gx, goal_gy):
            dist = math.hypot(goal_world[0] - start_world[0], goal_world[1] - start_world[1])
            res = (np.array([start_world, goal_world], dtype=np.float64), dist)
            self._cache[cache_key] = res
            return res

        # 1. Fast C++ Theta* Solver
        c_solver = _get_c_solver()
        if c_solver is not None:
            grid_u8 = np.ascontiguousarray(grid.astype(np.uint8))
            max_len = 4096
            out_x = (ctypes.c_float * max_len)()
            out_y = (ctypes.c_float * max_len)()
            out_len = ctypes.c_float(0.0)
            grid_ptr = grid_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

            n_pts = c_solver.solve_geometric_theta_star(
                grid_ptr,
                self.width,
                self.height,
                ctypes.c_float(self.resolution),
                start_gx,
                start_gy,
                goal_gx,
                goal_gy,
                out_x,
                out_y,
                max_len,
                ctypes.byref(out_len),
            )
            if n_pts > 0:
                world_pts = []
                for k in range(n_pts):
                    wx, wy = self.grid_to_world(int(round(out_x[k])), int(round(out_y[k])))
                    world_pts.append((wx, wy))
                path_arr = np.array(world_pts, dtype=np.float64)
                path_arr[0] = start_world
                path_arr[-1] = goal_world
                total_len = float(out_len.value)
                res = (path_arr, total_len)
                self._cache[cache_key] = res
                return res

        # 2. Python Fallback
        start = (start_gx, start_gy)
        goal = (goal_gx, goal_gy)

        g_score: dict[tuple[int, int], float] = {start: 0.0}
        parent: dict[tuple[int, int], tuple[int, int]] = {start: start}

        def h(pos: tuple[int, int]) -> float:
            return math.hypot(pos[0] - goal[0], pos[1] - goal[1])

        open_set: list[tuple[float, float, tuple[int, int]]] = []
        heapq.heappush(open_set, (h(start), 0.0, start))
        closed_set: set[tuple[int, int]] = set()

        neighbors = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

        found = False
        max_iters = max(self.width * self.height * 2, 10000)
        iter_count = 0

        while open_set and iter_count < max_iters:
            iter_count += 1
            _, g_curr, curr = heapq.heappop(open_set)

            if curr == goal:
                found = True
                break

            if curr in closed_set:
                continue
            closed_set.add(curr)

            for dx, dy in neighbors:
                nbr = (curr[0] + dx, curr[1] + dy)

                if not (0 <= nbr[0] < self.width and 0 <= nbr[1] < self.height):
                    continue
                if grid[nbr[1], nbr[0]]:
                    continue
                if nbr in closed_set:
                    continue

                curr_parent = parent[curr]

                # Theta* Line of Sight Optimization
                if _line_of_sight(grid, curr_parent[0], curr_parent[1], nbr[0], nbr[1]):
                    step_cost = math.hypot(nbr[0] - curr_parent[0], nbr[1] - curr_parent[1])
                    tentative_g = g_score[curr_parent] + step_cost
                    if tentative_g < g_score.get(nbr, float("inf")):
                        g_score[nbr] = tentative_g
                        parent[nbr] = curr_parent
                        f_score = tentative_g + h(nbr)
                        heapq.heappush(open_set, (f_score, tentative_g, nbr))
                else:
                    step_cost = math.hypot(nbr[0] - curr[0], nbr[1] - curr[1])
                    tentative_g = g_curr + step_cost
                    if tentative_g < g_score.get(nbr, float("inf")):
                        g_score[nbr] = tentative_g
                        parent[nbr] = curr
                        f_score = tentative_g + h(nbr)
                        heapq.heappush(open_set, (f_score, tentative_g, nbr))

        if not found:
            dist = math.hypot(goal_world[0] - start_world[0], goal_world[1] - start_world[1])
            res = (np.array([start_world, goal_world], dtype=np.float64), dist)
            self._cache[cache_key] = res
            return res

        path_grid = [goal]
        curr = goal
        visited_bt = {goal}
        while curr != start and curr in parent:
            next_curr = parent[curr]
            if next_curr == curr or next_curr in visited_bt:
                break
            visited_bt.add(next_curr)
            curr = next_curr
            path_grid.append(curr)
        if path_grid[-1] != start:
            path_grid.append(start)
        path_grid.reverse()

        world_pts = [start_world]
        for gx, gy in path_grid[1:-1]:
            world_pts.append(self.grid_to_world(gx, gy))
        world_pts.append(goal_world)

        path_arr = np.array(world_pts, dtype=np.float64)
        total_len = float(np.sum(np.sqrt(np.sum(np.diff(path_arr, axis=0) ** 2, axis=1))))

        res = (path_arr, total_len)
        self._cache[cache_key] = res
        return res


_solver_instances: dict[str, GeometricThetaStar] = {}


def compute_theta_star_path(
    occupancy_grid: np.ndarray,
    start_pos: tuple[float, float],
    goal_pos: tuple[float, float],
    resolution: float = 0.05,
    origin: tuple[float, float] = (0.0, 0.0),
    robot_radius: float = 0.3,
    map_id: str = "",
) -> ThetaStarResult:
    """Compute Theta* path given an explicit occupancy grid."""
    solver = GeometricThetaStar(
        occupancy_grid=occupancy_grid,
        resolution=resolution,
        origin=origin,
        robot_radius=robot_radius,
    )
    pts, length = solver.solve(start_pos, goal_pos, map_id=map_id)
    return ThetaStarResult(
        success=True,
        path_x=pts[:, 0],
        path_y=pts[:, 1],
        geodesic_length=length,
        path_points=pts,
    )


def compute_theta_star_for_episode(
    map_name: str,
    start_pos: tuple[float, float],
    goal_pos: tuple[float, float],
    robot_radius: float = 0.3,
    run_dir: str | None = None,
) -> ThetaStarResult:
    """Compute Theta* reference path for a named map in the registry or fallback to Euclidean."""
    if not map_name or map_name == "unknown":
        eucl_dist = math.hypot(goal_pos[0] - start_pos[0], goal_pos[1] - start_pos[1])
        pts = np.array([start_pos, goal_pos], dtype=np.float64)
        return ThetaStarResult(
            success=True,
            path_x=pts[:, 0],
            path_y=pts[:, 1],
            geodesic_length=eucl_dist,
            path_points=pts,
        )

    solver_key = f"{map_name}_{round(robot_radius, 2)}"
    if solver_key not in _solver_instances:
        try:
            from arena_evaluation.processing.map_registry import MapRegistry
            map_info = MapRegistry.get_map(map_name, run_dir=run_dir)
            if map_info and "png_path" in map_info and Path(map_info["png_path"]).exists():
                png_path = map_info["png_path"]
                img = Image.open(png_path).convert("L")
                grid_raw = np.array(img)
                # In ROS maps, black (<128) is obstacle, white (>250) is free
                # ROS map origin is bottom-left; if image loaded top-down, flip vertically
                grid_obstacle = np.flipud(grid_raw < 128)
                resolution = float(map_info.get("resolution", 0.05))
                origin = (float(map_info["origin"][0]), float(map_info["origin"][1]))
                _solver_instances[solver_key] = GeometricThetaStar(
                    occupancy_grid=grid_obstacle,
                    resolution=resolution,
                    origin=origin,
                    robot_radius=robot_radius,
                )
        except Exception:
            pass

    solver = _solver_instances.get(solver_key)
    if solver is not None:
        pts, length = solver.solve(start_pos, goal_pos, map_id=map_name)
        return ThetaStarResult(
            success=True,
            path_x=pts[:, 0],
            path_y=pts[:, 1],
            geodesic_length=length,
            path_points=pts,
        )
    else:
        eucl_dist = math.hypot(goal_pos[0] - start_pos[0], goal_pos[1] - start_pos[1])
        pts = np.array([start_pos, goal_pos], dtype=np.float64)
        return ThetaStarResult(
            success=True,
            path_x=pts[:, 0],
            path_y=pts[:, 1],
            geodesic_length=eucl_dist,
            path_points=pts,
        )
