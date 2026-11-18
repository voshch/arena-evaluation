"""Theta* route planning on static lethal grids: the one route solver of the evaluator.

A grid is a boolean lethal mask in a map frame. Cell (gx, gy) covers [origin + g * resolution, origin + (g + 1) * resolution)
on each axis, so its centre is at origin + (g + 0.5) * resolution. The search is the C library built from solver.cpp, and there
is no second engine. An endpoint that cannot be planned from, or a route that does not exist, is a RouteResult with a null
reason; a missing library raises.
"""

from __future__ import annotations

import ctypes
import functools
import itertools
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

MAX_PATH_CELLS = 4096
MIN_SEARCH_ITERATIONS: Final = 10000
_SEARCH_REASONS: Final[dict[int, str]] = {0: "no_path", -1: "iteration_cap", -2: "broken_parent_chain"}
_PACKAGE_DIR = Path(__file__).parent
_SOLVER_SOURCE = _PACKAGE_DIR / "solver.cpp"
_SOLVER_LIBRARY = _PACKAGE_DIR / "solver.so"


@dataclass(frozen=True, slots=True, kw_only=True)
class StaticGrid:
    """Lethal mask of a map frame. Row index is the y cell, column index the x cell."""

    origin_xy: tuple[float, float]
    resolution_m: float
    lethal: np.ndarray

    def cell_of(self, x: float, y: float) -> tuple[int, int] | None:
        """The cell containing a point; None outside the grid and for a non-finite point."""
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        gx = math.floor((x - self.origin_xy[0]) / self.resolution_m)
        gy = math.floor((y - self.origin_xy[1]) / self.resolution_m)
        rows, cols = self.lethal.shape
        if 0 <= gx < cols and 0 <= gy < rows:
            return gx, gy
        return None

    def cell_centre(self, gx: int, gy: int) -> tuple[float, float]:
        return (
            self.origin_xy[0] + (gx + 0.5) * self.resolution_m,
            self.origin_xy[1] + (gy + 0.5) * self.resolution_m,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RouteResult:
    """A route in the map frame from the exact start to the exact goal, or the reason none was planned."""

    points: np.ndarray | None
    length_m: float | None
    null_reason: str | None


def plan_route(grid: StaticGrid, start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> RouteResult:
    """Shortest any-angle route between two points whose cells all stay clear."""
    if not (_is_finite(start_xy) and _is_finite(goal_xy)):
        return _refused("endpoint_not_finite")
    start_cell = grid.cell_of(*start_xy)
    if start_cell is None:
        return _refused("start_outside_grid")
    goal_cell = grid.cell_of(*goal_xy)
    if goal_cell is None:
        return _refused("goal_outside_grid")
    if _is_lethal(grid, start_cell):
        return _refused("start_in_lethal_cell")
    if _is_lethal(grid, goal_cell):
        return _refused("goal_in_lethal_cell")
    cells: tuple[tuple[int, int], ...]
    if _line_of_sight(grid.lethal, start_cell[0], start_cell[1], goal_cell[0], goal_cell[1]):
        cells = (start_cell, goal_cell)
    else:
        searched = _search(grid.lethal, grid.resolution_m, start_cell, goal_cell, max_iterations=_search_budget(grid.lethal))
        if isinstance(searched, str):
            return _refused(searched)
        cells = searched
    interior = [grid.cell_centre(gx, gy) for gx, gy in cells[1:-1]]
    points = np.array([start_xy, *interior, goal_xy], dtype=np.float64)
    return RouteResult(points=points, length_m=_polyline_length(points), null_reason=None)


def _search_budget(lethal: np.ndarray) -> int:
    """Expansions the search may make before it reports an iteration cap: twice the cell count, at least MIN_SEARCH_ITERATIONS."""
    return max(2 * lethal.size, MIN_SEARCH_ITERATIONS)


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
            if (0 <= y < h and 0 <= x + x_inc < w and grid[y, x + x_inc]) or (0 <= y + y_inc < h and 0 <= x < w and grid[y + y_inc, x]):
                return False
            x += x_inc
            y += y_inc
            error += dx - dy
    return True


@functools.cache
def _solver_library() -> ctypes.CDLL:
    """The search library, built from solver.cpp when it is missing or older than the source. A failed build raises: there is no second engine to fall back to."""
    if not _SOLVER_LIBRARY.exists() or _SOLVER_SOURCE.stat().st_mtime > _SOLVER_LIBRARY.stat().st_mtime:
        try:
            subprocess.run(
                ["g++", "-O3", "-shared", "-fPIC", "-std=c++17", str(_SOLVER_SOURCE), "-o", str(_SOLVER_LIBRARY)],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(f"Theta* search library {_SOLVER_LIBRARY} is missing or older than {_SOLVER_SOURCE}, and its build failed: {error}") from error
    library = ctypes.CDLL(str(_SOLVER_LIBRARY))
    library.solve_geometric_theta_star.argtypes = [
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
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    library.solve_geometric_theta_star.restype = ctypes.c_int
    return library


def _search(
    lethal: np.ndarray,
    resolution_m: float,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    max_iterations: int,
) -> tuple[tuple[int, int], ...] | str:
    """The route's cells, or the reason the search returned none: no_path, iteration_cap or broken_parent_chain."""
    height, width = lethal.shape
    grid_u8 = np.ascontiguousarray(lethal, dtype=np.uint8)
    out_x = (ctypes.c_float * MAX_PATH_CELLS)()
    out_y = (ctypes.c_float * MAX_PATH_CELLS)()
    out_length = ctypes.c_float(0.0)
    count = _solver_library().solve_geometric_theta_star(
        grid_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        width,
        height,
        ctypes.c_float(resolution_m),
        start[0],
        start[1],
        goal[0],
        goal[1],
        out_x,
        out_y,
        MAX_PATH_CELLS,
        max_iterations,
        ctypes.byref(out_length),
    )
    if count <= 0:
        return _SEARCH_REASONS[count]
    if count >= MAX_PATH_CELLS:
        raise RuntimeError(f"Theta* route exceeds {MAX_PATH_CELLS} cells")
    return tuple((int(round(out_x[k])), int(round(out_y[k]))) for k in range(count))


def _is_finite(point_xy: tuple[float, float]) -> bool:
    return math.isfinite(point_xy[0]) and math.isfinite(point_xy[1])


def _is_lethal(grid: StaticGrid, cell: tuple[int, int]) -> bool:
    return bool(grid.lethal[cell[1], cell[0]])


def _polyline_length(points: np.ndarray) -> float:
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in itertools.pairwise(points)))


def _refused(reason: str) -> RouteResult:
    return RouteResult(points=None, length_m=None, null_reason=reason)
