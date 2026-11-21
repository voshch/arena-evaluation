import math

import numpy as np
import polars as pl
import pytest
import shapely

from arena_evaluation.processing.metrics.performance.path_metrics import (
    PathMetricsCalculator,
)
from arena_evaluation.processing.metrics.performance.vln_metrics import (
    VlnMetricsCalculator,
    _within,
    geodesic_distances,
    geodesic_field,
    goal_inputs,
    route,
)
from arena_evaluation.processing.path.static_environment import (
    build_static_grid_from_parts,
)
from arena_evaluation.processing.path.theta_star import StaticGrid
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from robot_params_factory import make_robot_params

RES = 0.1
SIZE = 200
RADIUS = 0.3
START = (2.0, 10.0)
GOAL = (15.0, 10.0)
TOLERANCE = 3.0
# A leg or goal is reached within this radius, as in the mission reference.
ARRIVAL_RADIUS_M = 1.0
OCTILE_MAX_RATIO = math.sqrt(4.0 - 2.0 * math.sqrt(2.0))


def _grid(wall: bool) -> StaticGrid:
    walls = shapely.box(8.0, 0.0, 8.2, 18.0) if wall else shapely.GeometryCollection()
    return build_static_grid_from_parts(
        free=shapely.box(0.0, 0.0, SIZE * RES, SIZE * RES),
        walls=walls,
        doors=shapely.Polygon(),
        footprints=(),
        resolution_m=RES,
        robot_radius_m=RADIUS,
        padding_cells=5,
    )


def _field(grid: StaticGrid, goal: tuple[float, float]) -> np.ndarray:
    cell = grid.cell_of(*goal)
    assert cell is not None
    return geodesic_field(grid.lethal, cell, grid.resolution_m)


def _geodesic(grid: StaticGrid, field: np.ndarray, x: float, y: float) -> float:
    return float(geodesic_distances(field, np.array([x]), np.array([y]), grid.resolution_m, grid.origin_xy)[0])


def _densify(waypoints: list[tuple[float, float]], step: float) -> np.ndarray:
    pts = [np.array(waypoints[0], dtype=float)]
    for a, b in zip(waypoints[:-1], waypoints[1:], strict=True):
        a, b = np.array(a, dtype=float), np.array(b, dtype=float)
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        pts.extend(a + (b - a) * k / n for k in range(1, n + 1))
    return np.array(pts)


def _phases(gotos: list[tuple[float, float]], tolerance: float | None) -> dict:
    phase = {} if tolerance is None else {"tolerance_radius": tolerance}
    return {
        "phases": [{"goto": [x, y, 0.0], **phase} for x, y in gotos],
        "conditions": [],
        "map_poses": [[x, y, 0.0] for x, y in gotos],
    }


def _episode(
    waypoints: list[tuple[float, float]],
    map_name: str,
    dwell_s: float = 0.0,
    goal: tuple[float, float] = GOAL,
    tolerance: float | None = TOLERANCE,
    legs: tuple[tuple[float, float], ...] = (),
) -> AlignedEpisodeBundle:
    pts = _densify(waypoints, 0.1)
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    t = np.concatenate(([0.0], np.cumsum(steps)))
    yaw = np.arctan2(np.diff(pts[:, 1]), np.diff(pts[:, 0]))
    yaw = np.concatenate((yaw[:1], yaw))
    if dwell_s > 0:
        extra = np.arange(0.1, dwell_s + 1e-9, 0.1)
        pts = np.vstack((pts, np.repeat(pts[-1:], len(extra), axis=0)))
        t = np.concatenate((t, t[-1] + extra))
        yaw = np.concatenate((yaw, np.full(len(extra), yaw[-1])))
    data = pl.DataFrame(
        {
            "t_ns": (t * 1e9).astype(np.int64),
            "x": pts[:, 0],
            "y": pts[:, 1],
            "yaw": yaw,
        }
    )
    return AlignedEpisodeBundle(
        episode_id=0,
        data=data,
        start_pos=[pts[0, 0], pts[0, 1], yaw[0]],
        goal_pos=list(goal),
        phases=_phases([*legs, goal], tolerance),
        map=map_name,
    )


def _run(episode: AlignedEpisodeBundle, grid: StaticGrid, success: bool) -> dict:
    params = make_robot_params(robot_radius=RADIUS)
    prior = PathMetricsCalculator(params).calculate(episode, {})
    prior.update(success=success)
    return VlnMetricsCalculator(params).calculate_on_grid(episode, prior, grid)


def test_goto_without_tolerance_keeps_only_ne() -> None:
    out = _run(
        _episode([START, (11.0, 10.0)], "vln_free_no_tolerance", tolerance=None),
        _grid(wall=False),
        success=True,
    )
    assert out["ne"] == pytest.approx(4.0, abs=RES)
    assert all(out[k] is None for k in ("osr", "success_geodesic", "spl_geodesic", "ndtw", "sdtw"))


def test_goal_behind_wall_within_euclidean_threshold_fails() -> None:
    goal = (9.5, 10.0)
    out = _run(
        _episode([START, (7.0, 10.0)], "vln_wall_behind", goal=goal),
        _grid(wall=True),
        success=True,
    )
    assert math.hypot(goal[0] - 7.0, 0.0) < TOLERANCE
    assert out["ne"] > TOLERANCE
    assert out["osr"] is False
    assert out["success_geodesic"] is False
    assert out["sdtw"] == 0.0


def test_stuck_flags_failed_dwell_and_ignores_success() -> None:
    grid = _grid(wall=False)
    long_dwell = _episode([START, (11.0, 10.0)], "vln_free_dwell_long", dwell_s=15.0)
    assert _run(long_dwell, grid, success=False)["stuck"] is True
    short_dwell = _episode([START, (11.0, 10.0)], "vln_free_dwell_short", dwell_s=5.0)
    assert _run(short_dwell, grid, success=False)["stuck"] is False
    at_goal = _episode([START, (14.0, 10.0)], "vln_free_dwell_goal", dwell_s=15.0)
    assert _run(at_goal, grid, success=True)["stuck"] is False


def _arrival_point(start: tuple[float, float], leg: tuple[float, float], radius_m: float) -> tuple[float, float]:
    """The point where a straight approach from start first enters the radius around leg."""
    length = math.dist(start, leg)
    return (
        leg[0] - (leg[0] - start[0]) * radius_m / length,
        leg[1] - (leg[1] - start[1]) * radius_m / length,
    )


def test_route_through_leg_goals_scores_ndtw_against_the_chained_reference() -> None:
    leg = (8.0, 16.0)
    # The chained reference turns where the route enters the leg's arrival radius and stops where it
    # enters the goal's, so a driver that follows it does the same, not the exact points.
    turn = _arrival_point(START, leg, ARRIVAL_RADIUS_M)
    stop = _arrival_point(turn, GOAL, ARRIVAL_RADIUS_M)
    path = [START, turn, stop]
    chained = _run(
        _episode(path, "vln_free_two_legs", legs=(leg,)),
        _grid(wall=False),
        success=True,
    )
    direct = _run(_episode(path, "vln_free_two_legs_direct"), _grid(wall=False), success=True)
    assert chained["ndtw"] == pytest.approx(1.0, abs=0.02)
    assert chained["spl_geodesic"] == pytest.approx(1.0, abs=0.02)
    assert direct["ndtw"] < chained["ndtw"] - 0.2
    assert direct["spl_geodesic"] < 0.8


def test_skipping_a_leg_goal_lowers_ndtw_but_not_success() -> None:
    leg = (8.0, 16.0)
    out = _run(
        _episode([START, GOAL], "vln_free_skip_leg", legs=(leg,)),
        _grid(wall=False),
        success=True,
    )
    assert out["success_geodesic"] is True
    assert out["ndtw"] < 0.8
    # The commanded route is longer than the shortcut that was driven: NULL, not a clamped 1.0.
    assert out["spl_geodesic"] is None
    assert out["spl_geodesic_null_reason"] == "invalid_episode:reference_exceeds_path"


def test_route_is_empty_without_a_goto() -> None:
    assert route(None) == ([], None)
    assert route({"phases": [{"gesture": "wave"}], "conditions": [], "map_poses": [None]}) == ([], None)


@pytest.mark.parametrize(
    ("inputs", "gotos", "instructions", "label"),
    [
        (["pose"], [[4.0, 4.0, 0.0]], [None], "pose"),
        (["pose"], ["ward"], [None], "pose:oracle"),
        (
            ["instruction"],
            ["ward"],
            [{"source": "target", "text": "Go to the ward."}],
            "instruction",
        ),
        (
            ["instruction"],
            [[4.0, 4.0, 0.0]],
            [{"source": "authored", "text": "Find the bed."}],
            "instruction",
        ),
        (
            ["instruction"],
            ["ward", [4.0, 4.0, 0.0]],
            [
                {"source": "target", "text": "Go to the ward."},
                {"source": "route", "text": "Walk about 3 meters and stop."},
            ],
            "instruction:route",
        ),
        (
            ["instruction"],
            [[4.0, 4.0, 0.0], [6.0, 4.0, 0.0]],
            [
                {"source": "route", "text": "Walk about 3 meters and stop."},
                {"source": "coordinates", "text": "Go to (6.0, 4.0)."},
            ],
            "instruction:coordinates",
        ),
        (
            ["pose", "instruction"],
            ["ward"],
            [{"source": "target", "text": "Go to the ward."}],
            "pose:oracle+instruction",
        ),
    ],
)
def test_goal_inputs_label_names_derived_inputs(inputs, gotos, instructions, label) -> None:
    phases = {
        "phases": [{"goto": goto} for goto in gotos],
        "map_poses": [[4.0, 4.0, 0.0]] * len(gotos),
        "goal_inputs": inputs,
        "instructions": instructions,
    }
    assert goal_inputs(phases) == label


@pytest.mark.parametrize(
    ("final", "goal", "reason"),
    [
        ((14.0, 10.0), (3.0, 3.0), "start_outside_grid"),
        ((2.0, 3.0), (15.0, 10.0), "goal_outside_grid"),
    ],
)
def test_refused_ne_route_carries_the_planner_reason(final, goal, reason) -> None:
    grid = StaticGrid(origin_xy=(0.0, 0.0), resolution_m=RES, lethal=np.zeros((50, 50), dtype=bool))
    out = _run(_episode([START, final], "vln_refused_ne_route", goal=goal), grid, success=False)
    assert out["ne"] is None
    assert out["ne_null_reason"] == reason


def test_refused_route_returns_the_planner_reason() -> None:
    grid = StaticGrid(origin_xy=(0.0, 0.0), resolution_m=RES, lethal=np.zeros((50, 50), dtype=bool))
    assert _within(grid, (14.0, 10.0), (3.0, 3.0), TOLERANCE) == (
        False,
        "start_outside_grid",
    )
