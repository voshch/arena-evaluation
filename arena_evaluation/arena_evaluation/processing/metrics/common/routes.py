from __future__ import annotations

import itertools
import math
import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.path.theta_star import RouteResult, StaticGrid, plan_route

# Waypoints closer than 0.25 m are the same goal: below the planner's 0.5 m arrival tolerance.
_ROUTE_WAYPOINT_TOLERANCE_M: typing.Final[float] = 0.25

# A planned route longer than the sampled travelled path is a sampling anomaly, not a ratio above one.
REFERENCE_EXCEEDS_PATH: typing.Final[str] = "invalid_episode:reference_exceeds_path"


def _phase_waypoints(phases: typing.Mapping[str, object] | None) -> tuple[tuple[float, float], ...]:
    """The recorded task goal poses of this robot in mission order; () when the record carries none."""
    if not phases:
        return ()
    raw_poses = phases.get("map_poses")
    if not isinstance(raw_poses, (list, tuple)):
        return ()
    waypoints: list[tuple[float, float]] = []
    for pose in raw_poses:
        if not isinstance(pose, (list, tuple)) or len(pose) < 2:
            continue
        try:
            point = (float(pose[0]), float(pose[1]))
        except (TypeError, ValueError):
            continue
        if math.isfinite(point[0]) and math.isfinite(point[1]):
            waypoints.append(point)
    return tuple(waypoints)


def _planned_waypoints(topics: typing.Mapping[str, pl.DataFrame] | None) -> tuple[tuple[float, float], ...]:
    """Distinct commanded plan goals in command order, from the recorded robot_plans topic."""
    if not topics:
        return ()
    plans = topics.get("robot_plans")
    if plans is None or len(plans) == 0:
        return ()
    # agent-contracts: ignore recording schema evolution at the MCAP topic boundary
    required = {"t_ns", "plan_id", "waypoint_idx", "x", "y"}
    if not required <= set(plans.columns):
        return ()
    last_per_plan = plans.sort(["t_ns", "plan_id", "waypoint_idx"]).group_by("plan_id", maintain_order=True).agg(pl.col("t_ns").min().alias("t_ns"), pl.col("x").last().alias("x"), pl.col("y").last().alias("y")).sort("t_ns")
    waypoints: list[tuple[float, float]] = []
    for raw_x, raw_y in zip(last_per_plan["x"].to_numpy(), last_per_plan["y"].to_numpy(), strict=True):
        point = (float(raw_x), float(raw_y))
        if not (math.isfinite(point[0]) and math.isfinite(point[1])):
            continue
        if not waypoints or math.dist(point, waypoints[-1]) > _ROUTE_WAYPOINT_TOLERANCE_M:
            waypoints.append(point)
    return tuple(waypoints)


def route_waypoints(
    *,
    start_pt: tuple[float, float],
    goal_pt: tuple[float, float] | None,
    phases: typing.Mapping[str, object] | None,
    topics: typing.Mapping[str, pl.DataFrame] | None,
) -> tuple[tuple[float, float], ...]:
    """Mission waypoints in order: the start and every commanded task goal.

    The route is the mission the robot was given, so a goal it did not reach stays in the
    reference: filtering to reached goals would shorten the mission after the fact and make
    ``path_efficiency`` and ``spl_geodesic`` disagree about the same episode. The recorded task
    phases are authoritative; archived recordings made before phases were recorded fall back to
    the distinct plan goals, and a single-leg episode then reduces to start -> goal. Without any
    commanded goal the final goal is the reference.
    """
    commanded = _phase_waypoints(phases) or _planned_waypoints(topics)
    points: list[tuple[float, float]] = [start_pt]
    for waypoint in commanded:
        if math.dist(waypoint, points[-1]) > _ROUTE_WAYPOINT_TOLERANCE_M:
            points.append(waypoint)
    if len(points) == 1 and goal_pt is not None and math.dist(goal_pt, start_pt) > _ROUTE_WAYPOINT_TOLERANCE_M:
        points.append(goal_pt)
    return tuple(points)


def plan_mission(*, grid: StaticGrid | None, waypoints: tuple[tuple[float, float], ...]) -> RouteResult:
    """Whole-mission route: the per-leg routes between the waypoints in order, refused at the first leg that is."""
    if grid is None:
        return RouteResult(points=None, length_m=None, null_reason="no_map")
    if len(waypoints) < 2:
        return RouteResult(points=None, length_m=None, null_reason="degenerate_mission")
    chunks: list[np.ndarray] = []
    length_m = 0.0
    for leg_start, leg_goal in itertools.pairwise(waypoints):
        leg = plan_route(grid, leg_start, leg_goal)
        if leg.points is None or leg.length_m is None:
            return leg
        chunks.append(leg.points if not chunks else leg.points[1:])
        length_m += leg.length_m
    return RouteResult(points=np.vstack(chunks), length_m=length_m, null_reason=None)
