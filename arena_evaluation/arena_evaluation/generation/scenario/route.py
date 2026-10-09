"""The robot's route in a base scenario: the world's shipped route, cut or shuttled to fit the episode.

The robot spec is copied, never invented: every world ships a scenario with a working route.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .populate import _centroid, _polygon

#: Shortest episode worth measuring: the robot's route is extended until it takes this long at `ROBOT_SPEED_MPS`.
MIN_EPISODE_S = 30.0

#: Hard episode budget: the planner queue dwells 60 sim-seconds per scenario (2026-09-01), so travel
#: plus the case's own delays must fit inside it. It outranks `MIN_EPISODE_S`.
MAX_EPISODE_S = 60.0
#: Budget reserved for the case to play out: onsets and holds are capped at 30 s and partly overlap the drive.
CASE_RESERVE_S = 20.0

#: The faster of nav2's controller defaults for `max_linear_vel` (mppi/rpp/hateb 0.5, dwb 0.26), so the
#: floor holds for the slower ones too.
ROBOT_SPEED_MPS = 0.5

#: Margin for the goal tolerance (1.0 m per leg) and the corners a planner cuts.
ROUTE_MARGIN = 1.2

#: A world whose endpoints are 2 m apart would otherwise get a 20-leg shuttle: a robot pacing a doorway.
MAX_LEGS = 8

MAX_ROUTE_M = (MAX_EPISODE_S - CASE_RESERVE_S) * ROBOT_SPEED_MPS


def _endpoints(robot: Mapping[str, Any]) -> list[Any]:
    out: list[Any] = [robot["start"]]
    for phase in robot.get("phases") or []:
        if isinstance(phase, Mapping) and (goto := phase.get("goto")) is not None:
            out.append(goto)
    if (goal := robot.get("goal")) is not None:
        out.append(goal)
    return out


def _xy(pose: str | Sequence[float], centroids: Mapping[str, tuple[float, float]]) -> tuple[float, float] | None:
    if isinstance(pose, str):
        return centroids.get(pose)
    if isinstance(pose, Sequence) and len(pose) >= 2:
        return (float(pose[0]), float(pose[1]))
    return None


def route_length(endpoints: Sequence[Any], zones: Sequence[Mapping[str, Any]]) -> float:
    """Metres along a polyline of points and zone names; names resolve to their centroids."""
    centroids = {str(z["name"]): _centroid(_polygon(z)) for z in zones if isinstance(z, Mapping) and z.get("name") and len(_polygon(z)) >= 3}
    pts = [p for p in (_xy(e, centroids) for e in endpoints) if p is not None]
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:], strict=False))


def robot_for(
    robot: Mapping[str, Any],
    zones: Sequence[Mapping[str, Any]],
    *,
    min_seconds: float = MIN_EPISODE_S,
    speed: float = ROBOT_SPEED_MPS,
) -> dict[str, Any]:
    """The shipped route in `phases` form (`goal:` is deprecated): its shortest prefix that takes `min_seconds`
    at `speed`, or shuttled between its endpoints until it does, within `MAX_ROUTE_M` and `MAX_LEGS`."""
    ends = _endpoints(robot)

    need = min_seconds * speed * ROUTE_MARGIN
    # a prefix, not the whole route: the acoustics corridors ship a 200-leg shuttle, a 33-minute episode verbatim
    seq = [ends[0]]
    for point in ends[1:]:
        seq.append(point)
        if route_length(seq, zones) >= min(need, MAX_ROUTE_M) or len(seq) - 1 >= MAX_LEGS:
            break

    if len(ends) >= 2:
        # the episode budget outranks the floor: never bounce past `MAX_ROUTE_M`
        back, forth = list(reversed(ends[:-1])), list(ends[1:])
        bounce = [back, forth]
        turn = 0
        capped = False
        while not capped and route_length(seq, zones) < need and len(seq) - 1 < MAX_LEGS:
            added = False
            for point in bounce[turn % 2]:
                if len(seq) - 1 >= MAX_LEGS:
                    break
                if route_length([*seq, point], zones) > MAX_ROUTE_M:
                    capped = True
                    break
                seq.append(point)
                added = True
                if route_length(seq, zones) >= need:
                    break
            turn += 1
            if not added:
                break
        del seq[MAX_LEGS + 1 :]

    spec: dict[str, Any] = {"start": seq[0], "phases": [{"goto": p} for p in seq[1:]]}
    for key in ("start_floor", "goal_floor"):
        if robot.get(key):
            spec[key] = robot[key]
    return spec


def route_seconds(robot: Mapping[str, Any], zones: Sequence[Mapping[str, Any]], speed: float = ROBOT_SPEED_MPS) -> float:
    return route_length(_endpoints(robot), zones) / speed if speed > 0 else float("inf")


__all__ = [
    "CASE_RESERVE_S",
    "MAX_EPISODE_S",
    "MAX_LEGS",
    "MAX_ROUTE_M",
    "MIN_EPISODE_S",
    "ROBOT_SPEED_MPS",
    "ROUTE_MARGIN",
    "robot_for",
    "route_length",
    "route_seconds",
]
