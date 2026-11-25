"""The mission reference ends where the mission does: a reached goal is measured to its arrival radius, an unreached one whole."""

import math

import numpy as np
import pytest

from arena_evaluation.processing.metrics.common import routes
from arena_evaluation.processing.path.theta_star import RouteResult


def test_route_is_measured_to_the_arrival_radius() -> None:
    points = np.array([[0.0, 0.0], [10.0, 0.0]])

    length, truncated = routes._truncate_to_goal_region(points, (10.0, 0.0), 1.0)

    assert length == pytest.approx(9.0)
    assert truncated[-1] == pytest.approx([9.0, 0.0])


def test_route_starting_inside_the_radius_has_no_length() -> None:
    points = np.array([[0.0, 0.0], [0.5, 0.0]])

    length, _ = routes._truncate_to_goal_region(points, (0.5, 0.0), 1.0)

    assert length == pytest.approx(0.0)


def test_unreached_route_keeps_its_whole_length() -> None:
    points = np.array([[0.0, 0.0], [5.0, 0.0]])

    length, truncated = routes._truncate_to_goal_region(points, (10.0, 0.0), 1.0)

    assert length == pytest.approx(5.0)
    assert truncated[-1] == pytest.approx([5.0, 0.0])


def test_entry_on_a_diagonal_segment_is_found_on_the_segment() -> None:
    points = np.array([[0.0, 0.0], [3.0, 4.0]])

    length, _ = routes._truncate_to_goal_region(points, (3.0, 4.0), 1.0)

    assert length == pytest.approx(4.0)


def test_second_leg_starts_at_the_first_arrival(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legs: list[tuple[tuple[float, float], tuple[float, float]]] = []

    def straight(grid: object, start: tuple[float, float], goal: tuple[float, float]) -> RouteResult:
        legs.append((start, goal))
        length = math.dist(start, goal)
        return RouteResult(
            points=np.array([start, goal], dtype=float),
            length_m=length,
            null_reason=None,
        )

    monkeypatch.setattr(routes, "plan_route", straight)

    mission = routes.plan_mission(grid=object(), waypoints=((0.0, 0.0), (10.0, 0.0), (10.0, 5.0)))

    assert mission.length_m == pytest.approx(9.0 + (math.hypot(1.0, 5.0) - 1.0))
    assert legs[1][0] == pytest.approx((9.0, 0.0))
