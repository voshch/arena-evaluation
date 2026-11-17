from __future__ import annotations

import typing

import numpy as np
import polars as pl
import pytest
import shapely
from polars.testing import assert_frame_equal

from arena_evaluation.processing.metrics.common.context import EvalContext
from arena_evaluation.processing.metrics.common.events import reconstruct_events
from arena_evaluation.processing.metrics.common.routes import (
    route_waypoints,
)
from arena_evaluation.processing.metrics.common.zones import (
    DoorGeometry,
    ZoneGeometry,
    extract_door_geometry,
    extract_zone_geometry,
)

_EVENT_INPUT_SCHEMA = {
    "t_ns": pl.Int64,
    "entity_name": pl.String,
    "kind": pl.String,
    "field_name": pl.String,
    "field_kind": pl.String,
    "value_bool": pl.Boolean,
    "value_num": pl.Float64,
    "value_str": pl.String,
}
_EVENT_OUTPUT_SCHEMA = {
    "t_ns": pl.Int64,
    "entity": pl.String,
    "kind": pl.String,
    "field": pl.String,
    "previous": pl.String,
    "current": pl.String,
}
_SNAPSHOT = pl.DataFrame(
    {
        "t_ns": [1, 2, 3, 5],
        "entity_name": ["a", "a", "a", "b"],
        "kind": ["robot"] * 4,
        "field_name": ["state"] * 4,
        "field_kind": ["state"] * 4,
        "value_bool": [None] * 4,
        "value_num": [None] * 4,
        "value_str": ["go", "stop", "go", "idle"],
    },
    schema=_EVENT_INPUT_SCHEMA,
)


class _Corner(typing.NamedTuple):
    x: float
    y: float


class _Semantic(typing.NamedTuple):
    role: str
    name: str
    value: object = None


class _Zone(typing.NamedTuple):
    name: str
    polygon: list[_Corner]
    semantics: list[_Semantic]


class _Door(typing.NamedTuple):
    name: str
    start: _Corner
    end: _Corner
    activation_distance: list[float]


class _Level(typing.NamedTuple):
    zones: list[_Zone]
    all_doors: list[_Door]


def _zone_level() -> _Level:
    triangle = [_Corner(0.0, 0.0), _Corner(2.0, 0.0), _Corner(2.0, 2.0)]
    return _Level(
        zones=[
            _Zone(
                "speed_zone",
                triangle,
                [
                    _Semantic("state", "max_speed", 1.5),
                    _Semantic("predicate", "quiet", True),
                ],
            ),
            _Zone("quiet_zone", triangle, []),
            _Zone("degenerate", [_Corner(0.0, 0.0), _Corner(1.0, 1.0)], []),
        ],
        all_doors=[_Door("door", _Corner(0.0, 0.0), _Corner(4.0, 2.0), [0.5, 2.0])],
    )


def _overlapping_zones() -> list[ZoneGeometry]:
    return [
        ZoneGeometry("inner", shapely.box(0.0, 0.0, 2.0, 2.0), None, False, False),
        ZoneGeometry("outer", shapely.box(0.0, 0.0, 4.0, 4.0), None, False, False),
    ]


def _context_events() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t_ns": [10, 10, 20, 20],
            "entity": ["env_0/jackal_0"] * 4,
            "field": ["state", "pose.x", "state", "pose.y"],
            "current": ["go", "1", "stop", "2"],
        }
    )


def test_reconstruct_events_returns_empty_frame_for_unusable_snapshot() -> None:
    expected = pl.DataFrame(schema=_EVENT_OUTPUT_SCHEMA)
    assert_frame_equal(reconstruct_events(None), expected)
    assert_frame_equal(reconstruct_events(pl.DataFrame()), expected)
    assert_frame_equal(reconstruct_events(pl.DataFrame({"t_ns": [1]})), expected)


@pytest.mark.parametrize(
    ("require_annotation", "expected_names"),
    [(True, ["speed_zone"]), (False, ["speed_zone", "quiet_zone"])],
)
def test_extract_zone_geometry_filters_by_annotation_and_skips_degenerate_polygons(
    require_annotation: bool, expected_names: list[str]
) -> None:
    zones = extract_zone_geometry(_zone_level(), require_annotation=require_annotation)
    assert [zone.name for zone in zones] == expected_names
    assert (zones[0].max_speed, zones[0].quiet, zones[0].restricted) == (
        1.5,
        True,
        False,
    )
    assert zones[0].polygon.equals(
        shapely.Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)])
    )


def test_extract_door_geometry_uses_the_midpoint_and_largest_activation_distance() -> (
    None
):
    assert extract_door_geometry(_zone_level()) == [DoorGeometry("door", 2.0, 1.0, 2.0)]


def test_route_waypoints_keeps_every_commanded_goal_when_none_is_reached() -> None:
    waypoints = route_waypoints(
        start_pt=(0.0, 0.0),
        goal_pt=None,
        phases={"map_poses": [[10.0, 0.0], [20.0, 0.0]]},
        topics=None,
    )
    assert waypoints == ((0.0, 0.0), (10.0, 0.0), (20.0, 0.0))


def _eval_context() -> EvalContext:
    return EvalContext(
        events=_context_events(),
        t_ns=np.array([5, 15, 20, 25]),
        pos_x=np.array([0.0, 1.0, 2.0, 3.0]),
        pos_y=np.zeros(4),
        data=pl.DataFrame(),
        zones_by_name={},
        entity_roster={"jackal_0": "env_0/jackal_0"},
        ped_roster={},
    )
