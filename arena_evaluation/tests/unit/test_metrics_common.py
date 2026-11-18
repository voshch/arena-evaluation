from __future__ import annotations

import typing

import numpy as np
import polars as pl
import pytest
import shapely
from polars.testing import assert_frame_equal, assert_series_equal

from arena_evaluation.processing.metrics.common.alignment import TRACK_ALIGN_TOLERANCE_NS
from arena_evaluation.processing.metrics.common.caching import world_cache
from arena_evaluation.processing.metrics.common.context import EvalContext
from arena_evaluation.processing.metrics.common.events import reconstruct_events
from arena_evaluation.processing.metrics.common.parsing import parse_bool, parse_float
from arena_evaluation.processing.metrics.common.resampling import resample_fleet
from arena_evaluation.processing.metrics.common.rosters import entity_roster, ped_roster
from arena_evaluation.processing.metrics.common.routes import plan_mission, route_waypoints
from arena_evaluation.processing.metrics.common.series import null_float_series
from arena_evaluation.processing.metrics.common.zones import DoorGeometry, ZoneGeometry, extract_door_geometry, extract_zone_geometry, zone_membership
from arena_evaluation.processing.path.theta_star import StaticGrid

_EVENT_INPUT_SCHEMA = {"t_ns": pl.Int64, "entity_name": pl.String, "kind": pl.String, "field_name": pl.String, "field_kind": pl.String, "value_bool": pl.Boolean, "value_num": pl.Float64, "value_str": pl.String}
_EVENT_OUTPUT_SCHEMA = {"t_ns": pl.Int64, "entity": pl.String, "kind": pl.String, "field": pl.String, "previous": pl.String, "current": pl.String}
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
        zones=[_Zone("speed_zone", triangle, [_Semantic("state", "max_speed", 1.5), _Semantic("predicate", "quiet", True)]), _Zone("quiet_zone", triangle, []), _Zone("degenerate", [_Corner(0.0, 0.0), _Corner(1.0, 1.0)], [])],
        all_doors=[_Door("door", _Corner(0.0, 0.0), _Corner(4.0, 2.0), [0.5, 2.0])],
    )


def _overlapping_zones() -> list[ZoneGeometry]:
    return [ZoneGeometry("inner", shapely.box(0.0, 0.0, 2.0, 2.0), None, False, False), ZoneGeometry("outer", shapely.box(0.0, 0.0, 4.0, 4.0), None, False, False)]


def _context_events() -> pl.DataFrame:
    return pl.DataFrame({"t_ns": [10, 10, 20, 20], "entity": ["env_0/jackal_0"] * 4, "field": ["state", "pose.x", "state", "pose.y"], "current": ["go", "1", "stop", "2"]})


def test_reconstruct_events_returns_empty_frame_for_unusable_snapshot() -> None:
    expected = pl.DataFrame(schema=_EVENT_OUTPUT_SCHEMA)
    assert_frame_equal(reconstruct_events(None), expected)
    assert_frame_equal(reconstruct_events(pl.DataFrame()), expected)
    assert_frame_equal(reconstruct_events(pl.DataFrame({"t_ns": [1]})), expected)


def test_reconstruct_events_keeps_only_change_points_per_entity_and_field() -> None:
    expected = pl.DataFrame({"t_ns": [1, 2, 3, 5], "entity": ["a", "a", "a", "b"], "kind": ["robot"] * 4, "field": ["state"] * 4, "previous": ["", "go", "stop", ""], "current": ["go", "stop", "go", "idle"]}, schema=_EVENT_OUTPUT_SCHEMA)
    assert_frame_equal(reconstruct_events(_SNAPSHOT), expected)


@pytest.mark.parametrize(
    ("field_kind", "value_bool", "value_num", "value_str", "expected"),
    [
        ("predicate", True, None, None, "true"),
        ("predicate", False, None, None, "false"),
        ("predicate", None, None, None, None),
        ("continuous", None, 1.5, None, "1.5"),
        ("state", None, None, None, ""),
        ("state", None, None, "idle", "idle"),
    ],
)
def test_reconstruct_events_serializes_snapshot_values_to_the_wire_convention(field_kind: str, value_bool: bool | None, value_num: float | None, value_str: str | None, expected: str) -> None:
    snapshot = pl.DataFrame({"t_ns": [1], "entity_name": ["a"], "kind": ["robot"], "field_name": ["state"], "field_kind": [field_kind], "value_bool": [value_bool], "value_num": [value_num], "value_str": [value_str]}, schema=_EVENT_INPUT_SCHEMA)
    result = reconstruct_events(snapshot)
    assert_series_equal(result["previous"], pl.Series("previous", [""], dtype=pl.String))
    assert_series_equal(result["current"], pl.Series("current", [expected], dtype=pl.String))


@pytest.mark.parametrize(("require_annotation", "expected_names"), [(True, ["speed_zone"]), (False, ["speed_zone", "quiet_zone"])])
def test_extract_zone_geometry_filters_by_annotation_and_skips_degenerate_polygons(require_annotation: bool, expected_names: list[str]) -> None:
    zones = extract_zone_geometry(_zone_level(), require_annotation=require_annotation)
    assert [zone.name for zone in zones] == expected_names
    assert (zones[0].max_speed, zones[0].quiet, zones[0].restricted) == (1.5, True, False)
    assert zones[0].polygon.equals(shapely.Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)]))


def test_extract_door_geometry_uses_the_midpoint_and_largest_activation_distance() -> None:
    assert extract_door_geometry(_zone_level()) == [DoorGeometry("door", 2.0, 1.0, 2.0)]


@pytest.mark.parametrize(("x", "y", "expected"), [(0.5, 0.5, 0), (3.0, 3.0, 1), (5.0, 5.0, -1), (2.0, 1.0, 0)])
def test_zone_membership_returns_the_first_covering_zone(x: float, y: float, expected: int) -> None:
    result = zone_membership(np.array([x]), np.array([y]), _overlapping_zones())
    assert result.tolist() == [expected]
    assert result.dtype == np.int64


@pytest.mark.parametrize(("token", "expected"), [("true", True), (" TRUE ", True), ("1", True), ("false", False), ("0", False)])
def test_parse_bool_accepts_true_and_one_case_insensitively(token: str, expected: bool) -> None:
    assert parse_bool(token) is expected


@pytest.mark.parametrize(("token", "expected"), [("1.5", 1.5), ("-2.25", -2.25)])
def test_parse_float_parses_numbers(token: str, expected: float) -> None:
    assert parse_float(token) == expected


def test_route_waypoints_returns_start_and_reached_phase_goals() -> None:
    waypoints = route_waypoints(start_pt=(0.0, 0.0), goal_pt=None, phases={"map_poses": [[0.0, 0.0], [10.0, 0.0]]}, topics=None, track_x=np.array([0.0, 1.0, 10.0]), track_y=np.array([0.0, 0.0, 0.0]))
    assert waypoints == ((0.0, 0.0), (10.0, 0.0))


def test_route_waypoints_uses_the_last_commanded_goal_when_none_is_reached() -> None:
    waypoints = route_waypoints(start_pt=(0.0, 0.0), goal_pt=None, phases={"map_poses": [[10.0, 0.0], [20.0, 0.0]]}, topics=None, track_x=np.array([0.0]), track_y=np.array([0.0]))
    assert waypoints == ((0.0, 0.0), (20.0, 0.0))


@pytest.mark.parametrize(("goal", "expected"), [((0.0, 0.0), ((0.0, 0.0),)), ((0.25, 0.0), ((0.0, 0.0),)), ((0.2501, 0.0), ((0.0, 0.0), (0.2501, 0.0)))])
def test_route_waypoints_omits_goals_within_the_waypoint_tolerance(goal: tuple[float, float], expected: tuple[tuple[float, float], ...]) -> None:
    waypoints = route_waypoints(start_pt=(0.0, 0.0), goal_pt=goal, phases=None, topics=None, track_x=np.array([]), track_y=np.array([]))
    assert waypoints == expected


def test_route_waypoints_reads_distinct_plan_goals_when_phases_are_absent() -> None:
    plans = pl.DataFrame({"t_ns": [1, 2, 3, 4], "plan_id": [1, 1, 2, 2], "waypoint_idx": [0, 1, 0, 1], "x": [0.0, 5.0, 5.0, 10.0], "y": [0.0, 0.0, 0.0, 0.0]})
    waypoints = route_waypoints(start_pt=(0.0, 0.0), goal_pt=None, phases=None, topics={"robot_plans": plans}, track_x=np.array([0.0, 5.0, 10.0]), track_y=np.array([0.0, 0.0, 0.0]))
    assert waypoints == ((0.0, 0.0), (5.0, 0.0), (10.0, 0.0))


def test_mission_length_sums_the_legs_and_is_refused_without_a_grid_or_a_leg() -> None:
    open_grid = StaticGrid(origin_xy=(0.0, 0.0), resolution_m=0.1, lethal=np.zeros((50, 50), dtype=bool))

    assert plan_mission(grid=open_grid, waypoints=((1.0, 1.0), (4.0, 1.0), (4.0, 4.0))).length_m == pytest.approx(6.0)
    assert plan_mission(grid=None, waypoints=((0.0, 0.0), (1.0, 0.0))).null_reason == "no_map"
    assert plan_mission(grid=open_grid, waypoints=((0.0, 0.0),)).null_reason == "degenerate_mission"
    assert plan_mission(grid=open_grid, waypoints=((1.0, 1.0), (1.0, 1.0), (-9.0, 1.0))).null_reason == "goal_outside_grid"


def test_entity_roster_resolves_env_prefixed_and_world_aliases() -> None:
    snapshot = pl.DataFrame({"entity_name": ["env_0/jackal_0", "env_1/world_door_1/level_1", "env_0/env_dup", "env_1/env_dup", "world_elevator_2"]})
    door = "env_1/world_door_1/level_1"
    assert entity_roster(snapshot) == {"jackal_0": "env_0/jackal_0", "world_door_1/level_1": door, "world_door_1": door, "door_1/level_1": door, "world_elevator_2": "world_elevator_2", "elevator_2": "world_elevator_2"}


@pytest.mark.parametrize("snapshot", [None, pl.DataFrame(schema={"entity_name": pl.String})])
def test_entity_roster_is_empty_without_samples(snapshot: pl.DataFrame | None) -> None:
    assert entity_roster(snapshot) == {}


def test_ped_roster_keeps_only_names_that_resolve_uniquely() -> None:
    peds = pl.DataFrame({"ped_name": ["env_0/ped_a", "env_0/ped_b", "env_1/ped_b", "ped_c", "env_0/ped_d/level_1"]})
    assert ped_roster(peds) == {"ped_a": "env_0/ped_a", "ped_c": "ped_c", "ped_d/level_1": "env_0/ped_d/level_1", "ped_d": "env_0/ped_d/level_1"}


def _eval_context() -> EvalContext:
    return EvalContext(events=_context_events(), t_ns=np.array([5, 15, 20, 25]), pos_x=np.array([0.0, 1.0, 2.0, 3.0]), pos_y=np.zeros(4), data=pl.DataFrame(), zones_by_name={}, entity_roster={"jackal_0": "env_0/jackal_0"}, ped_roster={})


@pytest.mark.parametrize(("t_ns", "expected"), [(5, "go"), (15, "go"), (20, "stop"), (25, "stop")])
def test_eval_context_field_at_holds_the_latest_value_at_or_before_the_instant(t_ns: int, expected: str) -> None:
    assert _eval_context().field_at("jackal_0", "state", t_ns) == expected


def test_eval_context_field_at_is_none_for_an_unrostered_entity() -> None:
    assert _eval_context().field_at("unknown", "state", 15) is None


def test_eval_context_family_at_returns_every_member_at_the_instant() -> None:
    assert _eval_context().family_at("jackal_0", "pose", 15) == {"x": "1", "y": "2"}


@pytest.mark.parametrize(("index", "expected"), [(0, {"ped_a": (1.0, 2.0)}), (1, {"ped_a": (3.0, 4.0)}), (2, {})])
def test_eval_context_peds_at_uses_the_latest_frame_within_the_alignment_tolerance(index: int, expected: dict[str, tuple[float, float]]) -> None:
    context = _eval_context()
    context.t_ns = np.array([50_000_000, 150_000_000, 250_000_000])
    context.peds_frames = [pl.DataFrame({"ped_name": ["ped_a"], "x": [1.0], "y": [2.0]}), pl.DataFrame({"ped_name": ["ped_a"], "x": [3.0], "y": [4.0]})]
    context.peds_times = np.array([0, TRACK_ALIGN_TOLERANCE_NS], dtype=np.int64)
    assert context.peds_at(index) == expected


def test_eval_context_sample_reports_the_ego_pose_on_the_time_axis() -> None:
    context = _eval_context()
    context.t_ns = np.array([1_000_000_000])
    context.yaw = np.array([0.5])
    sampled = context.sample(0)
    assert (sampled.t, sampled.robots) == (1.0, {"robot": (0.0, 0.0, 0.5)})


def test_resample_fleet_holds_other_robot_poses_onto_the_ego_axis() -> None:
    fleet = {
        "env_0_jackal_0": pl.DataFrame({"t_ns": [0, 10], "pos_x": [9.0, 9.0], "pos_y": [9.0, 9.0], "yaw": [0.0, 0.0]}),
        "env_1_jackal_1": pl.DataFrame({"t_ns": [0, 10], "pos_x": [1.0, 2.0], "pos_y": [0.0, 1.0], "yaw": [0.0, 0.5]}),
        "env_2_jackal_2": pl.DataFrame({"t_ns": [10], "pos_x": [5.0], "pos_y": [5.0], "yaw": [0.0]}),
    }
    result = resample_fleet(fleet, np.array([5, 10]), ego="env_0_jackal_0")
    assert set(result) == {"jackal_1"}
    assert [array.tolist() for array in result["jackal_1"]] == [[1.0, 2.0], [0.0, 1.0], [0.0, 0.5]]


def test_world_cache_is_a_mutable_module_level_mapping() -> None:
    assert isinstance(world_cache, dict)
    world_cache["world"] = None
    try:
        assert world_cache["world"] is None
    finally:
        del world_cache["world"]


def test_track_align_tolerance_is_one_tenth_of_a_second_in_nanoseconds() -> None:
    assert (TRACK_ALIGN_TOLERANCE_NS, isinstance(TRACK_ALIGN_TOLERANCE_NS, int)) == (100_000_000, True)


@pytest.mark.parametrize(("values", "expected"), [([], []), ([np.nan], [None]), ([1.0, np.nan, 3.0], [1.0, None, 3.0])])
def test_null_float_series_replaces_nan_with_an_explicit_null(values: list[float], expected: list[float | None]) -> None:
    assert_series_equal(null_float_series(np.array(values, dtype=np.float64), "speed_mps"), pl.Series("speed_mps", expected, dtype=pl.Float64))
