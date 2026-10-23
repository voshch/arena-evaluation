import numpy as np
import pytest

pl = pytest.importorskip("polars")
shapely = pytest.importorskip("shapely")

from arena_evaluation.processing.metrics.ecological.compliance_metrics import (
    ComplianceMetricsCalculator,
    _DoorGeometry,
    _ZoneGeometry,
    _door_open_series,
    _extract_zone_geometry,
    _offset_doors,
    _reconstruct_events,
    _speed_zone_metrics,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

SEC = 1_000_000_000
LEFT = 0.5
RIGHT = -0.5


def _calc(convention: str = "right") -> ComplianceMetricsCalculator:
    calc = ComplianceMetricsCalculator(RobotParams(0.2, 0.0, 10.0))
    calc.passing_convention = convention
    return calc


def _robot(xs: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time_ns": [i * SEC for i in range(len(xs))],
            "pos_x_gt": [float(x) for x in xs],
            "pos_y_gt": [0.0] * len(xs),
            "yaw_gt": [0.0] * len(xs),
        }
    )


def _episode(peds: pl.DataFrame, robot: pl.DataFrame | None) -> AlignedEpisodeBundle:
    topics = {"peds": peds}
    if robot is not None:
        topics["tf_gt"] = robot
    return AlignedEpisodeBundle(1, pl.DataFrame(), [], [], topics=topics)


def _head_on(side: float, closed: bool) -> tuple[pl.DataFrame, pl.DataFrame]:
    positions = [[[4.0, side]], [[3.0, side], [3.5, -side]], [[2.5, side]], [[1.5, side]]]
    twists = [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, -1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    if closed:
        positions.append([[1.0, side]])
        twists.append([0.0, 0.0, 0.0])
    n = len(positions)
    peds = pl.DataFrame({"time_ns": [i * SEC for i in range(n)], "peds_positions": positions, "peds_twists": twists, "num_pedestrians": [len(p) for p in positions]})
    return peds, _robot(list(range(n)))


@pytest.mark.parametrize(
    ("side", "convention", "closed", "expected"),
    [
        (LEFT, "right", True, 1.0),
        (RIGHT, "right", True, 0.0),
        (RIGHT, "left", True, 1.0),
        (LEFT, "left", True, 0.0),
        (LEFT, "right", False, 1.0),
        (RIGHT, "right", False, 0.0),
        (RIGHT, "left", False, 1.0),
        (LEFT, "left", False, 0.0),
    ],
)
def test_head_on_encounter_is_judged_by_the_side_at_closest_approach(side: float, convention: str, closed: bool, expected: float):
    peds, robot = _head_on(side, closed)

    assert _calc(convention)._compute_passing_compliance(_episode(peds, robot)) == expected


def test_two_encounters_on_opposite_sides_score_one_half():
    positions = [[[3.0, LEFT]], [[2.0, LEFT]], [[2.0, LEFT]], [[5.0, RIGHT]], [[4.0, RIGHT]], [[4.0, RIGHT]]]
    twists = [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    peds = pl.DataFrame({"time_ns": [i * SEC for i in range(6)], "peds_positions": positions, "peds_twists": twists})

    assert _calc()._compute_passing_compliance(_episode(peds, _robot([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))) == 0.5


def test_pedestrian_walking_the_same_way_is_no_encounter():
    positions = [[[3.0, LEFT]], [[4.0, LEFT]], [[5.0, LEFT]]]
    twists = [[1.0, 0.0, 0.0]] * 3
    peds = pl.DataFrame({"time_ns": [0, SEC, 2 * SEC], "peds_positions": positions, "peds_twists": twists})

    assert _calc()._compute_passing_compliance(_episode(peds, _robot([0.0, 1.0, 2.0]))) == 1.0


def test_string_twists_parse_and_unparseable_ones_fall_back_to_finite_differences():
    positions = [[[4.0, RIGHT]], [[3.0, RIGHT]], [[2.5, RIGHT]], [[2.5, RIGHT]]]
    twists = ["[-1.0, 0.0, 0.0]", "oops", None, "[0.0, 0.0, 0.0]"]
    peds = pl.DataFrame({"time_ns": [i * SEC for i in range(4)], "peds_positions": positions, "peds_twists": twists})

    assert _calc()._compute_passing_compliance(_episode(peds, _robot([0.0, 1.0, 2.0, 3.0]))) == 0.0


def test_pedestrian_velocity_comes_from_finite_differences_without_twists():
    positions = [[], [[3.0, RIGHT]], [[2.0, RIGHT]], [[2.0, RIGHT]]]
    peds = pl.DataFrame({"time_ns": [i * SEC for i in range(4)], "peds_positions": positions}, schema={"time_ns": pl.Int64, "peds_positions": pl.List(pl.List(pl.Float64))})

    assert _calc()._compute_passing_compliance(_episode(peds, _robot([0.0, 1.0, 2.0, 3.0]))) == 0.0


def test_passing_compliance_needs_a_robot_pose():
    peds = pl.DataFrame({"time_ns": [0], "peds_positions": [[[1.0, 0.0]]]})

    assert _calc()._compute_passing_compliance(_episode(peds, None)) is None


def test_passing_compliance_needs_pedestrian_samples():
    peds = pl.DataFrame({"time_ns": [], "peds_positions": []}, schema={"time_ns": pl.Int64, "peds_positions": pl.List(pl.List(pl.Float64))})

    assert _calc()._compute_passing_compliance(_episode(peds, _robot([0.0, 1.0]))) is None


def test_zone_extraction_skips_degenerate_polygons_and_reads_restricted_and_unset_caps():
    from arena_simulation_setup.shared.semantics import SemanticCfg
    from arena_simulation_setup.tree.World import Level, Zone
    from arena_simulation_setup.utils.geometry import Position

    sliver = Zone(name="sliver", polygon=[Position(0.0, 0.0), Position(1.0, 0.0)], semantics=[SemanticCfg(role="predicate", name="restricted", value=True)])
    vault = Zone(
        name="vault",
        polygon=[Position(0.0, 0.0), Position(0.0, 2.0), Position(2.0, 2.0)],
        semantics=[SemanticCfg(role="state", name="max_speed", value=None), SemanticCfg(role="predicate", name="restricted", value=True), SemanticCfg(role="state", name="occupancy", value=3.0)],
    )

    zones = _extract_zone_geometry(Level(zones=[sliver, vault]))

    assert [z.name for z in zones] == ["vault"]
    assert zones[0].max_speed is None
    assert zones[0].restricted is True
    assert zones[0].quiet is False


def test_zero_offset_keeps_the_door_list():
    doors = [_DoorGeometry(name="d", center_x=1.0, center_y=2.0, radius=1.0)]

    assert _offset_doors(doors, (0.0, 0.0)) is doors


def test_speed_violations_ignore_samples_outside_every_zone():
    zones = [_ZoneGeometry(name="corridor", polygon=None, max_speed=1.0, quiet=False, restricted=False)]

    violations, seconds = _speed_zone_metrics(np.array([-1, 0, -1]), np.array([5.0, 2.0, 5.0]), np.array([1.0, 1.0]), zones)

    assert (violations, seconds) == (1, pytest.approx(1.0))


def test_members_only_snapshot_reconstructs_the_empty_members_seed():
    snapshot = pl.DataFrame({"time_ns": [0], "entity": ["env_0/lobby"], "kind": ["zone"], "field": ["members"], "field_kind": ["members"], "value_str": [None], "value_num": [None], "value_bool": [None]})

    assert _reconstruct_events(snapshot).rows() == [(0, "env_0/lobby", "zone", "members", "", "")]


def test_door_series_is_empty_without_door_state_events():
    events = pl.DataFrame({"time_ns": [0], "entity": ["env_0/elevator_1"], "kind": ["elevator"], "field": ["state"], "current": ["open"]})

    assert _door_open_series(events) == {}


def _snapshot(entity: str, state: str) -> pl.DataFrame:
    return pl.DataFrame({"time_ns": [0], "entity": [entity], "kind": ["door"], "field": ["state"], "field_kind": ["discrete"], "value_str": [state], "value_num": [None], "value_bool": [None]})


def _door_episode(xs: list[float], ys: list[float], vel: list[float], snapshot: pl.DataFrame | None) -> AlignedEpisodeBundle:
    data = pl.DataFrame({"time_ns": [i * SEC for i in range(len(xs))], "pos_x": xs, "pos_y": ys, "yaw": [0.0] * len(xs), "vel_linear": vel})
    return AlignedEpisodeBundle(1, data, [xs[0], ys[0], 0.0], [], semantic_snapshot=snapshot)


def _blocking(episode: AlignedEpisodeBundle, doors: list[_DoorGeometry]) -> float:
    calc = _calc()
    pos_x, pos_y, _, _, _, _ = calc.resolve_robot_pose(episode)
    return calc._doorway_blocking_time(episode, pos_x, pos_y, np.abs(episode.data["vel_linear"].to_numpy()), episode.data["time_ns"].to_numpy(), doors)


def test_doorway_blocking_needs_two_samples():
    episode = _door_episode([0.0], [0.0], [0.0], _snapshot("env_0/door_1", "open"))

    assert _blocking(episode, [_DoorGeometry("door_1", 0.0, 0.0, 1.0)]) == 0.0


def test_doorway_blocking_checks_every_door_and_counts_an_interval_once():
    episode = _door_episode([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], _snapshot("env_0/door_1", "open"))
    doors = [_DoorGeometry("far_door", 10.0, 10.0, 1.0), _DoorGeometry("door_1", 0.0, 0.0, 1.0), _DoorGeometry("door_1", 0.5, 0.0, 1.0)]

    assert _blocking(episode, doors) == pytest.approx(2.0)


def test_doorway_blocking_ignores_open_doors_out_of_reach():
    episode = _door_episode([5.0, 5.0], [0.0, 0.0], [0.0, 0.0], _snapshot("env_0/door_1", "open"))

    assert _blocking(episode, [_DoorGeometry("door_1", 0.0, 0.0, 1.0)]) == 0.0


def test_cached_world_without_velocity_column_leaves_zone_metrics_unset():
    calc = _calc()
    calc.world = "compliance_test_world"
    calc._world_cache["compliance_test_world"] = ([], [])
    data = pl.DataFrame({"time_ns": [0, SEC], "pos_x": [0.0, 1.0], "pos_y": [0.0, 0.0], "yaw": [0.0, 0.0]})

    res = calc.calculate(AlignedEpisodeBundle(1, data, [0.0, 0.0, 0.0], []), {})

    assert res == dict.fromkeys(ComplianceMetricsCalculator.output_keys())


def test_world_is_loaded_from_the_shipped_assets_once_and_cached():
    calc = _calc()
    calc.world = "map_empty"
    data = pl.DataFrame({"time_ns": [0, SEC, 2 * SEC], "pos_x": [15.0, 15.0, 15.0], "pos_y": [0.2, 0.2, 0.2], "yaw": [0.0] * 3, "vel_linear": [0.0] * 3})
    episode = AlignedEpisodeBundle(1, data, [15.0, 0.2, 0.0], [], semantic_snapshot=_snapshot("env_0/south_door", "open"))

    res = calc.calculate(episode, {})
    loaded = calc._world_cache["map_empty"]
    calc.calculate(episode, {})

    zones, doors = loaded
    assert calc._world_cache["map_empty"] is loaded
    assert zones == []
    assert {d.name for d in doors} >= {"north_door", "west_door", "south_door"}
    assert res["speed_zone_violations"] == 0
    assert res["quiet_zone_dwell_seconds"] == 0.0
    assert res["restricted_zone_entries"] == 0
    assert res["doorway_blocking_time"] == pytest.approx(2.0)
    assert res["passing_rule_compliance"] is None


def test_multi_level_world_flattens_each_level_door_to_its_own_place():
    _, doors = _calc()._load_world("three_storied_residential")

    edges = {d.name: (d.center_x, d.center_y) for d in doors if d.name.startswith("door_edge_")}
    assert set(edges) == {"door_edge_1_1", "door_edge_2_1", "door_edge_3_1"}
    assert len(set(edges.values())) == 3
