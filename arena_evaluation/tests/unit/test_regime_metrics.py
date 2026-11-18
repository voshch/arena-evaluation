import numpy as np
import pytest
from robot_params_factory import make_robot_params

pl = pytest.importorskip("polars")

from arena_evaluation.processing.metrics.base import Direction, ScopeSlice
from arena_evaluation.processing.metrics.common.zones import (
    DoorGeometry as _DoorGeometry,
)
from arena_evaluation.processing.metrics.ecological.regime_metrics import (
    RegimeMetricsCalculator,
    _CapZoneGeometry,
    _active_at,
    _cmd_vel_change_times,
    _entered_over_cap_zone,
    _extract_occupancy_zone_geometry,
    _latency_distribution,
    _overlaps,
    _ran_red_signal,
    _replan_triggers,
    _used_elevator_during_alarm,
    _windows,
)
from arena_evaluation.storage.registry import MetricScope
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams
from snapshot_rows import snapshot_frame


def _calc():
    return RegimeMetricsCalculator(
        make_robot_params(robot_radius=0.2, laser_min_range=0.0, laser_max_range=10.0)
    )


def _start_pos(data):
    if data is not None and "pos_x" in data.columns and len(data) > 0:
        return [data["pos_x"][0], data["pos_y"][0], data["yaw"][0]]
    return []


def _episode(data, semantic_snapshot=None):
    if data is not None and "pos_x" in data.columns and "pos_x_gt" not in data.columns:
        data = data.with_columns(
            pl.col("pos_x").alias("pos_x_gt"),
            pl.col("pos_y").alias("pos_y_gt"),
            pl.col("yaw").alias("yaw_gt"),
        )
    return AlignedEpisodeBundle(
        episode_id=1,
        data=data,
        start_pos=_start_pos(data),
        goal_pos=[],
        semantic_snapshot=semantic_snapshot,
    )


def _robot_results(calc, episode):
    """Robot-scope values, computed the way the registry hands the robot slice over."""
    scope_slice = ScopeSlice(
        scope=MetricScope.ROBOT,
        slice_id="robot",
        t_start_ns=0,
        t_end_ns=0,
        identity={},
        window_bundle=episode,
        full_episode=episode,
    )
    return calc.calculate_for_scope(episode, {}, scope_slice)


def _bool(token: str) -> bool:
    return token.strip().lower() in ("true", "1")


# -- _windows --------------------------------------------------------------


def test_windows_pairs_transitions_and_holds_open_at_end():
    events = pl.DataFrame(
        {
            "t_ns": [1_000_000_000, 3_000_000_000, 4_000_000_000],
            "entity": ["env_0/sched_1"] * 3,
            "kind": ["schedule"] * 3,
            "field": ["active"] * 3,
            "current": ["true", "false", "true"],
        }
    )

    windows = _windows(events, "schedule", "active", _bool, end_t_ns=6_000_000_000)

    assert windows["sched_1"] == [
        (1_000_000_000, 3_000_000_000),
        (4_000_000_000, 6_000_000_000),
    ]


def test_windows_empty_without_matching_rows():
    events = pl.DataFrame(
        {"t_ns": [], "entity": [], "kind": [], "field": [], "current": []}
    )

    assert _windows(events, "schedule", "active", _bool, end_t_ns=None) == {}


def test_windows_does_not_hold_open_past_a_closing_transition():
    events = pl.DataFrame(
        {
            "t_ns": [1_000_000_000, 2_000_000_000],
            "entity": ["env_0/sig_1", "env_0/sig_1"],
            "kind": ["signal", "signal"],
            "field": ["stop", "stop"],
            "current": ["true", "false"],
        }
    )

    windows = _windows(events, "signal", "stop", _bool, end_t_ns=10_000_000_000)

    assert windows["sig_1"] == [(1_000_000_000, 2_000_000_000)]


# -- _active_at / _overlaps -----------------------------------------------


def test_active_at_checks_half_open_interval():
    intervals = [(1_000_000_000, 3_000_000_000)]

    assert _active_at(intervals, 1_000_000_000) is True
    assert _active_at(intervals, 2_999_999_999) is True
    assert _active_at(intervals, 3_000_000_000) is False
    assert _active_at(intervals, 0) is False


def test_overlaps_detects_intersection():
    assert _overlaps(0, 10, 5, 15) is True
    assert _overlaps(0, 10, 10, 15) is False
    assert _overlaps(0, 10, 20, 30) is False


# -- used_elevator_during_alarm -------------------------------------------


def test_used_elevator_during_alarm_counts_overlapping_window():
    events = pl.DataFrame(
        {
            "t_ns": [1_000_000_000, 5_000_000_000, 2_000_000_000, 3_000_000_000],
            "entity": [
                "env_0/fire_alarm",
                "env_0/fire_alarm",
                "env_0/main_elevator",
                "env_0/main_elevator",
            ],
            "kind": ["schedule", "schedule", "elevator", "elevator"],
            "field": ["active", "active", "occupants", "occupants"],
            "current": ["true", "false", "1.0", "0.0"],
        }
    )

    assert _used_elevator_during_alarm(events, end_t_ns=6_000_000_000) == 1


def test_used_elevator_during_alarm_no_overlap_not_counted():
    events = pl.DataFrame(
        {
            "t_ns": [1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000],
            "entity": [
                "env_0/fire_alarm",
                "env_0/fire_alarm",
                "env_0/main_elevator",
                "env_0/main_elevator",
            ],
            "kind": ["schedule", "schedule", "elevator", "elevator"],
            "field": ["active", "active", "occupants", "occupants"],
            "current": ["true", "false", "1.0", "0.0"],
        }
    )

    assert _used_elevator_during_alarm(events, end_t_ns=5_000_000_000) == 0


# -- entered_over_cap_zone ------------------------------------------------


def test_entered_over_cap_zone_counts_transitions_during_over_cap_window():
    events = pl.DataFrame(
        {
            "t_ns": [1_000_000_000, 3_000_000_000],
            "entity": ["env_0/lobby", "env_0/lobby"],
            "kind": ["occupancy_cap", "occupancy_cap"],
            "field": ["over_cap", "over_cap"],
            "current": ["true", "false"],
        }
    )
    zone_idx = np.array([-1, 0, -1, 0, 0])
    t_ns = np.array([0, 1_500_000_000, 2_000_000_000, 2_500_000_000, 5_000_000_000])

    count = _entered_over_cap_zone(
        events, zone_idx, t_ns, end_t_ns=6_000_000_000, zone_names=["lobby"]
    )

    assert count == 2


def test_entered_over_cap_zone_ignores_entries_outside_window():
    events = pl.DataFrame(
        {
            "t_ns": [1_000_000_000, 2_000_000_000],
            "entity": ["env_0/lobby", "env_0/lobby"],
            "kind": ["occupancy_cap", "occupancy_cap"],
            "field": ["over_cap", "over_cap"],
            "current": ["true", "false"],
        }
    )
    zone_idx = np.array([-1, -1, 0])
    t_ns = np.array([0, 2_500_000_000, 3_000_000_000])

    count = _entered_over_cap_zone(
        events, zone_idx, t_ns, end_t_ns=4_000_000_000, zone_names=["lobby"]
    )

    assert count == 0


def test_entered_over_cap_zone_zero_without_any_over_cap_window():
    events = pl.DataFrame(
        {"t_ns": [], "entity": [], "kind": [], "field": [], "current": []}
    )
    zone_idx = np.array([-1, 0])
    t_ns = np.array([0, 1_000_000_000])

    count = _entered_over_cap_zone(
        events, zone_idx, t_ns, end_t_ns=2_000_000_000, zone_names=["lobby"]
    )

    assert count == 0


# -- ran_red_signal -------------------------------------------------------


def test_ran_red_signal_counts_entries_during_stop_window():
    events = pl.DataFrame(
        {
            "t_ns": [1_000_000_000, 3_000_000_000],
            "entity": ["env_0/crossing_1", "env_0/crossing_1"],
            "kind": ["signal", "signal"],
            "field": ["stop", "stop"],
            "current": ["true", "false"],
        }
    )
    doors = [_DoorGeometry(name="crossing_1", center_x=0.0, center_y=0.0, radius=1.0)]
    pos_x = np.array([5.0, 0.0, 5.0])
    pos_y = np.array([5.0, 0.0, 5.0])
    t_ns = np.array([0, 1_500_000_000, 2_000_000_000])

    count = _ran_red_signal(
        events, pos_x, pos_y, t_ns, end_t_ns=4_000_000_000, doors=doors
    )

    assert count == 1


def test_ran_red_signal_ignores_entries_outside_stop_window():
    events = pl.DataFrame(
        {
            "t_ns": [1_000_000_000, 2_000_000_000],
            "entity": ["env_0/crossing_1", "env_0/crossing_1"],
            "kind": ["signal", "signal"],
            "field": ["stop", "stop"],
            "current": ["true", "false"],
        }
    )
    doors = [_DoorGeometry(name="crossing_1", center_x=0.0, center_y=0.0, radius=1.0)]
    pos_x = np.array([5.0, 0.0])
    pos_y = np.array([5.0, 0.0])
    t_ns = np.array([0, 3_000_000_000])

    count = _ran_red_signal(
        events, pos_x, pos_y, t_ns, end_t_ns=4_000_000_000, doors=doors
    )

    assert count == 0


def test_ran_red_signal_skips_signal_without_matching_door():
    events = pl.DataFrame(
        {
            "t_ns": [1_000_000_000],
            "entity": ["env_0/unknown_signal"],
            "kind": ["signal"],
            "field": ["stop"],
            "current": ["true"],
        }
    )
    doors = [_DoorGeometry(name="crossing_1", center_x=0.0, center_y=0.0, radius=1.0)]
    pos_x = np.array([0.0])
    pos_y = np.array([0.0])
    t_ns = np.array([2_000_000_000])

    count = _ran_red_signal(
        events, pos_x, pos_y, t_ns, end_t_ns=3_000_000_000, doors=doors
    )

    assert count == 0


# -- cmd_vel change detection / replan triggers / latency ----------------


def test_cmd_vel_change_times_detects_change_above_epsilon():
    data = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
            "linear_x": [0.0, 0.0, 0.5, 0.5],
            "angular_z": [0.0, 0.0, 0.0, 0.0],
        }
    )

    changes = _cmd_vel_change_times(data, epsilon=0.05)

    assert changes == [2_000_000_000]


def test_cmd_vel_change_times_none_without_cmd_columns():
    data = pl.DataFrame({"t_ns": [0, 1_000_000_000]})

    assert _cmd_vel_change_times(data, epsilon=0.05) is None


def test_replan_triggers_collects_all_four_kinds_and_direction():
    events = pl.DataFrame(
        {
            "t_ns": [
                1_000_000_000,
                2_000_000_000,
                3_000_000_000,
                4_000_000_000,
                5_000_000_000,
            ],
            "entity": [
                "env_0/door_1",
                "env_0/gate_1",
                "env_0/sig_1",
                "env_0/sched_1",
                "env_0/door_1",
            ],
            "kind": ["door", "gate", "signal", "schedule", "door"],
            "field": ["open", "locked", "state", "active", "open"],
            "previous": ["false", "true", "red", "false", "true"],
            "current": ["true", "false", "green", "true", "false"],
        }
    )

    triggers = _replan_triggers(events)

    # the last row (door open true->false) is not a trigger direction
    assert triggers == [1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000]


def test_latency_distribution_drops_trigger_without_subsequent_change():
    triggers = [1_000_000_000, 5_000_000_000]
    changes = [2_000_000_000]

    latencies = _latency_distribution(triggers, changes)

    assert latencies == [pytest.approx(1.0)]


# -- occupancy zone geometry extraction -----------------------------------


def test_extract_occupancy_zone_geometry_filters_by_annotation():
    from arena_simulation_setup.shared.semantics import SemanticCfg
    from arena_simulation_setup.tree.World import Level, Zone
    from arena_simulation_setup.utils.geometry import Position

    annotated = Zone(
        name="lobby",
        polygon=[
            Position(0.0, 0.0),
            Position(0.0, 4.0),
            Position(4.0, 4.0),
            Position(4.0, 0.0),
        ],
        semantics=[SemanticCfg(role="state", name="cap", value=2.0)],
    )
    plain = Zone(
        name="plain_room",
        polygon=[
            Position(0.0, 0.0),
            Position(0.0, 2.0),
            Position(2.0, 2.0),
            Position(2.0, 0.0),
        ],
    )
    level = Level(zones=[annotated, plain])

    zones = _extract_occupancy_zone_geometry(level)

    assert len(zones) == 1
    assert zones[0].name == "lobby"


# -- end-to-end calculate() / robot scope ---------------------------------


def test_calculate_returns_none_defaults_without_semantic_snapshot():
    calc = _calc()
    data = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000],
            "pos_x": [0.0, 0.0],
            "pos_y": [0.0, 0.0],
            "yaw": [0.0, 0.0],
        }
    )
    episode = _episode(data, semantic_snapshot=None)

    assert calc.calculate(episode, {}) == {"used_elevator_during_alarm": None}
    robot = _robot_results(calc, episode)
    assert all(v is None for v in robot.values())
    assert set(robot) == set(calc.output_keys_for_scope(MetricScope.ROBOT))


def test_calculate_used_elevator_during_alarm_end_to_end_without_world():
    calc = _calc()
    data = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000, 6_000_000_000],
            "pos_x": [0.0, 0.0, 0.0],
            "pos_y": [0.0, 0.0, 0.0],
            "yaw": [0.0, 0.0, 0.0],
        }
    )
    snapshot = snapshot_frame(
        [
            (
                1_000_000_000,
                "env_0/fire_alarm",
                "schedule",
                "active",
                "predicate",
                True,
            ),
            (
                5_000_000_000,
                "env_0/fire_alarm",
                "schedule",
                "active",
                "predicate",
                False,
            ),
            (
                2_000_000_000,
                "env_0/main_elevator",
                "elevator",
                "occupants",
                "continuous",
                1.0,
            ),
            (
                3_000_000_000,
                "env_0/main_elevator",
                "elevator",
                "occupants",
                "continuous",
                0.0,
            ),
        ]
    )
    episode = _episode(data, semantic_snapshot=snapshot)

    assert calc.calculate(episode, {}) == {"used_elevator_during_alarm": 1}
    robot = _robot_results(calc, episode)
    assert robot["ran_red_signal"] is None
    assert robot["entered_over_cap_zone"] is None
    assert robot["replan_latency_after_state_change_median"] is None


def test_calculate_replan_latency_end_to_end():
    calc = _calc()
    data = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000],
            "pos_x": [0.0] * 5,
            "pos_y": [0.0] * 5,
            "yaw": [0.0] * 5,
            "linear_x": [0.0, 0.0, 0.0, 0.5, 0.5],
            "angular_z": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    snapshot = snapshot_frame(
        [
            (0, "env_0/door_1", "door", "open", "predicate", False),
            (1_000_000_000, "env_0/door_1", "door", "open", "predicate", True),
        ]
    )

    robot = _robot_results(calc, _episode(data, semantic_snapshot=snapshot))

    assert robot["replan_latency_after_state_change_median"] == pytest.approx(2.0)
    assert robot["replan_latency_after_state_change_p95"] == pytest.approx(2.0)
    assert robot["replan_unanswered_triggers"] == 0


def test_calculate_replan_unanswered_triggers_counts_censored_triggers():
    calc = _calc()
    data = pl.DataFrame(
        {
            "t_ns": [
                0,
                1_000_000_000,
                2_000_000_000,
                3_000_000_000,
                4_000_000_000,
                5_000_000_000,
                6_000_000_000,
            ],
            "pos_x": [0.0] * 7,
            "pos_y": [0.0] * 7,
            "yaw": [0.0] * 7,
            "linear_x": [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5],
            "angular_z": [0.0] * 7,
        }
    )
    snapshot = snapshot_frame(
        [
            (0, "env_0/door_1", "door", "open", "predicate", False),
            (1_000_000_000, "env_0/door_1", "door", "open", "predicate", True),
            (0, "env_0/gate_1", "gate", "locked", "predicate", True),
            (5_000_000_000, "env_0/gate_1", "gate", "locked", "predicate", False),
            (0, "env_0/door_2", "door", "open", "predicate", False),
            (6_000_000_000, "env_0/door_2", "door", "open", "predicate", True),
        ]
    )

    robot = _robot_results(calc, _episode(data, semantic_snapshot=snapshot))

    # Only the 1 s door trigger has a later cmd_vel change (at 3 s).
    assert robot["replan_unanswered_triggers"] == 2
    assert robot["replan_latency_after_state_change_median"] == pytest.approx(2.0)
    assert robot["replan_latency_after_state_change_p95"] == pytest.approx(2.0)


def test_calculate_replan_metrics_are_null_without_a_required_stream():
    calc = _calc()
    data = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000],
            "pos_x": [0.0, 0.0],
            "pos_y": [0.0, 0.0],
            "yaw": [0.0, 0.0],
        }
    )
    snapshot = snapshot_frame(
        [
            (0, "env_0/door_1", "door", "open", "predicate", False),
            (1_000_000_000, "env_0/door_1", "door", "open", "predicate", True),
        ]
    )
    episode = _episode(data, semantic_snapshot=snapshot)

    robot = _robot_results(calc, episode)

    assert robot["replan_unanswered_triggers"] is None
    assert robot["replan_latency_after_state_change_median"] is None
    assert robot["replan_latency_after_state_change_p95"] is None
    assert calc.missing_input_provenance(episode) == {
        "replan_latency_after_state_change_median": "replan_latency_after_state_change_median:topic_missing:robot_cmd_vel",
        "replan_latency_after_state_change_p95": "replan_latency_after_state_change_p95:topic_missing:robot_cmd_vel",
        "replan_unanswered_triggers": "replan_unanswered_triggers:topic_missing:robot_cmd_vel",
    }

    with_cmd_vel = AlignedEpisodeBundle(
        episode_id=1,
        data=data.with_columns(
            pl.lit(0.0).alias("linear_x"),
            pl.lit(0.0).alias("angular_z"),
        ),
        start_pos=_start_pos(data),
        goal_pos=[],
        topics={"robot_cmd_vel": pl.DataFrame({"t_ns": [0]})},
    )

    assert _robot_results(calc, with_cmd_vel)["replan_unanswered_triggers"] is None
    assert calc.missing_input_provenance(with_cmd_vel) == {
        "replan_latency_after_state_change_median": "replan_latency_after_state_change_median:topic_missing:world_semantics",
        "replan_latency_after_state_change_p95": "replan_latency_after_state_change_p95:topic_missing:world_semantics",
        "replan_unanswered_triggers": "replan_unanswered_triggers:topic_missing:world_semantics",
    }


def test_calculate_entered_over_cap_zone_with_cached_world():
    import shapely

    calc = _calc()
    calc.world = "synthetic_world_cap"
    zone = _CapZoneGeometry(
        name="lobby", polygon=shapely.Polygon([(0, 0), (0, 4), (4, 4), (4, 0)])
    )
    calc._world_cache["synthetic_world_cap"] = ([zone], [], [])

    data = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
            "pos_x": [10.0, 1.0, 10.0, 1.0],
            "pos_y": [10.0, 1.0, 10.0, 1.0],
            "yaw": [0.0, 0.0, 0.0, 0.0],
        }
    )
    snapshot = snapshot_frame(
        [
            (0, "env_0/lobby", "occupancy_cap", "over_cap", "predicate", True),
            (
                4_000_000_000,
                "env_0/lobby",
                "occupancy_cap",
                "over_cap",
                "predicate",
                False,
            ),
        ]
    )

    robot = _robot_results(calc, _episode(data, semantic_snapshot=snapshot))

    assert robot["entered_over_cap_zone"] == 2


def test_calculate_ran_red_signal_with_cached_world():
    calc = _calc()
    calc.world = "synthetic_world_signal"
    door = _DoorGeometry(name="crossing_1", center_x=0.0, center_y=0.0, radius=1.0)
    calc._world_cache["synthetic_world_signal"] = ([], [door], [])

    data = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000, 2_000_000_000],
            "pos_x": [5.0, 0.0, 5.0],
            "pos_y": [5.0, 0.0, 5.0],
            "yaw": [0.0, 0.0, 0.0],
        }
    )
    snapshot = snapshot_frame(
        [
            (0, "env_0/crossing_1", "signal", "stop", "predicate", True),
            (3_000_000_000, "env_0/crossing_1", "signal", "stop", "predicate", False),
        ]
    )

    robot = _robot_results(calc, _episode(data, semantic_snapshot=snapshot))

    assert robot["ran_red_signal"] == 1


def test_calculate_world_not_locally_present_leaves_geometry_metrics_none() -> None:
    import logging

    from arena_evaluation.processing.metrics.common import caching

    caching.loaded_worlds.pop("definitely_nonexistent_world_xyz", None)

    records: list[logging.LogRecord] = []

    class _CollectingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _CollectingHandler()
    caching.logger.addHandler(handler)
    try:
        calc = _calc()
        calc.world = "definitely_nonexistent_world_xyz"
        data = pl.DataFrame(
            {
                "t_ns": [0, 1_000_000_000, 6_000_000_000],
                "pos_x": [0.0, 0.0, 0.0],
                "pos_y": [0.0, 0.0, 0.0],
                "yaw": [0.0, 0.0, 0.0],
            }
        )
        snapshot = snapshot_frame(
            [
                (
                    1_000_000_000,
                    "env_0/fire_alarm",
                    "schedule",
                    "active",
                    "predicate",
                    True,
                ),
                (
                    5_000_000_000,
                    "env_0/fire_alarm",
                    "schedule",
                    "active",
                    "predicate",
                    False,
                ),
                (
                    2_000_000_000,
                    "env_0/main_elevator",
                    "elevator",
                    "occupants",
                    "continuous",
                    1.0,
                ),
                (
                    3_000_000_000,
                    "env_0/main_elevator",
                    "elevator",
                    "occupants",
                    "continuous",
                    0.0,
                ),
            ]
        )
        episode = _episode(data, semantic_snapshot=snapshot)
        results = calc.calculate(episode, {})
        robot = _robot_results(calc, episode)
    finally:
        caching.logger.removeHandler(handler)

    assert results == {"used_elevator_during_alarm": 1}
    assert robot["entered_over_cap_zone"] is None
    assert robot["ran_red_signal"] is None
    assert any(
        "definitely_nonexistent_world_xyz" in record.getMessage() for record in records
    )


def _members_events(rows):
    """Reconstructed-event frame from (t_ns, entity, kind, field, current) rows."""
    return pl.DataFrame(
        {
            "t_ns": [r[0] for r in rows],
            "entity": [r[1] for r in rows],
            "kind": [r[2] for r in rows],
            "field": [r[3] for r in rows],
            "previous": [""] * len(rows),
            "current": [r[4] for r in rows],
        }
    )


def test_windows_keys_level_suffixed_entities_by_authored_name():
    events = _members_events(
        [
            (1_000_000_000, "env_0/sig_1/0", "signal", "stop", "true"),
            (2_000_000_000, "env_0/sig_1/0", "signal", "stop", "false"),
            (3_000_000_000, "env_0/sig_1/1", "signal", "stop", "true"),
        ]
    )

    windows = _windows(events, "signal", "stop", _bool, end_t_ns=4_000_000_000)

    assert windows == {
        "sig_1": [(1_000_000_000, 2_000_000_000), (3_000_000_000, 4_000_000_000)]
    }


def test_ran_red_signal_prefers_signal_position_and_radius():
    from arena_evaluation.processing.metrics.ecological.mechanism_common import (
        SignalGeom,
    )

    events = _members_events([(0, "env_0/crossing_1/0", "signal", "stop", "true")])
    door = _DoorGeometry(name="crossing_1", center_x=0.0, center_y=0.0, radius=1.0)
    signal = SignalGeom("crossing_1", "0", 10.0, 0.0, 0.5)
    pos_x = np.array([5.0, 0.0, 5.0, 10.0])
    pos_y = np.zeros(4)
    t_ns = np.arange(4) * 1_000_000_000

    assert _ran_red_signal(events, pos_x, pos_y, t_ns, 4_000_000_000, [door]) == 1
    assert (
        _ran_red_signal(events, pos_x, pos_y, t_ns, 4_000_000_000, [door], [signal])
        == 1
    )
    assert (
        _ran_red_signal(
            events, pos_x[:3], pos_y[:3], t_ns[:3], 4_000_000_000, [door], [signal]
        )
        == 0
    )


def test_ran_red_signal_radius_falls_back_to_door_then_default():
    from arena_evaluation.processing.metrics.ecological.mechanism_common import (
        SignalGeom,
    )
    from arena_evaluation.processing.metrics.ecological.regime_metrics import (
        DEFAULT_SIGNAL_RADIUS,
        _signal_area,
    )

    door = _DoorGeometry(name="s", center_x=0.0, center_y=0.0, radius=2.5)
    placed = SignalGeom("s", "0", 3.0, 4.0, None)

    assert _signal_area("s", {"s": placed}, {"s": door}) == (3.0, 4.0, 2.5)
    assert _signal_area("s", {"s": placed}, {}) == (3.0, 4.0, DEFAULT_SIGNAL_RADIUS)
    assert _signal_area("s", {}, {"s": door}) == (0.0, 0.0, 2.5)
    assert _signal_area("s", {}, {}) is None


def test_calculate_ran_red_signal_with_cached_signal_geometry():
    from arena_evaluation.processing.metrics.ecological.mechanism_common import (
        SignalGeom,
    )

    calc = _calc()
    calc.world = "synthetic_world_signal_pos"
    calc._world_cache["synthetic_world_signal_pos"] = (
        [],
        [],
        [SignalGeom("crossing_1", "0", 20.0, 20.0, 1.0)],
    )
    data = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000, 2_000_000_000],
            "pos_x": [25.0, 20.0, 25.0],
            "pos_y": [20.0, 20.0, 20.0],
            "yaw": [0.0] * 3,
        }
    )
    snapshot = snapshot_frame(
        [(0, "env_0/crossing_1/0", "signal", "stop", "predicate", True)]
    )

    assert (
        _robot_results(calc, _episode(data, semantic_snapshot=snapshot))[
            "ran_red_signal"
        ]
        == 1
    )


def test_used_elevator_during_alarm_robot_counts_only_robot_membership():
    from arena_evaluation.processing.metrics.ecological.mechanism_common import (
        RobotIdentity,
    )

    ident = RobotIdentity("env_0_jackal")
    ped_only = _members_events(
        [
            (1_000_000_000, "env_0/fire_alarm/0", "schedule", "active", "true"),
            (5_000_000_000, "env_0/fire_alarm/0", "schedule", "active", "false"),
            (2_000_000_000, "env_0/elev/0", "elevator", "occupants", "1.0"),
            (
                2_000_000_000,
                "env_0/elev/0",
                "elevator",
                "members",
                "env_0/Pedestrian_1",
            ),
        ]
    )
    assert _used_elevator_during_alarm(ped_only, 6_000_000_000) == 1
    assert _used_elevator_during_alarm(ped_only, 6_000_000_000, ident) == 0

    robot = _members_events(
        [
            (1_000_000_000, "env_0/fire_alarm/0", "schedule", "active", "true"),
            (5_000_000_000, "env_0/fire_alarm/0", "schedule", "active", "false"),
            (
                3_000_000_000,
                "env_0/elev/0",
                "elevator",
                "members",
                "env_0/Pedestrian_1,env_0/jackal",
            ),
            (4_000_000_000, "env_0/elev/0", "elevator", "members", ""),
        ]
    )
    assert _used_elevator_during_alarm(robot, 6_000_000_000, ident) == 1

    no_members = _members_events(
        [(1_000_000_000, "env_0/fire_alarm/0", "schedule", "active", "true")]
    )
    assert _used_elevator_during_alarm(no_members, 6_000_000_000, ident) is None


def test_robot_scope_used_elevator_during_alarm_needs_robot_members():
    data = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000, 6_000_000_000],
            "pos_x": [0.0] * 3,
            "pos_y": [0.0] * 3,
            "yaw": [0.0] * 3,
        }
    )
    snapshot = snapshot_frame(
        [
            (
                1_000_000_000,
                "env_0/fire_alarm",
                "schedule",
                "active",
                "predicate",
                True,
            ),
            (
                2_000_000_000,
                "env_0/main_elevator",
                "elevator",
                "occupants",
                "continuous",
                1.0,
            ),
        ]
    )
    episode = _episode(data, semantic_snapshot=snapshot)

    assert _robot_results(_calc(), episode)["used_elevator_during_alarm"] is None
    episode.robot_name = "env_0_jackal"
    assert _robot_results(_calc(), episode)["used_elevator_during_alarm"] is None


def test_regime_outputs_declare_their_better_side():
    assert RegimeMetricsCalculator.DIRECTIONS == {
        "ran_red_signal": Direction.LOWER,
        "entered_over_cap_zone": Direction.LOWER,
        "used_elevator_during_alarm": Direction.LOWER,
        "replan_latency_after_state_change_median": Direction.LOWER,
        "replan_latency_after_state_change_p95": Direction.LOWER,
    }
    assert set(RegimeMetricsCalculator.PRIMARY_OUTPUTS) <= set(
        RegimeMetricsCalculator.DIRECTIONS
    )
