import numpy as np
import pytest
from robot_params_factory import make_robot_params

import polars as pl

from arena_evaluation.processing.metrics.base import ScopeSlice
from arena_evaluation.processing.metrics.common.zones import (
    DoorGeometry as _DoorGeometry,
)
from arena_evaluation.processing.metrics.ecological.regime_metrics import (
    RegimeMetricsCalculator,
    _CapZoneGeometry,
    _entered_over_cap_zone,
    _ran_red_signal,
    _used_elevator_during_alarm,
)
from arena_evaluation.storage.registry import MetricScope
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from snapshot_rows import snapshot_frame


def _calc():
    return RegimeMetricsCalculator(
        make_robot_params(robot_radius=0.2, laser_min_range=0.0, laser_max_range=10.0)
    )


def _start_pos(data):
    if data is not None and "x" in data.columns and len(data) > 0:
        return [data["x"][0], data["y"][0], data["yaw"][0]]
    return []


def _episode(data, semantic_snapshot=None):
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








def test_calculate_used_elevator_during_alarm_end_to_end_without_world():
    calc = _calc()
    data = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000, 6_000_000_000],
            "x": [0.0, 0.0, 0.0],
            "y": [0.0, 0.0, 0.0],
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

    assert calc.calculate(episode, {}) == {"used_elevator_during_alarm_anyone": 1}
    robot = _robot_results(calc, episode)
    assert robot["ran_red_signal"] is None
    assert robot["entered_over_cap_zone"] is None
    assert robot["replan_latency_after_state_change_median"] is None


def test_calculate_replan_latency_end_to_end():
    calc = _calc()
    data = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000],
            "x": [0.0] * 5,
            "y": [0.0] * 5,
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


def test_calculate_replan_metrics_are_null_without_a_required_stream():
    calc = _calc()
    data = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000],
            "x": [0.0, 0.0],
            "y": [0.0, 0.0],
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
            "x": [10.0, 1.0, 10.0, 1.0],
            "y": [10.0, 1.0, 10.0, 1.0],
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
            "x": [5.0, 0.0, 5.0],
            "y": [5.0, 0.0, 5.0],
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
                "x": [0.0, 0.0, 0.0],
                "y": [0.0, 0.0, 0.0],
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

    assert results == {"used_elevator_during_alarm_anyone": 1}
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
            "x": [25.0, 20.0, 25.0],
            "y": [20.0, 20.0, 20.0],
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
            "x": [0.0] * 3,
            "y": [0.0] * 3,
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

    assert _robot_results(_calc(), episode)["used_elevator_during_alarm_robot"] is None
    episode.robot_name = "env_0_jackal"
    assert _robot_results(_calc(), episode)["used_elevator_during_alarm_robot"] is None
