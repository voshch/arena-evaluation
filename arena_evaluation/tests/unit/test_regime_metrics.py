import numpy as np
import pytest

pl = pytest.importorskip("polars")

from arena_evaluation.processing.metrics.ecological.compliance_metrics import _DoorGeometry
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
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams


def _calc():
    return RegimeMetricsCalculator(RobotParams(0.2, 0.0, 10.0))


def _start_pos(data):
    if data is not None and "pos_x" in data.columns and len(data) > 0:
        return [data["pos_x"][0], data["pos_y"][0], data["yaw"][0]]
    return []


def _episode(data, semantic_snapshot=None):
    return AlignedEpisodeBundle(
        episode_id=1,
        data=data,
        start_pos=_start_pos(data),
        goal_pos=[],
        semantic_snapshot=semantic_snapshot,
    )


def _snap(rows):
    """Synthetic long-format snapshot rows: (time_ns, entity, kind, field, field_kind, value)."""
    time_ns, entity, kind, field, field_kind = [], [], [], [], []
    value_str, value_num, value_bool = [], [], []
    for t, e, k, f, fk, v in rows:
        time_ns.append(t)
        entity.append(e)
        kind.append(k)
        field.append(f)
        field_kind.append(fk)
        value_str.append(v if fk == "discrete" else None)
        value_num.append(float(v) if fk == "continuous" else None)
        value_bool.append(bool(v) if fk == "predicate" else None)
    return pl.DataFrame(
        {
            "time_ns": time_ns,
            "entity": entity,
            "kind": kind,
            "field": field,
            "field_kind": field_kind,
            "value_str": value_str,
            "value_num": value_num,
            "value_bool": value_bool,
        }
    )


def _bool(token: str) -> bool:
    return token.strip().lower() in ("true", "1")


# -- _windows --------------------------------------------------------------


def test_windows_pairs_transitions_and_holds_open_at_end():
    events = pl.DataFrame(
        {
            "time_ns": [1_000_000_000, 3_000_000_000, 4_000_000_000],
            "entity": ["env_0/sched_1"] * 3,
            "kind": ["schedule"] * 3,
            "field": ["active"] * 3,
            "current": ["true", "false", "true"],
        }
    )

    windows = _windows(events, "schedule", "active", _bool, end_time_ns=6_000_000_000)

    assert windows["sched_1"] == [(1_000_000_000, 3_000_000_000), (4_000_000_000, 6_000_000_000)]


def test_windows_empty_without_matching_rows():
    events = pl.DataFrame({"time_ns": [], "entity": [], "kind": [], "field": [], "current": []})

    assert _windows(events, "schedule", "active", _bool, end_time_ns=None) == {}


def test_windows_does_not_hold_open_past_a_closing_transition():
    events = pl.DataFrame(
        {
            "time_ns": [1_000_000_000, 2_000_000_000],
            "entity": ["env_0/sig_1", "env_0/sig_1"],
            "kind": ["signal", "signal"],
            "field": ["stop", "stop"],
            "current": ["true", "false"],
        }
    )

    windows = _windows(events, "signal", "stop", _bool, end_time_ns=10_000_000_000)

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
            "time_ns": [1_000_000_000, 5_000_000_000, 2_000_000_000, 3_000_000_000],
            "entity": ["env_0/fire_alarm", "env_0/fire_alarm", "env_0/main_elevator", "env_0/main_elevator"],
            "kind": ["schedule", "schedule", "elevator", "elevator"],
            "field": ["active", "active", "occupants", "occupants"],
            "current": ["true", "false", "1.0", "0.0"],
        }
    )

    assert _used_elevator_during_alarm(events, end_time_ns=6_000_000_000) == 1


def test_used_elevator_during_alarm_no_overlap_not_counted():
    events = pl.DataFrame(
        {
            "time_ns": [1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000],
            "entity": ["env_0/fire_alarm", "env_0/fire_alarm", "env_0/main_elevator", "env_0/main_elevator"],
            "kind": ["schedule", "schedule", "elevator", "elevator"],
            "field": ["active", "active", "occupants", "occupants"],
            "current": ["true", "false", "1.0", "0.0"],
        }
    )

    assert _used_elevator_during_alarm(events, end_time_ns=5_000_000_000) == 0


# -- entered_over_cap_zone ------------------------------------------------


def test_entered_over_cap_zone_counts_transitions_during_over_cap_window():
    events = pl.DataFrame(
        {
            "time_ns": [1_000_000_000, 3_000_000_000],
            "entity": ["env_0/lobby", "env_0/lobby"],
            "kind": ["occupancy_cap", "occupancy_cap"],
            "field": ["over_cap", "over_cap"],
            "current": ["true", "false"],
        }
    )
    zone_idx = np.array([-1, 0, -1, 0, 0])
    time_ns = np.array([0, 1_500_000_000, 2_000_000_000, 2_500_000_000, 5_000_000_000])

    count = _entered_over_cap_zone(events, zone_idx, time_ns, end_time_ns=6_000_000_000, zone_names=["lobby"])

    assert count == 2


def test_entered_over_cap_zone_ignores_entries_outside_window():
    events = pl.DataFrame(
        {
            "time_ns": [1_000_000_000, 2_000_000_000],
            "entity": ["env_0/lobby", "env_0/lobby"],
            "kind": ["occupancy_cap", "occupancy_cap"],
            "field": ["over_cap", "over_cap"],
            "current": ["true", "false"],
        }
    )
    zone_idx = np.array([-1, -1, 0])
    time_ns = np.array([0, 2_500_000_000, 3_000_000_000])

    count = _entered_over_cap_zone(events, zone_idx, time_ns, end_time_ns=4_000_000_000, zone_names=["lobby"])

    assert count == 0


def test_entered_over_cap_zone_zero_without_any_over_cap_window():
    events = pl.DataFrame({"time_ns": [], "entity": [], "kind": [], "field": [], "current": []})
    zone_idx = np.array([-1, 0])
    time_ns = np.array([0, 1_000_000_000])

    count = _entered_over_cap_zone(events, zone_idx, time_ns, end_time_ns=2_000_000_000, zone_names=["lobby"])

    assert count == 0


# -- ran_red_signal -------------------------------------------------------


def test_ran_red_signal_counts_entries_during_stop_window():
    events = pl.DataFrame(
        {
            "time_ns": [1_000_000_000, 3_000_000_000],
            "entity": ["env_0/crossing_1", "env_0/crossing_1"],
            "kind": ["signal", "signal"],
            "field": ["stop", "stop"],
            "current": ["true", "false"],
        }
    )
    doors = [_DoorGeometry(name="crossing_1", center_x=0.0, center_y=0.0, radius=1.0)]
    pos_x = np.array([5.0, 0.0, 5.0])
    pos_y = np.array([5.0, 0.0, 5.0])
    time_ns = np.array([0, 1_500_000_000, 2_000_000_000])

    count = _ran_red_signal(events, pos_x, pos_y, time_ns, end_time_ns=4_000_000_000, doors=doors)

    assert count == 1


def test_ran_red_signal_ignores_entries_outside_stop_window():
    events = pl.DataFrame(
        {
            "time_ns": [1_000_000_000, 2_000_000_000],
            "entity": ["env_0/crossing_1", "env_0/crossing_1"],
            "kind": ["signal", "signal"],
            "field": ["stop", "stop"],
            "current": ["true", "false"],
        }
    )
    doors = [_DoorGeometry(name="crossing_1", center_x=0.0, center_y=0.0, radius=1.0)]
    pos_x = np.array([5.0, 0.0])
    pos_y = np.array([5.0, 0.0])
    time_ns = np.array([0, 3_000_000_000])

    count = _ran_red_signal(events, pos_x, pos_y, time_ns, end_time_ns=4_000_000_000, doors=doors)

    assert count == 0


def test_ran_red_signal_skips_signal_without_matching_door():
    events = pl.DataFrame(
        {
            "time_ns": [1_000_000_000],
            "entity": ["env_0/unknown_signal"],
            "kind": ["signal"],
            "field": ["stop"],
            "current": ["true"],
        }
    )
    doors = [_DoorGeometry(name="crossing_1", center_x=0.0, center_y=0.0, radius=1.0)]
    pos_x = np.array([0.0])
    pos_y = np.array([0.0])
    time_ns = np.array([2_000_000_000])

    count = _ran_red_signal(events, pos_x, pos_y, time_ns, end_time_ns=3_000_000_000, doors=doors)

    assert count == 0


# -- cmd_vel change detection / replan triggers / latency ----------------


def test_cmd_vel_change_times_detects_change_above_epsilon():
    data = pl.DataFrame(
        {
            "time_ns": [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
            "linear_x": [0.0, 0.0, 0.5, 0.5],
            "angular_z": [0.0, 0.0, 0.0, 0.0],
        }
    )

    changes = _cmd_vel_change_times(data, epsilon=0.05)

    assert changes == [2_000_000_000]


def test_cmd_vel_change_times_none_without_cmd_columns():
    data = pl.DataFrame({"time_ns": [0, 1_000_000_000]})

    assert _cmd_vel_change_times(data, epsilon=0.05) is None


def test_replan_triggers_collects_all_four_kinds_and_direction():
    events = pl.DataFrame(
        {
            "time_ns": [1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000, 5_000_000_000],
            "entity": ["env_0/door_1", "env_0/gate_1", "env_0/sig_1", "env_0/sched_1", "env_0/door_1"],
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
    from arena_simulation_setup.tree.World.World import LevelDescription
    from arena_simulation_setup.utils.geometry import Position

    annotated = LevelDescription.Zone(
        name="lobby",
        corners=[Position(0.0, 0.0), Position(0.0, 4.0), Position(4.0, 4.0), Position(4.0, 0.0)],
        semantics=[SemanticCfg(role="state", name="cap", value=2.0)],
    )
    plain = LevelDescription.Zone(
        name="plain_room",
        corners=[Position(0.0, 0.0), Position(0.0, 2.0), Position(2.0, 2.0), Position(2.0, 0.0)],
    )
    level = LevelDescription(zones=[annotated, plain])

    zones = _extract_occupancy_zone_geometry(level)

    assert len(zones) == 1
    assert zones[0].name == "lobby"


# -- end-to-end calculate() -----------------------------------------------


def test_calculate_returns_none_defaults_without_semantic_snapshot():
    calc = _calc()
    data = pl.DataFrame(
        {
            "time_ns": [0, 1_000_000_000],
            "pos_x": [0.0, 0.0],
            "pos_y": [0.0, 0.0],
            "yaw": [0.0, 0.0],
        }
    )

    results = calc.calculate(_episode(data, semantic_snapshot=None), {})

    assert all(v is None for v in results.values())
    assert set(results) == set(calc.output_keys())


def test_calculate_used_elevator_during_alarm_end_to_end_without_world():
    calc = _calc()
    data = pl.DataFrame(
        {
            "time_ns": [0, 1_000_000_000, 6_000_000_000],
            "pos_x": [0.0, 0.0, 0.0],
            "pos_y": [0.0, 0.0, 0.0],
            "yaw": [0.0, 0.0, 0.0],
        }
    )
    snapshot = _snap(
        [
            (1_000_000_000, "env_0/fire_alarm", "schedule", "active", "predicate", True),
            (5_000_000_000, "env_0/fire_alarm", "schedule", "active", "predicate", False),
            (2_000_000_000, "env_0/main_elevator", "elevator", "occupants", "continuous", 1.0),
            (3_000_000_000, "env_0/main_elevator", "elevator", "occupants", "continuous", 0.0),
        ]
    )

    results = calc.calculate(_episode(data, semantic_snapshot=snapshot), {})

    assert results["used_elevator_during_alarm"] == 1
    assert results["ran_red_signal"] is None
    assert results["entered_over_cap_zone"] is None
    assert results["replan_latency_after_state_change_median"] is None


def test_calculate_replan_latency_end_to_end():
    calc = _calc()
    data = pl.DataFrame(
        {
            "time_ns": [0, 1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000],
            "pos_x": [0.0] * 5,
            "pos_y": [0.0] * 5,
            "yaw": [0.0] * 5,
            "linear_x": [0.0, 0.0, 0.0, 0.5, 0.5],
            "angular_z": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    snapshot = _snap(
        [
            (0, "env_0/door_1", "door", "open", "predicate", False),
            (1_000_000_000, "env_0/door_1", "door", "open", "predicate", True),
        ]
    )

    results = calc.calculate(_episode(data, semantic_snapshot=snapshot), {})

    assert results["replan_latency_after_state_change_median"] == pytest.approx(2.0)
    assert results["replan_latency_after_state_change_p95"] == pytest.approx(2.0)


def test_calculate_entered_over_cap_zone_with_cached_world():
    import shapely

    calc = _calc()
    calc.world = "synthetic_world_cap"
    zone = _CapZoneGeometry(name="lobby", polygon=shapely.Polygon([(0, 0), (0, 4), (4, 4), (4, 0)]))
    calc._world_cache["synthetic_world_cap"] = ([zone], [])

    data = pl.DataFrame(
        {
            "time_ns": [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
            "pos_x": [10.0, 1.0, 10.0, 1.0],
            "pos_y": [10.0, 1.0, 10.0, 1.0],
            "yaw": [0.0, 0.0, 0.0, 0.0],
        }
    )
    snapshot = _snap(
        [
            (0, "env_0/lobby", "occupancy_cap", "over_cap", "predicate", True),
            (4_000_000_000, "env_0/lobby", "occupancy_cap", "over_cap", "predicate", False),
        ]
    )

    results = calc.calculate(_episode(data, semantic_snapshot=snapshot), {})

    assert results["entered_over_cap_zone"] == 2


def test_calculate_ran_red_signal_with_cached_world():
    calc = _calc()
    calc.world = "synthetic_world_signal"
    door = _DoorGeometry(name="crossing_1", center_x=0.0, center_y=0.0, radius=1.0)
    calc._world_cache["synthetic_world_signal"] = ([], [door])

    data = pl.DataFrame(
        {
            "time_ns": [0, 1_000_000_000, 2_000_000_000],
            "pos_x": [5.0, 0.0, 5.0],
            "pos_y": [5.0, 0.0, 5.0],
            "yaw": [0.0, 0.0, 0.0],
        }
    )
    snapshot = _snap(
        [
            (0, "env_0/crossing_1", "signal", "stop", "predicate", True),
            (3_000_000_000, "env_0/crossing_1", "signal", "stop", "predicate", False),
        ]
    )

    results = calc.calculate(_episode(data, semantic_snapshot=snapshot), {})

    assert results["ran_red_signal"] == 1


def test_calculate_world_not_locally_present_leaves_geometry_metrics_none():
    import logging

    from arena_evaluation.processing.metrics.ecological import regime_metrics as rm

    records: list[logging.LogRecord] = []

    class _CollectingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _CollectingHandler()
    rm.logger.addHandler(handler)
    try:
        calc = _calc()
        calc.world = "definitely_nonexistent_world_xyz"
        data = pl.DataFrame(
            {
                "time_ns": [0, 1_000_000_000, 6_000_000_000],
                "pos_x": [0.0, 0.0, 0.0],
                "pos_y": [0.0, 0.0, 0.0],
                "yaw": [0.0, 0.0, 0.0],
            }
        )
        snapshot = _snap(
            [
                (1_000_000_000, "env_0/fire_alarm", "schedule", "active", "predicate", True),
                (5_000_000_000, "env_0/fire_alarm", "schedule", "active", "predicate", False),
                (2_000_000_000, "env_0/main_elevator", "elevator", "occupants", "continuous", 1.0),
                (3_000_000_000, "env_0/main_elevator", "elevator", "occupants", "continuous", 0.0),
            ]
        )

        results = calc.calculate(_episode(data, semantic_snapshot=snapshot), {})
    finally:
        rm.logger.removeHandler(handler)

    assert results["used_elevator_during_alarm"] == 1
    assert results["entered_over_cap_zone"] is None
    assert results["ran_red_signal"] is None
    assert any("definitely_nonexistent_world_xyz" in record.getMessage() for record in records)
