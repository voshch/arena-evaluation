import numpy as np
import pytest
from robot_params_factory import make_robot_params

pytest.importorskip("polars")

import polars as pl

from arena_evaluation.processing.metrics.base import NULL_REASON_SUFFIX, Direction
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from arena_evaluation.processing.metrics.common.zones import (
    DoorGeometry as _DoorGeometry,
    ZoneGeometry as _ZoneGeometry,
    extract_door_geometry as _extract_door_geometry,
    extract_zone_geometry as _extract_zone_geometry,
)
from arena_evaluation.processing.metrics.ecological.compliance_metrics import (
    ComplianceMetricsCalculator,
    _door_open_series,
    _is_open_at,
    _restricted_zone_entries,
)
from snapshot_rows import snapshot_frame


def _calc():
    return ComplianceMetricsCalculator(
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


def test_restricted_zone_entries_counts_transitions_not_dwell():
    zones = [
        _ZoneGeometry(
            name="pharmacy", polygon=None, max_speed=None, quiet=False, restricted=True
        )
    ]
    # outside, enter, stay, leave, re-enter
    zone_idx = np.array([-1, 0, 0, -1, 0])

    entries = _restricted_zone_entries(zone_idx, zones)

    assert entries == 2


def test_door_open_series_and_query():
    events = pl.DataFrame(
        {
            "t_ns": [1_000_000_000, 3_000_000_000],
            "entity": ["env_0/door_1", "env_0/door_1"],
            "kind": ["door", "door"],
            "field": ["state", "state"],
            "current": ["opening", "open"],
        }
    )

    series = _door_open_series(events)

    assert _is_open_at(series, "door_1", 0) is False  # seeded closed before first event
    assert _is_open_at(series, "door_1", 1_000_000_000) is False  # opening != open
    assert _is_open_at(series, "door_1", 3_500_000_000) is True
    assert _is_open_at(series, "unknown_door", 5_000_000_000) is False


def test_doorway_blocking_time_requires_stationary_and_open():
    doors = [_DoorGeometry(name="door_1", center_x=0.0, center_y=0.0, radius=1.0)]
    snapshot = snapshot_frame(
        [
            (0, "env_0/door_1", "door", "state", "discrete", "open"),
            (2_000_000_000, "env_0/door_1", "door", "state", "discrete", "open"),
        ]
    )
    data = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
            "x": [0.0, 0.0, 0.0, 0.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "yaw": [0.0, 0.0, 0.0, 0.0],
            "speed_mps": [0.0, 0.0, 1.0, 0.0],  # moving through [2s, 3s)
        }
    )
    episode = _episode(data, semantic_snapshot=snapshot)

    calc = _calc()
    pos_x, pos_y, _yaw, _ox, _oy, _oyaw = calc.resolve_robot_pose(episode)
    speed = np.abs(episode.data["speed_mps"].to_numpy())
    t_ns = episode.data["t_ns"].to_numpy()

    total = calc._doorway_blocking_time(episode, pos_x, pos_y, speed, t_ns, doors)

    # stationary intervals [0,1) and [1,2) both qualify, [2,3) does not (moving)
    assert total == pytest.approx(2.0)


def test_doorway_blocking_degrades_to_zero_without_snapshot():
    doors = [_DoorGeometry(name="door_1", center_x=0.0, center_y=0.0, radius=1.0)]
    data = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000],
            "x": [0.0, 0.0],
            "y": [0.0, 0.0],
            "yaw": [0.0, 0.0],
            "speed_mps": [0.0, 0.0],
        }
    )
    episode = _episode(data, semantic_snapshot=None)
    calc = _calc()

    pos_x, pos_y, _yaw, _ox, _oy, _oyaw = calc.resolve_robot_pose(episode)
    speed = np.abs(episode.data["speed_mps"].to_numpy())
    t_ns = episode.data["t_ns"].to_numpy()

    total = calc._doorway_blocking_time(episode, pos_x, pos_y, speed, t_ns, doors)

    assert total == pytest.approx(0.0)


def test_calculate_end_to_end_with_cached_world():
    from arena_simulation_setup.shared.semantics import SemanticCfg
    from arena_simulation_setup.tree.World import Level, Zone
    from arena_simulation_setup.utils.geometry import Position

    zone = Zone(
        name="lobby",
        polygon=[
            Position(0.0, 0.0),
            Position(0.0, 4.0),
            Position(4.0, 4.0),
            Position(4.0, 0.0),
        ],
        semantics=[
            SemanticCfg(role="state", name="max_speed", value=1.0),
            SemanticCfg(role="predicate", name="quiet", value=True),
        ],
    )
    level = Level(zones=[zone])

    calc = _calc()
    calc.world = "synthetic_world"
    calc._world_cache["synthetic_world"] = (
        _extract_zone_geometry(level),
        _extract_door_geometry(level),
    )

    data = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000, 2_000_000_000],
            "x": [1.0, 1.0, 1.0],
            "y": [1.0, 1.0, 1.0],
            "yaw": [0.0, 0.0, 0.0],
            "speed_mps": [0.5, 2.0, 0.5],  # over 1.0 m/s cap during [1s, 2s)
        }
    )
    snapshot = snapshot_frame(
        [(0, "env_0/door_1", "door", "state", "discrete", "closed")]
    )
    episode = _episode(data, semantic_snapshot=snapshot)

    results = calc.calculate(episode, {})

    assert results["speed_zone_violations"] == 1
    assert results["speed_zone_violation_seconds"] == pytest.approx(1.0)
    assert results["quiet_zone_dwell_seconds"] == pytest.approx(2.0)
    assert results["restricted_zone_entries"] == 0
    assert results["doorway_blocking_time"] == pytest.approx(0.0)


def _quiet_zone_level():
    from arena_simulation_setup.shared.semantics import SemanticCfg
    from arena_simulation_setup.tree.World import Level, Zone
    from arena_simulation_setup.utils.geometry import Position

    zone = Zone(
        name="lobby",
        polygon=[
            Position(0.0, 0.0),
            Position(0.0, 4.0),
            Position(4.0, 4.0),
            Position(4.0, 0.0),
        ],
        semantics=[
            SemanticCfg(role="state", name="max_speed", value=1.0),
            SemanticCfg(role="predicate", name="quiet", value=True),
        ],
    )
    return Level(zones=[zone])


def _open_door_snapshot():
    return snapshot_frame([(0, "env_0/door_1", "door", "state", "discrete", "open")])


def _quiet_zone_calculator(world_name):
    level = _quiet_zone_level()
    calc = _calc()
    calc.world = world_name
    calc._world_cache[world_name] = (
        _extract_zone_geometry(level),
        _extract_door_geometry(level),
    )
    return calc


def test_compliance_outputs_declare_their_better_side():
    assert ComplianceMetricsCalculator.DIRECTIONS == {
        "speed_zone_violations": Direction.LOWER,
        "speed_zone_violation_seconds": Direction.LOWER,
        "quiet_zone_dwell_seconds": Direction.LOWER,
        "restricted_zone_entries": Direction.LOWER,
        "doorway_blocking_time": Direction.LOWER,
        "passing_rule_compliance": Direction.HIGHER,
    }
    assert set(ComplianceMetricsCalculator.PRIMARY_OUTPUTS) <= set(
        ComplianceMetricsCalculator.DIRECTIONS
    )


def test_speed_metrics_are_null_with_topic_missing_without_the_speed_stream():
    calc = _quiet_zone_calculator("synthetic_speed_missing")
    data = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000, 2_000_000_000],
            "x": [1.0, 1.0, 1.0],
            "y": [1.0, 1.0, 1.0],
            "yaw": [0.0, 0.0, 0.0],
        }
    )

    results = calc.calculate(
        _episode(data, semantic_snapshot=_open_door_snapshot()), {}
    )

    for key in (
        "speed_zone_violations",
        "speed_zone_violation_seconds",
        "doorway_blocking_time",
    ):
        assert results[key] is None, key
        assert results[f"{key}{NULL_REASON_SUFFIX}"] == "topic_missing:robot_odom", key
    assert results["quiet_zone_dwell_seconds"] == pytest.approx(2.0)
    assert results["restricted_zone_entries"] == 0


def test_speed_metrics_are_null_with_no_samples_when_a_speed_is_not_finite():
    calc = _quiet_zone_calculator("synthetic_speed_not_finite")
    data = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000, 2_000_000_000],
            "x": [1.0, 1.0, 1.0],
            "y": [1.0, 1.0, 1.0],
            "yaw": [0.0, 0.0, 0.0],
            "speed_mps": [0.5, float("nan"), 0.5],
        }
    )

    results = calc.calculate(
        _episode(data, semantic_snapshot=_open_door_snapshot()), {}
    )

    for key in (
        "speed_zone_violations",
        "speed_zone_violation_seconds",
        "doorway_blocking_time",
    ):
        assert results[key] is None, key
        assert results[f"{key}{NULL_REASON_SUFFIX}"] == "no_samples", key
    assert results["quiet_zone_dwell_seconds"] == pytest.approx(2.0)


