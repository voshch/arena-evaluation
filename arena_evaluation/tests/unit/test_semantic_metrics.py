import pytest

pl = pytest.importorskip("polars")

from arena_evaluation.storage.schemas import RobotParams, AlignedEpisodeBundle
from arena_evaluation.processing.metrics.ecological.semantic_metrics import SemanticInteractionMetricsCalculator


def _calc():
    return SemanticInteractionMetricsCalculator(RobotParams(0.2, 0.0, 10.0))


def _episode(snapshot, end_time_ns=6_000_000_000):
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame({"time_ns": [0, end_time_ns]}),
        start_pos=[],
        goal_pos=[],
        semantic_snapshot=snapshot,
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


def test_time_waiting_at_doors_counts_triggered_not_open():
    snapshot = _snap(
        [
            (1_000_000_000, "env_0/door_1", "door", "triggered", "predicate", True),
            (3_000_000_000, "env_0/door_1", "door", "state", "discrete", "opening"),
            (4_000_000_000, "env_0/door_1", "door", "state", "discrete", "open"),
            (5_000_000_000, "env_0/door_1", "door", "triggered", "predicate", False),
        ]
    )

    results = _calc().calculate(_episode(snapshot), {})

    # triggered+not-open holds [1s,3s) closed and [3s,4s) opening: 2s + 1s
    assert results["time_waiting_at_doors"] == pytest.approx(3.0)
    assert results["elevator_rides"] == 0


def test_time_waiting_at_doors_sums_across_doors():
    snapshot = _snap(
        [
            (1_000_000_000, "env_0/door_1", "door", "triggered", "predicate", True),
            (2_000_000_000, "env_0/door_1", "door", "state", "discrete", "open"),
            (1_000_000_000, "env_0/door_2", "door", "triggered", "predicate", True),
            (3_000_000_000, "env_0/door_2", "door", "state", "discrete", "open"),
        ]
    )

    results = _calc().calculate(_episode(snapshot), {})

    # door_1: triggered [1s,2s) not open -> 1s. door_2: triggered [1s,3s) not open -> 2s.
    assert results["time_waiting_at_doors"] == pytest.approx(3.0)


def test_elevator_rides_requires_nonzero_occupants():
    snapshot = _snap(
        [
            (1_000_000_000, "env_0/1_elevator", "elevator", "occupants", "continuous", 1.0),
            (2_000_000_000, "env_0/1_elevator", "elevator", "just_arrived", "predicate", True),
            (3_000_000_000, "env_0/1_elevator", "elevator", "just_arrived", "predicate", False),
            (4_000_000_000, "env_0/1_elevator", "elevator", "occupants", "continuous", 0.0),
            (5_000_000_000, "env_0/1_elevator", "elevator", "just_arrived", "predicate", True),
        ]
    )

    results = _calc().calculate(_episode(snapshot), {})

    # Only the first just_arrived=True edge has occupants > 0
    assert results["elevator_rides"] == 1
    assert results["time_waiting_at_doors"] == pytest.approx(0.0)


def test_no_semantic_snapshot_returns_zero_defaults():
    results = _calc().calculate(_episode(None), {})

    assert results["time_waiting_at_doors"] == 0.0
    assert results["elevator_rides"] == 0


def test_empty_semantic_snapshot_returns_zero_defaults():
    snapshot = pl.DataFrame(
        {
            "time_ns": [],
            "entity": [],
            "kind": [],
            "field": [],
            "field_kind": [],
            "value_str": [],
            "value_num": [],
            "value_bool": [],
        }
    )

    results = _calc().calculate(_episode(snapshot), {})

    assert results["time_waiting_at_doors"] == 0.0
    assert results["elevator_rides"] == 0
