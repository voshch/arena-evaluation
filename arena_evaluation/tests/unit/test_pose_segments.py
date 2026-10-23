import numpy as np
import pytest

pytest.importorskip("polars")

from arena_evaluation.processing.pose_segments import teleport_jumps


def test_sparse_ground_truth_is_one_segment():
    # 1 m/s robot sampled every 1.5 s: 1.5 m steps, no teleport
    t = np.arange(0, 30, 1.5)
    x = t * 1.0
    y = np.zeros_like(x)
    assert teleport_jumps(x, y, (t * 1e9).astype(np.int64)).size == 0


def test_reset_teleport_is_a_jump():
    t = np.arange(0, 10, 0.05)
    x = t * 1.0
    x[100:] += 20.0  # robot restaged 20 m away within one sample
    y = np.zeros_like(x)
    assert list(teleport_jumps(x, y, (t * 1e9).astype(np.int64))) == [99]


def test_distance_rule_without_time():
    x = np.array([0.0, 0.2, 0.4, 3.0, 3.2])
    y = np.zeros_like(x)
    assert list(teleport_jumps(x, y)) == [2]


def test_teleport_jumps_exempt_elevator_arrivals():
    import polars as pl

    from arena_evaluation.processing.pose_segments import elevator_arrival_times

    x = np.array([0.0, 0.5, 1.0, 101.0, 101.5])
    y = np.zeros(5)
    log_t = (np.arange(5) * 1e9).astype(np.int64)
    assert list(teleport_jumps(x, y, log_t)) == [2]
    assert list(teleport_jumps(x, y, log_t, exempt_ns=np.array([3_500_000_000]), exempt_time_ns=log_t)) == []
    assert list(teleport_jumps(x, y, log_t, exempt_ns=np.array([9_000_000_000]), exempt_time_ns=log_t)) == [2]
    assert list(teleport_jumps(x, y, exempt_ns=np.array([3_000_000_000]), exempt_time_ns=log_t)) == []
    assert list(teleport_jumps(x, y, exempt_ns=np.array([3_000_000_000]))) == [2]

    snap = pl.DataFrame(
        {
            "time_ns": [0, 2, 3, 5, 0],
            "entity": ["env_0/e/0"] * 4 + ["env_0/f/0"],
            "kind": ["elevator"] * 5,
            "field": ["just_arrived"] * 5,
            "value_bool": [False, True, True, False, True],
        }
    )
    assert list(elevator_arrival_times(snap)) == [0, 2]
    assert list(elevator_arrival_times(None)) == []
    assert list(elevator_arrival_times(pl.DataFrame({"time_ns": [1]}))) == []
