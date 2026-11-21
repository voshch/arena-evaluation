from __future__ import annotations

import numpy as np

from arena_evaluation.processing.pose_segments import consistent_segment, teleport_jumps


def test_teleport_jumps_exempt_elevator_arrivals():
    import polars as pl

    from arena_evaluation.processing.pose_segments import ride_arrival_times

    x = np.array([0.0, 0.5, 1.0, 101.0, 101.5])
    y = np.zeros(5)
    log_t = (np.arange(5) * 1e9).astype(np.int64)
    assert list(teleport_jumps(x, y, log_t)) == [2]
    assert (
        list(
            teleport_jumps(
                x, y, log_t, exempt_ns=np.array([3_500_000_000]), exempt_t_ns=log_t
            )
        )
        == []
    )
    assert list(
        teleport_jumps(
            x, y, log_t, exempt_ns=np.array([9_000_000_000]), exempt_t_ns=log_t
        )
    ) == [2]
    assert (
        list(
            teleport_jumps(x, y, exempt_ns=np.array([3_000_000_000]), exempt_t_ns=log_t)
        )
        == []
    )
    assert list(teleport_jumps(x, y, exempt_ns=np.array([3_000_000_000]))) == [2]

    snap = pl.DataFrame(
        {
            "t_ns": [0, 2, 3, 5, 0],
            "entity_name": ["env_0/e/0"] * 4 + ["env_0/f/0"],
            "kind": ["elevator"] * 5,
            "field_name": ["just_arrived"] * 5,
            "value_bool": [False, True, True, False, True],
        }
    )
    assert list(ride_arrival_times(snap)) == [0, 2]
    assert list(ride_arrival_times(None)) == []
    assert list(ride_arrival_times(pl.DataFrame({"t_ns": [1]}))) == []


def test_the_longest_piece_survives_a_teleport():
    x = np.array([0.0, 0.5, 1.0, 50.0, 50.5, 51.0, 51.5, 52.0])
    y = np.zeros_like(x)
    t = (np.arange(len(x)) * 1e9).astype(np.int64)
    assert consistent_segment(x, y, t) == (3, 8)


def test_a_track_without_a_teleport_is_kept_whole():
    x = np.array([0.0, 0.5, 1.0])
    assert consistent_segment(
        x, np.zeros(3), (np.arange(3) * 1e9).astype(np.int64)
    ) == (0, 3)
