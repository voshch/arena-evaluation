"""Unit tests for the vectorized scope-window detectors.

Covers ``processing/windows.py``: pedestrian / fleet pair construction, encounter-bout
detection (distance threshold, interior-gap merging, ordering, approach classification,
versioned detector metadata) and contact-bout kind normalization and encounter
linkage. Every fixture is in-memory: no disk I/O, no mocks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from arena_evaluation.processing.windows import (
    APPROACH_TYPE_DTYPE,
    CONTACT_WINDOW_SCHEMA,
    DEFAULT_DETECTOR_PARAMS,
    DETECTOR_VERSION,
    ENCOUNTER_EDGE_DISTANCE_M,
    ENCOUNTER_WINDOW_SCHEMA,
    PAIR_SCHEMA,
    DetectorParams,
    contact_windows_lf,
    detect_encounter_windows_lf,
    empty_contact_windows,
    empty_encounter_windows,
    empty_pairs,
    fleet_pairs_lf,
    pedestrian_pairs_lf,
)
from arena_evaluation.storage.exceptions import MetricInputError

NS_PER_S: Final[int] = 1_000_000_000
ROBOT_RADIUS_M: Final[float] = 0.25
ACTOR_RADIUS_M: Final[float] = 0.3
DEFAULT_PARAMS_HASH: Final[str] = "b66b842c3b4c"

_ROBOT_TRACKS_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "robot_id": pl.String,
        "t_ns": pl.Int64,
        "x": pl.Float64,
        "y": pl.Float64,
        "yaw": pl.Float64,
        "vx": pl.Float64,
        "vy": pl.Float64,
    }
)

_PEDESTRIAN_TRACKS_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "ped_id": pl.String,
        "ped_name": pl.String,
        "t_ns": pl.Int64,
        "x": pl.Float64,
        "y": pl.Float64,
        "vx": pl.Float64,
        "vy": pl.Float64,
        "radius_m": pl.Float64,
    }
)

_FLEET_POSE_SCHEMA: Final[pl.Schema] = pl.Schema(
    {"t_ns": pl.Int64, "pos_x": pl.Float64, "pos_y": pl.Float64, "radius_m": pl.Float64}
)

_PAIR_INPUT_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "robot_id": pl.String,
        "actor_id": pl.String,
        "actor_name": pl.String,
        "actor_kind": pl.String,
        "t_ns": pl.Int64,
        "robot_x": pl.Float64,
        "robot_y": pl.Float64,
        "robot_yaw": pl.Float64,
        "robot_vx": pl.Float64,
        "robot_vy": pl.Float64,
        "actor_x": pl.Float64,
        "actor_y": pl.Float64,
        "actor_vx": pl.Float64,
        "actor_vy": pl.Float64,
        "actor_radius_m": pl.Float64,
    }
)

_ROBOT_DEFAULTS: Final[dict[str, object]] = {
    "robot_id": "r1",
    "x": 0.0,
    "y": 0.0,
    "yaw": 0.0,
    "vx": 0.0,
    "vy": 0.0,
}
_PEDESTRIAN_DEFAULTS: Final[dict[str, object]] = {
    "ped_id": "p1",
    "ped_name": "Pedestrian 1",
    "x": 0.0,
    "y": 0.0,
    "vx": 0.0,
    "vy": 0.0,
    "radius_m": ACTOR_RADIUS_M,
}
_FLEET_DEFAULTS: Final[dict[str, object]] = {"pos_y": 0.0, "radius_m": ROBOT_RADIUS_M}
_PAIR_DEFAULTS: Final[dict[str, object]] = {
    "robot_id": "r1",
    "actor_id": "p1",
    "actor_name": "Pedestrian 1",
    "actor_kind": "pedestrian",
    "robot_x": 0.0,
    "robot_y": 0.0,
    "robot_yaw": 0.0,
    "robot_vx": 0.0,
    "robot_vy": 0.0,
    "actor_x": 0.0,
    "actor_y": 0.0,
    "actor_vx": 0.0,
    "actor_vy": 0.0,
    "actor_radius_m": ACTOR_RADIUS_M,
}

_ENCOUNTER_DEFAULTS: Final[dict[str, object]] = {
    "encounter_idx": 0,
    "robot_id": "r1",
    "actor_id": "Pedestrian_1",
    "actor_name": "Pedestrian_1",
    "actor_kind": "pedestrian",
    "t_start_ns": 0,
    "t_end_ns": 1_000_000_000,
    "min_distance_m": 0.5,
    "min_clearance_m": 0.2,
    "approach_type": "crossing",
    "detector_version": DETECTOR_VERSION,
    "detector_params_hash": DEFAULT_PARAMS_HASH,
}


def _build(
    rows: Sequence[dict[str, object]], defaults: dict[str, object], schema: pl.Schema
) -> pl.DataFrame:
    return pl.DataFrame([{**defaults, **row} for row in rows], schema=schema)


def _robot_tracks(rows: Sequence[dict[str, object]]) -> pl.DataFrame:
    return _build(rows, _ROBOT_DEFAULTS, _ROBOT_TRACKS_SCHEMA)


def _pedestrian_tracks(rows: Sequence[dict[str, object]]) -> pl.DataFrame:
    return _build(rows, _PEDESTRIAN_DEFAULTS, _PEDESTRIAN_TRACKS_SCHEMA)


def _fleet_poses(rows: Sequence[dict[str, object]]) -> pl.DataFrame:
    return _build(rows, _FLEET_DEFAULTS, _FLEET_POSE_SCHEMA)


def _encounter_table(rows: Sequence[dict[str, object]]) -> pl.DataFrame:
    return _build(rows, _ENCOUNTER_DEFAULTS, ENCOUNTER_WINDOW_SCHEMA)


def _pairs_lf(rows: Sequence[dict[str, object]]) -> pl.LazyFrame:
    """Pair rows from compact samples, deriving centre distance and edge clearance like the detector."""
    frame = _build(rows, _PAIR_DEFAULTS, _PAIR_INPUT_SCHEMA)
    return (
        frame.with_columns(
            distance_m=(
                (pl.col("actor_x") - pl.col("robot_x")).pow(2)
                + (pl.col("actor_y") - pl.col("robot_y")).pow(2)
            ).sqrt(),
        )
        .with_columns(
            edge_clearance_m=pl.col("distance_m")
            - (pl.lit(ROBOT_RADIUS_M) + pl.col("actor_radius_m"))
        )
        .select(PAIR_SCHEMA.names())
        .lazy()
    )


def _detect_windows(
    pairs: pl.LazyFrame, params: DetectorParams = DEFAULT_DETECTOR_PARAMS
) -> pl.DataFrame:
    return detect_encounter_windows_lf(pairs, params).collect()


# ---------------------------------------------------------------------------
# pedestrian_pairs_lf
# ---------------------------------------------------------------------------


def test_pedestrian_pairs_known_geometry() -> None:
    robot = _robot_tracks(
        [{"t_ns": 0, "x": 0.0, "y": 0.0, "yaw": 0.3, "vx": 1.0, "vy": 0.5}]
    )
    pedestrians = _pedestrian_tracks(
        [
            {
                "t_ns": 50_000_000,
                "x": 3.0,
                "y": 4.0,
                "vx": -0.5,
                "vy": 0.1,
                "radius_m": 0.4,
            }
        ]
    )

    result = pedestrian_pairs_lf(robot, pedestrians, robot_radius_m=0.26).collect()

    expected = pl.DataFrame(
        {
            "robot_id": ["r1"],
            "actor_id": ["p1"],
            "actor_name": ["Pedestrian 1"],
            "actor_kind": ["pedestrian"],
            "t_ns": [50_000_000],
            "robot_x": [0.0],
            "robot_y": [0.0],
            "robot_yaw": [0.3],
            "robot_vx": [1.0],
            "robot_vy": [0.5],
            "actor_x": [3.0],
            "actor_y": [4.0],
            "actor_vx": [-0.5],
            "actor_vy": [0.1],
            "actor_radius_m": [0.4],
            "distance_m": [5.0],
            "edge_clearance_m": [5.0 - 0.26 - 0.4],
        },
        schema=PAIR_SCHEMA,
    )
    assert_frame_equal(result, expected)


def test_pedestrian_pairs_refuse_a_pedestrian_without_a_recorded_radius() -> None:
    robot = _robot_tracks([{"t_ns": 0}])
    pedestrians = _pedestrian_tracks([{"t_ns": 0, "x": 5.0, "radius_m": None}])

    with pytest.raises(MetricInputError, match="radius_m"):
        pedestrian_pairs_lf(robot, pedestrians, robot_radius_m=0.5)


@pytest.mark.parametrize(
    ("offset_ns", "expected_height"),
    [(-100_000_000, 0), (0, 1), (100_000_000, 1), (150_000_000, 0)],
)
def test_pedestrian_pairs_alignment_tolerance(
    offset_ns: int, expected_height: int
) -> None:
    robot = _robot_tracks([{"t_ns": 0}])
    pedestrians = _pedestrian_tracks([{"t_ns": offset_ns, "x": 5.0}])

    result = pedestrian_pairs_lf(robot, pedestrians, ROBOT_RADIUS_M).collect()

    assert result.height == expected_height
    assert result.schema == PAIR_SCHEMA


def test_pedestrian_pairs_accepts_lazy_input() -> None:
    robot = _robot_tracks([{"t_ns": 0}])
    pedestrians = _pedestrian_tracks([{"t_ns": 0, "x": 1.0}])

    eager = pedestrian_pairs_lf(robot, pedestrians, ROBOT_RADIUS_M).collect()
    lazy = pedestrian_pairs_lf(
        robot.lazy(), pedestrians.lazy(), ROBOT_RADIUS_M
    ).collect()

    assert_frame_equal(lazy, eager)


# ---------------------------------------------------------------------------
# detect_encounter_windows_lf
# ---------------------------------------------------------------------------


def test_encounter_straight_pass_by_exact_bounds() -> None:
    rows = [
        {"t_ns": i * 100_000_000, "actor_x": -5.0 + i / 10.0, "actor_vx": 1.0}
        for i in range(100)
    ]

    windows = _detect_windows(_pairs_lf(rows))

    expected = pl.DataFrame(
        {
            "t_start_ns": [2_500_000_000],
            "t_end_ns": [7_500_000_000],
            "min_distance_m": [0.0],
            "min_clearance_m": [0.0 - ROBOT_RADIUS_M - ACTOR_RADIUS_M],
        },
        schema={
            "t_start_ns": pl.Int64,
            "t_end_ns": pl.Int64,
            "min_distance_m": pl.Float64,
            "min_clearance_m": pl.Float64,
        },
    )
    assert_frame_equal(
        windows.select("t_start_ns", "t_end_ns", "min_distance_m", "min_clearance_m"),
        expected,
    )


@pytest.mark.parametrize(
    ("edge_clearance_m", "expected_height"), [(1.999, 1), (2.0, 1), (2.001, 0)]
)
def test_encounter_edge_threshold_is_inclusive(
    edge_clearance_m: float, expected_height: int
) -> None:
    windows = _detect_windows(_pair_frame([edge_clearance_m]))

    assert windows.height == expected_height
    if expected_height == 1:
        assert windows["min_clearance_m"][0] == edge_clearance_m
        assert windows["min_distance_m"][0] == pytest.approx(
            edge_clearance_m + ROBOT_RADIUS_M + ACTOR_RADIUS_M
        )


@pytest.mark.parametrize(("ped_radius_m", "expected_height"), [(0.3, 0), (0.6, 1)])
def test_encounter_edge_threshold_is_radius_aware(
    ped_radius_m: float, expected_height: int
) -> None:
    robot = _robot_tracks([{"t_ns": 0}])
    pedestrians = _pedestrian_tracks([{"t_ns": 0, "x": 2.7, "radius_m": ped_radius_m}])

    windows = _detect_windows(pedestrian_pairs_lf(robot, pedestrians, ROBOT_RADIUS_M))

    assert windows.height == expected_height


@pytest.mark.parametrize(
    "far_rows",
    [
        [{"t_ns": 0, "actor_x": 10.0}],
        [{"t_ns": 0, "actor_x": 10.0}, {"t_ns": 500_000_000, "actor_x": 10.5}],
    ],
)
def test_encounter_never_within_threshold_has_no_window(
    far_rows: list[dict[str, object]],
) -> None:
    windows = _detect_windows(_pairs_lf(far_rows))

    assert_frame_equal(windows, empty_encounter_windows())


def test_encounter_leading_shoulder_does_not_extend_start() -> None:
    rows = [
        {"t_ns": t_s * NS_PER_S, "actor_x": x}
        for t_s, x in ((0, 5.0), (1, 5.0), (2, 2.0), (3, 5.0), (4, 5.0), (5, 5.0))
    ]

    windows = _detect_windows(_pairs_lf(rows))

    expected = pl.DataFrame(
        {
            "t_start_ns": [2_000_000_000],
            "t_end_ns": [2_000_000_000],
            "min_distance_m": [2.0],
        },
        schema={
            "t_start_ns": pl.Int64,
            "t_end_ns": pl.Int64,
            "min_distance_m": pl.Float64,
        },
    )
    assert_frame_equal(
        windows.select("t_start_ns", "t_end_ns", "min_distance_m"), expected
    )


def test_encounter_one_sample_gap_longer_than_the_merge_window_does_not_bridge() -> (
    None
):
    rows = [
        {"t_ns": t_s * NS_PER_S, "actor_x": x}
        for t_s, x in ((0, 2.0), (1, 5.0), (2, 2.0))
    ]

    windows = _detect_windows(_pairs_lf(rows))

    expected = pl.DataFrame(
        {
            "t_start_ns": [0, 2_000_000_000],
            "t_end_ns": [0, 2_000_000_000],
            "min_distance_m": [2.0, 2.0],
            "min_clearance_m": [
                2.0 - ROBOT_RADIUS_M - ACTOR_RADIUS_M,
                2.0 - ROBOT_RADIUS_M - ACTOR_RADIUS_M,
            ],
        },
        schema={
            "t_start_ns": pl.Int64,
            "t_end_ns": pl.Int64,
            "min_distance_m": pl.Float64,
            "min_clearance_m": pl.Float64,
        },
    )
    assert_frame_equal(
        windows.select("t_start_ns", "t_end_ns", "min_distance_m", "min_clearance_m"),
        expected,
    )


def test_encounter_bridges_short_interior_gap() -> None:
    rows = [
        {"t_ns": t_ns, "actor_x": x}
        for t_ns, x in ((0, 2.0), (450_000_000, 5.0), (900_000_000, 2.0))
    ]

    windows = _detect_windows(_pairs_lf(rows))

    expected = pl.DataFrame(
        {
            "t_start_ns": [0],
            "t_end_ns": [900_000_000],
            "min_distance_m": [2.0],
            "min_clearance_m": [2.0 - ROBOT_RADIUS_M - ACTOR_RADIUS_M],
        },
        schema={
            "t_start_ns": pl.Int64,
            "t_end_ns": pl.Int64,
            "min_distance_m": pl.Float64,
            "min_clearance_m": pl.Float64,
        },
    )
    assert_frame_equal(
        windows.select("t_start_ns", "t_end_ns", "min_distance_m", "min_clearance_m"),
        expected,
    )


def test_encounter_gap_of_exactly_the_merge_window_does_not_bridge() -> None:
    rows = [
        {"t_ns": t_ns, "actor_x": x}
        for t_ns, x in ((0, 2.0), (500_000_000, 5.0), (1_000_000_000, 2.0))
    ]

    windows = _detect_windows(_pairs_lf(rows))

    expected = pl.DataFrame(
        {
            "encounter_idx": pl.Series([0, 1], dtype=pl.UInt32),
            "t_start_ns": [0, 1_000_000_000],
            "t_end_ns": [0, 1_000_000_000],
        },
        schema={
            "encounter_idx": pl.UInt32,
            "t_start_ns": pl.Int64,
            "t_end_ns": pl.Int64,
        },
    )
    assert_frame_equal(
        windows.select("encounter_idx", "t_start_ns", "t_end_ns"), expected
    )


def test_encounter_splits_long_interior_gap() -> None:
    rows = [
        {"t_ns": t_s * NS_PER_S, "actor_x": x}
        for t_s, x in ((0, 2.0), (1, 5.0), (2, 5.0), (3, 5.0), (4, 2.0))
    ]

    windows = _detect_windows(_pairs_lf(rows))

    expected = pl.DataFrame(
        {
            "encounter_idx": pl.Series([0, 1], dtype=pl.UInt32),
            "t_start_ns": [0, 4_000_000_000],
            "t_end_ns": [0, 4_000_000_000],
            "min_distance_m": [2.0, 2.0],
            "min_clearance_m": [
                2.0 - ROBOT_RADIUS_M - ACTOR_RADIUS_M,
                2.0 - ROBOT_RADIUS_M - ACTOR_RADIUS_M,
            ],
        },
        schema={
            "encounter_idx": pl.UInt32,
            "t_start_ns": pl.Int64,
            "t_end_ns": pl.Int64,
            "min_distance_m": pl.Float64,
            "min_clearance_m": pl.Float64,
        },
    )
    assert_frame_equal(
        windows.select(
            "encounter_idx",
            "t_start_ns",
            "t_end_ns",
            "min_distance_m",
            "min_clearance_m",
        ),
        expected,
    )


def test_encounter_two_pedestrians_get_stable_zero_based_ids() -> None:
    rows = [
        {"t_ns": 0, "actor_id": "p1", "actor_name": "Pedestrian 1", "actor_x": 2.0},
        {
            "t_ns": 1_000_000_000,
            "actor_id": "p2",
            "actor_name": "Pedestrian 2",
            "actor_x": 2.0,
        },
    ]

    windows = _detect_windows(_pairs_lf(rows))

    expected = pl.DataFrame(
        {
            "encounter_idx": pl.Series([0, 1], dtype=pl.UInt32),
            "actor_id": ["p1", "p2"],
            "t_start_ns": [0, 1_000_000_000],
            "t_end_ns": [0, 1_000_000_000],
        },
        schema={
            "encounter_idx": pl.UInt32,
            "actor_id": pl.String,
            "t_start_ns": pl.Int64,
            "t_end_ns": pl.Int64,
        },
    )
    assert_frame_equal(
        windows.select("encounter_idx", "actor_id", "t_start_ns", "t_end_ns"), expected
    )


@pytest.mark.parametrize(
    ("robot_yaw", "robot_vx", "actor_x", "actor_y", "actor_vx", "expected"),
    [
        (0.0, 1.0, 2.0, 0.0, -1.0, "head_on"),
        (0.0, 1.0, 0.0, 2.0, -1.0, "crossing"),
        (0.0, 2.0, 1.0, 2.0, 1.0, "overtaking"),
        (0.0, 1.0, 2.0, 0.5, 0.0, "stationary"),
    ],
)
def test_encounter_approach_type(
    robot_yaw: float,
    robot_vx: float,
    actor_x: float,
    actor_y: float,
    actor_vx: float,
    expected: str,
) -> None:
    rows = [
        {
            "t_ns": 0,
            "robot_yaw": robot_yaw,
            "robot_vx": robot_vx,
            "actor_x": actor_x,
            "actor_y": actor_y,
            "actor_vx": actor_vx,
        }
    ]

    windows = _detect_windows(_pairs_lf(rows))

    assert windows.height == 1
    assert_series_equal(
        windows["approach_type"],
        pl.Series("approach_type", [expected], dtype=APPROACH_TYPE_DTYPE),
    )


@pytest.mark.parametrize(
    ("stationary_speed_mps", "expected"), [(0.15, "crossing"), (0.25, "stationary")]
)
def test_encounter_stationary_threshold_is_a_detector_parameter(
    stationary_speed_mps: float, expected: str
) -> None:
    rows = [{"t_ns": 0, "actor_x": 0.0, "actor_y": 2.0, "actor_vx": 0.2}]

    windows = _detect_windows(
        _pairs_lf(rows), DetectorParams(stationary_speed_mps=stationary_speed_mps)
    )

    assert windows["approach_type"].to_list() == [expected]


def test_encounter_detector_metadata_constants() -> None:
    windows = _detect_windows(_pairs_lf([{"t_ns": 0, "actor_x": 1.0}]))

    assert DETECTOR_VERSION == "wp2-scope-detectors-v4"
    assert_series_equal(
        windows["detector_version"],
        pl.Series("detector_version", ["wp2-scope-detectors-v4"], dtype=pl.String),
    )
    assert_series_equal(
        windows["detector_params_hash"],
        pl.Series("detector_params_hash", [DEFAULT_PARAMS_HASH], dtype=pl.String),
    )


def test_detector_params_fingerprint_is_stable_and_param_sensitive() -> None:
    assert DEFAULT_DETECTOR_PARAMS == DetectorParams()
    fingerprint = DEFAULT_DETECTOR_PARAMS.fingerprint()
    assert fingerprint == "b66b842c3b4c"
    assert len(fingerprint) == 12
    assert all(character in "0123456789abcdef" for character in fingerprint)
    assert DetectorParams(encounter_edge_distance_m=1.5).fingerprint() != fingerprint
    assert DetectorParams(contact_merge_gap_s=0.5).fingerprint() != fingerprint
    assert DetectorParams(stationary_speed_mps=0.2).fingerprint() != fingerprint


def test_encounter_custom_params_change_threshold_and_hash() -> None:
    rows = [{"t_ns": 0, "actor_x": 2.5}]
    custom_params = DetectorParams(encounter_edge_distance_m=1.9)

    default_windows = _detect_windows(_pairs_lf(rows))
    custom_windows = _detect_windows(_pairs_lf(rows), custom_params)

    assert default_windows.height == 1
    assert custom_windows.height == 0
    assert_series_equal(
        default_windows["detector_params_hash"],
        pl.Series("detector_params_hash", [DEFAULT_PARAMS_HASH], dtype=pl.String),
    )
    assert DEFAULT_PARAMS_HASH != custom_params.fingerprint()


def test_encounter_empty_pairs_returns_canonical_empty_frame() -> None:
    windows = _detect_windows(pl.LazyFrame(schema=PAIR_SCHEMA))

    assert_frame_equal(windows, empty_encounter_windows())


def test_empty_frame_helpers_have_canonical_schemas() -> None:
    assert_frame_equal(empty_pairs(), pl.DataFrame(schema=PAIR_SCHEMA))
    assert_frame_equal(
        empty_encounter_windows(), pl.DataFrame(schema=ENCOUNTER_WINDOW_SCHEMA)
    )
    assert_frame_equal(
        empty_contact_windows(), pl.DataFrame(schema=CONTACT_WINDOW_SCHEMA)
    )


# ---------------------------------------------------------------------------
# fleet_pairs_lf
# ---------------------------------------------------------------------------


def test_fleet_pairs_derive_robot_velocity_from_pose_gradient() -> None:
    robot = _robot_tracks([{"t_ns": 0}, {"t_ns": 100_000_000}, {"t_ns": 200_000_000}])
    fleet = {
        "r2": _fleet_poses(
            [
                {"t_ns": 0, "pos_x": 5.0},
                {"t_ns": 100_000_000, "pos_x": 4.9},
                {"t_ns": 200_000_000, "pos_x": 4.8},
            ]
        )
    }

    pairs = fleet_pairs_lf(robot, fleet, robot_radius_m=0.4).collect()

    expected = pl.DataFrame(
        {
            "robot_id": ["r1", "r1", "r1"],
            "actor_id": ["r2", "r2", "r2"],
            "actor_name": ["r2", "r2", "r2"],
            "actor_kind": ["robot", "robot", "robot"],
            "t_ns": [0, 100_000_000, 200_000_000],
            "actor_vx": [None, -1.0, -1.0],
            "actor_vy": [None, 0.0, 0.0],
            "actor_radius_m": [0.25, 0.25, 0.25],
            "distance_m": [5.0, 4.9, 4.8],
            "edge_clearance_m": [5.0 - 0.4 - 0.25, 4.9 - 0.4 - 0.25, 4.8 - 0.4 - 0.25],
        },
        schema={
            "robot_id": pl.String,
            "actor_id": pl.String,
            "actor_name": pl.String,
            "actor_kind": pl.String,
            "t_ns": pl.Int64,
            "actor_vx": pl.Float64,
            "actor_vy": pl.Float64,
            "actor_radius_m": pl.Float64,
            "distance_m": pl.Float64,
            "edge_clearance_m": pl.Float64,
        },
    )
    assert_frame_equal(
        pairs.select(
            "robot_id",
            "actor_id",
            "actor_name",
            "actor_kind",
            "t_ns",
            "actor_vx",
            "actor_vy",
            "actor_radius_m",
            "distance_m",
            "edge_clearance_m",
        ),
        expected,
    )


@pytest.mark.parametrize(
    ("actor_radius_m", "expected_radius_m"), [(0.25, 0.25), (0.5, 0.5)]
)
def test_fleet_pairs_edge_clearance_uses_robot_and_actor_radius(
    actor_radius_m: float, expected_radius_m: float
) -> None:
    robot = _robot_tracks([{"t_ns": 0}])
    fleet = {
        "r2": _fleet_poses([{"t_ns": 0, "pos_x": 4.0, "radius_m": actor_radius_m}])
    }

    pairs = fleet_pairs_lf(robot, fleet, robot_radius_m=0.4).collect()

    expected = pl.DataFrame(
        {
            "actor_radius_m": [expected_radius_m],
            "distance_m": [4.0],
            "edge_clearance_m": [4.0 - 0.4 - expected_radius_m],
        },
        schema={
            "actor_radius_m": pl.Float64,
            "distance_m": pl.Float64,
            "edge_clearance_m": pl.Float64,
        },
    )
    assert_frame_equal(
        pairs.select("actor_radius_m", "distance_m", "edge_clearance_m"), expected
    )


def test_fleet_pairs_empty_fleet_returns_canonical_empty_frame() -> None:
    pairs = fleet_pairs_lf(
        _robot_tracks([{"t_ns": 0}]), {}, robot_radius_m=0.4
    ).collect()

    assert_frame_equal(pairs, empty_pairs())


def test_fleet_encounter_detects_robot_actor_window() -> None:
    robot = _robot_tracks([{"t_ns": t_s * NS_PER_S} for t_s in (0, 2, 4, 6, 8, 10)])
    fleet = {
        "r2": _fleet_poses(
            [
                {"t_ns": t_s * NS_PER_S, "pos_x": x}
                for t_s, x in (
                    (0, 5.0),
                    (2, 4.0),
                    (4, 3.0),
                    (6, 2.0),
                    (8, 1.0),
                    (10, 0.0),
                )
            ]
        )
    }

    windows = _detect_windows(fleet_pairs_lf(robot, fleet, robot_radius_m=0.4))

    expected = pl.DataFrame(
        {
            "actor_id": ["r2"],
            "actor_kind": ["robot"],
            "t_start_ns": [6_000_000_000],
            "t_end_ns": [10_000_000_000],
            "min_distance_m": [0.0],
            "min_clearance_m": [0.0 - 0.4 - 0.25],
            "approach_type": ["crossing"],
        },
        schema={
            "actor_id": pl.String,
            "actor_kind": pl.String,
            "t_start_ns": pl.Int64,
            "t_end_ns": pl.Int64,
            "min_distance_m": pl.Float64,
            "min_clearance_m": pl.Float64,
            "approach_type": APPROACH_TYPE_DTYPE,
        },
    )
    assert_frame_equal(
        windows.select(
            "actor_id",
            "actor_kind",
            "t_start_ns",
            "t_end_ns",
            "min_distance_m",
            "min_clearance_m",
            "approach_type",
        ),
        expected,
    )


def _pair_frame(
    edge_clearance_m: Sequence[float], t_s: Sequence[float] | None = None
) -> pl.LazyFrame:
    """Pair rows with an explicit surface clearance, distance = clearance + the fixture radii."""
    count = len(edge_clearance_m)
    times_s = list(range(count)) if t_s is None else list(t_s)
    surface_margin_m = ROBOT_RADIUS_M + ACTOR_RADIUS_M
    return pl.DataFrame(
        {
            "robot_id": ["r1"] * count,
            "actor_id": ["p1"] * count,
            "actor_name": ["Pedestrian 1"] * count,
            "actor_kind": ["pedestrian"] * count,
            "t_ns": [int(time_s * NS_PER_S) for time_s in times_s],
            "robot_x": [0.0] * count,
            "robot_y": [0.0] * count,
            "robot_yaw": [0.0] * count,
            "robot_vx": [0.0] * count,
            "robot_vy": [0.0] * count,
            "actor_x": [clearance + surface_margin_m for clearance in edge_clearance_m],
            "actor_y": [0.0] * count,
            "actor_vx": [0.0] * count,
            "actor_vy": [0.0] * count,
            "actor_radius_m": [ACTOR_RADIUS_M] * count,
            "distance_m": [
                clearance + surface_margin_m for clearance in edge_clearance_m
            ],
            "edge_clearance_m": list(edge_clearance_m),
        },
        schema=PAIR_SCHEMA,
    ).lazy()


def test_two_robot_fixture_does_not_pair_the_robot_with_itself() -> None:
    robot = _robot_tracks([{"t_ns": 0, "x": 0.0, "y": 0.0}])
    fleet = {
        "r1": _fleet_poses([{"t_ns": 0, "pos_x": 0.0, "radius_m": 0.25}]),
        "env_0_r2": _fleet_poses([{"t_ns": 0, "pos_x": 1.0, "radius_m": 0.25}]),
    }

    windows = _detect_windows(fleet_pairs_lf(robot, fleet, robot_radius_m=0.25))

    assert windows.height == 1
    assert_series_equal(
        windows["actor_id"], pl.Series("actor_id", ["env_0_r2"], dtype=pl.String)
    )


def test_fleet_pairs_carry_each_actor_its_own_radius() -> None:
    robot = _robot_tracks([{"t_ns": 0}])
    fleet = {
        "env_0_r2": _fleet_poses([{"t_ns": 0, "pos_x": 4.0, "radius_m": 0.2}]),
        "env_1_r2": _fleet_poses([{"t_ns": 0, "pos_x": 5.0, "radius_m": 0.6}]),
    }

    pairs = fleet_pairs_lf(robot, fleet, robot_radius_m=0.4).collect().sort("actor_id")

    assert_series_equal(
        pairs["actor_id"],
        pl.Series("actor_id", ["env_0_r2", "env_1_r2"], dtype=pl.String),
    )
    assert_series_equal(
        pairs["actor_radius_m"],
        pl.Series("actor_radius_m", [0.2, 0.6], dtype=pl.Float64),
    )
    assert_series_equal(
        pairs["edge_clearance_m"],
        pl.Series(
            "edge_clearance_m", [4.0 - 0.4 - 0.2, 5.0 - 0.4 - 0.6], dtype=pl.Float64
        ),
    )


def test_one_robot_fixture_encounter_count_at_the_new_threshold() -> None:
    """A straight pass still yields one encounter at 2.0 m edge clearance.

    Membership moved from a 3.0 m centre distance to a 2.0 m surface clearance (2.55 m centre
    for the default 0.25 m robot and 0.3 m pedestrian), so the bout keeps its count but is
    shorter: here onset 2 s and end 4 s where the old centre rule spanned 1 s to 5 s.
    """
    rows = [
        {"t_ns": t_s * NS_PER_S, "actor_x": x}
        for t_s, x in (
            (0, 4.0),
            (1, 3.0),
            (2, 2.5),
            (3, 2.0),
            (4, 2.5),
            (5, 3.0),
            (6, 4.0),
        )
    ]

    windows = _detect_windows(_pairs_lf(rows))

    expected = pl.DataFrame(
        {"t_start_ns": [2_000_000_000], "t_end_ns": [4_000_000_000]},
        schema={"t_start_ns": pl.Int64, "t_end_ns": pl.Int64},
    )
    assert_frame_equal(windows.select("t_start_ns", "t_end_ns"), expected)


# ---------------------------------------------------------------------------
# contact_windows_lf: one row per contact bout, linked to its encounter
# ---------------------------------------------------------------------------

_BOUT_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "t_onset_ns": pl.Int64,
        "t_end_ns": pl.Int64,
        "collider_kind": pl.String,
        "obstacle_id": pl.String,
        "obstacle_x": pl.Float64,
        "obstacle_y": pl.Float64,
    }
)


def _bouts(rows: Sequence[dict[str, object]]) -> pl.DataFrame:
    return _build(rows, {"obstacle_x": None, "obstacle_y": None}, _BOUT_SCHEMA)


@pytest.mark.parametrize(
    ("kind", "expected_kind"),
    [
        ("pedestrian", "pedestrian"),
        ("ped_human", "pedestrian"),
        ("human", "pedestrian"),
        ("PEDESTRIAN", "pedestrian"),
        ("robot", "robot"),
        ("wall", "wall"),
        ("static", "static"),
        ("obstacle", "static"),
    ],
)
def test_bout_kind_is_normalized_to_its_collider_class(
    kind: str, expected_kind: str
) -> None:
    bouts = _bouts(
        [
            {
                "t_onset_ns": 0,
                "t_end_ns": 0,
                "collider_kind": kind,
                "obstacle_id": "box_2",
            }
        ]
    )

    windows = contact_windows_lf(bouts, empty_encounter_windows()).collect()

    assert_series_equal(
        windows["collider_kind"],
        pl.Series("collider_kind", [expected_kind], dtype=pl.String),
    )


def test_entity_ids_drop_the_env_prefix_and_path_suffix() -> None:
    bouts = _bouts(
        [
            {
                "t_onset_ns": 0,
                "t_end_ns": 0,
                "collider_kind": "pedestrian",
                "obstacle_id": "env_0/Pedestrian_1/pose",
            },
            {
                "t_onset_ns": 0,
                "t_end_ns": 0,
                "collider_kind": "robot",
                "obstacle_id": "env_12/Robot_3",
            },
        ]
    )

    windows = contact_windows_lf(bouts, empty_encounter_windows()).collect()

    assert_series_equal(
        windows["obstacle_id"],
        pl.Series("obstacle_id", ["Pedestrian_1", "Robot_3"], dtype=pl.String),
    )


def test_bouts_link_to_the_encounter_enclosing_their_onset() -> None:
    encounters = _encounter_table(
        [
            {
                "actor_id": "Pedestrian_1",
                "encounter_idx": 7,
                "t_start_ns": 0,
                "t_end_ns": 1_000_000_000,
            }
        ]
    )
    bouts = _bouts(
        [
            {
                "t_onset_ns": 500_000_000,
                "t_end_ns": 500_000_000,
                "collider_kind": "pedestrian",
                "obstacle_id": "Pedestrian_1",
            },
            {
                "t_onset_ns": 500_000_000,
                "t_end_ns": 500_000_000,
                "collider_kind": "wall",
                "obstacle_id": "env_0/wall_7",
            },
            {
                "t_onset_ns": 2_000_000_000,
                "t_end_ns": 2_000_000_000,
                "collider_kind": "pedestrian",
                "obstacle_id": "Pedestrian_1",
            },
        ]
    )

    windows = contact_windows_lf(bouts, encounters).collect()

    assert_series_equal(
        windows["encounter_idx"],
        pl.Series("encounter_idx", [7, None, None], dtype=pl.Int64),
    )


def test_numeric_gazebo_ids_and_actor_names_resolve_to_the_encounter_actor() -> None:
    encounters = _encounter_table(
        [
            {
                "actor_id": "Pedestrian_7",
                "actor_name": "alice",
                "encounter_idx": 4,
                "t_start_ns": 0,
                "t_end_ns": 1_000_000_000,
            }
        ]
    )
    bouts = _bouts(
        [
            {
                "t_onset_ns": 200_000_000,
                "t_end_ns": 200_000_000,
                "collider_kind": "pedestrian",
                "obstacle_id": "env_0/7",
            },
            {
                "t_onset_ns": 300_000_000,
                "t_end_ns": 300_000_000,
                "collider_kind": "pedestrian",
                "obstacle_id": "env_0/alice",
            },
        ]
    )

    windows = contact_windows_lf(bouts, encounters).collect()

    assert_series_equal(
        windows["obstacle_id"],
        pl.Series("obstacle_id", ["Pedestrian_7", "Pedestrian_7"], dtype=pl.String),
    )
    assert_series_equal(
        windows["encounter_idx"], pl.Series("encounter_idx", [4, 4], dtype=pl.Int64)
    )


def test_no_bouts_give_the_canonical_empty_frame() -> None:
    windows = contact_windows_lf(
        pl.DataFrame(schema=_BOUT_SCHEMA), empty_encounter_windows()
    ).collect()

    assert_frame_equal(windows, empty_contact_windows())


def test_canonical_encounter_edge_distance_constant_is_shared() -> None:
    assert ENCOUNTER_EDGE_DISTANCE_M == 2.0
    assert (
        DEFAULT_DETECTOR_PARAMS.encounter_edge_distance_m == ENCOUNTER_EDGE_DISTANCE_M
    )
