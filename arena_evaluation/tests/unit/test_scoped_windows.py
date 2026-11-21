"""Unit tests for the vectorized scope-window detectors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from arena_evaluation.processing.windows import (
    DEFAULT_DETECTOR_PARAMS,
    DETECTOR_VERSION,
    ENCOUNTER_WINDOW_SCHEMA,
    PAIR_SCHEMA,
    DetectorParams,
    detect_encounter_windows_lf,
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

_FLEET_POSE_SCHEMA: Final[pl.Schema] = pl.Schema({"t_ns": pl.Int64, "pos_x": pl.Float64, "pos_y": pl.Float64, "radius_m": pl.Float64})

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


def _build(rows: Sequence[dict[str, object]], defaults: dict[str, object], schema: pl.Schema) -> pl.DataFrame:
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
            distance_m=((pl.col("actor_x") - pl.col("robot_x")).pow(2) + (pl.col("actor_y") - pl.col("robot_y")).pow(2)).sqrt(),
        )
        .with_columns(edge_clearance_m=pl.col("distance_m") - (pl.lit(ROBOT_RADIUS_M) + pl.col("actor_radius_m")))
        .select(PAIR_SCHEMA.names())
        .lazy()
    )


def _detect_windows(pairs: pl.LazyFrame, params: DetectorParams = DEFAULT_DETECTOR_PARAMS) -> pl.DataFrame:
    return detect_encounter_windows_lf(pairs, params).collect()


# pedestrian_pairs_lf


def test_pedestrian_pairs_refuse_a_pedestrian_without_a_recorded_radius() -> None:
    robot = _robot_tracks([{"t_ns": 0}])
    pedestrians = _pedestrian_tracks([{"t_ns": 0, "x": 5.0, "radius_m": None}])

    with pytest.raises(MetricInputError, match="radius_m"):
        pedestrian_pairs_lf(robot, pedestrians, robot_radius_m=0.5)


# detect_encounter_windows_lf


# fleet_pairs_lf


def test_fleet_pairs_empty_fleet_returns_canonical_empty_frame() -> None:
    pairs = fleet_pairs_lf(_robot_tracks([{"t_ns": 0}]), {}, robot_radius_m=0.4).collect()

    assert_frame_equal(pairs, pl.DataFrame(schema=PAIR_SCHEMA))


def _pair_frame(edge_clearance_m: Sequence[float], t_s: Sequence[float] | None = None) -> pl.LazyFrame:
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
            "distance_m": [clearance + surface_margin_m for clearance in edge_clearance_m],
            "edge_clearance_m": list(edge_clearance_m),
        },
        schema=PAIR_SCHEMA,
    ).lazy()


# contact_windows_lf: one row per contact bout, linked to its encounter

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
