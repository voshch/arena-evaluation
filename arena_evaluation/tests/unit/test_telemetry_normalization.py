"""Flat relational telemetry ingestion: schemas, native timestamps, ground-truth kinematics and pedestrian identity."""

from __future__ import annotations

import math
import pathlib

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from arena_evaluation.processing.mcap_reader import MCAPReader
from test_mcap_reader_topics import ODOMETRY, PEDESTRIANS, SEC, TF_MESSAGE, _every_topic_bag, _header, _transform, _write_bag

ROBOT = "env_0_jackal"
GROUND_TRUTH_STEPS = 21
GROUND_TRUTH_DT_NS = SEC // 10


def _ground_truth_pose(step: int) -> tuple[float, float, float]:
    seconds = step * GROUND_TRUTH_DT_NS / SEC
    return 0.8 * seconds, 0.3 * seconds**2, 0.5 * seconds


def _odometry(x: float, y: float) -> dict:
    return {"header": _header(frame_id="odom"), "pose": {"pose": {"position": {"x": 1.5 * x, "y": -y}, "orientation": {"w": 1.0}}}, "twist": {"twist": {}}}


def _ground_truth_bag(path: pathlib.Path) -> pathlib.Path:
    records = []
    for step in range(GROUND_TRUTH_STEPS):
        x, y, yaw = _ground_truth_pose(step)
        t_ns = step * GROUND_TRUTH_DT_NS
        rotation = {"z": math.sin(yaw / 2), "w": math.cos(yaw / 2)}
        records.append(("/tf", TF_MESSAGE, {"transforms": [_transform("map", "env_0/jackal/base_link", x, y, rotation)]}, t_ns))
        records.append(("/env_0/jackal/odom", ODOMETRY, _odometry(x, y), t_ns))
    return _write_bag(path, records)


def _pedestrian_bag(path: pathlib.Path) -> pathlib.Path:
    pedestrians = [
        {"id": 3, "name": "Pedestrian_3", "pose": {"position": {"x": 1.0, "y": 2.0, "z": 0.0}, "orientation": {"w": 1.0}}, "twist": {"linear": {"x": 0.1}}, "animation_state": 0, "interaction_id": -1, "interaction_type": 0, "agent_type": "adult", "gait_phase": 0.0, "gait_cadence": 0.0, "gestures": []},
        {"id": 7, "name": "Pedestrian_7", "pose": {"position": {"x": -4.0, "y": 0.5, "z": 0.0}, "orientation": {"w": 1.0}}, "twist": {"linear": {"y": -0.2}}, "animation_state": 0, "interaction_id": -1, "interaction_type": 0, "agent_type": "adult", "gait_phase": 0.0, "gait_cadence": 0.0, "gestures": []},
        {"id": 9, "name": "Pedestrian_9", "pose": {"position": {"x": 6.0, "y": -3.0, "z": 0.0}, "orientation": {"w": 1.0}}, "twist": {}, "animation_state": 0, "interaction_id": -1, "interaction_type": 0, "agent_type": "adult", "gait_phase": 0.0, "gait_cadence": 0.0, "gestures": []},
    ]
    frame = {"header": _header(frame_id="map"), "pedestrians": pedestrians}
    return _write_bag(path, [("/env_0/jackal/odom", ODOMETRY, _odometry(0.0, 0.0), SEC), ("/env_0/peds", PEDESTRIANS, frame, SEC)])


def _odometry_only_bag(path: pathlib.Path) -> pathlib.Path:
    return _write_bag(path, [("/env_0/jackal/odom", ODOMETRY, _odometry(2.0, 0.0), SEC)])


@pytest.fixture(scope="module")
def ground_truth(tmp_path_factory: pytest.TempPathFactory) -> tuple[pathlib.Path, dict]:
    root = tmp_path_factory.mktemp("ground_truth")
    out = root / "telemetry"
    return out, MCAPReader(_ground_truth_bag(root / "run.mcap")).read(out)


@pytest.fixture(scope="module")
def pedestrian_frame(tmp_path_factory: pytest.TempPathFactory) -> tuple[pathlib.Path, dict]:
    root = tmp_path_factory.mktemp("pedestrian_frame")
    out = root / "telemetry"
    return out, MCAPReader(_pedestrian_bag(root / "run.mcap")).read(out)


@pytest.fixture(scope="module")
def every_topic_out(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    root = tmp_path_factory.mktemp("every_topic_flat")
    out = root / "telemetry"
    MCAPReader(_every_topic_bag(root / "run.mcap")).read(out)
    return out


def test_robot_tracks_positions_are_the_tf_ground_truth(ground_truth: tuple[pathlib.Path, dict]):
    _, bundles = ground_truth

    tracks = bundles[ROBOT].robot_tracks.collect()

    expected_xy = [_ground_truth_pose(step)[:2] for step in range(GROUND_TRUTH_STEPS)]
    np.testing.assert_allclose(tracks.select("x", "y").to_numpy(), expected_xy)


def test_robot_tracks_are_sourced_from_tf_ground_truth_only(ground_truth: tuple[pathlib.Path, dict]):
    _, bundles = ground_truth

    tracks = bundles[ROBOT].robot_tracks.collect()

    assert tracks["pose_source"].unique().to_list() == ["tf_gt"]


def test_velocities_are_numerical_derivatives_of_the_ground_truth_pose(ground_truth: tuple[pathlib.Path, dict]):
    _, bundles = ground_truth
    tracks = bundles[ROBOT].robot_tracks.collect()
    t_s = tracks["t_ns"].to_numpy() / SEC
    expected = np.column_stack(
        [
            np.gradient(tracks["x"].to_numpy(), t_s),
            np.gradient(tracks["y"].to_numpy(), t_s),
            np.gradient(np.unwrap(tracks["yaw"].to_numpy()), t_s),
        ]
    )

    np.testing.assert_allclose(tracks.select("vx", "vy", "omega_z").to_numpy(), expected, atol=1e-9)


def test_robot_without_tf_ground_truth_gets_no_robot_tracks(tmp_path: pathlib.Path):
    bundles = MCAPReader(_odometry_only_bag(tmp_path / "odom_only.mcap")).read(tmp_path / "telemetry")

    assert bundles[ROBOT].robot_tracks is None


def test_each_pedestrian_in_a_frame_becomes_its_own_row(pedestrian_frame: tuple[pathlib.Path, dict]):
    _, bundles = pedestrian_frame

    peds = bundles[ROBOT].pedestrian_arena_peds.collect()

    assert sorted(peds["ped_id"].to_list()) == ["Pedestrian_3", "Pedestrian_7", "Pedestrian_9"]


def test_each_pedestrian_row_carries_that_pedestrians_position(pedestrian_frame: tuple[pathlib.Path, dict]):
    _, bundles = pedestrian_frame

    peds = bundles[ROBOT].pedestrian_arena_peds.collect().sort("ped_id")

    np.testing.assert_allclose(peds.select("x", "y").to_numpy(), [[1.0, 2.0], [-4.0, 0.5], [6.0, -3.0]])


def test_no_ingested_table_holds_a_list_column(every_topic_out: pathlib.Path):
    list_columns = [f"{path.name}.{field.name}" for path in sorted(every_topic_out.rglob("*.parquet")) for field in pq.read_schema(path) if pa.types.is_list(field.type)]

    assert list_columns == []


def test_every_ingested_table_leads_with_a_non_null_monotonic_int64_t_ns(every_topic_out: pathlib.Path):
    problems = []
    for path in sorted(every_topic_out.rglob("*.parquet")):
        schema = pq.read_schema(path)
        if schema.names[0] != "t_ns" or schema.field("t_ns").type != pa.int64():
            problems.append(f"{path.name}: t_ns is not the leading int64 column")
            continue
        stamps = pq.read_table(path, columns=["t_ns"])["t_ns"]
        if stamps.null_count:
            problems.append(f"{path.name}: null t_ns")
        elif np.any(np.diff(stamps.to_numpy()) < 0):
            problems.append(f"{path.name}: t_ns decreases")

    assert problems == []


def test_laser_scan_summary_keeps_the_nearest_and_farthest_valid_return(every_topic_out: pathlib.Path):
    summary = pl.read_parquet(every_topic_out / "robot_scan_summary.parquet").filter(pl.col("robot_id") == ROBOT).sort("t_ns")

    assert summary.select("scan_min_m", "scan_max_m", "n_valid_rays").row(0) == (1.5, 2.0, 2)


def test_ray_detail_is_not_written_by_default(tmp_path: pathlib.Path):
    out = tmp_path / "telemetry"
    MCAPReader(_every_topic_bag(tmp_path / "run.mcap")).read(out)

    assert not (out / "robot_scan_rays.parquet").exists()


def test_ray_detail_has_one_row_per_ray_when_requested(tmp_path: pathlib.Path):
    out = tmp_path / "telemetry"
    MCAPReader(_every_topic_bag(tmp_path / "run.mcap"), include_scan_rays=True).read(out)

    rays = pl.read_parquet(out / "robot_scan_rays.parquet").filter(pl.col("robot_id") == ROBOT)

    assert rays.height == 5
