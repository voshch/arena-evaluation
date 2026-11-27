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
from test_mcap_reader_topics import (
    ODOMETRY,
    PEDESTRIANS,
    SEC,
    TF_MESSAGE,
    _every_topic_bag,
    _header,
    _transform,
    _write_bag,
)

ROBOT = "env_0_jackal"
GROUND_TRUTH_STEPS = 21
GROUND_TRUTH_DT_NS = SEC // 10


def _ground_truth_pose(step: int) -> tuple[float, float, float]:
    seconds = step * GROUND_TRUTH_DT_NS / SEC
    return 0.8 * seconds, 0.3 * seconds**2, 0.5 * seconds


def _odometry(x: float, y: float) -> dict:
    return {
        "header": _header(frame_id="odom"),
        "pose": {"pose": {"position": {"x": 1.5 * x, "y": -y}, "orientation": {"w": 1.0}}},
        "twist": {"twist": {}},
    }


def _ground_truth_bag(path: pathlib.Path) -> pathlib.Path:
    return _write_bag(path, _ground_truth_records())


def _ground_truth_records() -> list[tuple[str, tuple[str, str], dict, int]]:
    records = []
    for step in range(GROUND_TRUTH_STEPS):
        x, y, yaw = _ground_truth_pose(step)
        t_ns = step * GROUND_TRUTH_DT_NS
        rotation = {"z": math.sin(yaw / 2), "w": math.cos(yaw / 2)}
        records.append(
            (
                "/tf",
                TF_MESSAGE,
                {"transforms": [_transform("map", "env_0/jackal/base_link", x, y, rotation)]},
                t_ns,
            )
        )
        records.append(("/env_0/jackal/odom", ODOMETRY, _odometry(x, y), t_ns))
    return records


def _pedestrian_bag(path: pathlib.Path) -> pathlib.Path:
    pedestrians = [
        {
            "id": 3,
            "name": "Pedestrian_3",
            "pose": {
                "position": {"x": 1.0, "y": 2.0, "z": 0.0},
                "orientation": {"w": 1.0},
            },
            "twist": {"linear": {"x": 0.1}},
            "animation_state": 0,
            "interaction_id": -1,
            "interaction_type": 0,
            "agent_type": "adult",
            "gait_phase": 0.0,
            "gait_cadence": 0.0,
            "gestures": [],
        },
        {
            "id": 7,
            "name": "Pedestrian_7",
            "pose": {
                "position": {"x": -4.0, "y": 0.5, "z": 0.0},
                "orientation": {"w": 1.0},
            },
            "twist": {"linear": {"y": -0.2}},
            "animation_state": 0,
            "interaction_id": -1,
            "interaction_type": 0,
            "agent_type": "adult",
            "gait_phase": 0.0,
            "gait_cadence": 0.0,
            "gestures": [],
        },
        {
            "id": 9,
            "name": "Pedestrian_9",
            "pose": {
                "position": {"x": 6.0, "y": -3.0, "z": 0.0},
                "orientation": {"w": 1.0},
            },
            "twist": {},
            "animation_state": 0,
            "interaction_id": -1,
            "interaction_type": 0,
            "agent_type": "adult",
            "gait_phase": 0.0,
            "gait_cadence": 0.0,
            "gestures": [],
        },
    ]
    frame = {"header": _header(frame_id="map"), "pedestrians": pedestrians}
    return _write_bag(
        path,
        [
            ("/env_0/jackal/odom", ODOMETRY, _odometry(0.0, 0.0), SEC),
            ("/env_0/peds", PEDESTRIANS, frame, SEC),
        ],
    )


def _pedestrian_tf_bag(path: pathlib.Path) -> pathlib.Path:
    pedestrian = {
        "id": 1,
        "name": "env_0/Pedestrian_0",
        "pose": {"position": {"x": 3.0, "y": 1.0, "z": 0.0}, "orientation": {"w": 1.0}},
        "twist": {},
        "animation_state": 0,
        "interaction_id": -1,
        "interaction_type": 0,
        "agent_type": "adult",
        "gait_phase": 0.0,
        "gait_cadence": 0.0,
        "gestures": [],
    }
    records = [
        (
            "/tf",
            TF_MESSAGE,
            {"transforms": [_transform("map", "env_0/Pedestrian_0/base_link", 3.0 + 0.1 * step, 1.0)]},
            step * GROUND_TRUTH_DT_NS,
        )
        for step in range(GROUND_TRUTH_STEPS)
    ]
    records.append(("/env_0/peds", PEDESTRIANS, {"header": _header(frame_id="map"), "pedestrians": [pedestrian]}, (GROUND_TRUTH_STEPS - 1) * GROUND_TRUTH_DT_NS))
    return _write_bag(path, [*records, *_ground_truth_records()])


def _odometry_only_bag(path: pathlib.Path) -> pathlib.Path:
    return _write_bag(path, [("/env_0/jackal/odom", ODOMETRY, _odometry(2.0, 0.0), SEC)])


@pytest.fixture(scope="module")
def ground_truth(tmp_path_factory: pytest.TempPathFactory) -> tuple[pathlib.Path, dict]:
    root = tmp_path_factory.mktemp("ground_truth")
    out = root / "telemetry"
    return out, MCAPReader(_ground_truth_bag(root / "run.mcap")).read(out)


@pytest.fixture(scope="module")
def pedestrian_frame(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[pathlib.Path, dict]:
    root = tmp_path_factory.mktemp("pedestrian_frame")
    out = root / "telemetry"
    return out, MCAPReader(_pedestrian_bag(root / "run.mcap")).read(out)


@pytest.fixture(scope="module")
def every_topic_out(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    root = tmp_path_factory.mktemp("every_topic_flat")
    out = root / "telemetry"
    MCAPReader(_every_topic_bag(root / "run.mcap")).read(out)
    return out


def test_robot_tracks_are_sourced_from_tf_ground_truth_only(
    ground_truth: tuple[pathlib.Path, dict],
):
    _, bundles = ground_truth

    tracks = bundles[ROBOT].robot_tracks.collect()

    assert tracks["pose_source"].unique().to_list() == ["tf_gt"]


def test_velocities_are_numerical_derivatives_of_the_ground_truth_pose(
    ground_truth: tuple[pathlib.Path, dict],
):
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


def test_each_pedestrian_row_carries_that_pedestrians_position(
    pedestrian_frame: tuple[pathlib.Path, dict],
):
    _, bundles = pedestrian_frame

    peds = bundles[ROBOT].pedestrian_arena_peds.collect().sort("ped_id")

    np.testing.assert_allclose(peds.select("x", "y").to_numpy(), [[1.0, 2.0], [-4.0, 0.5], [6.0, -3.0]])


def test_a_pedestrian_tf_base_frame_is_not_a_robot(tmp_path: pathlib.Path) -> None:
    bundles = MCAPReader(_pedestrian_tf_bag(tmp_path / "run.mcap")).read(tmp_path / "telemetry")

    assert sorted(bundles) == [ROBOT]
    assert pl.read_parquet(tmp_path / "telemetry" / "robot_tracks.parquet")["robot_id"].unique().to_list() == [ROBOT]


def test_every_ingested_table_leads_with_a_non_null_monotonic_int64_t_ns(
    every_topic_out: pathlib.Path,
):
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


def test_laser_scan_summary_keeps_the_nearest_and_farthest_valid_return(
    every_topic_out: pathlib.Path,
):
    summary = pl.read_parquet(every_topic_out / "robot_scan_summary.parquet").filter(pl.col("robot_id") == ROBOT).sort("t_ns")

    assert summary.select("scan_min_m", "scan_max_m", "n_valid_rays").row(0) == (
        1.5,
        2.0,
        2,
    )


def test_scan_topics_extract_without_a_scan_rays_table(tmp_path: pathlib.Path) -> None:
    from arena_evaluation.storage.registry import registered_tables

    out = tmp_path / "telemetry"
    MCAPReader(_every_topic_bag(tmp_path / "run.mcap")).read(out)

    assert "robot_scan_rays" not in {table.table_name for table in registered_tables()}
    assert (out / "robot_scan_summary.parquet").exists()
    assert not (out / "robot_scan_rays.parquet").exists()


def _out_of_order_tf_bag(path: pathlib.Path) -> pathlib.Path:
    return _write_bag(
        path,
        [
            ("/env_0/jackal/odom", ODOMETRY, _odometry(0.0, 0.0), SEC),
            (
                "/tf",
                TF_MESSAGE,
                {"transforms": [_transform("map", "env_0/jackal/base_link", 2.0, 0.0)]},
                2 * SEC,
            ),
            (
                "/tf",
                TF_MESSAGE,
                {"transforms": [_transform("map", "env_0/jackal/base_link", 1.0, 0.0)]},
                SEC,
            ),
        ],
    )


def test_param_value_decodes_byte_array_and_rejects_unknown_type() -> None:
    from rcl_interfaces.msg import ParameterValue

    from arena_evaluation.processing.telemetry.messages import (
        parameter_value as _param_value,
    )

    assert _param_value(ParameterValue(type=5, byte_array_value=[b"\x01", b"\x02"])) == [1, 2]
    with pytest.raises(ValueError, match="unsupported ROS parameter type 99"):
        _param_value(ParameterValue(type=99))


def test_fleet_recorder_writes_descriptor_and_params_once_per_robot(
    tmp_path: pathlib.Path,
) -> None:
    from arena_runtime_msgs.msg import RobotDescriptor, RobotFleet, RobotState
    from diagnostic_msgs.msg import KeyValue

    from arena_evaluation.processing.telemetry.context import FleetRecorder
    from arena_evaluation.processing.telemetry.writer import TableSink

    fleet = RobotFleet(
        robots=[
            RobotState(
                descriptor=RobotDescriptor(
                    name="jackal",
                    model="jackal",
                    ns="/env_0/jackal",
                    frame="env_0/jackal/base_link",
                    base_frame="base_link",
                ),
                params=[KeyValue(key="robot_radius", value="0.25")],
            )
        ]
    )
    out = tmp_path / "topics"
    sink = TableSink(out)
    recorder = FleetRecorder(sink)
    recorder.record(100, "env_0", fleet.robots)
    recorder.record(200, "env_0", fleet.robots)
    sink.close()

    descriptors = pl.read_parquet(out / "robot_descriptor.parquet")
    params = pl.read_parquet(out / "robot_params.parquet")
    assert descriptors.height == 1
    assert descriptors["robot_id"].to_list() == ["env_0_jackal"]
    assert params["param_key"].to_list() == ["robot_radius"]
    assert params["param_value"].to_list() == ["0.25"]
