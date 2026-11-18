"""The recorded goal pose and occupancy grids: extraction, dense reconstruction and the offline route grid."""

from __future__ import annotations

import math
import pathlib

import numpy as np
import polars as pl
import pytest

pytest.importorskip("mcap_ros2.writer")

from mcap.records import Schema

from arena_evaluation.ingestion.topics import get_topics
from arena_evaluation.processing.maps import load_snapshot
from arena_evaluation.processing.mcap_reader import (
    IGNORED_TOPIC_MARKERS,
    MCAPReader,
    _Ingest,
)
from arena_evaluation.processing.path.theta_star import plan_route
from arena_evaluation.processing.telemetry.writer import TableSink
from test_mcap_reader_topics import (
    HEADER,
    POINT,
    POSE,
    QUAT,
    TIME,
    _header,
    _msgdef,
    _write_bag,
)

SEC = 1_000_000_000
GOAL_POSE = (
    "geometry_msgs/msg/PoseStamped",
    _msgdef(
        "std_msgs/Header header\ngeometry_msgs/Pose pose",
        HEADER,
        TIME,
        POSE,
        POINT,
        QUAT,
    ),
)
OCCUPANCY_GRID = (
    "nav_msgs/msg/OccupancyGrid",
    _msgdef(
        "std_msgs/Header header\nMapMetaData info\nint8[] data",
        HEADER,
        TIME,
        "MSG: nav_msgs/MapMetaData\nbuiltin_interfaces/Time map_load_time\nfloat32 resolution\nuint32 width\nuint32 height\ngeometry_msgs/Pose origin",
        POSE,
        POINT,
        QUAT,
    ),
)
# A 4x3 grid at 0.5 m: one occupied cell and one unknown cell block row 0.
GRID = [0, 100, 0, -1, 0, 0, 100, 0, 0, 0, 0, 0]


def _grid_message() -> dict:
    return {
        "header": _header(2, 0, "env_0/map"),
        "info": {
            "resolution": 0.5,
            "width": 4,
            "height": 3,
            "origin": {
                "position": {"x": 1.0, "y": 2.0, "z": 0.0},
                "orientation": {"w": 1.0},
            },
        },
        "data": GRID,
    }


def _goal_message() -> dict:
    return {
        "header": _header(3, 0, "env_0/map"),
        "pose": {
            "position": {"x": 3.0, "y": 4.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 1.0, "w": 0.0},
        },
    }


@pytest.fixture()
def topics(tmp_path: pathlib.Path) -> pathlib.Path:
    bag = _write_bag(
        tmp_path / "run.mcap",
        [
            ("/env_0/map", OCCUPANCY_GRID, _grid_message(), 2 * SEC),
            ("/env_0/jackal/goal_pose", GOAL_POSE, _goal_message(), 3 * SEC),
        ],
    )
    out = tmp_path / "topics"
    MCAPReader(bag).read(out)
    return out


def test_the_grid_and_the_goal_pose_are_extracted(topics: pathlib.Path) -> None:
    metadata = pl.read_parquet(topics / "environment_map.parquet")
    cells = pl.read_parquet(topics / "environment_map_cells.parquet")
    goal = pl.read_parquet(topics / "robot_goal_pose.parquet")

    assert metadata.height == 1
    assert metadata["kind"].to_list() == ["map"]
    assert metadata["resolution_m"].to_list() == [0.5]
    assert sorted(cells["occupancy"].to_list()) == [-1, 100, 100]
    row = goal.row(0, named=True)
    assert (row["pos_x"], row["pos_y"], row["pos_z"]) == (3.0, 4.0, 0.0)
    assert row["yaw"] == pytest.approx(math.pi)
    assert row["frame_id"] == "env_0/map"


def test_the_sparse_cells_reconstruct_the_dense_grid_cell_for_cell(
    topics: pathlib.Path,
) -> None:
    snapshot = load_snapshot(topics)

    assert snapshot is not None
    expected = np.array([[0, 100, 0, -1], [0, 0, 100, 0], [0, 0, 0, 0]], dtype=np.int8)
    np.testing.assert_array_equal(snapshot.cells, expected)
    assert snapshot.width_cells == 4
    assert snapshot.height_cells == 3
    assert (snapshot.origin_x, snapshot.origin_y) == (1.0, 2.0)


def test_theta_star_plans_on_the_recorded_grid_without_an_installed_world(
    topics: pathlib.Path,
) -> None:
    snapshot = load_snapshot(topics)
    assert snapshot is not None

    result = plan_route(snapshot.static_grid(), (1.25, 2.25), (2.25, 2.25))

    assert result.points is not None, result.null_reason
    assert result.length_m is not None and result.length_m > 1.0


def test_a_door_mask_snapshot_is_kept_apart_from_the_map(topics: pathlib.Path) -> None:
    assert load_snapshot(topics, kind="door_mask") is None


def _schema_for(message_type: type) -> Schema:
    """The recorded ROS schema name of a generated message class, e.g. nav_msgs/msg/OccupancyGrid."""
    package = message_type.__module__.split(".")[0]
    return Schema(
        id=1,
        data=b"",
        encoding="ros2msg",
        name=f"{package}/msg/{message_type.__name__}",
    )


@pytest.mark.parametrize("name", sorted(get_topics("env_0/jackal", "env_0")))
def test_every_declared_recording_topic_has_a_route_or_an_ignore(
    name: str, tmp_path: pathlib.Path
) -> None:
    definition = get_topics("env_0/jackal", "env_0")[name]
    ingest = _Ingest(TableSink(tmp_path))
    accepted = ingest.accepts(
        definition.name_template, _schema_for(definition.msg_type)
    )

    ignored = any(
        marker in definition.name_template for marker in IGNORED_TOPIC_MARKERS
    )
    assert accepted or ignored, definition.name_template


@pytest.mark.parametrize(
    "topic",
    ["/env_0/humans/tf", "/env_0/heard_sound_events", "/env_0/audio/raw_array"],
)
def test_the_documented_ignored_topics_are_not_routed(
    topic: str, tmp_path: pathlib.Path
) -> None:
    ingest = _Ingest(TableSink(tmp_path))

    assert ingest.accepts(topic, _schema_for(type("Unused", (), {}))) is False
    assert any(marker in topic for marker in IGNORED_TOPIC_MARKERS)
