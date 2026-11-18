"""The recorded goal pose and occupancy grids: extraction, dense reconstruction and the offline route grid."""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("mcap_ros2.writer")

from mcap.records import Schema

from arena_evaluation.processing.maps import load_snapshot
from arena_evaluation.processing.mcap_reader import (
    MCAPReader,
)
from arena_evaluation.processing.path.theta_star import plan_route
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


