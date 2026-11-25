"""Environment alignment at extraction: every stored map-frame position moves into the environment frame once."""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("polars")

import polars as pl

from arena_evaluation.processing import mcap_reader

ENV = "env_3"
JACKAL = "env_3_jackal"
PEDESTRIAN_FRAMES = {(ENV, "pedestrian_arena_peds"): frozenset({"map", ""})}
AT_50_NS = 50


def _tf(t_ns: int, parent: str, child: str, tx: float, ty: float) -> dict:
    return {
        "t_ns": t_ns,
        "parent_frame": parent,
        "child_frame": child,
        "tx": tx,
        "ty": ty,
        "tz": 0.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }


def _robot(t_ns: int, **columns: object) -> dict:
    return {"t_ns": t_ns, "env_id": ENV, "robot_id": JACKAL, **columns}


def _env(t_ns: int, **columns: object) -> dict:
    return {"t_ns": t_ns, "env_id": ENV, **columns}


def _write(topics: pathlib.Path, name: str, rows: list[dict]) -> None:
    pl.DataFrame(rows).write_parquet(topics / f"{name}.parquet")


def _stored(topics: pathlib.Path, name: str) -> dict:
    return pl.read_parquet(topics / f"{name}.parquet").row(0, named=True)


@pytest.fixture
def topics(tmp_path: pathlib.Path) -> pathlib.Path:
    """One environment whose map frame sits at (10, 20) until t=100 ns and at (30, 40) from then on."""
    topics = tmp_path / "topics"
    topics.mkdir()
    _write(
        topics,
        "tf_static",
        [_tf(0, "map", "env_3/map", 10.0, 20.0), _tf(100, "", "env_3", 30.0, 40.0)],
    )
    _write(
        topics,
        "robot_tracks",
        [_robot(AT_50_NS, x=10.5, y=20.5, yaw=0.0, pose_source="tf_gt")],
    )
    _write(topics, "robot_odom", [_robot(AT_50_NS, odom_x=10.5, odom_y=20.5, odom_yaw=0.0)])
    _write(topics, "robot_task_pose", [_robot(AT_50_NS, pos_x=12.0, pos_y=22.0, yaw=0.0)])
    _write(
        topics,
        "robot_collision_contacts",
        [
            _robot(
                AT_50_NS,
                obstacle_id="wall_3",
                obstacle_x=11.0,
                obstacle_y=21.0,
                obstacle_z=0.0,
            )
        ],
    )
    _write(
        topics,
        "pedestrian_arena_peds",
        [_env(AT_50_NS, ped_id="Pedestrian_11", x=11.0, y=22.0)],
    )
    _write(topics, "environment_map", [_env(0, origin_x=15.0, origin_y=25.0)])
    return topics


def test_stored_robot_tracks_are_in_the_environment_frame(topics: pathlib.Path) -> None:
    mcap_reader.align_to_environment(topics, PEDESTRIAN_FRAMES)

    track = _stored(topics, "robot_tracks")

    assert (track["x"], track["y"]) == pytest.approx((0.5, 0.5))


def test_stored_obstacles_of_collision_contacts_are_in_the_environment_frame(
    topics: pathlib.Path,
) -> None:
    mcap_reader.align_to_environment(topics, PEDESTRIAN_FRAMES)

    contact = _stored(topics, "robot_collision_contacts")

    assert (contact["obstacle_x"], contact["obstacle_y"]) == pytest.approx((1.0, 1.0))
    assert contact["obstacle_z"] == 0.0


def test_stored_odometry_is_in_the_environment_frame(topics: pathlib.Path) -> None:
    mcap_reader.align_to_environment(topics, PEDESTRIAN_FRAMES)

    odom = _stored(topics, "robot_odom")

    assert (odom["odom_x"], odom["odom_y"]) == pytest.approx((0.5, 0.5))


def test_stored_task_poses_stay_where_the_judge_published_them(topics: pathlib.Path) -> None:
    mcap_reader.align_to_environment(topics, PEDESTRIAN_FRAMES)

    task = _stored(topics, "robot_task_pose")

    assert (task["pos_x"], task["pos_y"]) == pytest.approx((12.0, 22.0))


def test_stored_task_poses_meet_the_ground_truth_track_after_alignment(topics: pathlib.Path) -> None:
    _write(topics, "robot_task_pose", [_robot(AT_50_NS, pos_x=0.5, pos_y=0.5, yaw=0.0)])
    mcap_reader.align_to_environment(topics, PEDESTRIAN_FRAMES)

    task = _stored(topics, "robot_task_pose")
    track = _stored(topics, "robot_tracks")

    assert (task["pos_x"], task["pos_y"]) == pytest.approx((track["x"], track["y"]))


def test_stored_pedestrians_and_the_environment_map_are_in_the_environment_frame(
    topics: pathlib.Path,
) -> None:
    mcap_reader.align_to_environment(topics, PEDESTRIAN_FRAMES)

    pedestrian = _stored(topics, "pedestrian_arena_peds")
    environment = _stored(topics, "environment_map")

    assert (pedestrian["x"], pedestrian["y"]) == pytest.approx((1.0, 2.0))
    assert (environment["origin_x"], environment["origin_y"]) == pytest.approx((5.0, 5.0))


def test_stored_engine_agent_states_keep_their_recorded_frame(
    topics: pathlib.Path,
) -> None:
    _write(
        topics,
        "pedestrian_agent_states",
        [_env(AT_50_NS, ped_id="Pedestrian_11", x=1.0, y=2.0)],
    )

    mcap_reader.align_to_environment(topics, PEDESTRIAN_FRAMES)

    agent = _stored(topics, "pedestrian_agent_states")

    assert (agent["x"], agent["y"]) == pytest.approx((1.0, 2.0))


def test_loading_bundles_does_not_move_the_stored_positions(
    topics: pathlib.Path,
) -> None:
    track = mcap_reader.load_bundles(topics)[JACKAL].robot_tracks.collect().row(0, named=True)

    assert (track["x"], track["y"]) == pytest.approx((10.5, 20.5))


def test_loading_bundles_after_alignment_keeps_the_aligned_positions(
    topics: pathlib.Path,
) -> None:
    mcap_reader.align_to_environment(topics, PEDESTRIAN_FRAMES)

    track = mcap_reader.load_bundles(topics)[JACKAL].robot_tracks.collect().row(0, named=True)

    assert (track["x"], track["y"]) == pytest.approx((0.5, 0.5))
