from __future__ import annotations

import json
import pathlib

import pytest

pytest.importorskip("polars")

import polars as pl

from arena_evaluation.processing.mcap_reader import align_to_environment, load_bundles

ENV = "env_3"
JACKAL = "env_3_jackal"
PEDESTRIAN_FRAMES = {(ENV, "pedestrian_arena_peds"): frozenset({"map", ""})}


def _tf(t_ns: int, parent: str, child: str, tx: float, ty: float) -> dict:
    return {"t_ns": t_ns, "parent_frame": parent, "child_frame": child, "tx": tx, "ty": ty, "tz": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}


def _robot(t_ns: int, env: str, robot: str, **columns: object) -> dict:
    return {"t_ns": t_ns, "env_id": env, "robot_id": robot, **columns}


def _env(t_ns: int, env: str, **columns: object) -> dict:
    return {"t_ns": t_ns, "env_id": env, **columns}


def _write(topics: pathlib.Path, name: str, rows: list[dict]) -> None:
    pl.DataFrame(rows).write_parquet(topics / f"{name}.parquet")


def _load(topics: pathlib.Path, frames: dict | None = None) -> dict:
    align_to_environment(topics, PEDESTRIAN_FRAMES if frames is None else frames)
    return load_bundles(topics)


@pytest.fixture
def shifted_env(tmp_path: pathlib.Path) -> pathlib.Path:
    topics = tmp_path / "topics"
    topics.mkdir()
    _write(
        topics,
        "tf_static",
        [
            _tf(0, "map", "env_3/map", 10.0, 20.0),
            _tf(0, "map", "env_3/map", 10.0, 20.0),
            _tf(0, "map", "env_3/jackal/base_link", 99.0, 99.0),
            _tf(0, "world", "camera", 5.0, 5.0),
            _tf(100, "", "env_3", 30.0, 40.0),
        ],
    )
    _write(topics, "robot_odom", [_robot(0, ENV, JACKAL, odom_x=0.0, odom_y=0.0, odom_yaw=0.0)])
    _write(topics, "robot_initialpose", [_robot(50, ENV, JACKAL, pos_x=12.0, pos_y=22.0, yaw=0.0), _robot(150, ENV, JACKAL, pos_x=33.0, pos_y=44.0, yaw=0.0)])
    _write(topics, "robot_tracks", [_robot(50, ENV, JACKAL, x=10.5, y=20.5, yaw=0.0, pose_source="tf_gt")])
    _write(
        topics,
        "robot_plans",
        [
            _robot(150, ENV, JACKAL, plan_id=0, waypoint_idx=0, x=30.0, y=40.0, yaw=0.0),
            _robot(150, ENV, JACKAL, plan_id=0, waypoint_idx=1, x=31.0, y=42.0, yaw=0.0),
        ],
    )
    _write(
        topics,
        "pedestrian_arena_peds",
        [_env(50, ENV, ped_id="Pedestrian_11", x=11.0, y=22.0), _env(50, ENV, ped_id="Pedestrian_13", x=13.0, y=24.0), _env(150, ENV, ped_id="Pedestrian_32", x=32.0, y=43.0)],
    )
    _write(
        topics,
        "pedestrian_gestures",
        [
            _env(50, ENV, ped_id="Pedestrian_1", slot="arm", clip="", hand="r", at_x=11.0, at_y=22.0, at_z=1.0, render_pose_override=False),
            _env(150, ENV, ped_id="Pedestrian_1", slot="arm", clip="", hand="r", at_x=32.0, at_y=43.0, at_z=1.0, render_pose_override=False),
        ],
    )
    phases = {"jackal": {"phases": [{"goto": "kitchen"}], "map_poses": [[12.0, 23.0, 0.5], None]}}
    _write(
        topics,
        "episode_lifecycle",
        [_env(50, ENV, episode_id=1, outcome_state=2, phases_yaml=json.dumps(phases)), _env(150, ENV, episode_id=2, outcome_state=2, phases_yaml="")],
    )
    return topics


def test_episode_phases_move_by_the_anchor_in_effect(shifted_env: pathlib.Path):
    record = _load(shifted_env)[JACKAL].episode_lifecycle.collect().sort("t_ns", maintain_order=True)

    assert json.loads(record["phases_yaml"][0])["jackal"]["map_poses"] == [[2.0, 3.0, 0.5], None]
    assert record["phases_yaml"][1] == ""


def test_unparseable_phases_leave_the_episode_unshifted(shifted_env: pathlib.Path):
    _write(shifted_env, "episode_lifecycle", [_env(50, ENV, episode_id=1, phases_yaml="not json")])

    record = _load(shifted_env)[JACKAL].episode_lifecycle.collect()

    assert record["phases_yaml"].to_list() == ["not json"]


def test_pedestrians_in_another_frame_refuse_the_map_offset(shifted_env: pathlib.Path):
    with pytest.raises(ValueError, match="only valid for frame map"):
        _load(shifted_env, {(ENV, "pedestrian_arena_peds"): frozenset({"env_3/odom"})})


def test_robot_of_an_env_without_an_anchor_is_refused(shifted_env: pathlib.Path):
    _write(shifted_env, "robot_odom", [_robot(0, ENV, JACKAL, odom_x=0.0, odom_y=0.0, odom_yaw=0.0), _robot(0, "env_5", "env_5_burger", odom_x=0.0, odom_y=0.0, odom_yaw=0.0)])

    with pytest.raises(ValueError, match="env_5: no map -> env_5/map anchor"):
        _load(shifted_env)


def test_dynamic_tf_anchors_apply_when_tf_static_has_none(tmp_path: pathlib.Path):
    topics = tmp_path / "topics"
    topics.mkdir()
    _write(topics, "tf", [_tf(0, "map", "env_4/map", 2.0, 3.0), _tf(0, "map", "env_4", 7.0, 7.0)])
    _write(topics, "robot_tracks", [_robot(0, "env_4", "env_4_jackal", x=5.0, y=5.0, yaw=0.0, pose_source="tf_gt")])

    bundle = _load(topics, {})["env_4_jackal"]

    assert bundle.robot_tracks.collect().select("x", "y").rows() == [(3.0, 2.0)]


