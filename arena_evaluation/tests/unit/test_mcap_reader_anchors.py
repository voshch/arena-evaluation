from __future__ import annotations

import json
import pathlib

import pytest

pl = pytest.importorskip("polars")

from arena_evaluation.processing.mcap_reader import MCAPReader

TF_COLUMNS = ("trans_z", "rot_x", "rot_y", "rot_z")


def _tf(rows: list[tuple[int, str, str, float, float]]) -> pl.DataFrame:
    frame = {
        "time_ns": [r[0] for r in rows],
        "frame_id": [r[1] for r in rows],
        "child_frame_id": [r[2] for r in rows],
        "trans_x": [r[3] for r in rows],
        "trans_y": [r[4] for r in rows],
    }
    frame.update({c: [0.0] * len(rows) for c in TF_COLUMNS})
    frame["rot_w"] = [1.0] * len(rows)
    return pl.DataFrame(frame)


def _odom(robot_dir: pathlib.Path) -> None:
    robot_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"time_ns": [0], "pos_x": [0.0], "pos_y": [0.0], "yaw": [0.0]}).write_parquet(robot_dir / "odom.parquet")


def _load(topics: pathlib.Path) -> dict:
    return MCAPReader(topics / "run.mcap").load_bundles(topics, topics)


@pytest.fixture
def shifted_env(tmp_path: pathlib.Path) -> pathlib.Path:
    topics = tmp_path / "topics"
    env_dir = topics / "env_3"
    robot_dir = topics / "env_3_jackal"
    env_dir.mkdir(parents=True)
    _odom(robot_dir)
    _tf(
        [
            (0, "map", "env_3/map", 10.0, 20.0),
            (0, "map", "env_3/map", 10.0, 20.0),
            (0, "map", "env_3/jackal/base_link", 99.0, 99.0),
            (0, "world", "camera", 5.0, 5.0),
            (100, "", "env_3", 30.0, 40.0),
        ]
    ).write_parquet(topics / "tf_static.parquet")
    phases = {"jackal": {"phases": [{"goto": "kitchen"}], "map_poses": [[12.0, 23.0, 0.5], None]}}
    pl.DataFrame(
        {
            "time_ns": [50, 150],
            "episode_id": [1, 2],
            "start_pos": [json.dumps([11.0, 21.0, 0.3]), json.dumps([31.0, 41.0])],
            "goal_pos": [json.dumps([15.0, 25.0]), json.dumps([1.0])],
            "phases": [json.dumps(phases), ""],
        }
    ).write_parquet(env_dir / "episode_record.parquet")
    pl.DataFrame({"time_ns": [50, 150], "peds_frame_id": ["/map", ""], "peds_positions": [[11.0, 22.0, 0.0, 13.0, 24.0, 1.0], [32.0, 43.0, 0.0]]}).write_parquet(env_dir / "peds.parquet")
    pl.DataFrame({"time_ns": [50, 150], "pos_x": [12.0, 33.0], "pos_y": [22.0, 44.0], "yaw": [0.0, 0.0]}).write_parquet(robot_dir / "initialpose.parquet")
    pl.DataFrame({"time_ns": [50], "pos_x_gt": [10.5], "pos_y_gt": [20.5], "yaw_gt": [0.0]}).write_parquet(robot_dir / "tf_gt.parquet")
    pl.DataFrame({"time_ns": [150], "poses_x": [[30.0, 31.0]], "poses_y": [[40.0, 42.0]]}).write_parquet(robot_dir / "plan.parquet")
    pl.DataFrame({"time_ns": [0], "note": ["kept out of the bundle"]}).write_parquet(robot_dir / "notes.parquet")
    return topics


def test_episode_record_start_goal_and_map_poses_move_by_the_anchor_in_effect(shifted_env: pathlib.Path):
    record = _load(shifted_env)["env_3_jackal"].episode_record.collect()

    assert [json.loads(s) for s in record["start_pos"]] == [[1.0, 1.0, 0.3], [1.0, 1.0]]
    assert [json.loads(g) for g in record["goal_pos"]] == [[5.0, 5.0], [1.0]]
    assert json.loads(record["phases"][0])["jackal"]["map_poses"] == [[2.0, 3.0, 0.5], None]
    assert record["phases"][1] == ""


def test_pose_topics_and_pedestrians_move_by_the_anchor_in_effect(shifted_env: pathlib.Path):
    bundle = _load(shifted_env)["env_3_jackal"]

    assert bundle.initialpose.collect().select("pos_x", "pos_y").rows() == [(2.0, 2.0), (3.0, 4.0)]
    assert bundle.tf_gt.collect().select("pos_x_gt", "pos_y_gt").rows() == [(0.5, 0.5)]
    plan = bundle.plan.collect()
    assert plan["poses_x"].to_list() == [[0.0, 1.0]]
    assert plan["poses_y"].to_list() == [[0.0, 2.0]]
    assert bundle.peds.collect()["peds_positions"].to_list() == [[1.0, 2.0, 0.0, 3.0, 4.0, 1.0], [2.0, 3.0, 0.0]]


def test_parquet_files_outside_the_bundle_fields_are_ignored(shifted_env: pathlib.Path):
    bundle = _load(shifted_env)["env_3_jackal"]

    assert bundle.available() >= {"odom", "episode_record", "peds", "initialpose", "tf_gt", "plan", "tf_static"}
    assert "notes" not in vars(bundle)


def test_unparseable_start_pose_leaves_the_episode_record_unshifted(shifted_env: pathlib.Path):
    env_dir = shifted_env / "env_3"
    pl.DataFrame({"time_ns": [50], "start_pos": ["not json"], "goal_pos": ["[]"]}).write_parquet(env_dir / "episode_record.parquet")

    bundle = _load(shifted_env)["env_3_jackal"]

    assert bundle.episode_record.collect()["start_pos"].to_list() == ["not json"]
    assert bundle.tf_gt.collect().select("pos_x_gt", "pos_y_gt").rows() == [(0.5, 0.5)]


def test_record_without_pose_columns_gains_empty_ones(shifted_env: pathlib.Path):
    pl.DataFrame({"time_ns": [50], "episode_id": [1]}).write_parquet(shifted_env / "env_3" / "episode_record.parquet")

    record = _load(shifted_env)["env_3_jackal"].episode_record.collect()

    assert record.select("start_pos", "goal_pos").rows() == [("[]", "[]")]
    assert "phases" not in record.columns


def test_pedestrians_in_another_frame_refuse_the_map_offset(shifted_env: pathlib.Path):
    pl.DataFrame({"time_ns": [50], "peds_frame_id": ["env_3/odom"], "peds_positions": [[1.0, 2.0, 0.0]]}).write_parquet(shifted_env / "env_3" / "peds.parquet")

    with pytest.raises(ValueError, match="only valid for frame map"):
        _load(shifted_env)


def test_robot_of_an_env_without_an_anchor_is_refused(shifted_env: pathlib.Path):
    _odom(shifted_env / "env_5_burger")

    with pytest.raises(ValueError, match="env_5: no map -> env_5/map anchor"):
        _load(shifted_env)


def test_dynamic_tf_anchors_apply_when_tf_static_has_none(tmp_path: pathlib.Path):
    topics = tmp_path / "topics"
    robot_dir = topics / "env_4_jackal"
    _odom(robot_dir)
    _tf([(0, "map", "env_4/map", 2.0, 3.0), (0, "map", "env_4", 7.0, 7.0), (0, "odom", "env_4/jackal/base_link", 9.0, 9.0)]).write_parquet(topics / "tf.parquet")
    pl.DataFrame({"time_ns": [0], "pos_x_gt": [5.0], "pos_y_gt": [5.0], "yaw_gt": [0.0]}).write_parquet(robot_dir / "tf_gt.parquet")

    gt = _load(topics)["env_4_jackal"].tf_gt.collect()

    assert gt.select("pos_x_gt", "pos_y_gt").rows() == [(3.0, 2.0)]


def test_malformed_tf_tables_are_ignored_and_poses_stay_unshifted(tmp_path: pathlib.Path):
    topics = tmp_path / "topics"
    robot_dir = topics / "env_6_jackal"
    _odom(robot_dir)
    pl.DataFrame({"time_ns": [0], "frame_id": ["map"]}).write_parquet(topics / "tf_static.parquet")
    pl.DataFrame({"time_ns": [0], "frame_id": ["map"]}).write_parquet(topics / "tf.parquet")
    pl.DataFrame({"time_ns": [0], "pos_x_gt": [5.0], "pos_y_gt": [6.0], "yaw_gt": [0.0]}).write_parquet(robot_dir / "tf_gt.parquet")

    bundle = _load(topics)["env_6_jackal"]

    assert bundle.tf_gt.collect().select("pos_x_gt", "pos_y_gt").rows() == [(5.0, 6.0)]
    assert bundle.tf_static is not None


def test_env_directory_without_odometry_is_not_a_robot_when_robots_exist(tmp_path: pathlib.Path):
    topics = tmp_path / "topics"
    _odom(topics / "env_0_jackal")
    (topics / "env_0").mkdir()
    pl.DataFrame({"time_ns": [0], "peds_positions": [[1.0, 1.0, 0.0]]}).write_parquet(topics / "env_0" / "peds.parquet")

    bundles = _load(topics)

    assert set(bundles) == {"env_0_jackal"}
    assert bundles["env_0_jackal"].peds.collect()["peds_positions"].to_list() == [[1.0, 1.0, 0.0]]


def test_empty_episode_record_passes_through_the_offset(shifted_env: pathlib.Path):
    pl.DataFrame({"time_ns": [], "start_pos": []}, schema={"time_ns": pl.Int64, "start_pos": pl.Utf8}).write_parquet(shifted_env / "env_3" / "episode_record.parquet")

    record = _load(shifted_env)["env_3_jackal"].episode_record.collect()

    assert record.columns == ["time_ns", "start_pos"]
    assert len(record) == 0
