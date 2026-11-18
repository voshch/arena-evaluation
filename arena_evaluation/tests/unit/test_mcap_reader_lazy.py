import pathlib

import polars as pl
import pytest

pytest.importorskip("mcap_ros2.writer")

from arena_evaluation.processing.mcap_reader import MCAPReader, load_bundles
from arena_evaluation.storage.schemas import TopicBundle
from test_mcap_reader_topics import ODOMETRY, TF_MESSAGE, _header, _transform, _write_bag


def _odom(x: float, y: float) -> dict:
    return {"header": _header(frame_id="odom"), "pose": {"pose": {"position": {"x": x, "y": y}, "orientation": {"w": 1.0}}}, "twist": {"twist": {"linear": {"x": 1.0}, "angular": {"z": 0.5}}}}


def _tf_row(t_ns: int, parent: str, child: str, tx: float, ty: float) -> dict:
    return {"t_ns": t_ns, "parent_frame": parent, "child_frame": child, "tx": tx, "ty": ty, "tz": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}


def test_robot_odometry_is_read_lazily_over_every_message(tmp_path: pathlib.Path):
    bag = _write_bag(tmp_path / "episode.mcap", [("/env_0/odom", ODOMETRY, _odom(float(i), float(i * 2)), i * 1000) for i in range(10005)])

    bundle = MCAPReader(bag).read(tmp_path / "raw")["env_0"]

    assert (tmp_path / "raw" / "robot_odom.parquet").exists()
    assert isinstance(bundle, TopicBundle)
    assert isinstance(bundle.robot_odom, pl.LazyFrame)
    odom = bundle.robot_odom.collect()
    assert len(odom) == 10005
    assert odom["odom_x"][0] == 0.0
    assert odom["odom_x"][10004] == 10004.0
    assert odom["odom_y"][10004] == 20008.0
    assert odom["t_ns"][10004] == 10004000


def test_map_frame_ground_truth_is_read_as_robot_tracks(tmp_path: pathlib.Path):
    transforms = [_transform("map", "env_0/map", 0.0, 0.0), _transform("map", "env_0/base_link", 5.5, 10.0)]
    bag = _write_bag(tmp_path / "episode.mcap", [("/tf", TF_MESSAGE, {"transforms": transforms}, 999999)])

    bundle = MCAPReader(bag).read(tmp_path / "raw")["env_0"]

    assert isinstance(bundle.robot_tracks, pl.LazyFrame)
    assert bundle.robot_tracks.collect().select("t_ns", "x", "y", "yaw", "pose_source").rows() == [(999999, 5.5, 10.0, 0.0, "tf_gt")]


def test_map_anchor_moves_ground_truth_into_the_map_but_never_odometry(tmp_path: pathlib.Path):
    topics = tmp_path / "topics"
    topics.mkdir()
    pl.DataFrame([_tf_row(1000, "map", "env_1/map", 100.0, 50.0)]).write_parquet(topics / "tf_static.parquet")
    pl.DataFrame([{"t_ns": 1000, "env_id": "env_1", "robot_id": "env_1_jackal", "x": 105.0, "y": 52.0, "yaw": 0.0, "pose_source": "tf_gt"}]).write_parquet(topics / "robot_tracks.parquet")
    pl.DataFrame([{"t_ns": 1000, "env_id": "env_1", "robot_id": "env_1_jackal", "odom_x": 5.0, "odom_y": 2.0, "odom_yaw": 0.0}]).write_parquet(topics / "robot_odom.parquet")

    bundle = load_bundles(topics, {})["env_1_jackal"]

    assert bundle.robot_tracks.collect().select("x", "y").rows() == [(5.0, 2.0)]
    assert bundle.robot_odom.collect().select("odom_x", "odom_y").rows() == [(5.0, 2.0)]


def test_each_row_moves_by_the_map_anchor_in_effect_at_its_time(tmp_path: pathlib.Path):
    topics = tmp_path / "topics"
    topics.mkdir()
    pl.DataFrame(
        [
            _tf_row(1000, "map", "env_11/map", 4.975, 152.075),
            _tf_row(1000, "map", "env_11/map", 4.975, 152.075),
            _tf_row(2000, "map", "env_11/map", 4.975, 152.075),
            _tf_row(2000, "map", "env_11/map", 773.354, 5.0),
        ]
    ).write_parquet(topics / "tf_static.parquet")
    pl.DataFrame(
        [
            {"t_ns": 1500, "env_id": "env_11", "robot_id": "env_11_jackal", "x": 5.975, "y": 154.075, "yaw": 0.0, "pose_source": "tf_gt"},
            {"t_ns": 2000, "env_id": "env_11", "robot_id": "env_11_jackal", "x": 774.354, "y": 7.0, "yaw": 0.0, "pose_source": "tf_gt"},
            {"t_ns": 2500, "env_id": "env_11", "robot_id": "env_11_jackal", "x": 775.354, "y": 8.0, "yaw": 0.0, "pose_source": "tf_gt"},
        ]
    ).write_parquet(topics / "robot_tracks.parquet")
    pl.DataFrame(
        [
            {"t_ns": 1500, "env_id": "env_11", "robot_id": "env_11_jackal", "plan_id": 0, "waypoint_idx": 0, "x": 4.975, "y": 152.075, "yaw": 0.0},
            {"t_ns": 1500, "env_id": "env_11", "robot_id": "env_11_jackal", "plan_id": 0, "waypoint_idx": 1, "x": 6.975, "y": 153.075, "yaw": 0.0},
            {"t_ns": 2500, "env_id": "env_11", "robot_id": "env_11_jackal", "plan_id": 0, "waypoint_idx": 0, "x": 773.354, "y": 5.0, "yaw": 0.0},
            {"t_ns": 2500, "env_id": "env_11", "robot_id": "env_11_jackal", "plan_id": 0, "waypoint_idx": 1, "x": 799.179, "y": 9.175, "yaw": 0.0},
        ]
    ).write_parquet(topics / "robot_plans.parquet")
    pl.DataFrame(
        [
            {"t_ns": 1500, "env_id": "env_11", "ped_id": "Pedestrian_1", "x": 5.975, "y": 153.075},
            {"t_ns": 2500, "env_id": "env_11", "ped_id": "Pedestrian_2", "x": 774.354, "y": 6.0},
        ]
    ).write_parquet(topics / "pedestrian_arena_peds.parquet")

    bundle = load_bundles(topics, {})["env_11_jackal"]

    assert bundle.robot_tracks.collect().sort("t_ns")["x"].to_list() == pytest.approx([1.0, 1.0, 2.0])
    assert bundle.robot_tracks.collect().sort("t_ns")["y"].to_list() == pytest.approx([2.0, 2.0, 3.0])
    plan = bundle.robot_plans.collect().sort(["t_ns", "waypoint_idx"])
    assert plan["x"].to_list() == pytest.approx([0.0, 2.0, 0.0, 25.825])
    assert plan["y"].to_list() == pytest.approx([0.0, 1.0, 0.0, 4.175])
    assert bundle.pedestrian_arena_peds.collect().sort("t_ns")["x"].to_list() == pytest.approx([1.0, 1.0])
    assert bundle.pedestrian_arena_peds.collect().sort("t_ns")["y"].to_list() == pytest.approx([1.0, 1.0])
