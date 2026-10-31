import pathlib

import polars as pl
import pytest

pytest.importorskip("mcap_ros2.writer")

from arena_evaluation.processing.mcap_reader import MCAPReader
from arena_evaluation.storage.schemas import TopicBundle
from test_mcap_reader_topics import ODOMETRY, TF_MESSAGE, _header, _transform, _write_bag


def _odom(x: float, y: float) -> dict:
    return {"header": _header(frame_id="odom"), "pose": {"pose": {"position": {"x": x, "y": y}, "orientation": {"w": 1.0}}}, "twist": {"twist": {"linear": {"x": 1.0}, "angular": {"z": 0.5}}}}


def test_mcap_reader_lazy_chunking(tmp_path: pathlib.Path):
    bag = _write_bag(tmp_path / "episode.mcap", [("/env_0/odom", ODOMETRY, _odom(float(i), float(i * 2)), i * 1000) for i in range(10005)])

    bundle = MCAPReader(bag).read(tmp_path / "raw")["env_0"]

    odom_parquet = tmp_path / "raw" / "env_0" / "odom.parquet"
    assert odom_parquet.exists()

    assert isinstance(bundle, TopicBundle)
    assert isinstance(bundle.odom, pl.LazyFrame)

    odom_df = bundle.odom.collect()
    assert len(odom_df) == 10005
    assert odom_df["pos_x"][0] == 0.0
    assert odom_df["pos_x"][10004] == 10004.0
    assert odom_df["pos_y"][10004] == 20008.0
    assert odom_df["time_ns"][10004] == 10004000


def test_mcap_reader_tf_gt_extraction(tmp_path: pathlib.Path):
    transforms = [_transform("map", "env_0/map", 0.0, 0.0), _transform("map", "env_0/base_link", 5.5, 10.0)]
    bag = _write_bag(tmp_path / "episode.mcap", [("/tf", TF_MESSAGE, {"transforms": transforms}, 999999)])

    bundle = MCAPReader(bag).read(tmp_path / "raw")["env_0"]

    tf_gt_parquet = tmp_path / "raw" / "env_0" / "tf_gt.parquet"
    assert tf_gt_parquet.exists()

    assert bundle.tf_gt is not None
    assert isinstance(bundle.tf_gt, pl.LazyFrame)

    tf_gt_df = bundle.tf_gt.collect()
    assert len(tf_gt_df) == 1
    assert tf_gt_df["time_ns"][0] == 999999
    assert tf_gt_df["pos_x_gt"][0] == 5.5
    assert tf_gt_df["pos_y_gt"][0] == 10.0
    assert tf_gt_df["yaw_gt"][0] == 0.0


def test_mcap_reader_env_offset_auto_detection(tmp_path: pathlib.Path):
    topics_dir = tmp_path / "topics"
    env_1_dir = topics_dir / "env_1"
    env_1_dir.mkdir(parents=True)

    tf_static_df = pl.DataFrame(
        {
            "time_ns": [1000],
            "frame_id": ["map"],
            "child_frame_id": ["env_1/map"],
            "trans_x": [100.0],
            "trans_y": [50.0],
            "trans_z": [0.0],
            "rot_x": [0.0],
            "rot_y": [0.0],
            "rot_z": [0.0],
            "rot_w": [1.0],
        }
    )
    tf_static_df.write_parquet(topics_dir / "tf_static.parquet")

    tf_gt_df = pl.DataFrame(
        {
            "time_ns": [1000],
            "pos_x_gt": [105.0],
            "pos_y_gt": [52.0],
            "yaw_gt": [0.0],
        }
    )
    tf_gt_df.write_parquet(env_1_dir / "tf_gt.parquet")

    odom_df = pl.DataFrame(
        {
            "time_ns": [1000],
            "pos_x": [5.0],
            "pos_y": [2.0],
            "yaw": [0.0],
        }
    )
    odom_df.write_parquet(env_1_dir / "odom.parquet")

    reader = MCAPReader(tmp_path / "dummy.mcap")
    bundles = reader.load_bundles(topics_dir, tmp_path)
    bundle = bundles["env_1"]

    assert bundle.tf_gt is not None
    gt_res = bundle.tf_gt.collect()
    assert abs(gt_res["pos_x_gt"][0] - 5.0) < 1e-5
    assert abs(gt_res["pos_y_gt"][0] - 2.0) < 1e-5

    assert bundle.odom is not None
    odom_res = bundle.odom.collect()
    assert abs(odom_res["pos_x"][0] - 5.0) < 1e-5
    assert abs(odom_res["pos_y"][0] - 2.0) < 1e-5


def test_mcap_reader_shifts_each_row_by_the_anchor_in_effect(tmp_path: pathlib.Path) -> None:
    topics_dir = tmp_path / "topics"
    env_dir = topics_dir / "env_11"
    robot_dir = topics_dir / "env_11_jackal"
    env_dir.mkdir(parents=True)
    robot_dir.mkdir()
    n = 4
    pl.DataFrame(
        {
            "time_ns": [1000, 1000, 2000, 2000],
            "frame_id": ["map"] * n,
            "child_frame_id": ["env_11/map"] * n,
            "trans_x": [4.975, 4.975, 4.975, 773.354],
            "trans_y": [152.075, 152.075, 152.075, 5.0],
            "trans_z": [0.0] * n,
            "rot_x": [0.0] * n,
            "rot_y": [0.0] * n,
            "rot_z": [0.0] * n,
            "rot_w": [1.0] * n,
        }
    ).write_parquet(topics_dir / "tf_static.parquet")
    pl.DataFrame({"time_ns": [1500, 2000, 2500], "pos_x_gt": [5.975, 774.354, 775.354], "pos_y_gt": [154.075, 7.0, 8.0], "yaw_gt": [0.0] * 3}).write_parquet(robot_dir / "tf_gt.parquet")
    pl.DataFrame({"time_ns": [1500, 2500], "poses_x": [[4.975, 6.975], [773.354, 799.179]], "poses_y": [[152.075, 153.075], [5.0, 9.175]], "poses_yaw": [[0.0, 0.0], [0.0, 0.0]]}).write_parquet(robot_dir / "plan.parquet")
    pl.DataFrame({"time_ns": [1500, 2500], "peds_positions": [[5.975, 153.075, 0.0], [774.354, 6.0, 0.0]]}).write_parquet(env_dir / "peds.parquet")
    pl.DataFrame({"time_ns": [1500, 2500], "pos_x": [0.0, 0.0], "pos_y": [0.0, 0.0], "yaw": [0.0, 0.0]}).write_parquet(robot_dir / "odom.parquet")

    bundle = MCAPReader(tmp_path / "dummy.mcap").load_bundles(topics_dir, tmp_path)["env_11_jackal"]

    gt = bundle.tf_gt.collect()
    assert gt["pos_x_gt"].to_list() == pytest.approx([1.0, 1.0, 2.0])
    assert gt["pos_y_gt"].to_list() == pytest.approx([2.0, 2.0, 3.0])
    plan = bundle.plan.collect()
    assert plan["poses_x"].to_list() == [pytest.approx([0.0, 2.0]), pytest.approx([0.0, 25.825])]
    assert plan["poses_y"].to_list() == [pytest.approx([0.0, 1.0]), pytest.approx([0.0, 4.175])]
    peds = bundle.peds.collect()
    assert peds["peds_positions"].to_list() == [pytest.approx([1.0, 1.0, 0.0]), pytest.approx([1.0, 1.0, 0.0])]
