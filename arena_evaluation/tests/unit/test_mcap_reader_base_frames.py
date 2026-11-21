from __future__ import annotations

import pathlib

import pytest
import yaml

pl = pytest.importorskip("polars")
pytest.importorskip("mcap_ros2.writer")

from test_mcap_reader_topics import ODOMETRY, SEC, TF_MESSAGE, _header, _transform, _write_bag

from arena_evaluation.processing.mcap_reader import MCAPReader, recorded_base_frames
from arena_evaluation.processing.telemetry.tracks import LEGACY_BASE_FRAMES, is_base_frame


def _snapshot(run_dir: pathlib.Path, robot: str, base_frame: str) -> None:
    path = run_dir / "robots" / robot / "caps_snapshot.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump({"format": "arena-caps/1", "model": robot, "effective": {"body": {"base_frame": base_frame}}}))


def _run(run_dir: pathlib.Path) -> None:
    run_dir.mkdir()
    odom = {"header": _header(), "pose": {"pose": {"position": {}, "orientation": {"w": 1.0}}}, "twist": {"twist": {}}}
    tf = {"transforms": [_transform("map", "env_0/rover/base", 1.0, 2.0), _transform("map", "env_0/rover/base_link", 3.0, 4.0)]}
    _write_bag(run_dir / "run.mcap", [("/env_0/rover/odom", ODOMETRY, odom, 0), ("/tf", TF_MESSAGE, tf, SEC)])


def test_a_run_without_snapshots_knows_the_conventional_base_frames(tmp_path: pathlib.Path):
    assert recorded_base_frames(tmp_path) == LEGACY_BASE_FRAMES


def test_the_base_frames_are_the_ones_the_snapshots_name(tmp_path: pathlib.Path):
    _snapshot(tmp_path, "env_0_rover", "Base")
    _snapshot(tmp_path, "env_0_other", "chassis")
    assert recorded_base_frames(tmp_path) == {"base", "chassis"}


def test_a_base_frame_is_matched_by_its_last_path_segment():
    names = frozenset({"base"})
    assert is_base_frame("env_0/rover/base", names)
    assert is_base_frame("/Base", names)
    assert not is_base_frame("env_0/rover/base_link", names)
    assert not is_base_frame("env_0/base/odom", names)


def test_the_ground_truth_pose_follows_the_recorded_base_frame(tmp_path: pathlib.Path):
    run = tmp_path / "run"
    _run(run)
    _snapshot(run, "env_0_rover", "base")

    tracks = MCAPReader(run).read(tmp_path / "out")["env_0_rover"].robot_tracks.collect()

    assert tracks.filter(pl.col("pose_source") == "tf_gt").select("x", "y").rows() == [(1.0, 2.0)]


def test_a_run_without_snapshots_keeps_following_the_conventional_frames(tmp_path: pathlib.Path):
    run = tmp_path / "run"
    _run(run)

    tracks = MCAPReader(run).read(tmp_path / "out")["env_0_rover"].robot_tracks.collect()

    assert tracks.filter(pl.col("pose_source") == "tf_gt").select("x", "y").rows() == [(3.0, 4.0)]
