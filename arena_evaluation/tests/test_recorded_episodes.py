from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arena_evaluation.processing.recorded_episodes import RecordedEpisodes
from musohu_bag import BEAMS, COUNT, cmd_vel, pose, scan_ranges, velocity

TOPICS = "episodes/episode_000/topics/env_0_human"
ODOM_COLUMNS = ["time_ns", "stamp_ns", "pos_x", "pos_y", "yaw", "vel_linear", "vel_angular"]
SCAN_COLUMNS = ["scan_ranges", "scan_min", "scan_range_max"]
CMD_COLUMNS = ["linear_x", "linear_y", "linear_z", "angular_x", "angular_y", "angular_z"]


def test_metadata(converted: Path):
    episodes = RecordedEpisodes(converted)
    assert len(episodes) == 1
    assert episodes.episode(0).planner == "human"
    assert episodes.metadata(0).robot_model == ["human"]
    assert list(episodes.robots(0)) == ["env_0_human"]
    frames = episodes.frames(0)
    assert frames.height == COUNT
    assert frames.columns == ODOM_COLUMNS + SCAN_COLUMNS + CMD_COLUMNS


def test_frames_carry_scan_pose_velocity_and_command_per_step(converted: Path):
    frames = RecordedEpisodes(converted).frames(0)
    row = frames.row(5, named=True)
    scan = np.asarray(row["scan_ranges"], dtype=np.float32)
    assert scan.shape == (BEAMS,)
    np.testing.assert_allclose(scan[1:], scan_ranges(5)[1:])
    assert (row["pos_x"], row["pos_y"], row["yaw"]) == pytest.approx(pose(5))
    assert (row["vel_linear"], row["vel_angular"]) == pytest.approx((velocity(5)[0], velocity(5)[2]))
    assert (row["linear_x"], row["angular_z"]) == pytest.approx(cmd_vel(4))
    assert frames["linear_x"][0] is None


def test_frames_match_extracted_topics(converted: Path):
    import polars as pl

    frames = RecordedEpisodes(converted).frames(0)
    odom = pl.read_parquet(converted / TOPICS / "odom.parquet").sort("time_ns")
    scan = pl.read_parquet(converted / TOPICS / "scan.parquet").sort("time_ns")
    assert frames.select(ODOM_COLUMNS).equals(odom)
    assert frames["scan_ranges"].to_list() == scan["scan_ranges"].to_list()
    assert frames["scan_min"].to_list() == scan["scan_min"].to_list()


def test_missing_episode_raises(converted: Path):
    with pytest.raises(IndexError, match="no episode 3"):
        RecordedEpisodes(converted).frames(3)


def test_unknown_robot_lists_the_recorded_ones(converted: Path):
    with pytest.raises(KeyError, match="env_0_human"):
        RecordedEpisodes(converted).frames(0, robot="env_0_jackal")
