import polars as pl
import pytest

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from robot_params_factory import make_robot_params


class _Calc(BaseMetricCalculator):
    NAME = "pose_probe"
    CATEGORY = "test"
    REQUIRED_TOPICS = ["odom"]

    @classmethod
    def output_keys(cls):
        return []

    def calculate(self, episode, prior_results):
        return {}


def _episode(data, start_pos):
    if data is not None and "pos_x" in data.columns and "pos_x_gt" not in data.columns:
        data = data.with_columns(
            pl.col("pos_x").alias("pos_x_gt"),
            pl.col("pos_y").alias("pos_y_gt"),
            pl.col("yaw").alias("yaw_gt"),
        )
    return AlignedEpisodeBundle(
        episode_id=1,
        data=data,
        start_pos=start_pos,
        goal_pos=[],
    )


def test_map_transform_anchors_at_episode_start_not_segment_start():
    data = pl.DataFrame(
        {
            "t_ns": [0, 100_000_000, 200_000_000, 300_000_000, 400_000_000, 500_000_000, 600_000_000],
            "pos_x": [0.0, 0.05, 0.1, 5.0, 5.3, 5.6, 5.9],
            "pos_y": [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0],
            "yaw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    calc = _Calc(make_robot_params(robot_radius=0.2, laser_min_range=0.0, laser_max_range=10.0))
    pos_x, pos_y, _yaw, _ox, _oy, _oyaw = calc.resolve_robot_pose(_episode(data, start_pos=[0.0, 0.0, 0.0]))

    assert pos_x[0] == pytest.approx(5.0)
    assert pos_x[-1] == pytest.approx(5.9)
    assert pos_y[0] == pytest.approx(5.0)


def test_map_transform_identity_without_teleport():
    data = pl.DataFrame(
        {
            "t_ns": [0, 100_000_000, 200_000_000],
            "pos_x": [1.0, 1.2, 1.4],
            "pos_y": [2.0, 2.0, 2.0],
            "yaw": [0.0, 0.0, 0.0],
        }
    )
    calc = _Calc(make_robot_params(robot_radius=0.2, laser_min_range=0.0, laser_max_range=10.0))
    pos_x, pos_y, *_ = calc.resolve_robot_pose(_episode(data, start_pos=[1.0, 2.0, 0.0]))

    assert pos_x[0] == pytest.approx(1.0)
    assert pos_x[-1] == pytest.approx(1.4)
    assert pos_y[0] == pytest.approx(2.0)
