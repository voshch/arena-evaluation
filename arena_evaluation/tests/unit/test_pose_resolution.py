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
    return AlignedEpisodeBundle(
        episode_id=1,
        data=data,
        start_pos=start_pos,
        goal_pos=[],
    )


def test_the_native_tf_track_is_not_cut_at_a_teleport():
    data = pl.DataFrame(
        {
            "t_ns": [
                0,
                100_000_000,
                200_000_000,
                300_000_000,
                400_000_000,
                500_000_000,
                600_000_000,
            ],
            "x": [0.0, 0.05, 0.1, 5.0, 5.3, 5.6, 5.9],
            "y": [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0],
            "yaw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    calc = _Calc(
        make_robot_params(robot_radius=0.2, laser_min_range=0.0, laser_max_range=10.0)
    )
    pos_x, pos_y, _yaw, _ox, _oy, _oyaw = calc.resolve_robot_pose(
        _episode(data, start_pos=[0.0, 0.0, 0.0])
    )

    # D3: the aligned frame carries the native /tf track for the whole episode window.
    assert pos_x[0] == pytest.approx(0.0)
    assert pos_x[-1] == pytest.approx(5.9)
    assert pos_y[0] == pytest.approx(0.0)


def test_map_transform_identity_without_teleport():
    data = pl.DataFrame(
        {
            "t_ns": [0, 100_000_000, 200_000_000],
            "x": [1.0, 1.2, 1.4],
            "y": [2.0, 2.0, 2.0],
            "yaw": [0.0, 0.0, 0.0],
        }
    )
    calc = _Calc(
        make_robot_params(robot_radius=0.2, laser_min_range=0.0, laser_max_range=10.0)
    )
    pos_x, pos_y, *_ = calc.resolve_robot_pose(
        _episode(data, start_pos=[1.0, 2.0, 0.0])
    )

    assert pos_x[0] == pytest.approx(1.0)
    assert pos_x[-1] == pytest.approx(1.4)
    assert pos_y[0] == pytest.approx(2.0)
