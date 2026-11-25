import numpy as np
import polars as pl
import pytest

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from robot_params_factory import make_robot_params

SEC = 1_000_000_000


class _Probe(BaseMetricCalculator):
    """Concrete subclass so the native-rate helpers can be exercised directly."""

    NAME = "probe"

    @classmethod
    def output_keys(cls) -> list[str]:
        return []

    def calculate(self, episode, prior_results):
        return {}


@pytest.fixture
def probe():
    return _Probe(make_robot_params(robot_radius=0.25))


def _bundle(topics=None, data=None, start_pos=None):
    return AlignedEpisodeBundle(
        episode_id=1,
        data=data if data is not None else pl.DataFrame(),
        start_pos=start_pos if start_pos is not None else [0.0, 0.0, 0.0],
        goal_pos=[1.0, 0.0],
        topics=topics,
    )


def test_pose_at_times_without_reference_is_zero_filled(probe):
    empty = np.array([])
    x, y, yaw = probe.pose_at_times(np.array([0, SEC]), empty, empty, empty, empty)
    assert len(x) == len(y) == len(yaw) == 2
    assert np.isnan(x).all()
    assert np.isnan(y).all()
    assert np.isnan(yaw).all()


def test_speed_from_pose_single_sample_is_zero(probe):
    out = probe.speed_from_pose(np.array([1.0]), np.array([1.0]), np.array([0]))
    assert len(out) == 1
    assert np.isnan(out[0])
