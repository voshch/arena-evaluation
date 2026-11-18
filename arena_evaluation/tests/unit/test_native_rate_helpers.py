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


def test_native_topics_absent_is_empty_dict(probe):
    assert probe.native_topics(_bundle()) == {}
    assert probe.native_topics(_bundle(topics={"odom": pl.DataFrame()})).keys() == {"odom"}


def test_resolve_native_pose_from_robot_tracks(probe):
    topics = {
        "robot_tracks": pl.DataFrame(
            {
                "t_ns": [0, SEC],
                "x": [5.0, 6.0],
                "y": [7.0, 7.0],
                "yaw": [0.1, 0.1],
            }
        ),
    }
    x, y, yaw, t = probe.resolve_native_pose(_bundle(topics=topics))
    assert x.tolist() == [5.0, 6.0]
    assert y.tolist() == [7.0, 7.0]
    assert yaw.tolist() == [0.1, 0.1]
    assert t.tolist() == [0, SEC]


def test_resolve_native_pose_never_falls_back_to_odometry(probe):
    topics = {
        "robot_odom": pl.DataFrame(
            {
                "t_ns": [0, SEC],
                "odom_x": [0.0, 1.0],
                "odom_y": [0.0, 0.0],
                "odom_yaw": [0.0, 0.0],
            }
        ),
    }
    x, y, yaw, t = probe.resolve_native_pose(_bundle(topics=topics))
    assert len(x) == len(y) == len(yaw) == len(t) == 0


def test_resolve_native_pose_without_topics_is_empty(probe):
    x, y, yaw, t = probe.resolve_native_pose(_bundle())
    assert len(x) == len(y) == len(yaw) == len(t) == 0


def test_pose_at_times_backward_asof(probe):
    pose_t = np.array([0, 2 * SEC, 4 * SEC])
    px = np.array([0.0, 2.0, 4.0])
    py = np.array([0.0, 0.0, 0.0])
    yaw = np.array([0.0, 0.5, 1.0])

    qx, qy, qyaw = probe.pose_at_times(np.array([0, 2 * SEC, 4 * SEC]), px, py, yaw, pose_t)
    assert qx.tolist() == [0.0, 2.0, 4.0]
    assert qyaw.tolist() == [0.0, 0.5, 1.0]

    # 50 ms after a sample is inside the 100 ms tolerance and holds its value.
    qx, _qy, _qyaw = probe.pose_at_times(np.array([2 * SEC + 50_000_000]), px, py, yaw, pose_t)
    assert qx.tolist() == [2.0]


def test_pose_at_times_beyond_tolerance_is_zero_filled(probe):
    pose_t = np.array([0])
    out_x, out_y, out_yaw = probe.pose_at_times(np.array([5 * SEC]), np.array([3.0]), np.array([4.0]), np.array([1.0]), pose_t)
    assert np.isnan(out_x).all()
    assert np.isnan(out_y).all()
    assert np.isnan(out_yaw).all()


def test_pose_at_times_without_reference_is_zero_filled(probe):
    empty = np.array([])
    x, y, yaw = probe.pose_at_times(np.array([0, SEC]), empty, empty, empty, empty)
    assert len(x) == len(y) == len(yaw) == 2
    assert np.isnan(x).all()
    assert np.isnan(y).all()
    assert np.isnan(yaw).all()


def test_values_at_times_nan_outside_tolerance(probe):
    values = np.array([1.0, 2.0])
    values_t = np.array([0, SEC])
    out = probe.values_at_times(values, values_t, np.array([0, SEC, SEC + 5 * SEC]))
    assert out[0] == 1.0
    assert out[1] == 2.0
    assert np.isnan(out[2])


def test_speed_and_velocity_from_pose(probe):
    t = np.array([0, SEC, 2 * SEC])
    x = np.array([0.0, 1.0, 1.0])
    y = np.array([0.0, 0.0, 3.0])

    speed = probe.speed_from_pose(x, y, t)
    assert np.isnan(speed[0])
    assert speed[1:].tolist() == [1.0, 3.0]

    vx, vy = probe.velocity_from_pose(x, y, t)
    assert np.isnan(vx[0])
    assert vx[1:].tolist() == [1.0, 0.0]
    assert np.isnan(vy[0])
    assert vy[1:].tolist() == [0.0, 3.0]


def test_speed_from_pose_single_sample_is_zero(probe):
    out = probe.speed_from_pose(np.array([1.0]), np.array([1.0]), np.array([0]))
    assert len(out) == 1
    assert np.isnan(out[0])


def test_native_ped_frame_sorted_by_time(probe):
    peds = pl.DataFrame({"t_ns": [2 * SEC, 0, SEC], "ped_id": ["p1", "p2", "p3"], "x": [1.0, 2.0, 3.0], "y": [0.0, 0.0, 0.0]})
    out = probe.native_ped_frame(_bundle(topics={"pedestrian_tracks": peds}))
    assert out is not None
    assert out["t_ns"].to_list() == [0, SEC, 2 * SEC]
    assert probe.native_ped_frame(_bundle()) is None
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        probe.native_ped_frame(_bundle(topics={"pedestrian_tracks": pl.DataFrame({"x": [1]})}))
