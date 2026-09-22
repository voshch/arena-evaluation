import numpy as np
import polars as pl
import pytest

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

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
    return _Probe(RobotParams(robot_radius=0.25))


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


def test_resolve_native_pose_prefers_tf_gt(probe):
    topics = {
        "odom": pl.DataFrame(
            {
                "time_ns": [0, SEC],
                "pos_x": [0.0, 1.0],
                "pos_y": [0.0, 0.0],
                "yaw": [0.0, 0.0],
            }
        ),
        "tf_gt": pl.DataFrame(
            {
                "time_ns": [0, SEC],
                "pos_x_gt": [5.0, 6.0],
                "pos_y_gt": [7.0, 7.0],
                "yaw_gt": [0.1, 0.1],
            }
        ),
    }
    x, y, yaw, t = probe.resolve_native_pose(_bundle(topics=topics))
    assert x.tolist() == [5.0, 6.0]
    assert y.tolist() == [7.0, 7.0]
    assert yaw.tolist() == [0.1, 0.1]
    assert t.tolist() == [0, SEC]


def test_resolve_native_pose_odom_fallback_is_world_framed(probe):
    """Odom starts at its own origin; the start pose maps it into the world."""
    topics = {
        "odom": pl.DataFrame(
            {
                "time_ns": [0, SEC, 2 * SEC],
                "pos_x": [0.0, 1.0, 2.0],
                "pos_y": [0.0, 0.0, 0.0],
                "yaw": [0.0, 0.0, 0.0],
            }
        ),
    }
    x, y, _yaw, _t = probe.resolve_native_pose(_bundle(topics=topics, start_pos=[10.0, 20.0, 0.0]))
    assert np.allclose(x, [10.0, 11.0, 12.0])
    assert np.allclose(y, [20.0, 20.0, 20.0])


def test_resolve_native_pose_odom_fallback_rotates_by_start_yaw(probe):
    topics = {
        "odom": pl.DataFrame(
            {
                "time_ns": [0, SEC],
                "pos_x": [0.0, 1.0],
                "pos_y": [0.0, 0.0],
                "yaw": [0.0, 0.0],
            }
        ),
    }
    x, y, _yaw, _t = probe.resolve_native_pose(_bundle(topics=topics, start_pos=[0.0, 0.0, np.pi / 2]))
    assert np.allclose(x, [0.0, 0.0], atol=1e-9)
    assert np.allclose(y, [0.0, 1.0], atol=1e-9)


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
    assert out_x.tolist() == [0.0]
    assert out_y.tolist() == [0.0]
    assert out_yaw.tolist() == [0.0]


def test_pose_at_times_without_reference_is_zero_filled(probe):
    empty = np.array([])
    x, y, yaw = probe.pose_at_times(np.array([0, SEC]), empty, empty, empty, empty)
    assert x.tolist() == [0.0, 0.0]
    assert y.tolist() == [0.0, 0.0]
    assert yaw.tolist() == [0.0, 0.0]


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
    assert speed.tolist() == [0.0, 1.0, 3.0]

    vx, vy = probe.velocity_from_pose(x, y, t)
    assert vx.tolist() == [0.0, 1.0, 0.0]
    assert vy.tolist() == [0.0, 0.0, 3.0]


def test_speed_from_pose_single_sample_is_zero(probe):
    out = probe.speed_from_pose(np.array([1.0]), np.array([1.0]), np.array([0]))
    assert out.tolist() == [0.0]


def test_parse_peds_reads_flat_triples_and_pairs(probe):
    """The pedestrian count disambiguates a flat list."""
    triples = probe._parse_peds([1.0, 2.0, 0.5, 3.0, 4.0, 1.5], 2)
    assert triples.shape == (2, 3)
    assert triples[1].tolist() == [3.0, 4.0, 1.5]

    pairs = probe._parse_peds([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3)
    assert pairs.shape == (3, 2)
    assert pairs[2].tolist() == [5.0, 6.0]


def test_parse_peds_without_a_count_prefers_triples(probe):
    assert probe._parse_peds([1.0, 2.0, 0.5, 3.0, 4.0, 1.5]).shape == (2, 3)
    assert probe._parse_peds([1.0, 2.0, 3.0, 4.0]).shape == (2, 2)


def test_parse_peds_accepts_already_shaped_input(probe):
    out = probe._parse_peds([[1.0, 2.0], [3.0, 4.0]])
    assert out.shape == (2, 2)


def test_parse_peds_parses_a_stringified_list(probe):
    out = probe._parse_peds("[[1.0, 2.0], [3.0, 4.0]]")
    assert out.shape == (2, 2)
    assert out[0].tolist() == [1.0, 2.0]


@pytest.mark.parametrize("bad", [None, [], "", "not a list", [1.0], [1.0, 2.0, 3.0, 4.0, 5.0]])
def test_parse_peds_returns_empty_for_unusable_input(probe, bad):
    out = probe._parse_peds(bad)
    assert out.shape == (0, 2)


def test_parse_peds_ignores_a_count_that_fits_neither_layout(probe):
    assert probe._parse_peds([1.0, 2.0, 3.0, 4.0], 7).shape == (0, 2)


def test_native_ped_frame_sorted_by_time(probe):
    peds = pl.DataFrame({"time_ns": [2 * SEC, 0, SEC], "peds_positions": [[], [], []]})
    out = probe.native_ped_frame(_bundle(topics={"peds": peds}))
    assert out["time_ns"].to_list() == [0, SEC, 2 * SEC]
    assert probe.native_ped_frame(_bundle()) is None
    assert probe.native_ped_frame(_bundle(topics={"peds": pl.DataFrame({"x": [1]})})) is None
