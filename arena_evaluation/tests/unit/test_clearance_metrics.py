import polars as pl
import pytest

from arena_evaluation.processing.metrics.performance.clearance_metrics import (
    ClearanceMetricsCalculator,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

SEC = 1_000_000_000
RADIUS = 0.25
PED_RADIUS = ClearanceMetricsCalculator._PED_RADIUS


def _episode(times_ns, scan=None, peds=None):
    n = len(times_ns)
    data = pl.DataFrame(
        {
            "time_ns": times_ns,
            "pos_x": [0.0] * n,
            "pos_y": [0.0] * n,
            "yaw": [0.0] * n,
        }
    )
    topics = {}
    if scan is not None:
        topics["scan"] = pl.DataFrame({"time_ns": times_ns, **scan})
    if peds is not None:
        topics["peds"] = pl.DataFrame(
            {
                "time_ns": times_ns,
                "peds_positions": peds,
                "num_pedestrians": [len(p) // 2 for p in peds],
            }
        )
    return AlignedEpisodeBundle(
        episode_id=1,
        data=data,
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[5.0, 0.0],
        num_pedestrians=1,
        topics=topics,
    )


@pytest.fixture
def calc():
    return ClearanceMetricsCalculator(RobotParams(robot_radius=RADIUS))


def test_no_topics_yields_all_none(calc):
    results = calc.calculate(_episode([0, SEC]), {})
    assert set(results) == set(calc.output_keys())
    assert all(v is None for v in results.values())


def test_obstacle_clearance_subtracts_the_robot_radius(calc):
    scan = {"scan_min": [2.0, 3.0], "scan_range_max": [10.0, 10.0]}
    results = calc.calculate(_episode([0, SEC], scan=scan), {})
    assert results["min_obstacle_clearance"] == pytest.approx(2.0 - RADIUS)
    assert results["mean_obstacle_clearance"] == pytest.approx(2.5 - RADIUS)


def test_scan_at_max_range_is_no_detection_not_a_far_obstacle(calc):
    """The sentinel frame must not be averaged in as a huge clearance."""
    scan = {"scan_min": [2.0, 10.0], "scan_range_max": [10.0, 10.0]}
    results = calc.calculate(_episode([0, SEC], scan=scan), {})
    assert results["min_obstacle_clearance"] == pytest.approx(2.0 - RADIUS)
    assert results["mean_obstacle_clearance"] == pytest.approx(2.0 - RADIUS)
    assert results["clearance_timeseries"][1] is None


def test_every_frame_at_max_range_leaves_no_obstacle_clearance(calc):
    scan = {"scan_min": [10.0, 10.0], "scan_range_max": [10.0, 10.0]}
    results = calc.calculate(_episode([0, SEC], scan=scan), {})
    assert results["min_obstacle_clearance"] is None
    assert results["mean_obstacle_clearance"] is None


def test_pedestrian_clearance_is_edge_to_edge(calc):
    results = calc.calculate(_episode([0, SEC], peds=[[1.5, 0.0], [3.0, 0.0]]), {})
    assert results["min_pedestrian_clearance"] == pytest.approx(1.5 - RADIUS - PED_RADIUS)


def test_pedestrian_clearance_is_clamped_at_zero(calc):
    results = calc.calculate(_episode([0, SEC], peds=[[0.1, 0.0], [0.1, 0.0]]), {})
    assert results["min_pedestrian_clearance"] == 0.0


def test_nearest_pedestrian_wins(calc):
    peds = [[3.0, 0.0, 1.0, 0.0], [3.0, 0.0, 1.0, 0.0]]
    results = calc.calculate(_episode([0, SEC], peds=peds), {})
    assert results["min_pedestrian_clearance"] == pytest.approx(1.0 - RADIUS - PED_RADIUS)


def test_timeseries_takes_the_minimum_of_both_sources(calc):
    scan = {"scan_min": [2.0, 0.5], "scan_range_max": [10.0, 10.0]}
    peds = [[1.0, 0.0], [5.0, 0.0]]
    results = calc.calculate(_episode([0, SEC], scan=scan, peds=peds), {})
    assert results["clearance_timeseries"][0] == pytest.approx(1.0 - RADIUS - PED_RADIUS)
    assert results["clearance_timeseries"][1] == pytest.approx(0.5 - RADIUS)


def test_frames_without_pedestrians_fall_back_to_the_scan(calc):
    scan = {"scan_min": [2.0, 2.0], "scan_range_max": [10.0, 10.0]}
    results = calc.calculate(_episode([0, SEC], scan=scan, peds=[[], []]), {})
    assert results["min_pedestrian_clearance"] is None
    assert results["clearance_timeseries"] == [
        pytest.approx(2.0 - RADIUS),
        pytest.approx(2.0 - RADIUS),
    ]
