import polars as pl
import pytest
from flat_peds import flat_peds
from polars.testing import assert_series_equal

from arena_evaluation.processing.metrics.performance.clearance_metrics import (
    ClearanceMetricsCalculator,
)
from arena_evaluation.storage.exceptions import MetricInputError
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from robot_params_factory import make_robot_params

SEC = 1_000_000_000
RADIUS = 0.25
PED_RADIUS = 0.3


def _episode(times_ns, scan=None, peds=None, radius_m=PED_RADIUS, radii=None):
    """Robot at the origin for every sample; scan is the robot_scan_summary columns, peds one list of (x, y) per sample."""
    n = len(times_ns)
    data = pl.DataFrame(
        {
            "t_ns": times_ns,
            "x": [0.0] * n,
            "y": [0.0] * n,
            "yaw": [0.0] * n,
        }
    )
    topics = {"robot_tracks": pl.DataFrame({"t_ns": times_ns, "x": [0.0] * n, "y": [0.0] * n, "yaw": [0.0] * n})}
    if scan is not None:
        topics["robot_scan_summary"] = pl.DataFrame({"t_ns": times_ns, **scan})
    frames = None
    if peds is not None:
        frames = flat_peds([(t, [{"x": x, "y": y} for x, y in sample]) for t, sample in zip(times_ns, peds, strict=True)])
        if radius_m is not None:
            radius_expr = pl.lit(radius_m, dtype=pl.Float64)
            for ped_id, ped_radius in (radii or {}).items():
                radius_expr = pl.when(pl.col("ped_id") == ped_id).then(pl.lit(ped_radius, dtype=pl.Float64)).otherwise(radius_expr)
            frames = frames.with_columns(radius_m=radius_expr)
        topics["pedestrian_arena_peds"] = frames
    return AlignedEpisodeBundle(
        episode_id=1,
        data=data,
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[5.0, 0.0],
        num_pedestrians=1,
        topics=topics,
        peds=frames,
    )


@pytest.fixture
def calc():
    return ClearanceMetricsCalculator(make_robot_params(robot_radius=RADIUS))


def _series(calc, episode, column):
    return calc.series_table(episode, run_id="run", robot_id="robot").get_column(column)


def test_pedestrian_clearance_requires_the_recorded_radius(calc):
    episode = _episode([0, SEC], peds=[[(1.5, 0.0)], [(1.5, 0.0)]], radius_m=None)
    with pytest.raises(MetricInputError, match="radius_m"):
        calc.calculate(episode, {})


def test_timeseries_takes_the_minimum_of_both_sources(calc):
    episode = _episode(
        [0, SEC],
        scan={"scan_min_m": [2.0, 0.5], "range_max_m": [10.0, 10.0]},
        peds=[[(1.0, 0.0)], [(5.0, 0.0)]],
    )
    calc.calculate(episode, {})
    expected = pl.Series(
        "combined_clearance_m",
        [1.0 - RADIUS - PED_RADIUS, 0.5 - RADIUS],
        dtype=pl.Float64,
        nan_to_null=True,
    )
    assert_series_equal(
        _series(calc, episode, "combined_clearance_m"),
        expected,
        check_names=False,
        rel_tol=1e-9,
    )
