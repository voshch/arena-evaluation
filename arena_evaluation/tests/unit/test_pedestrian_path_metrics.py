import math

import polars as pl

from arena_evaluation.processing.metrics.performance.pedestrian_path_metrics import PedestrianPathMetricsCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams


def _episode(rows: list) -> AlignedEpisodeBundle:
    """Episode whose aligned frame carries one flat x,y,z peds_positions list per row."""
    df = pl.DataFrame({"peds_positions": rows}, schema={"peds_positions": pl.List(pl.Float64)})
    return AlignedEpisodeBundle(episode_id=1, data=df, start_pos=[], goal_pos=[])


def _paths(rows: list) -> list:
    return PedestrianPathMetricsCalculator(RobotParams()).calculate(_episode(rows), {})["pedestrian_path"]


def _is_nan_point(point: list) -> bool:
    return len(point) == 3 and all(math.isnan(v) for v in point)


def test_output_keys_and_unit():
    assert PedestrianPathMetricsCalculator.output_keys() == ["pedestrian_path"]
    assert PedestrianPathMetricsCalculator.UNITS == {"pedestrian_path": "m"}


def test_missing_frame_yields_no_paths():
    episode = AlignedEpisodeBundle(episode_id=1, data=None, start_pos=[], goal_pos=[])
    assert PedestrianPathMetricsCalculator(RobotParams()).calculate(episode, {}) == {"pedestrian_path": []}


def test_frame_without_peds_column_yields_no_paths():
    episode = AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame({"time_ns": [0, 1]}), start_pos=[], goal_pos=[])
    assert PedestrianPathMetricsCalculator(RobotParams()).calculate(episode, {}) == {"pedestrian_path": []}


def test_empty_frame_yields_no_paths():
    assert _paths([]) == []


def test_all_null_rows_yield_no_paths():
    assert _paths([None, None, None]) == []


def test_rows_shorter_than_one_triple_yield_no_paths():
    assert _paths([[1.0, 2.0], []]) == []


def test_two_pedestrians_are_split_into_time_aligned_xyz_paths():
    assert _paths([[0.0, 0.0, 0.0, 5.0, 5.0, 0.5], [1.0, 0.0, 0.0, 5.0, 6.0, 0.5]]) == [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[5.0, 5.0, 0.5], [5.0, 6.0, 0.5]],
    ]


def test_null_row_repeats_previous_position_of_every_pedestrian():
    assert _paths([[0.0, 0.0, 0.0, 3.0, 3.0, 0.0], None, [2.0, 0.0, 0.0, 3.0, 4.0, 0.0]]) == [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [[3.0, 3.0, 0.0], [3.0, 3.0, 0.0], [3.0, 4.0, 0.0]],
    ]


def test_leading_null_row_stays_nan():
    (path,) = _paths([None, [1.0, 2.0, 0.0]])
    assert len(path) == 2
    assert _is_nan_point(path[0])
    assert path[1] == [1.0, 2.0, 0.0]


def test_pedestrian_dropping_out_holds_its_last_position():
    assert _paths([[0.0, 0.0, 0.0, 5.0, 5.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]) == [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [[5.0, 5.0, 0.0], [5.0, 5.0, 0.0], [5.0, 5.0, 0.0]],
    ]


def test_pedestrian_appearing_late_is_nan_before_first_sample():
    first, second = _paths([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 7.0, 7.0, 0.0]])
    assert first == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert _is_nan_point(second[0])
    assert second[1] == [7.0, 7.0, 0.0]


def test_trailing_partial_triple_is_ignored():
    assert _paths([[0.0, 1.0, 2.0, 9.0], [3.0, 4.0, 5.0]]) == [[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]]


def test_pedestrian_with_nan_x_throughout_is_dropped():
    assert _paths([[0.0, 0.0, 0.0, float("nan"), 1.0, 0.0], [1.0, 0.0, 0.0, float("nan"), 2.0, 0.0]]) == [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    ]


def test_single_sample_episode_yields_one_point_paths():
    assert _paths([[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]) == [[[4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0]]]
