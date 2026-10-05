import math

from arena_evaluation.processing.mcap_reader import nearest_return


def test_nearest_return_is_closest_valid_range():
    assert nearest_return([3.0, 1.2, 5.0], 0.1, 10.0) == 1.2


def test_nearest_return_skips_invalid_readings():
    assert nearest_return([0.05, math.nan, math.inf, 2.5], 0.1, 10.0) == 2.5


def test_nearest_return_without_detection_is_range_max():
    assert nearest_return([math.inf, 10.0, math.nan], 0.1, 10.0) == 10.0
