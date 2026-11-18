import numpy as np
import pytest

from arena_evaluation.processing.metrics.common.segments import polyline_length_m


def test_polyline_length_sums_the_segments():
    assert polyline_length_m(np.array([0.0, 3.0, 3.0]), np.array([0.0, 0.0, 4.0])) == pytest.approx(7.0)


@pytest.mark.parametrize("points", [0, 1])
def test_polyline_length_is_zero_below_two_points(points):
    assert polyline_length_m(np.arange(points, dtype=np.float64), np.zeros(points)) == 0.0
