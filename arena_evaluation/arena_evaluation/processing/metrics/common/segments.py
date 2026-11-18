"""Planar polyline length of a recorded track or window."""

from __future__ import annotations

import numpy as np


def polyline_length_m(x: np.ndarray, y: np.ndarray) -> float:
    """Euclidean length of a planar polyline; zero for fewer than two points."""
    if len(x) < 2:
        return 0.0
    return float(np.sum(np.hypot(np.diff(x), np.diff(y))))
