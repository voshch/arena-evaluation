"""Teleport detection on a recorded robot pose track.

The longest-consistent-segment rule (``pipeline._resolve_odom_frame`` and
``MetricCalculator.resolve_robot_pose``) splits a track wherever the robot
"jumps" and keeps the longest piece, so a reset teleport caught in the
recording does not stretch the path across the map. A jump used to be any
step above 0.5 m between consecutive samples. That is fine on 20 Hz odometry
and wrong on the ground-truth stream: the gz pose bridge publishes at a few
Hz under load, with gaps of 1-2 s, so a robot driving at 1 m/s puts
consecutive samples 0.5-2 m apart and a healthy episode is chopped into
fragments, of which only one survives (observed: 24 m corridor transits
reported as 4-13 m paths).

A teleport is a step the platform cannot have driven: distance over the
sample interval above ``MAX_SPEED_MPS``. The interval must be sim time (the
message header stamp, ``stamp_ns`` / ``stamp_ns_gt``), never the recorder's
log time: under lockstep the wall clock runs at an arbitrary real-time
factor, and a reset happens with the sim paused, so in wall time it is a
slow drift rather than a jump. Without sim stamps the distance rule stands.
"""

from __future__ import annotations

import numpy as np

MIN_JUMP_M = 0.5
MAX_SPEED_MPS = 5.0  # above any ground robot in the fleet; a reset moves the robot within one sample


def teleport_jumps(x: np.ndarray, y: np.ndarray, time_ns: np.ndarray | None = None) -> np.ndarray:
    """Indices ``i`` where the step from sample ``i`` to ``i+1`` is a teleport."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return np.array([], dtype=np.int64)
    dists = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    big = dists > MIN_JUMP_M
    if time_ns is None:
        return np.where(big)[0]
    t = np.asarray(time_ns, dtype=np.float64)
    if len(t) != len(x):
        return np.where(big)[0]
    dt = np.diff(t) / 1e9
    speed = np.full_like(dists, np.inf)
    ok = dt > 0
    speed[ok] = dists[ok] / dt[ok]
    return np.where(big & (speed > MAX_SPEED_MPS))[0]
