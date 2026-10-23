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

An elevator ride is a teleport too, but one that belongs to the episode. A jump landing within
``ELEVATOR_ARRIVAL_TOLERANCE_NS`` of an elevator arrival (`just_arrived` rising in the semantic
snapshot, recorder log time) is exempt, so a ride does not cut the episode to one floor.
"""

from __future__ import annotations

import typing

import numpy as np

if typing.TYPE_CHECKING:
    import polars as pl

MIN_JUMP_M = 0.5
MAX_SPEED_MPS = 5.0  # above any ground robot in the fleet; a reset moves the robot within one sample


ELEVATOR_ARRIVAL_TOLERANCE_NS = 2_000_000_000


def _drop_exempt(jumps: np.ndarray, exempt_ns: np.ndarray | None, exempt_time_ns: np.ndarray | None, n: int) -> np.ndarray:
    """Jumps whose landing sample (log time) is not within tolerance of an exempt time."""
    if exempt_ns is None or exempt_time_ns is None or len(jumps) == 0:
        return jumps
    exempt = np.sort(np.asarray(exempt_ns, dtype=np.int64))
    log_t = np.asarray(exempt_time_ns, dtype=np.int64)
    if len(exempt) == 0 or len(log_t) != n:
        return jumps
    keep = []
    for i in jumps:
        t = log_t[i + 1]
        k = int(np.searchsorted(exempt, t))
        near = min((abs(int(exempt[j]) - int(t)) for j in (k - 1, k) if 0 <= j < len(exempt)), default=None)
        if near is None or near > ELEVATOR_ARRIVAL_TOLERANCE_NS:
            keep.append(i)
    return np.asarray(keep, dtype=np.int64)


def teleport_jumps(
    x: np.ndarray,
    y: np.ndarray,
    time_ns: np.ndarray | None = None,
    *,
    exempt_ns: np.ndarray | None = None,
    exempt_time_ns: np.ndarray | None = None,
) -> np.ndarray:
    """Indices ``i`` where the step from sample ``i`` to ``i+1`` is a teleport."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return np.array([], dtype=np.int64)
    dists = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    big = dists > MIN_JUMP_M
    if time_ns is None:
        return _drop_exempt(np.where(big)[0], exempt_ns, exempt_time_ns, len(x))
    t = np.asarray(time_ns, dtype=np.float64)
    if len(t) != len(x):
        return _drop_exempt(np.where(big)[0], exempt_ns, exempt_time_ns, len(x))
    dt = np.diff(t) / 1e9
    speed = np.full_like(dists, np.inf)
    ok = dt > 0
    speed[ok] = dists[ok] / dt[ok]
    return _drop_exempt(np.where(big & (speed > MAX_SPEED_MPS))[0], exempt_ns, exempt_time_ns, len(x))


def elevator_arrival_times(snapshot: pl.DataFrame | pl.LazyFrame | None) -> np.ndarray:
    """Log times (ns) where an elevator's `just_arrived` predicate rises."""
    import polars as pl

    if snapshot is None:
        return np.array([], dtype=np.int64)
    lazy = snapshot.lazy() if isinstance(snapshot, pl.DataFrame) else snapshot
    names = lazy.collect_schema().names()
    if not {"kind", "field", "value_bool", "entity", "time_ns"}.issubset(names):
        return np.array([], dtype=np.int64)
    rows = lazy.filter((pl.col("kind") == "elevator") & (pl.col("field") == "just_arrived")).select("time_ns", "entity", "value_bool").collect()
    times: list[int] = []
    for _entity, group in rows.group_by("entity"):
        prev = False
        for t, value in group.sort("time_ns").select("time_ns", "value_bool").iter_rows():
            value = bool(value)
            if value and not prev:
                times.append(int(t))
            prev = value
    return np.asarray(sorted(times), dtype=np.int64)
