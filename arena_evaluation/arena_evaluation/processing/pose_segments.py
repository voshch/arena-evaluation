"""Teleport detection on a recorded robot pose track.

The longest-consistent-segment rule (``pipeline._consistent_pose_segment`` and
``MetricCalculator.resolve_robot_pose``) splits a track wherever the robot
"jumps" and keeps the longest piece, so a reset teleport caught in the
recording does not stretch the path across the map. A jump used to be any
step above 0.5 m between consecutive samples. That is fine on 20 Hz odometry
and wrong on the ground-truth stream: the gz pose bridge publishes at a few
Hz under load, with gaps of 1-2 s, so a robot driving at 1 m/s puts
consecutive samples 0.5-2 m apart and a healthy episode is chopped into
fragments, of which only one survives (observed: 24 m corridor transits
reported as 4-13 m paths).

A teleport is a step the platform cannot have driven: distance over the sample
interval above ``MAX_SPEED_MPS``. The interval is the recorded log time
(``t_ns``) of the flat telemetry. Sim-time header stamps are not part of that
contract, so under lockstep a reset that happens with the sim paused can read
as slow drift rather than a jump.

An elevator ride or a stair crossing is a teleport too, but one that belongs to the episode. A
jump landing within ``RIDE_ARRIVAL_TOLERANCE_NS`` of a ride arrival (`just_arrived` rising on an
`elevator`, `just_crossed` rising on a `stairs` or `escalator` entity in the semantic snapshot,
recorder log time) is exempt, so neither cuts the episode to one floor.
"""

from __future__ import annotations

import typing

import numpy as np

if typing.TYPE_CHECKING:
    import polars as pl

MIN_JUMP_M = 0.5
MAX_SPEED_MPS = 5.0  # above any ground robot in the fleet; a reset moves the robot within one sample
MIN_SEGMENT_M = 0.2  # a piece shorter than this cannot be the episode's path


RIDE_ARRIVAL_TOLERANCE_NS = 2_000_000_000
RIDE_ARRIVALS = {"elevator": "just_arrived", "stairs": "just_crossed", "escalator": "just_crossed"}


def _drop_exempt(jumps: np.ndarray, exempt_ns: np.ndarray | None, exempt_t_ns: np.ndarray | None, n: int) -> np.ndarray:
    """Jumps whose landing sample (log time) is not within tolerance of an exempt time."""
    if exempt_ns is None or exempt_t_ns is None or len(jumps) == 0:
        return jumps
    exempt = np.sort(np.asarray(exempt_ns, dtype=np.int64))
    log_t = np.asarray(exempt_t_ns, dtype=np.int64)
    if len(exempt) == 0 or len(log_t) != n:
        return jumps
    keep = []
    for i in jumps:
        t = log_t[i + 1]
        k = int(np.searchsorted(exempt, t))
        near = min((abs(int(exempt[j]) - int(t)) for j in (k - 1, k) if 0 <= j < len(exempt)), default=None)
        if near is None or near > RIDE_ARRIVAL_TOLERANCE_NS:
            keep.append(i)
    return np.asarray(keep, dtype=np.int64)


def teleport_jumps(
    x: np.ndarray,
    y: np.ndarray,
    t_ns: np.ndarray | None = None,
    *,
    exempt_ns: np.ndarray | None = None,
    exempt_t_ns: np.ndarray | None = None,
) -> np.ndarray:
    """Indices ``i`` where the step from sample ``i`` to ``i+1`` is a teleport."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return np.array([], dtype=np.int64)
    dists = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    big = dists > MIN_JUMP_M
    if t_ns is None:
        return _drop_exempt(np.where(big)[0], exempt_ns, exempt_t_ns, len(x))
    t = np.asarray(t_ns, dtype=np.float64)
    if len(t) != len(x):
        return _drop_exempt(np.where(big)[0], exempt_ns, exempt_t_ns, len(x))
    dt = np.diff(t) / 1e9
    speed = np.full_like(dists, np.inf)
    ok = dt > 0
    speed[ok] = dists[ok] / dt[ok]
    return _drop_exempt(np.where(big & (speed > MAX_SPEED_MPS))[0], exempt_ns, exempt_t_ns, len(x))


def consistent_segment(x: np.ndarray, y: np.ndarray, t_ns: np.ndarray, exempt_ns: np.ndarray | None = None) -> tuple[int, int]:
    """[start, end) of the longest teleport-free piece of at least MIN_SEGMENT_M; the whole track when nothing is cut."""
    n = len(x)
    jumps = teleport_jumps(x, y, t_ns, exempt_ns=exempt_ns, exempt_t_ns=t_ns)
    if len(jumps) == 0:
        return 0, n
    bounds = np.concatenate(([0], np.asarray(jumps) + 1, [n]))
    best: tuple[int, int] | None = None
    best_len = -1.0
    for start, end in zip(bounds[:-1].tolist(), bounds[1:].tolist(), strict=True):
        seg_x, seg_y = x[start:end], y[start:end]
        seg_len = 0.0 if len(seg_x) < 2 else float(np.sum(np.hypot(np.diff(seg_x), np.diff(seg_y))))
        if seg_len >= MIN_SEGMENT_M and seg_len > best_len:
            best, best_len = (start, end), seg_len
    return best if best is not None else (0, n)


def ride_arrival_times(snapshot: pl.DataFrame | pl.LazyFrame | None) -> np.ndarray:
    """Log times (ns) where an elevator's `just_arrived` or a stair's `just_crossed` predicate rises."""
    import polars as pl

    if snapshot is None:
        return np.array([], dtype=np.int64)
    lazy = snapshot.lazy() if isinstance(snapshot, pl.DataFrame) else snapshot
    names = lazy.collect_schema().names()
    if not {"kind", "field_name", "value_bool", "entity_name", "t_ns"}.issubset(names):
        return np.array([], dtype=np.int64)
    arrival = pl.any_horizontal([(pl.col("kind") == kind) & (pl.col("field_name") == field) for kind, field in RIDE_ARRIVALS.items()])
    rows = lazy.filter(arrival).select("t_ns", "entity_name", "value_bool").collect()
    rising = (
        rows.sort("entity_name", "t_ns")
        .with_columns(high=pl.col("value_bool").fill_null(False))
        .with_columns(was_high=pl.col("high").shift(1, fill_value=False).over("entity_name"))
        .filter(pl.col("high") & ~pl.col("was_high"))
    )
    return rising.get_column("t_ns").sort().to_numpy().astype(np.int64)
