"""Teleport detection on a recorded robot pose track."""

from __future__ import annotations

import numpy as np
import polars as pl

MIN_JUMP_M = 0.5
MAX_SPEED_MPS = 5.0  # above any ground robot in the fleet; a reset moves the robot within one sample
MIN_SEGMENT_M = 0.2  # a piece shorter than this cannot be the episode's path
RIDE_ARRIVAL_TOLERANCE_NS = 2_000_000_000
RIDE_ARRIVALS = {
    "elevator": "just_arrived",
    "stairs": "just_crossed",
    "escalator": "just_crossed",
}


def _drop_exempt(
    jumps: np.ndarray,
    exempt_ns: np.ndarray | None,
    exempt_t_ns: np.ndarray | None,
    n: int,
) -> np.ndarray:
    """Jumps whose landing sample (log time) is not within tolerance of an exempt time."""
    if exempt_ns is None or exempt_t_ns is None or len(jumps) == 0:
        return jumps
    arrivals = np.sort(np.asarray(exempt_ns, dtype=np.int64))
    log_t = np.asarray(exempt_t_ns, dtype=np.int64)
    if len(arrivals) == 0 or len(log_t) != n:
        return jumps
    landing = log_t[jumps + 1]
    right = np.clip(np.searchsorted(arrivals, landing), 0, len(arrivals) - 1)
    left = np.clip(right - 1, 0, len(arrivals) - 1)
    distance = np.minimum(np.abs(arrivals[left] - landing), np.abs(arrivals[right] - landing))
    return jumps[distance > RIDE_ARRIVAL_TOLERANCE_NS]


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
    distance = np.hypot(np.diff(x), np.diff(y))
    teleport = distance > MIN_JUMP_M
    if t_ns is not None:
        t = np.asarray(t_ns, dtype=np.float64)
        if len(t) == len(x):
            dt = np.diff(t) / 1e9
            speed = np.divide(distance, dt, out=np.full_like(distance, np.inf), where=dt > 0)
            teleport &= speed > MAX_SPEED_MPS
    return _drop_exempt(np.flatnonzero(teleport), exempt_ns, exempt_t_ns, len(x))


def consistent_segment(
    x: np.ndarray,
    y: np.ndarray,
    t_ns: np.ndarray,
    exempt_ns: np.ndarray | None = None,
) -> tuple[int, int]:
    """[start, end) of the longest teleport-free piece of at least MIN_SEGMENT_M; the whole track when nothing is cut."""
    n = len(x)
    jumps = teleport_jumps(x, y, t_ns, exempt_ns=exempt_ns, exempt_t_ns=t_ns)
    if len(jumps) == 0:
        return 0, n
    bounds = np.concatenate(([0], jumps + 1, [n]))
    travelled = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
    lengths = travelled[bounds[1:] - 1] - travelled[bounds[:-1]]
    candidates = np.flatnonzero(lengths >= MIN_SEGMENT_M)
    if len(candidates) == 0:
        return 0, n
    best = int(candidates[np.argmax(lengths[candidates])])
    return int(bounds[best]), int(bounds[best + 1])


def ride_arrival_times(snapshot: pl.DataFrame | pl.LazyFrame | None) -> np.ndarray:
    """Log times (ns) where an elevator's `just_arrived` or a stair's `just_crossed` predicate rises."""
    if snapshot is None:
        return np.array([], dtype=np.int64)
    lazy = snapshot.lazy() if isinstance(snapshot, pl.DataFrame) else snapshot
    names = lazy.collect_schema().names()
    if not {"kind", "field_name", "value_bool", "entity_name", "t_ns"}.issubset(names):
        return np.array([], dtype=np.int64)
    arrival = pl.any_horizontal([(pl.col("kind") == kind) & (pl.col("field_name") == field) for kind, field in RIDE_ARRIVALS.items()])
    rises = lazy.filter(arrival).select("t_ns", "entity_name", "value_bool").collect().sort("entity_name", "t_ns").with_columns(high=pl.col("value_bool").fill_null(False)).with_columns(was_high=pl.col("high").shift(1, fill_value=False).over("entity_name")).filter(pl.col("high") & ~pl.col("was_high"))
    return rises.get_column("t_ns").sort().to_numpy().astype(np.int64)
