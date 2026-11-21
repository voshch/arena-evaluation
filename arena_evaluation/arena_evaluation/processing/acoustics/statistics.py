"""Level statistics shared by the robot-level and the pedestrian-level acoustic metrics."""

from __future__ import annotations

import numpy as np

from arena_evaluation.processing.acoustics.constants import NANOSECONDS_PER_SECOND

# A level below the human hearing threshold carries no energy worth publishing.
MIN_DBA: float = 1.0
# Guarantees a positive duration for zero-spaced samples.
MIN_DT_S: float = 1e-6


def frame_durations_s(t_ns: np.ndarray) -> np.ndarray:
    """Duration each sample stands for: the gap to the next sample, the last gap repeated."""
    times = np.asarray(t_ns, dtype=np.int64)
    if len(times) < 2:
        return np.full(len(times), MIN_DT_S, dtype=np.float64)
    gaps = np.diff(times) / NANOSECONDS_PER_SECOND
    durations = np.append(gaps, gaps[-1])
    return np.where(durations > 0.0, durations, MIN_DT_S)


def linear_energy(levels_dba: np.ndarray) -> np.ndarray:
    """Linear power ratio of A-weighted levels, floored at MIN_DBA."""
    return np.power(10.0, np.maximum(np.asarray(levels_dba, dtype=np.float64), MIN_DBA) / 10.0)


def energy_equivalent_level_dba(levels_dba: np.ndarray, durations_s: np.ndarray) -> float | None:
    """Duration-weighted energy-equivalent level of the samples, None without a positive duration."""
    levels = np.asarray(levels_dba, dtype=np.float64)
    durations = np.asarray(durations_s, dtype=np.float64)
    usable = np.isfinite(levels) & np.isfinite(durations) & (durations > 0.0)
    if not np.any(usable):
        return None
    total_duration_s = float(np.sum(durations[usable]))
    energy = float(np.sum(linear_energy(levels[usable]) * durations[usable]))
    if total_duration_s <= 0.0 or energy <= 0.0:
        return None
    level = 10.0 * np.log10(energy / total_duration_s)
    return None if (np.isnan(level) or np.isinf(level)) else float(level)


def rise_rates_dba_per_s(levels_dba: np.ndarray, t_ns: np.ndarray, *, gate_dba: float | None = None) -> np.ndarray:
    """Signed level change rates (dBA/s) between consecutive samples a positive time apart.

    A repeated timestamp carries no rate: it is dropped, never divided by a floor duration.
    `gate_dba` keeps only the pairs whose later sample is at or above that level.
    """
    levels = np.asarray(levels_dba, dtype=np.float64)
    times = np.asarray(t_ns, dtype=np.int64)
    if len(levels) < 2 or len(times) != len(levels):
        return np.empty(0, dtype=np.float64)
    gaps = np.diff(times) / NANOSECONDS_PER_SECOND
    usable = gaps > 0.0
    if gate_dba is not None:
        usable &= levels[1:] >= gate_dba
    if not np.any(usable):
        return np.empty(0, dtype=np.float64)
    rates = np.diff(levels)[usable] / gaps[usable]
    return rates[np.isfinite(rates)]


def max_positive_rise_dba_per_s(rates: np.ndarray) -> float:
    """The rise a metric publishes: the largest positive rate, 0.0 when the level never rose."""
    usable = np.asarray(rates, dtype=np.float64)
    if len(usable) == 0:
        return 0.0
    return max(0.0, float(np.max(usable)))
