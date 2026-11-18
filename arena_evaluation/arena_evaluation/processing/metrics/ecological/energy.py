"""Electrical energy, power and efficiency of the robot over an episode, a robot run or an encounter window.

Each scope slice is reduced to one series of power, battery and speed samples, and every output is read
from that series, so each quantity is integrated once. The outputs are the static, mechanical, thermal and
total energy integrals, the battery change, the cost of transport and energy per meter, the peak-to-mean
power ratio, the standstill penalty, the kinetic and friction terms, the mean and peak power, and the
encounter surge ratio. Energies are integrated in watt-hours; JOULES_PER_WH is the only Wh-to-joule
conversion, used by the cost of transport.
"""

from __future__ import annotations

import typing
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import yaml

from arena_evaluation.storage.registry.series import SeriesEnergy
from arena_evaluation.processing.metrics.base import (
    BaseMetricCalculator,
    Direction,
    ScopeSlice,
)
from arena_evaluation.processing.metrics.common.alignment import (
    TRACK_ALIGN_TOLERANCE_NS,
    asof_values,
)
from arena_evaluation.storage.registry import MetricScope, define_metric
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

JOULES_PER_WH: Final[float] = 3600.0
NANOSECONDS_PER_SECOND: Final[float] = 1e9
STANDSTILL_SPEED_MPS: Final[float] = 0.01
MIN_STANDSTILL_BLOCK_S: Final[float] = 0.5
MIN_PATH_LENGTH_M: Final[float] = 0.1
MIN_MEAN_POWER_W: Final[float] = 0.01
GRAVITY_MPS2: Final[float] = 9.81
#: Longest NaN gap, seconds, that the series may be filled across; a longer gap stays NaN, so the
#: integral is undefined instead of inventing power, velocity or battery over the gap.
MAX_FILL_GAP_S: Final[float] = 1.0

SERIES_SCHEMA: Final[pl.Schema] = SeriesEnergy.polars_schema()

ENERGY_STATIC_WH = define_metric(
    "energy_static_wh",
    pl.Float64,
    scopes=(MetricScope.ROBOT, MetricScope.ENCOUNTER),
    unit="Wh",
    description="Static (idle) electrical energy drawn over the scope.",
    null_reason=True,
)
ENERGY_MECHANICAL_WH = define_metric(
    "energy_mechanical_wh",
    pl.Float64,
    scopes=(MetricScope.ROBOT, MetricScope.ENCOUNTER),
    unit="Wh",
    description="Mechanical energy delivered to the drivetrain over the scope: the episode at robot scope, the window at encounter scope.",
    null_reason=True,
)
ENERGY_THERMAL_WH = define_metric(
    "energy_thermal_wh",
    pl.Float64,
    scopes=(MetricScope.ROBOT, MetricScope.ENCOUNTER),
    unit="Wh",
    description="Thermal energy dissipated in the drivetrain over the scope: the episode at robot scope, the window at encounter scope.",
    null_reason=True,
)
ENERGY_TOTAL_WH = define_metric(
    "energy_total_wh",
    pl.Float64,
    scopes=(MetricScope.ROBOT, MetricScope.ENCOUNTER),
    unit="Wh",
    description="Energy the robot drew over its episode at ROBOT scope, and over the window at ENCOUNTER scope: the integral of recorded total power (D-15), not the device energy counter, which is kept only as a cross-check in a test.",
    null_reason=True,
)
BATTERY_SOC_FINAL = define_metric(
    "battery_soc_final",
    pl.Float64,
    scopes=(MetricScope.ROBOT, MetricScope.ENCOUNTER),
    unit="%",
    description="Battery state of charge at the end of the scope, as recorded: not validated against 0-100 % and not normalised to 100 at the start.",
    null_reason=True,
)
BATTERY_SOC_DROP_PCT = define_metric(
    "battery_soc_drop_pct",
    pl.Float64,
    scopes=(MetricScope.ROBOT, MetricScope.ENCOUNTER),
    unit="%",
    description="Battery state-of-charge drop over the scope, floored at zero.",
    null_reason=True,
)
SPECIFIC_COST_OF_TRANSPORT = define_metric(
    "specific_cost_of_transport",
    pl.Float64,
    scopes=(MetricScope.ROBOT, MetricScope.ENCOUNTER),
    unit="1",
    description="Energy per unit weight and distance over the scope's own path; NULL without a declared mass.",
    null_reason=True,
)
ENERGY_PER_METER = define_metric(
    "energy_per_meter",
    pl.Float64,
    scopes=(MetricScope.ENCOUNTER,),
    unit="Wh/m",
    description="Energy per metre of the window's own path.",
    null_reason=True,
)
PEAK_TO_MEAN_POWER_RATIO = define_metric(
    "peak_to_mean_power_ratio",
    pl.Float64,
    scopes=(MetricScope.ROBOT, MetricScope.ENCOUNTER),
    unit="1",
    description="Peak over mean total power of the scope, after a three-sample median filter.",
    null_reason=True,
)
STANDSTILL_ENERGY_PENALTY_WH = define_metric(
    "standstill_energy_penalty_wh",
    pl.Float64,
    scopes=(MetricScope.ROBOT, MetricScope.ENCOUNTER),
    unit="Wh",
    description="Energy drawn while standing still, in blocks of at least half a second.",
    null_reason=True,
)
KINETIC_ENERGY_DEMAND_J = define_metric(
    "kinetic_energy_demand_j",
    pl.Float64,
    scopes=(MetricScope.ROBOT, MetricScope.ENCOUNTER),
    unit="J",
    description="Sum of the absolute changes in kinetic energy over the scope; NULL without angular velocity.",
    null_reason=True,
)
FRICTION_DISSIPATION_J = define_metric(
    "friction_dissipation_j",
    pl.Float64,
    scopes=(MetricScope.ROBOT, MetricScope.ENCOUNTER),
    unit="J",
    description="Energy dissipated by rolling friction over the scope; NULL without a declared rolling-resistance coefficient.",
    null_reason=True,
)
MEAN_POWER_W = define_metric(
    "mean_power_w",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="W",
    description="Mean electrical power of the robot over the episode.",
    null_reason=True,
)
POWER_PEAK_W = define_metric(
    "power_peak_w",
    pl.Float64,
    scopes=(MetricScope.ROBOT, MetricScope.ENCOUNTER),
    unit="W",
    description="Peak total electrical power over the scope: the episode at robot scope, the window at encounter scope.",
    null_reason=True,
)
ENERGY_SURGE_RATIO = define_metric(
    "energy_surge_ratio",
    pl.Float64,
    scopes=(MetricScope.ENCOUNTER,),
    unit="1",
    description="Mean power of the encounter window over the mean power of the episode.",
    null_reason=True,
)

_METRICS: Final[tuple[typing.Any, ...]] = (
    ENERGY_STATIC_WH,
    ENERGY_MECHANICAL_WH,
    ENERGY_THERMAL_WH,
    ENERGY_TOTAL_WH,
    BATTERY_SOC_FINAL,
    BATTERY_SOC_DROP_PCT,
    SPECIFIC_COST_OF_TRANSPORT,
    ENERGY_PER_METER,
    PEAK_TO_MEAN_POWER_RATIO,
    STANDSTILL_ENERGY_PENALTY_WH,
    KINETIC_ENERGY_DEMAND_J,
    FRICTION_DISSIPATION_J,
    MEAN_POWER_W,
    POWER_PEAK_W,
    ENERGY_SURGE_RATIO,
)
_NOT_LOWER_IS_BETTER: Final[frozenset[str]] = frozenset(
    {"energy_static_wh", "battery_soc_final"}
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Series:
    """Samples of one scope on its own clock; absent sensors are all-NaN and gaps are filled."""

    time_ns: np.ndarray
    time_s: np.ndarray
    total_w: np.ndarray
    static_w: np.ndarray
    mechanical_w: np.ndarray
    thermal_w: np.ndarray
    battery_pct: np.ndarray
    velocity_mps: np.ndarray
    angular_velocity_radps: np.ndarray


class EnergyCalculator(BaseMetricCalculator):
    """Electrical energy, power, battery, efficiency and surge metrics of the robot."""

    NAME = "energy"
    CATEGORY = "ecological"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = ["path_metrics", "motion_metrics"]
    REQUIRED_TOPICS = [("robot_power", "robot_energy", "robot_odom")]
    PARAMS = ()

    OUTPUTS = _METRICS
    DIRECTIONS = {
        metric.name: Direction.LOWER
        for metric in _METRICS
        if metric.name not in _NOT_LOWER_IS_BETTER
    }
    PRIMARY_OUTPUTS = [
        "energy_total_wh",
        "energy_mechanical_wh",
        "power_peak_w",
        "specific_cost_of_transport",
        "energy_per_meter",
        "mean_power_w",
    ]
    SERIES_TABLE_STEM = "energy"

    def __init__(self, robot_params: RobotParams) -> None:
        super().__init__(robot_params)
        # None without the robot's power definition: friction is then NULL, never an invented default.
        self._rolling_resistance = _load_rolling_resistance(robot_params.model)

    @classmethod
    def required_topics_for_scope(
        cls, scope: MetricScope
    ) -> tuple[str | tuple[str, ...], ...]:
        # A window may legitimately lack an optional sensor stream; a missing stream is then per-metric
        # `no_samples`, not a row-level `topic_missing` that would block the whole encounter.
        return (
            ()
            if scope == MetricScope.ENCOUNTER
            else super().required_topics_for_scope(scope)
        )

    def calculate(
        self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        series = _episode_series(episode)
        if series is None:
            return dict.fromkeys(self.output_keys())
        span_ns = prior_results.get("time_diff")
        return self._outputs(
            series,
            path_m=prior_results.get("path_length_m"),
            duration_s=None
            if span_ns is None
            else float(span_ns) / NANOSECONDS_PER_SECOND,
            episode_mean_w=None,
        )

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        if scope_slice is None or scope_slice.scope == MetricScope.ROBOT:
            return self.calculate(episode, prior_results)
        if scope_slice.scope == MetricScope.ENCOUNTER:
            return self._window_summary(
                episode, scope_slice.full_episode, prior_results
            )
        raise ValueError(f"energy does not serve the {scope_slice.scope.value} scope")

    def series_table(
        self, episode: AlignedEpisodeBundle, run_id: str = "", robot_id: str = ""
    ) -> pl.DataFrame:
        """1NF flat series table of the power, energy and velocity samples, one row per sample."""
        series = _episode_series(episode)
        if series is None:
            return pl.DataFrame(schema=SERIES_SCHEMA)
        return (
            pl.DataFrame(
                {
                    "t_ns": series.time_ns,
                    "time_s": series.time_s,
                    "power_total_w": series.total_w,
                    "power_static_w": series.static_w,
                    "power_mechanical_w": series.mechanical_w,
                    "power_thermal_w": series.thermal_w,
                    "battery_soc_pct": series.battery_pct,
                    "velocity_linear_mps": series.velocity_mps,
                }
            )
            .with_columns(
                pl.col(
                    "time_s",
                    "power_total_w",
                    "power_static_w",
                    "power_mechanical_w",
                    "power_thermal_w",
                    "battery_soc_pct",
                    "velocity_linear_mps",
                ).fill_nan(None),
                pl.lit(run_id).alias("run_id"),
                pl.lit(int(episode.episode_id), dtype=pl.Int64).alias("episode_id"),
                pl.lit(robot_id or (episode.robot_name or "robot")).alias("robot_id"),
            )
            .select(list(SERIES_SCHEMA))
        )

    def _window_summary(
        self,
        episode: AlignedEpisodeBundle,
        full_episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        topics = self.native_topics(episode)
        series = _window_series(topics)
        if series is None:
            return dict.fromkeys(self.output_keys())
        episode_mean_w = prior_results.get("mean_power_w")
        if episode_mean_w is None:
            # A direct call has no ROBOT-scope prior; read the episode through the same series path.
            full_series = _window_series(self.native_topics(full_episode))
            episode_mean_w = (
                _mean_or_none(full_series.total_w) if full_series is not None else None
            )
        return self._outputs(
            series,
            path_m=prior_results.get("path_length_m"),
            duration_s=None,
            episode_mean_w=float(episode_mean_w)
            if episode_mean_w is not None
            else None,
        )

    def _outputs(
        self,
        series: _Series,
        *,
        path_m: float | None,
        duration_s: float | None,
        episode_mean_w: float | None,
    ) -> dict[str, typing.Any]:
        dt_s = _step_seconds(series.time_s)
        energy_wh = _integral_wh(series.total_w, dt_s)
        mechanical_wh = _integral_wh(series.mechanical_w, dt_s)
        thermal_wh = _integral_wh(series.thermal_w, dt_s)
        mean_w = _mean_or_none(series.total_w)
        friction_j = self._friction_dissipation_j(series.velocity_mps, dt_s)
        battery_final, battery_drop = _battery_soc(series.battery_pct)
        return {
            "energy_static_wh": _integral_wh(series.static_w, dt_s),
            "energy_mechanical_wh": mechanical_wh,
            "energy_thermal_wh": thermal_wh,
            "energy_total_wh": energy_wh,
            "battery_soc_final": battery_final,
            "battery_soc_drop_pct": battery_drop,
            "specific_cost_of_transport": self._cost_of_transport(energy_wh, path_m),
            "energy_per_meter": _energy_per_meter(energy_wh, path_m),
            "peak_to_mean_power_ratio": _peak_to_mean_ratio(series.total_w),
            "standstill_energy_penalty_wh": _standstill_energy_wh(
                series.velocity_mps, series.total_w, dt_s
            ),
            "kinetic_energy_demand_j": self._kinetic_energy_demand_j(
                series.velocity_mps, series.angular_velocity_radps
            ),
            "friction_dissipation_j": friction_j,
            "mean_power_w": _mean_power_w(energy_wh, duration_s, mean_w),
            "power_peak_w": _max_or_none(series.total_w),
            "energy_surge_ratio": _ratio(mean_w, episode_mean_w),
        }

    def _cost_of_transport(
        self, energy_wh: float | None, path_m: float | None
    ) -> float | None:
        mass_kg = self.robot_params.mass
        if (
            energy_wh is None
            or path_m is None
            or mass_kg <= 0.0
            or not _travels(path_m)
            or energy_wh <= 0.0
        ):
            return None
        return energy_wh * JOULES_PER_WH / (mass_kg * GRAVITY_MPS2 * path_m)

    def _kinetic_energy_demand_j(
        self, velocity_mps: np.ndarray, angular_velocity_radps: np.ndarray
    ) -> float | None:
        mass_kg = self.robot_params.mass
        if mass_kg <= 0.0 or len(velocity_mps) < 2:
            return None
        inertia_kgm2 = 0.5 * mass_kg * self.robot_params.robot_radius**2
        kinetic_j = (
            0.5 * mass_kg * velocity_mps**2
            + 0.5 * inertia_kgm2 * angular_velocity_radps**2
        )
        demand_j = float(np.sum(np.abs(np.diff(kinetic_j))))
        return demand_j if np.isfinite(demand_j) else None

    def _friction_dissipation_j(
        self, velocity_mps: np.ndarray, dt_s: np.ndarray
    ) -> float | None:
        mass_kg = self.robot_params.mass
        if self._rolling_resistance is None or mass_kg <= 0.0 or len(velocity_mps) < 2:
            return None
        return float(
            np.sum(
                self._rolling_resistance
                * mass_kg
                * GRAVITY_MPS2
                * np.abs(velocity_mps)
                * dt_s
            )
        )


def _load_rolling_resistance(model: str) -> float | None:
    try:
        from ament_index_python.packages import get_package_share_directory

        power_path = (
            Path(get_package_share_directory("arena_robots"))
            / "robots"
            / model
            / "telemetry"
            / "power.yaml"
        )
        data = yaml.safe_load(power_path.read_text())
    except (OSError, ImportError, KeyError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    power_system = data.get("power_system")
    if not isinstance(power_system, dict):
        return None
    coefficient = power_system.get("rolling_resistance_crr")
    if coefficient is None:
        return None
    try:
        return float(coefficient)
    except (TypeError, ValueError):
        return None


def _column(frame: pl.DataFrame, name: str) -> np.ndarray:
    if name not in frame.columns:
        return np.full(frame.height, np.nan)
    return frame[name].cast(pl.Float64).to_numpy()


def _asof_values(
    time_ns: np.ndarray, table: pl.DataFrame | None, column: str
) -> np.ndarray:
    if table is None or table.height == 0 or column not in table.columns:
        return np.full(len(time_ns), np.nan)
    reference = (
        table.select(pl.col("t_ns").cast(pl.Int64), pl.col(column).cast(pl.Float64))
        .drop_nulls("t_ns")
        .sort("t_ns")
    )
    return asof_values(
        reference["t_ns"].to_numpy(),
        reference[column].to_numpy(),
        time_ns,
        TRACK_ALIGN_TOLERANCE_NS,
    )


def _build_series(
    time_ns: np.ndarray,
    *,
    total_w: np.ndarray,
    static_w: np.ndarray,
    mechanical_w: np.ndarray,
    thermal_w: np.ndarray,
    battery_pct: np.ndarray,
    velocity_mps: np.ndarray,
    angular_velocity_radps: np.ndarray,
) -> _Series:
    time_s = (time_ns - time_ns.min()).astype(np.float64) / NANOSECONDS_PER_SECOND
    return _Series(
        time_ns=time_ns,
        time_s=time_s,
        total_w=_bounded_fill(total_w, time_s),
        static_w=_bounded_fill(static_w, time_s),
        mechanical_w=_bounded_fill(mechanical_w, time_s),
        thermal_w=_bounded_fill(thermal_w, time_s),
        battery_pct=_bounded_fill(battery_pct, time_s),
        velocity_mps=_bounded_fill(velocity_mps, time_s),
        angular_velocity_radps=_bounded_fill(angular_velocity_radps, time_s),
    )


def _episode_series(episode: AlignedEpisodeBundle) -> _Series | None:
    frame = episode.data
    if frame is None or frame.height == 0:
        return None
    return _build_series(
        frame["t_ns"].cast(pl.Int64).to_numpy(),
        total_w=_column(frame, "total_power_w"),
        static_w=_column(frame, "static_power_w"),
        mechanical_w=_column(frame, "total_mech_power_w"),
        thermal_w=_column(frame, "total_thermal_power_w"),
        battery_pct=_column(frame, "battery_soc_percent"),
        velocity_mps=_column(frame, "vel_linear"),
        angular_velocity_radps=_column(frame, "vel_angular"),
    )


def _window_series(topics: Mapping[str, pl.DataFrame | None]) -> _Series | None:
    power = topics.get("robot_power")
    if power is None or power.height == 0:
        return None
    power = power.sort("t_ns")
    time_ns = power["t_ns"].cast(pl.Int64).to_numpy()
    return _build_series(
        time_ns,
        total_w=_column(power, "total_power_w"),
        static_w=_column(power, "static_power_w"),
        mechanical_w=_column(power, "total_mech_power_w"),
        thermal_w=_column(power, "total_thermal_power_w"),
        battery_pct=_asof_values(
            time_ns, topics.get("robot_energy"), "battery_soc_percent"
        ),
        velocity_mps=_asof_values(time_ns, topics.get("robot_tracks"), "speed_mps"),
        angular_velocity_radps=_asof_values(
            time_ns, topics.get("robot_tracks"), "omega_z"
        ),
    )


def _step_seconds(time_s: np.ndarray) -> np.ndarray:
    steps = np.diff(time_s, prepend=time_s[0])
    steps[steps < 0.0] = 0.0
    return steps


def _bounded_fill(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """Fill a NaN gap only when the samples bracketing it are at most MAX_FILL_GAP_S apart.

    A longer gap keeps its NaN: the sample was never measured and no value is invented across it.
    An all-NaN array is returned unchanged.
    """
    missing = np.isnan(values)
    if not missing.any():
        return values
    filled = values.copy()
    finite = np.flatnonzero(~missing)
    if finite.size == 0:
        return filled

    for left, right in zip(finite[:-1], finite[1:], strict=True):
        if right - left > 1 and time_s[right] - time_s[left] <= MAX_FILL_GAP_S:
            filled[left + 1 : right] = values[left]

    first = int(finite[0])
    if first > 0 and time_s[first] - time_s[0] <= MAX_FILL_GAP_S:
        filled[:first] = values[first]

    last = int(finite[-1])
    if last < len(values) - 1 and time_s[-1] - time_s[last] <= MAX_FILL_GAP_S:
        filled[last + 1 :] = values[last]

    return filled


def _median_filter_window3(values: np.ndarray) -> np.ndarray:
    """3-sample moving median, to reject single-frame outliers."""
    length = len(values)
    if length < 3:
        return values.copy()
    filtered = np.empty_like(values)
    filtered[0] = np.median(values[:2])
    for index in range(1, length - 1):
        filtered[index] = np.median(values[index - 1 : index + 2])
    filtered[-1] = np.median(values[-2:])
    return filtered


def _integral_wh(values: np.ndarray, dt_s: np.ndarray) -> float | None:
    if np.isnan(values).any():
        # Either every sample is missing, or a gap longer than the fill bound leaves the
        # segment unmeasured: a partial integral would under-report the energy silently.
        return None
    return float(np.sum(values * dt_s) / JOULES_PER_WH)


def _mean_or_none(values: np.ndarray) -> float | None:
    return None if np.isnan(values).all() else float(np.nanmean(values))


def _max_or_none(values: np.ndarray) -> float | None:
    return None if np.isnan(values).all() else float(np.nanmax(values))


def _battery_soc(battery_pct: np.ndarray) -> tuple[float | None, float | None]:
    if (
        np.isnan(battery_pct).all()
        or not np.isfinite(battery_pct[0])
        or not np.isfinite(battery_pct[-1])
    ):
        return None, None
    return float(battery_pct[-1]), max(float(battery_pct[0] - battery_pct[-1]), 0.0)


def _ratio(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None or reference <= 0.0:
        return None
    return value / reference


def _mean_power_w(
    energy_wh: float | None, duration_s: float | None, mean_w: float | None
) -> float | None:
    if energy_wh is not None and duration_s is not None and float(duration_s) > 0.0:
        return energy_wh * JOULES_PER_WH / float(duration_s)
    return mean_w


def _travels(path_m: float) -> bool:
    return path_m > MIN_PATH_LENGTH_M


def _energy_per_meter(energy_wh: float | None, path_m: float | None) -> float | None:
    if energy_wh is None or path_m is None or not _travels(path_m) or energy_wh <= 0.0:
        return None
    return energy_wh / path_m


def _peak_to_mean_ratio(total_w: np.ndarray) -> float | None:
    valid = total_w[~np.isnan(total_w)]
    if valid.size == 0:
        return None
    filtered = _median_filter_window3(valid)
    mean_w = float(np.mean(filtered))
    if mean_w <= MIN_MEAN_POWER_W:
        return None
    return float(np.max(filtered)) / mean_w


def _standstill_energy_wh(
    velocity_mps: np.ndarray, total_w: np.ndarray, dt_s: np.ndarray
) -> float | None:
    if len(velocity_mps) == 0 or np.isnan(velocity_mps).all():
        return None
    stationary = np.abs(velocity_mps) < STANDSTILL_SPEED_MPS
    if not stationary.any():
        return 0.0
    blocks = pl.DataFrame(
        {"stationary": stationary, "power_w": total_w, "dt_s": dt_s}
    ).with_columns(
        block_id=(pl.col("stationary") != pl.col("stationary").shift(1))
        .fill_null(True)
        .cum_sum(),
    )
    billed = (
        blocks.filter(pl.col("stationary"))
        .group_by("block_id")
        .agg(
            duration_s=pl.col("dt_s").sum(),
            energy_wh=(pl.col("power_w") * pl.col("dt_s")).sum() / JOULES_PER_WH,
        )
        .filter(pl.col("duration_s") >= MIN_STANDSTILL_BLOCK_S)
    )
    if billed.height == 0:
        return 0.0
    energies = billed["energy_wh"].to_numpy()
    return float(np.sum(energies)) if np.isfinite(energies).all() else None
