"""Pedestrian deflection, velocity delay and patrol rounds, per pedestrian and across the episode."""

from __future__ import annotations

import typing
from typing import Final

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, Direction
from arena_evaluation.storage.registry import MetricScope, define_metric
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

PEDESTRIAN_MIN_POINTS: Final[int] = 5
DEFLECTION_MIN_CHORD_M: Final[float] = 0.5
NOMINAL_DESIRED_SPEED_MPS: Final[float] = 1.2
VALID_POSITION_LIMIT_M: Final[float] = 1.0e6
SUMMED_VALUE_DECIMALS: Final[int] = 3
UNDEFINED: Final[float] = float("nan")

PEDESTRIAN_GEOMETRY_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "ped_id": pl.String,
        "n_points": pl.UInt32,
        "first_t_ns": pl.Int64,
        "last_t_ns": pl.Int64,
        "path_length_m": pl.Float64,
        "chord_m": pl.Float64,
        "max_displacement_m": pl.Float64,
        "lateral_deviation_m": pl.Float64,
    }
)

PED_PATH_DEFLECTION_M = define_metric(
    "ped_path_deflection_m",
    pl.Float64,
    scopes=(MetricScope.PEDESTRIAN,),
    unit="m",
    description="Mean lateral deviation of the pedestrian's recorded path from the straight chord of the track.",
    null_reason=True,
)
PED_VELOCITY_DELAY_RATIO = define_metric(
    "ped_velocity_delay_ratio",
    pl.Float64,
    scopes=(MetricScope.PEDESTRIAN,),
    unit="1",
    description="Relative slowdown of the pedestrian's mean traveled speed against its recorded desired_speed, or the nominal 1.2 m/s when the recording carries none; null when the duration or motion is undefined.",
    null_reason=True,
)


def _track_frame(peds: pl.DataFrame) -> pl.DataFrame:
    """Finite, plausibly located pedestrian rows sorted by pedestrian and time."""
    return (
        peds.select(
            t_ns=pl.col("t_ns").cast(pl.Int64),
            ped_id=pl.col("ped_id").cast(pl.String),
            x_m=pl.col("x").cast(pl.Float64),
            y_m=pl.col("y").cast(pl.Float64),
        )
        .filter(
            pl.col("x_m").is_finite()
            & pl.col("y_m").is_finite()
            & (pl.col("x_m").abs() < VALID_POSITION_LIMIT_M)
            & (pl.col("y_m").abs() < VALID_POSITION_LIMIT_M)
        )
        .sort("ped_id", "t_ns")
    )


def pedestrian_geometry_frame(peds: pl.DataFrame | None) -> pl.DataFrame:
    """Per-pedestrian path length, straight chord, farthest displacement and mean lateral deviation."""
    if peds is None or peds.height == 0:
        return pl.DataFrame(schema=PEDESTRIAN_GEOMETRY_SCHEMA)
    track = _track_frame(peds)
    if track.height == 0:
        return pl.DataFrame(schema=PEDESTRIAN_GEOMETRY_SCHEMA)

    first_x_m = pl.col("x_m").first()
    first_y_m = pl.col("y_m").first()
    chord_x_m = pl.col("x_m").last() - first_x_m
    chord_y_m = pl.col("y_m").last() - first_y_m
    chord_m = (chord_x_m.pow(2) + chord_y_m.pow(2)).sqrt()
    lateral_m = (
        (pl.col("x_m") - first_x_m) * (chord_y_m / chord_m)
        - (pl.col("y_m") - first_y_m) * (chord_x_m / chord_m)
    ).abs()
    return (
        track.group_by("ped_id")
        .agg(
            n_points=pl.len(),
            first_t_ns=pl.col("t_ns").first(),
            last_t_ns=pl.col("t_ns").last(),
            path_length_m=(pl.col("x_m").diff().pow(2) + pl.col("y_m").diff().pow(2))
            .sqrt()
            .sum(),
            chord_m=chord_m,
            max_displacement_m=(
                (pl.col("x_m") - first_x_m).pow(2) + (pl.col("y_m") - first_y_m).pow(2)
            )
            .sqrt()
            .max(),
            lateral_deviation_m=pl.when(chord_m > DEFLECTION_MIN_CHORD_M)
            .then(lateral_m.mean())
            .otherwise(None)
            .alias("lateral_deviation_m"),
        )
        .select(
            "ped_id",
            "n_points",
            "first_t_ns",
            "last_t_ns",
            "path_length_m",
            "chord_m",
            "max_displacement_m",
            "lateral_deviation_m",
        )
    )


def desired_speed_mps(peds: pl.DataFrame) -> float | None:
    """Mean recorded desired_speed of the pedestrians, None when every recorded value is non-positive.

    The pedestrian track schema carries ``desired_speed``; a row that did not record one is null here.
    """
    values = peds["desired_speed"].drop_nulls().to_numpy().astype(np.float64)
    measured = values[np.isfinite(values) & (values > 0.0)]
    return float(np.mean(measured)) if measured.size > 0 else None


def velocity_delay_value(
    geometry: pl.DataFrame, duration_s: float | None, desired_speed: float | None = None
) -> float:
    """Relative slowdown of the mean traveled speed against the recorded (or nominal) desired speed; NaN when undefined."""
    if duration_s is None or duration_s <= 0.0:
        return UNDEFINED
    traveled = geometry.filter(pl.col("n_points") > 1)
    if traveled.height == 0:
        return UNDEFINED
    mean_distance_m = float(traveled.select(pl.col("path_length_m").mean()).item())
    mean_speed_mps = mean_distance_m / duration_s
    if mean_speed_mps <= 0.0:
        return UNDEFINED
    reference_speed_mps = (
        desired_speed
        if desired_speed is not None and desired_speed > 0.0
        else NOMINAL_DESIRED_SPEED_MPS
    )
    return max(0.0, 1.0 - mean_speed_mps / reference_speed_mps)


class PedestrianDisturbanceCalculator(BaseMetricCalculator):
    """Per-pedestrian path deflection and velocity delay ratio."""

    NAME = "pedestrian_disturbance"
    CATEGORY = "social"
    REQUIRES_PEDSIM = True
    REQUIRED_TOPICS = ["pedestrian_arena_peds"]

    OUTPUTS = (PED_PATH_DEFLECTION_M, PED_VELOCITY_DELAY_RATIO)
    PRIMARY_OUTPUTS = ["ped_path_deflection_m"]
    DIRECTIONS = {"ped_path_deflection_m": Direction.LOWER}

    @staticmethod
    def _compute_trajectory_deflection(
        actual_coords: np.ndarray, reference_coords: np.ndarray
    ) -> float:
        """Mean distance of the recorded points to their nearest reference point, NaN when either side has no finite point."""
        if len(actual_coords) == 0 or len(reference_coords) == 0:
            return UNDEFINED
        valid_actual = actual_coords[~np.isnan(actual_coords).any(axis=1)]
        valid_reference = reference_coords[~np.isnan(reference_coords).any(axis=1)]
        if len(valid_actual) == 0 or len(valid_reference) == 0:
            return UNDEFINED
        actual_m = np.ascontiguousarray(valid_actual[:, :2], dtype=np.float64)
        reference_m = np.ascontiguousarray(valid_reference[:, :2], dtype=np.float64)
        if np.isfinite(reference_m).all() and np.isfinite(actual_m).all():
            distances, _ = cKDTree(reference_m).query(actual_m, k=1)
            return float(np.mean(distances))
        dense = np.hypot(
            actual_m[:, None, 0] - reference_m[None, :, 0],
            actual_m[:, None, 1] - reference_m[None, :, 1],
        )
        return float(np.mean(np.min(dense, axis=1)))

    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: typing.Mapping[str, typing.Any],
    ) -> dict[str, typing.Any]:
        """Deflection and delay ratio of the single pedestrian of the PEDESTRIAN slice this bundle carries."""
        del prior_results
        peds = self.native_ped_frame(episode)
        if peds is None or peds.height == 0:
            return dict.fromkeys(self.output_keys())
        geometry = pedestrian_geometry_frame(peds)
        if geometry.height == 0:
            return {
                "ped_path_deflection_m": UNDEFINED,
                "ped_velocity_delay_ratio": UNDEFINED,
            }
        first_ns, last_ns = geometry.select(
            pl.col("first_t_ns").min(), pl.col("last_t_ns").max()
        ).row(0)
        measured = geometry.filter(
            (pl.col("n_points") >= PEDESTRIAN_MIN_POINTS)
            & pl.col("lateral_deviation_m").is_not_null()
        )
        return {
            "ped_path_deflection_m": float(measured["lateral_deviation_m"][0])
            if measured.height == 1
            else None,
            "ped_velocity_delay_ratio": round(
                velocity_delay_value(
                    geometry,
                    (int(last_ns) - int(first_ns)) / 1e9,
                    desired_speed_mps(peds),
                ),
                SUMMED_VALUE_DECIMALS,
            ),
        }
