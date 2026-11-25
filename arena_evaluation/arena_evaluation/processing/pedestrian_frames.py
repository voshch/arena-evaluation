"""Pedestrian frames of an episode: peak counts in windows and per-robot-time pedestrian states."""

from __future__ import annotations

import dataclasses
from typing import Final

import numpy as np
import polars as pl

from arena_evaluation.processing.alignment import (
    TRACK_ALIGN_TOLERANCE_NS,
    asof_join,
)

NS_PER_S: Final[float] = 1e9

FRAME_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "frame_ix": pl.Int64,
        "t_ns": pl.Int64,
        "dt_s": pl.Float64,
        "robot_x_m": pl.Float64,
        "robot_y_m": pl.Float64,
        "robot_yaw_rad": pl.Float64,
        "robot_vx_mps": pl.Float64,
        "robot_vy_mps": pl.Float64,
        "robot_speed_mps": pl.Float64,
    }
)

PEDESTRIAN_FRAME_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "frame_ix": pl.Int64,
        "t_ns": pl.Int64,
        "ped_id": pl.String,
        "ped_x_m": pl.Float64,
        "ped_y_m": pl.Float64,
        "ped_yaw_rad": pl.Float64,
        "ped_vx_mps": pl.Float64,
        "ped_vy_mps": pl.Float64,
        "robot_yaw_rad": pl.Float64,
        "dx_m": pl.Float64,
        "dy_m": pl.Float64,
        "distance_m": pl.Float64,
        "rel_vx_mps": pl.Float64,
        "rel_vy_mps": pl.Float64,
        "rel_speed_mps": pl.Float64,
    }
)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class FrameGeometry:
    """Robot state sampled onto the pedestrian frame axis, and the frame x pedestrian relative geometry."""

    frames: pl.DataFrame
    pedestrians: pl.DataFrame

    @property
    def observed(self) -> bool:
        """Whether at least one frame resolved a robot pose against its pedestrians."""
        return self.frames.filter(pl.col("robot_x_m").is_not_null()).height > 0


def _empty_geometry() -> FrameGeometry:
    return FrameGeometry(
        frames=pl.DataFrame(schema=FRAME_SCHEMA),
        pedestrians=pl.DataFrame(schema=PEDESTRIAN_FRAME_SCHEMA),
    )


def _robot_track(robot_tracks: pl.DataFrame) -> pl.DataFrame:
    """Robot pose and finite-difference velocity on the track's own time axis."""
    step_s = (pl.col("robot_t_ns") - pl.col("robot_t_ns").shift(1)).cast(pl.Float64) / NS_PER_S
    safe_step_s = pl.when(step_s > 0.0).then(step_s).otherwise(1e-6)
    return (
        robot_tracks.select(
            robot_t_ns=pl.col("t_ns").cast(pl.Int64),
            robot_x_m=pl.col("x").cast(pl.Float64),
            robot_y_m=pl.col("y").cast(pl.Float64),
            robot_yaw_rad=pl.col("yaw").cast(pl.Float64),
        )
        .drop_nulls(["robot_t_ns", "robot_x_m", "robot_y_m"])
        .sort("robot_t_ns")
        .with_columns(
            robot_vx_mps=((pl.col("robot_x_m") - pl.col("robot_x_m").shift(1)) / safe_step_s).fill_null(0.0),
            robot_vy_mps=((pl.col("robot_y_m") - pl.col("robot_y_m").shift(1)) / safe_step_s).fill_null(0.0),
        )
        .with_columns(robot_speed_mps=(pl.col("robot_vx_mps").pow(2) + pl.col("robot_vy_mps").pow(2)).sqrt())
    )


def _frames(peds: pl.DataFrame) -> pl.DataFrame:
    """One row per distinct pedestrian frame timestamp, in time order, with its integration step."""
    step_s = (pl.col("t_ns").shift(-1) - pl.col("t_ns")).cast(pl.Float64) / NS_PER_S
    return peds.select(pl.col("t_ns").cast(pl.Int64)).unique().sort("t_ns").with_row_index("frame_ix").with_columns(dt_s=pl.when(step_s.is_null() | (step_s == 0.0)).then(1e-6).otherwise(step_s))


def frame_geometry(
    peds: pl.DataFrame | None,
    robot_tracks: pl.DataFrame | None,
    *,
    tolerance_ns: int = TRACK_ALIGN_TOLERANCE_NS,
) -> FrameGeometry:
    """Sample the robot track onto the pedestrian frame axis and attach every pedestrian's relative geometry."""
    if peds is None or robot_tracks is None or peds.height == 0 or robot_tracks.height == 0:
        return _empty_geometry()
    sampled = asof_join(
        _frames(peds),
        _robot_track(robot_tracks),
        left_on="t_ns",
        right_on="robot_t_ns",
        tolerance_ns=tolerance_ns,
    )
    pedestrians = (
        peds.select(
            t_ns=pl.col("t_ns").cast(pl.Int64),
            ped_id=pl.col("ped_id").cast(pl.String),
            ped_x_m=pl.col("x").cast(pl.Float64),
            ped_y_m=pl.col("y").cast(pl.Float64),
            ped_yaw_rad=pl.col("yaw").cast(pl.Float64),
            ped_vx_mps=pl.col("vx").cast(pl.Float64),
            ped_vy_mps=pl.col("vy").cast(pl.Float64),
        )
        .join(sampled, on="t_ns", how="inner")
        .with_columns(
            dx_m=pl.col("ped_x_m") - pl.col("robot_x_m"),
            dy_m=pl.col("ped_y_m") - pl.col("robot_y_m"),
            rel_vx_mps=pl.col("ped_vx_mps") - pl.col("robot_vx_mps"),
            rel_vy_mps=pl.col("ped_vy_mps") - pl.col("robot_vy_mps"),
        )
        .with_columns(
            distance_m=(pl.col("dx_m").pow(2) + pl.col("dy_m").pow(2)).sqrt(),
            rel_speed_mps=(pl.col("rel_vx_mps").pow(2) + pl.col("rel_vy_mps").pow(2)).sqrt(),
        )
        .select(
            "frame_ix",
            "t_ns",
            "ped_id",
            "ped_x_m",
            "ped_y_m",
            "ped_yaw_rad",
            "ped_vx_mps",
            "ped_vy_mps",
            "robot_yaw_rad",
            "dx_m",
            "dy_m",
            "distance_m",
            "rel_vx_mps",
            "rel_vy_mps",
            "rel_speed_mps",
        )
    )
    return FrameGeometry(frames=sampled.select(*FRAME_SCHEMA.names()), pedestrians=pedestrians)


def frames_of(peds: pl.DataFrame) -> list[pl.DataFrame]:
    """The pedestrians of each frame, frames in time order: the flat table partitioned by t_ns."""
    if peds is None or len(peds) == 0:
        return []
    return peds.sort("t_ns").partition_by("t_ns", maintain_order=True)


def distances_to_pedestrians(
    t_ns: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    peds: pl.DataFrame,
    tolerance_ns: int = TRACK_ALIGN_TOLERANCE_NS,
) -> pl.DataFrame:
    """One row per robot sample and pedestrian in its frame: ``row`` (the sample index), ``ped_id`` and ``distance`` to the robot at (x, y)."""
    robot = pl.DataFrame({"row": np.arange(len(t_ns)), "t_ns": t_ns.astype(np.int64), "rx": x, "ry": y})
    frames = pedestrians_at(robot["t_ns"].unique(), peds, tolerance_ns).select("t_ns", "ped_id", "x", "y")
    return robot.join(frames, on="t_ns", how="inner").select(
        "row",
        "ped_id",
        ((pl.col("x") - pl.col("rx")).pow(2) + (pl.col("y") - pl.col("ry")).pow(2)).sqrt().alias("distance"),
    )


def pedestrians_at(t_ns: pl.Series, peds: pl.DataFrame, tolerance_ns: int = TRACK_ALIGN_TOLERANCE_NS) -> pl.DataFrame:
    """Every pedestrian of the latest frame at or before each robot time, within the tolerance: one row per robot time and pedestrian."""
    frames = peds.select(pl.col("t_ns").alias("frame_ns")).unique().sort("frame_ns")
    robot_times = pl.DataFrame({"t_ns": t_ns.cast(pl.Int64)}).sort("t_ns")
    matched = asof_join(
        robot_times,
        frames,
        left_on="t_ns",
        right_on="frame_ns",
        tolerance_ns=tolerance_ns,
    )
    return matched.drop_nulls("frame_ns").join(peds, left_on="frame_ns", right_on="t_ns", how="inner")
