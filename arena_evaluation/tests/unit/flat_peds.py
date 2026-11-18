"""Flat pedestrian telemetry for metric tests: one row per pedestrian per frame, as the recorder writes pedestrian_tracks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import polars as pl

SCHEMA = pl.Schema(
    {
        "t_ns": pl.Int64,
        "env_id": pl.String,
        "ped_id": pl.String,
        "ped_name": pl.String,
        "x": pl.Float64,
        "y": pl.Float64,
        "z": pl.Float64,
        "yaw": pl.Float64,
        "vx": pl.Float64,
        "vy": pl.Float64,
        "vz": pl.Float64,
        "speed_mps": pl.Float64,
        "radius_m": pl.Float64,
        "desired_speed": pl.Float64,
        "policy": pl.String,
        "agent_type": pl.String,
        "animation_state": pl.Int64,
        "interaction_id": pl.Int64,
        "interaction_type": pl.Int64,
        "gait_phase": pl.Float64,
        "gait_cadence": pl.Float64,
    }
)


def flat_peds(frames: Sequence[tuple[int, Sequence[Mapping[str, object]]]]) -> pl.DataFrame:
    """Pedestrian rows of frames given as (t_ns, pedestrians).

    A pedestrian gives x and y and may give ped_id, ped_name, yaw, vx, vy, animation_state, interaction_id and interaction_type;
    a frame without pedestrians writes no rows, as the recorder does.
    """
    rows = []
    for t_ns, pedestrians in frames:
        for index, ped in enumerate(pedestrians):
            vx_raw, vy_raw = ped.get("vx", 0.0), ped.get("vy", 0.0)
            vx = None if vx_raw is None else float(vx_raw)  # pyright: ignore[reportArgumentType]
            vy = None if vy_raw is None else float(vy_raw)  # pyright: ignore[reportArgumentType]
            rows.append(
                {
                    "t_ns": t_ns,
                    "env_id": "env_0",
                    "ped_id": str(ped.get("ped_id", f"Pedestrian_{index}")),
                    "ped_name": str(ped.get("ped_name", "")),
                    "x": float(ped["x"]),  # pyright: ignore[reportArgumentType]
                    "y": float(ped["y"]),  # pyright: ignore[reportArgumentType]
                    "z": 0.0,
                    "yaw": None if ped.get("yaw") is None else float(ped["yaw"]),  # pyright: ignore[reportArgumentType]
                    "vx": vx,
                    "vy": vy,
                    "vz": 0.0,
                    "speed_mps": None if vx is None or vy is None else math.hypot(vx, vy),
                    "radius_m": None,
                    "desired_speed": None,
                    "policy": None,
                    "agent_type": None,
                    "animation_state": ped.get("animation_state", 0),  # pyright: ignore[reportArgumentType]
                    "interaction_id": ped.get("interaction_id", -1),  # pyright: ignore[reportArgumentType]
                    "interaction_type": ped.get("interaction_type", 0),  # pyright: ignore[reportArgumentType]
                    "gait_phase": None,
                    "gait_cadence": None,
                }
            )
    return pl.DataFrame(rows, schema=SCHEMA)
