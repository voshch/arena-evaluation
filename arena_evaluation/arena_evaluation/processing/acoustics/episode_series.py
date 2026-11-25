"""Aligned acoustic episode series: the telemetry streams a field plot renders."""

from __future__ import annotations

import pathlib
import typing

import polars as pl

from arena_evaluation.processing.metrics.common.alignment import (
    TRACK_ALIGN_TOLERANCE_NS,
    asof_join,
)
from arena_evaluation.storage.star_schema import scan_telemetry_tables

__all__ = ["episode_number", "load_aligned_episode"]


def episode_number(episode_id: str | int) -> int:
    """The integer number of an episode id written as ``episode_NNN``."""
    if isinstance(episode_id, int):
        return episode_id
    return int(str(episode_id).removeprefix("episode_"))


def load_aligned_episode(
    benchmark_dir: pathlib.Path | str,
    episode_id: str | int,
    *,
    tolerance_ns: int | None = TRACK_ALIGN_TOLERANCE_NS,
) -> pl.DataFrame | None:
    """Episode telemetry aligned to the robot ground-truth samples, None when no pose was recorded."""
    tables = scan_telemetry_tables(pathlib.Path(benchmark_dir))
    ep_num = episode_number(episode_id)
    tracks_table = tables.get("robot_tracks")
    if tracks_table is None:
        return None

    tracks = tracks_table.filter(pl.col("episode_id") == ep_num).select("t_ns", "x", "y").sort("t_ns").collect()
    if tracks.is_empty():
        return None

    frame = _join_acoustics(tracks, tables, ep_num, tolerance_ns)
    frame = _join_collisions(frame, tables, ep_num, tolerance_ns)
    return _join_pedestrian_positions(frame, tables, ep_num, tolerance_ns)


def _join_acoustics(
    frame: pl.DataFrame,
    tables: dict[str, pl.LazyFrame],
    ep_num: int,
    tolerance_ns: int | None,
) -> pl.DataFrame:
    """Attach the recorded robot source level, never a default, NULL beyond the tolerance."""
    acoustics_table = tables.get("robot_acoustics")
    if acoustics_table is None:
        return frame
    acoustics = acoustics_table.filter(pl.col("episode_id") == ep_num).select("t_ns", pl.col("total_level_af_dba").alias("source_dba")).sort("t_ns").collect()
    if acoustics.is_empty():
        return frame
    return asof_join(frame, acoustics, tolerance_ns=tolerance_ns)


def _join_collisions(
    frame: pl.DataFrame,
    tables: dict[str, pl.LazyFrame],
    ep_num: int,
    tolerance_ns: int | None,
) -> pl.DataFrame:
    """Attach whether a collision was recorded at the nearest sample within the tolerance."""
    collisions_table = tables.get("robot_collision_events")
    if collisions_table is None:
        return frame
    collisions = collisions_table.filter(pl.col("episode_id") == ep_num).select("t_ns", (pl.col("event_count") > 0).alias("has_collision")).sort("t_ns").collect()
    if collisions.is_empty():
        return frame
    return asof_join(frame, collisions, tolerance_ns=tolerance_ns)


def _join_pedestrian_positions(
    frame: pl.DataFrame,
    tables: dict[str, pl.LazyFrame],
    ep_num: int,
    tolerance_ns: int | None,
) -> pl.DataFrame:
    """Attach each aligned sample's pedestrian positions as ``(x, y)`` pairs, NULL beyond the tolerance."""
    peds_table = tables.get("pedestrian_arena_peds")
    if peds_table is None:
        return frame
    peds = peds_table.filter(pl.col("episode_id") == ep_num).select(["t_ns", "x", "y"]).sort("t_ns").collect()
    if peds.is_empty():
        return frame

    rows: list[dict[str, typing.Any]] = []
    for part in peds.partition_by("t_ns", maintain_order=True):
        xs = part["x"].to_numpy()
        ys = part["y"].to_numpy()
        rows.append(
            {
                "t_ns": int(part["t_ns"][0]),
                "peds_positions": [(float(x), float(y)) for x, y in zip(xs, ys, strict=True)],
            }
        )
    grouped = pl.DataFrame(rows).sort("t_ns")
    return asof_join(frame, grouped, tolerance_ns=tolerance_ns)
