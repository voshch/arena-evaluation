"""In-memory sensitivity rerun of the pedestrian encounter detector over the published layout."""

from __future__ import annotations

import pathlib
from typing import Any, Final

import polars as pl

from arena_evaluation.processing.windows import (
    ACTOR_PEDESTRIAN,
    ENCOUNTER_EDGE_DISTANCE_M,
    ENCOUNTER_MERGE_GAP_S,
    DetectorParams,
    detect_encounter_windows_lf,
    pedestrian_pairs_lf,
)
from arena_evaluation.storage.exceptions import MetricInputError
from arena_evaluation.storage.registry import NullReason
from arena_evaluation.storage.registry.facts import FACT_TABLES
from arena_evaluation.storage.registry.telemetry import (
    PedestrianAgentStates,
    PedestrianArenaPeds,
    RobotTracks,
)
from arena_evaluation.storage.star_schema import (
    episode_directories,
    episode_directory_id,
    scan_dimensions,
)

_NS_PER_S: Final[float] = 1e9
_DETECTOR_KEYS: Final[frozenset[str]] = frozenset(
    {"encounter_edge_distance_m", "encounter_merge_gap_s"}
)
_TOPIC_MISSING: Final[str] = NullReason.TOPIC_MISSING.value
_NO_SAMPLES: Final[str] = NullReason.NO_SAMPLES.value
_NOT_RECORDED: Final[str] = NullReason.NOT_RECORDED.value
_ROW_SCHEMA: Final[dict[str, pl.DataType]] = {
    "episode_id": pl.Int64,
    "n_samples": pl.Int64,
    "encounter_edge_distance_m": pl.Float64,
    "encounter_merge_gap_s": pl.Float64,
    "published_n_encounters": pl.Int64,
    "rerun_n_encounters": pl.Int64,
    "published_min_clearance_m": pl.Float64,
    "rerun_min_clearance_m": pl.Float64,
    "published_total_duration_s": pl.Float64,
    "rerun_total_duration_s": pl.Float64,
    "null_reason": pl.String,
    "is_sensitivity": pl.Boolean,
    "variant": pl.String,
}

Published = tuple[int | None, float | None, float | None]


def _read(table: type, episode_dir: pathlib.Path) -> pl.DataFrame | None:
    """One episode's rows of a table, through the table's own scan; None when it does not record the table."""
    path = episode_dir / table.relative_path()
    return table.scan(episode_dir).collect() if path.is_file() else None


def _published(episode_dir: pathlib.Path) -> Published:
    """Published pedestrian encounter count, minimum clearance and total duration of one episode."""
    encounters = _read(FACT_TABLES["encounters"], episode_dir)
    if encounters is None:
        return None, None, None
    peds = encounters.filter(pl.col("actor_2_kind") == ACTOR_PEDESTRIAN)
    if peds.is_empty():
        return 0, None, None
    clearance = peds["min_clearance_m"].min()
    return (
        peds.height,
        None if clearance is None else float(clearance),
        float(peds["duration_s"].sum()),
    )


def _episode_state(
    robots: pl.DataFrame | None,
    arena: pl.DataFrame | None,
    agent: pl.DataFrame | None,
    episode_radii: pl.DataFrame,
) -> tuple[pl.DataFrame | None, str | None]:
    """The pedestrian stream to rerun and, when the episode cannot be rerun, why."""
    if robots is None:
        return None, f"{_TOPIC_MISSING}:{RobotTracks.table_name}"
    if robots.is_empty():
        return None, _NO_SAMPLES
    pedestrian = next(
        (
            frame
            for frame in (arena, agent)
            if frame is not None and not frame.is_empty()
        ),
        None,
    )
    if pedestrian is None:
        topic_missing = f"{_TOPIC_MISSING}:{PedestrianArenaPeds.table_name}"
        return None, topic_missing if arena is None and agent is None else _NO_SAMPLES
    unrecorded = (
        robots.select("env_id", "robot_id")
        .unique()
        .join(episode_radii, on=["env_id", "robot_id"], how="anti")
    )
    if not unrecorded.is_empty():
        return None, f"{_NOT_RECORDED}:robot_radius"
    return pedestrian, None


def _rerun_counts(
    robots: pl.DataFrame,
    pedestrians: pl.DataFrame,
    episode_radii: pl.DataFrame,
    params: DetectorParams,
) -> tuple[int, float | None, float | None]:
    """Encounter count, minimum clearance and total duration of the rerun, each robot against its own radius."""
    windows: list[pl.DataFrame] = []
    for robot_rows in robots.partition_by(["env_id", "robot_id"]):
        radius = float(
            robot_rows.select("env_id", "robot_id")
            .head(1)
            .join(episode_radii, on=["env_id", "robot_id"])["robot_radius"]
            .item()
        )
        windows.append(
            detect_encounter_windows_lf(
                pedestrian_pairs_lf(robot_rows, pedestrians, radius, params), params
            ).collect()
        )
    found = pl.concat(windows, how="vertical")
    if found.is_empty():
        return 0, None, None
    durations = (found["t_end_ns"] - found["t_start_ns"]) / _NS_PER_S
    return found.height, float(found["min_clearance_m"].min()), float(durations.sum())


def rerun_sensitivity(
    benchmark_dir: pathlib.Path,
    *,
    episode_ids: list[int] | None = None,
    detector_params: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """Rerun the pedestrian encounter detector in memory with the overridden parameters; nothing is written."""
    overrides = dict(detector_params or {})
    unknown = sorted(set(overrides) - _DETECTOR_KEYS)
    if unknown:
        raise MetricInputError(
            f"unknown detector parameter(s) {unknown}; the rerun takes {sorted(_DETECTOR_KEYS)}"
        )
    params = DetectorParams(
        encounter_edge_distance_m=float(
            overrides.get("encounter_edge_distance_m", ENCOUNTER_EDGE_DISTANCE_M)
        ),
        encounter_merge_gap_s=float(
            overrides.get("encounter_merge_gap_s", ENCOUNTER_MERGE_GAP_S)
        ),
    )
    radii = (
        scan_dimensions(benchmark_dir)["dim_robots"]
        .select("episode_id", "env_id", "robot_id", "robot_radius")
        .drop_nulls("robot_radius")
        .collect()
    )
    target_ids = None if episode_ids is None else set(episode_ids)
    rows: list[dict[str, Any]] = []
    for episode_dir in episode_directories(benchmark_dir):
        episode_id = episode_directory_id(episode_dir)
        if episode_id is None or (
            target_ids is not None and episode_id not in target_ids
        ):
            continue
        episode_radii = radii.filter(pl.col("episode_id") == episode_id)
        robots = _read(RobotTracks, episode_dir)
        pedestrians, reason = _episode_state(
            robots,
            _read(PedestrianArenaPeds, episode_dir),
            _read(PedestrianAgentStates, episode_dir),
            episode_radii,
        )
        rerun = (
            _rerun_counts(robots, pedestrians, episode_radii, params)
            if reason is None and robots is not None and pedestrians is not None
            else (None, None, None)
        )
        published = _published(episode_dir)
        rows.append(
            {
                "episode_id": episode_id,
                "n_samples": None if robots is None else robots.height,
                "encounter_edge_distance_m": params.encounter_edge_distance_m,
                "encounter_merge_gap_s": params.encounter_merge_gap_s,
                "published_n_encounters": published[0],
                "rerun_n_encounters": rerun[0],
                "published_min_clearance_m": published[1],
                "rerun_min_clearance_m": rerun[1],
                "published_total_duration_s": published[2],
                "rerun_total_duration_s": rerun[2],
                "null_reason": reason,
                "is_sensitivity": True,
                "variant": "sensitivity",
            }
        )
    return pl.DataFrame(rows, schema=_ROW_SCHEMA)
