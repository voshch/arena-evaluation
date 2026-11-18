"""In-memory sensitivity rerun of the pedestrian encounter detector over the published layout.

What this what-if reruns (the reduction of D-13 recorded in the FIX2 report): the encounter
detector only, for pedestrian encounters, with ``encounter_edge_distance_m`` and
``encounter_merge_gap_s`` overridden. Calculator reruns, variant tables and fleet (robot-robot)
encounters are not rerun. Each episode is read through its own table scans, the radii come from
the benchmark-root dimension, and nothing is written.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
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


def _detector_params(detector_params: Mapping[str, Any] | None) -> DetectorParams:
    """The rerun's detector parameters: the defaults, overridden by the known keys; an unknown key is refused."""
    params = dict(detector_params or {})
    unknown = sorted(set(params) - _DETECTOR_KEYS)
    if unknown:
        raise MetricInputError(
            f"unknown detector parameter(s) {unknown}; the rerun takes {sorted(_DETECTOR_KEYS)}"
        )
    return DetectorParams(
        encounter_edge_distance_m=float(
            params.get("encounter_edge_distance_m", ENCOUNTER_EDGE_DISTANCE_M)
        ),
        encounter_merge_gap_s=float(
            params.get("encounter_merge_gap_s", ENCOUNTER_MERGE_GAP_S)
        ),
    )


def _radii(benchmark_dir: pathlib.Path) -> pl.DataFrame:
    """The recorded radius of each robot of each episode, from the benchmark-root dimension table."""
    return (
        scan_dimensions(benchmark_dir)["dim_robots"]
        .select("episode_id", "env_id", "robot_id", "robot_radius")
        .drop_nulls("robot_radius")
        .collect()
    )


def _read(table: type, episode_dir: pathlib.Path) -> pl.DataFrame | None:
    """One episode's rows of a table, through the table's own scan; None when the episode does not record it."""
    if not (episode_dir / table.relative_path()).is_file():
        return None
    return table.scan(episode_dir).collect()


def _published(episode_dir: pathlib.Path) -> Published:
    """Published pedestrian encounter count, minimum clearance and total duration of one episode."""
    encounters = _read(FACT_TABLES["encounters"], episode_dir)
    if encounters is None:
        return None, None, None
    pedestrian = encounters.filter(pl.col("actor_2_kind") == ACTOR_PEDESTRIAN)
    if pedestrian.is_empty():
        return 0, None, None
    clearance = pedestrian.get_column("min_clearance_m").min()
    return (
        pedestrian.height,
        None if clearance is None else float(clearance),
        float(pedestrian.get_column("duration_s").sum()),
    )


def _pedestrian_frame(
    arena: pl.DataFrame | None, agent: pl.DataFrame | None
) -> tuple[pl.DataFrame | None, str | None]:
    """The pedestrian stream of an episode as the evaluator picks it (the arena peds, else the agent states), with its null reason."""
    present = [frame for frame in (arena, agent) if frame is not None]
    if not present:
        return None, f"{_TOPIC_MISSING}:{PedestrianArenaPeds.table_name}"
    for frame in present:
        if not frame.is_empty():
            return frame, None
    return None, _NO_SAMPLES


def _null_reason(
    robots: pl.DataFrame | None,
    pedestrian_reason: str | None,
    episode_radii: pl.DataFrame,
) -> str | None:
    """Why the episode cannot be rerun, or None when it can: every robot has its recorded radius."""
    if robots is None:
        return f"{_TOPIC_MISSING}:{RobotTracks.table_name}"
    if robots.is_empty():
        return _NO_SAMPLES
    if pedestrian_reason is not None:
        return pedestrian_reason
    unrecorded = (
        robots.select("env_id", "robot_id")
        .unique()
        .join(episode_radii, on=["env_id", "robot_id"], how="anti")
    )
    if unrecorded.is_empty():
        return None
    return f"{_NOT_RECORDED}:robot_radius"


def _robot_radius_m(robot_rows: pl.DataFrame, episode_radii: pl.DataFrame) -> float:
    """The recorded radius of the robot whose rows these are."""
    robot = robot_rows.select("env_id", "robot_id").head(1)
    return float(
        robot.join(episode_radii, on=["env_id", "robot_id"])
        .get_column("robot_radius")
        .item()
    )


def _rerun_counts(
    robots: pl.DataFrame,
    pedestrians: pl.DataFrame,
    episode_radii: pl.DataFrame,
    params: DetectorParams,
) -> tuple[int, float | None, float | None]:
    """Encounter count, minimum clearance and total duration of the rerun, each robot against its own radius."""
    windows = [
        detect_encounter_windows_lf(
            pedestrian_pairs_lf(
                robot_rows,
                pedestrians,
                _robot_radius_m(robot_rows, episode_radii),
                params,
            ),
            params,
        ).collect()
        for robot_rows in robots.partition_by(["env_id", "robot_id"])
    ]
    found = pl.concat(windows, how="vertical")
    if found.is_empty():
        return 0, None, None
    clearance = float(found.get_column("min_clearance_m").min())
    duration = float(
        (
            (found.get_column("t_end_ns") - found.get_column("t_start_ns")) / _NS_PER_S
        ).sum()
    )
    return found.height, clearance, duration


def _sensitivity_row(
    episode_id: int,
    *,
    params: DetectorParams,
    published: Published,
    n_samples: int | None,
    rerun: tuple[int | None, float | None, float | None],
    null_reason: str | None,
) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "n_samples": n_samples,
        "encounter_edge_distance_m": params.encounter_edge_distance_m,
        "encounter_merge_gap_s": params.encounter_merge_gap_s,
        "published_n_encounters": published[0],
        "rerun_n_encounters": rerun[0],
        "published_min_clearance_m": published[1],
        "rerun_min_clearance_m": rerun[1],
        "published_total_duration_s": published[2],
        "rerun_total_duration_s": rerun[2],
        "null_reason": null_reason,
        "is_sensitivity": True,
        "variant": "sensitivity",
    }


def rerun_sensitivity(
    benchmark_dir: pathlib.Path,
    *,
    episode_ids: list[int] | None = None,
    detector_params: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """Rerun the pedestrian encounter detector in memory with the overridden parameters; nothing is written."""
    params = _detector_params(detector_params)
    target_ids = None if episode_ids is None else set(episode_ids)
    radii = _radii(benchmark_dir)
    rows: list[dict[str, Any]] = []
    for episode_dir in episode_directories(benchmark_dir):
        episode_id = episode_directory_id(episode_dir)
        if episode_id is None or (
            target_ids is not None and episode_id not in target_ids
        ):
            continue
        robot_rows = _read(RobotTracks, episode_dir)
        pedestrians, pedestrian_reason = _pedestrian_frame(
            _read(PedestrianArenaPeds, episode_dir),
            _read(PedestrianAgentStates, episode_dir),
        )
        episode_radii = radii.filter(pl.col("episode_id") == episode_id)
        null_reason = _null_reason(robot_rows, pedestrian_reason, episode_radii)
        rerun: tuple[int | None, float | None, float | None] = (None, None, None)
        if null_reason is None and robot_rows is not None and pedestrians is not None:
            rerun = _rerun_counts(robot_rows, pedestrians, episode_radii, params)
        rows.append(
            _sensitivity_row(
                episode_id,
                params=params,
                published=_published(episode_dir),
                n_samples=None if robot_rows is None else robot_rows.height,
                rerun=rerun,
                null_reason=null_reason,
            )
        )
    return pl.DataFrame(rows, schema=_ROW_SCHEMA)
