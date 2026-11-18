"""In-memory sensitivity reruns over recorded telemetry without touching published files."""

from __future__ import annotations

import pathlib
from typing import Any

import polars as pl

from arena_evaluation.processing.windows import (
    ENCOUNTER_EDGE_DISTANCE_M,
    ENCOUNTER_MERGE_GAP_S,
    DetectorParams,
    detect_encounter_windows_lf,
    pedestrian_pairs_lf,
)
from arena_evaluation.storage.registry.dimensions import DIMENSION_TABLES
from arena_evaluation.storage.registry.facts import FACT_TABLES
from arena_evaluation.storage.registry.telemetry import (
    PedestrianAgentStates,
    RobotTracks,
)
from arena_evaluation.storage.star_schema import (
    episode_directories,
    episode_directory_id,
)

_NS_PER_S = 1e9


def _published_encounter_summary(
    ep_dir: pathlib.Path,
) -> tuple[int | None, float | None, float | None]:
    """Read published encounter count, min clearance, and total duration for one episode."""
    enc_path = ep_dir / FACT_TABLES["encounters"].relative_path()
    if not enc_path.is_file():
        return None, None, None
    df = pl.read_parquet(enc_path)
    if df.is_empty():
        return 0, None, None
    raw_clr = df.get_column("min_clearance_m").min()
    raw_dur = df.get_column("duration_s").sum()
    return (
        df.height,
        float(raw_clr) if raw_clr is not None else None,
        float(raw_dur) if raw_dur is not None else None,
    )


def _robot_radius_m(ep_dir: pathlib.Path) -> float | None:
    """Read ``robot_radius`` from episode ``dim_robots.parquet``, or None when absent."""
    robots_path = ep_dir / DIMENSION_TABLES["dim_robots"].relative_path()
    if not robots_path.is_file():
        return None
    df = pl.read_parquet(robots_path)
    if df.is_empty():
        return None
    val = df.get_column("robot_radius").drop_nulls().first()
    return float(val) if val is not None else None


def _sensitivity_row(
    ep_id: int,
    *,
    n_samples: int | None,
    edge_dist: float,
    merge_gap: float,
    pub: tuple[int | None, float | None, float | None],
    rerun_n: int | None,
    rerun_clr: float | None,
    rerun_dur: float | None,
    null_reason: str | None,
) -> dict[str, Any]:
    return {
        "episode_id": ep_id,
        "n_samples": n_samples,
        "encounter_edge_distance_m": edge_dist,
        "encounter_merge_gap_s": merge_gap,
        "published_n_encounters": pub[0],
        "rerun_n_encounters": rerun_n,
        "published_min_clearance_m": pub[1],
        "rerun_min_clearance_m": rerun_clr,
        "published_total_duration_s": pub[2],
        "rerun_total_duration_s": rerun_dur,
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
    """Rerun L2 encounter detection in memory with overridden ``detector_params`` without modifying disk files."""
    params = detector_params or {}
    edge_dist = float(
        params.get(
            "encounter_edge_distance_m",
            params.get("encounter_distance_m", ENCOUNTER_EDGE_DISTANCE_M),
        )
    )
    merge_gap = float(params.get("encounter_merge_gap_s", ENCOUNTER_MERGE_GAP_S))
    det_params = DetectorParams(
        encounter_edge_distance_m=edge_dist,
        encounter_merge_gap_s=merge_gap,
    )
    target_ids = set(episode_ids) if episode_ids is not None else None

    rows: list[dict[str, Any]] = []
    for ep_dir in episode_directories(benchmark_dir):
        ep_id = episode_directory_id(ep_dir)
        if target_ids is not None and ep_id not in target_ids:
            continue
        pub = _published_encounter_summary(ep_dir)
        tracks_path = ep_dir / RobotTracks.relative_path()
        if not tracks_path.is_file():
            rows.append(
                _sensitivity_row(
                    ep_id,
                    n_samples=None,
                    edge_dist=edge_dist,
                    merge_gap=merge_gap,
                    pub=pub,
                    rerun_n=None,
                    rerun_clr=None,
                    rerun_dur=None,
                    null_reason="telemetry:no_robot_tracks",
                )
            )
            continue

        robot_df = pl.read_parquet(tracks_path)
        if robot_df.is_empty():
            rows.append(
                _sensitivity_row(
                    ep_id,
                    n_samples=0,
                    edge_dist=edge_dist,
                    merge_gap=merge_gap,
                    pub=pub,
                    rerun_n=None,
                    rerun_clr=None,
                    rerun_dur=None,
                    null_reason="telemetry:empty_robot_tracks",
                )
            )
            continue

        ped_path = ep_dir / PedestrianAgentStates.relative_path()
        if not ped_path.is_file():
            rows.append(
                _sensitivity_row(
                    ep_id,
                    n_samples=robot_df.height,
                    edge_dist=edge_dist,
                    merge_gap=merge_gap,
                    pub=pub,
                    rerun_n=None,
                    rerun_clr=None,
                    rerun_dur=None,
                    null_reason="telemetry:no_pedestrian_states",
                )
            )
            continue

        ped_df = pl.read_parquet(ped_path)
        if ped_df.is_empty():
            rows.append(
                _sensitivity_row(
                    ep_id,
                    n_samples=robot_df.height,
                    edge_dist=edge_dist,
                    merge_gap=merge_gap,
                    pub=pub,
                    rerun_n=None,
                    rerun_clr=None,
                    rerun_dur=None,
                    null_reason="telemetry:empty_pedestrian_states",
                )
            )
            continue

        robot_radius = _robot_radius_m(ep_dir)
        if robot_radius is None:
            rows.append(
                _sensitivity_row(
                    ep_id,
                    n_samples=robot_df.height,
                    edge_dist=edge_dist,
                    merge_gap=merge_gap,
                    pub=pub,
                    rerun_n=None,
                    rerun_clr=None,
                    rerun_dur=None,
                    null_reason="telemetry:no_robot_radius",
                )
            )
            continue

        pairs = pedestrian_pairs_lf(robot_df, ped_df, robot_radius, det_params)
        windows = detect_encounter_windows_lf(pairs, det_params).collect()
        rerun_n = windows.height
        if rerun_n == 0:
            rerun_clr = None
            rerun_dur = None
        else:
            raw_clr = windows.get_column("min_clearance_m").min()
            rerun_clr = float(raw_clr) if raw_clr is not None else None
            raw_dur = (
                (windows.get_column("t_end_ns") - windows.get_column("t_start_ns"))
                / _NS_PER_S
            ).sum()
            rerun_dur = float(raw_dur) if raw_dur is not None else None

        rows.append(
            _sensitivity_row(
                ep_id,
                n_samples=robot_df.height,
                edge_dist=edge_dist,
                merge_gap=merge_gap,
                pub=pub,
                rerun_n=rerun_n,
                rerun_clr=rerun_clr,
                rerun_dur=rerun_dur,
                null_reason=None,
            )
        )
    return pl.DataFrame(rows)
