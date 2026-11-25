"""Synthetic benchmark generator with planted ground-truth differences, swaps, and shares."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

import polars as pl

from arena_evaluation.processing.metrics import MetricRegistry
from arena_evaluation.storage.registry import dimensions
from arena_evaluation.storage.registry.telemetry import RobotTracks
from star_benchmark import write_benchmark


@dataclasses.dataclass(frozen=True, slots=True)
class SyntheticTruth:
    """Ground-truth parameters planted in the synthetic benchmark fixture."""

    planted_diff_metric: str
    planted_diff_a: str
    planted_diff_b: str
    planted_diff_delta: float
    null_pair_a: str
    null_pair_b: str
    swap_condition_low: str
    swap_condition_high: str
    closer_shares: dict[str, dict[str, float]]
    all_null_metric: str
    hypotheses: tuple[dict[str, str], ...] = ()


def write_synthetic_benchmark(
    directory: pathlib.Path,
    *,
    include_swap: bool = True,
) -> SyntheticTruth:
    """Write a Star Schema benchmark with planted truth, contacts, telemetry, and truth.json."""
    MetricRegistry.discover_calculators_cls()
    planners = ("planner_a", "planner_b", "planner_c", "planner_d")
    worlds = ("world_1", "world_2")
    conditions = ("blind", "visible")
    seeds = tuple(range(1, 9))

    rows: list[dict[str, Any]] = []
    contacts: list[dict[str, object]] = []
    tracks: list[dict[str, object]] = []
    ep_id = 0
    contact_id = 0

    for seed in seeds:
        for world in worlds:
            for cond in conditions:
                for planner in planners:
                    base_ttg = 12.0 + (seed % 3) * 0.2 + (0.5 if world == "world_2" else 0.0)
                    if planner == "planner_a":
                        ttg = base_ttg - 2.0
                        if include_swap and cond == "visible":
                            ttg = base_ttg + 4.0
                    elif planner == "planner_b":
                        ttg = base_ttg + 2.0
                        if include_swap and cond == "visible":
                            ttg = base_ttg - 2.0
                    elif planner in ("planner_c", "planner_d"):
                        ttg = base_ttg
                    else:
                        ttg = base_ttg

                    rows.append(
                        {
                            "run_id": "synthetic_truth",
                            "episode_id": ep_id,
                            "robot_id": "robot_0",
                            "env_id": "env_0",
                            "planner": planner,
                            "stage": "s1",
                            "world": world,
                            "scenario": "corridor",
                            "robot_model": "jackal",
                            "crowd_size": 5,
                            "seed": seed,
                            "is_reference": False,
                            "success": True,
                            "invalid_reason": None,
                            "time_to_goal": ttg,
                            "path_efficiency": 0.9 if planner == "planner_a" else 0.75,
                            "energy_total_wh": 2.0 if planner == "planner_a" else 3.0,
                            "collision_amount_wall": 0,
                            "collision_amount_static": 0,
                            "ci_max": None,
                            "null_reason": "ci_max:not_recorded",
                            "robot_blind": cond == "blind",
                        }
                    )

                    for step in range(4):
                        tracks.append(
                            {
                                "run_id": "synthetic_truth",
                                "episode_id": ep_id,
                                "robot_id": "robot_0",
                                "env_id": "env_0",
                                "t_ns": step * 500_000_000,
                                "x": float(step) * 0.8,
                                "y": 0.0,
                                "z": 0.0,
                                "yaw": 0.0,
                                "vx": 0.8,
                                "vy": 0.0,
                                "vz": 0.0,
                                "omega_z": 0.0,
                                "speed_mps": 0.8,
                                "pose_source": "truth",
                            }
                        )

                    # Unbalanced contact counts so summed-parts share != mean of episode percentages:
                    # seed 1: 1 contact (robot_closing=1.0, target_closing=0.0)
                    # seed 2: 9 contacts (robot_closing=0.0, target_closing=1.0)
                    # Summed robot share across seed 1+2 = 1 / 10 = 0.10 (whereas mean of ep % = 0.50).
                    if planner == "planner_a" and world == "world_1":
                        if seed == 1:
                            contacts.append(
                                {
                                    "run_id": "synthetic_truth",
                                    "episode_id": ep_id,
                                    "robot_id": "robot_0",
                                    "contact_id": contact_id,
                                    "collider_kind": "pedestrian",
                                    "v_close_robot_mps": 1.0,
                                    "v_close_target_mps": 0.0,
                                    "v_robot_impact_mps": 1.0,
                                    "v_target_impact_mps": 0.0,
                                    "relative_impact_speed_mps": 1.0,
                                    "freeze_clearance_m": 0.5,
                                }
                            )
                            contact_id += 1
                        elif seed == 2:
                            for _ in range(9):
                                contacts.append(
                                    {
                                        "run_id": "synthetic_truth",
                                        "episode_id": ep_id,
                                        "robot_id": "robot_0",
                                        "contact_id": contact_id,
                                        "collider_kind": "pedestrian",
                                        "v_close_robot_mps": 0.0,
                                        "v_close_target_mps": 1.0,
                                        "v_robot_impact_mps": 0.0,
                                        "v_target_impact_mps": 1.0,
                                        "relative_impact_speed_mps": 1.0,
                                        "freeze_clearance_m": 0.5,
                                    }
                                )
                                contact_id += 1

                    ep_id += 1

    write_benchmark(directory, rows, robot_tracks=tracks, contacts=contacts)

    tracks_schema = RobotTracks.polars_schema()
    for ep_num in sorted({int(r["episode_id"]) for r in rows}):
        ep_tracks = [t for t in tracks if int(t["episode_id"]) == ep_num]
        tracks_path = directory / "episodes" / f"episode_{ep_num:03d}" / "topics" / "robot_tracks.parquet"
        pl.DataFrame(
            [{k: t.get(k) for k in tracks_schema.names()} for t in ep_tracks],
            schema=tracks_schema,
        ).write_parquet(tracks_path)

    ped_schema = dimensions.DimPedestrians.polars_schema()
    ped_rows = [
        {
            **{k: None for k in ped_schema.names()},
            "run_id": "synthetic_truth",
            "episode_id": r["episode_id"],
            "env_id": "env_0",
            "ped_id": "ped_0",
            "robot_blind": r["robot_blind"],
        }
        for r in rows
    ]
    pl.DataFrame(ped_rows, schema=ped_schema).write_parquet(directory / dimensions.DimPedestrians.relative_path())

    dm_schema = dimensions.DimMetrics.polars_schema()
    dm_rows = [
        {
            "metric": "time_to_goal",
            "scope": "robots",
            "fact_table": "fact_robots",
            "dtype": "Float64",
            "unit": "s",
            "description": "Time from start to goal.",
            "direction": "lower",
            "producer": "time_to_goal",
            "null_reason": None,
        },
        {
            "metric": "energy_total_wh",
            "scope": "robots",
            "fact_table": "fact_robots",
            "dtype": "Float64",
            "unit": "Wh",
            "description": "Total robot energy consumed.",
            "direction": "lower",
            "producer": "energy",
            "null_reason": None,
        },
        {
            "metric": "path_efficiency",
            "scope": "robots",
            "fact_table": "fact_robots",
            "dtype": "Float64",
            "unit": "ratio",
            "description": "Straight-line distance over path length.",
            "direction": "higher",
            "producer": "path_efficiency",
            "null_reason": None,
        },
        {
            "metric": "ci_max",
            "scope": "robots",
            "fact_table": "fact_robots",
            "dtype": "Float64",
            "unit": "",
            "description": "Maximum collision index.",
            "direction": "lower",
            "producer": "collision_index",
            "null_reason": None,
        },
    ]
    pl.DataFrame(dm_rows, schema=dm_schema).write_parquet(directory / dimensions.DimMetrics.relative_path())

    truth = SyntheticTruth(
        planted_diff_metric="time_to_goal",
        planted_diff_a="planner_a",
        planted_diff_b="planner_b",
        planted_diff_delta=-4.0 if not include_swap else -1.0,
        null_pair_a="planner_c",
        null_pair_b="planner_d",
        swap_condition_low="blind",
        swap_condition_high="visible",
        closer_shares={"planner_a": {"robot_closing_share": 0.1, "target_closing_share": 0.9}},
        all_null_metric="ci_max",
        hypotheses=(
            {
                "id": "H_diff",
                "claim": "planner_a achieves lower time_to_goal than planner_b",
                "metric": "time_to_goal",
                "planner_a": "planner_a",
                "planner_b": "planner_b",
                "expected_verdict": "verified",
            },
            {
                "id": "H_null",
                "claim": "planner_c differs from planner_d on time_to_goal",
                "metric": "time_to_goal",
                "planner_a": "planner_c",
                "planner_b": "planner_d",
                "expected_verdict": "refuted",
            },
            {
                "id": "H_missing",
                "claim": "planner_a achieves lower ci_max than planner_b",
                "metric": "ci_max",
                "planner_a": "planner_a",
                "planner_b": "planner_b",
                "expected_verdict": "needs_work",
            },
        ),
    )
    (directory / "truth.json").write_text(json.dumps(dataclasses.asdict(truth), indent=2), encoding="utf-8")
    return truth
