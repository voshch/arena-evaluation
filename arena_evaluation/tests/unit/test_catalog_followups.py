"""Catalog follow-ups: other-robot contacts, ranking-valid scorecard, crowd condition, and episode provenance."""

from __future__ import annotations

import pathlib

import polars as pl

from arena_evaluation.storage.registry import (
    FACT_TABLES,
    FactTable,
    NullReason,
    dimensions,
)
from arena_evaluation.storage.star_schema import write_benchmark_dimensions

RUN_ID = "bench-catalog-followups"
ROBOT_ID = "env_0_jackal_0"


def _frame(rows: list[dict[str, object]], schema: pl.Schema) -> pl.DataFrame:
    return pl.DataFrame(
        [{name: row.get(name) for name in schema.names()} for row in rows],
        schema=schema,
    )


def _write_benchmark(
    root: pathlib.Path, *, include_pedestrians: bool = True
) -> pathlib.Path:
    root.mkdir(parents=True)
    dim_ep_schema = dimensions.DimEpisodes.polars_schema()
    dim_rob_schema = dimensions.DimRobots.polars_schema()
    dim_ped_schema = dimensions.DimPedestrians.polars_schema()
    fact_ep_schema = FACT_TABLES[FactTable.EPISODES].polars_schema()
    fact_rob_schema = FACT_TABLES[FactTable.ROBOTS].polars_schema()
    fact_con_schema = FACT_TABLES[FactTable.CONTACTS].polars_schema()

    dims = {
        name: pl.DataFrame(schema=tbl.polars_schema())
        for name, tbl in dimensions.DIMENSION_TABLES.items()
    }
    dims["dim_episodes"] = _frame(
        [
            {
                "run_id": RUN_ID,
                "episode_id": episode_id,
                "env_id": "env_0",
                "robot_id": ROBOT_ID,
                "benchmark_id": RUN_ID,
                "suite_name": "fixture",
                "contest_name": "contest-ignored-by-local-planner",
                "stage": "s0",
                "world": "map_empty",
                "scenario": "s0",
                "planner": "dwb-standard",
                "inter_planner": "none",
                "dispatcher": "none",
                "ped_policy_default": "sfm",
                "crowd_size": 2,
                "seed": 7,
                "is_reference": False,
                "duration_s": 10.0,
                "outcome_state": 2,
                "outcome_info": "finished",
                "null_reason": None
                if include_pedestrians
                else f"robot_awareness:{NullReason.NOT_RECORDED}",
            }
            for episode_id in (0, 1, 2)
        ],
        dim_ep_schema,
    )
    dims["dim_robots"] = _frame(
        [
            {
                "run_id": RUN_ID,
                "episode_id": episode_id,
                "env_id": "env_0",
                "robot_id": ROBOT_ID,
                "robot_model": "jackal",
                "planner": "dwb-standard",
                "inter_planner": "none",
                "dispatcher": "none",
            }
            for episode_id in (0, 1, 2)
        ],
        dim_rob_schema,
    )
    if include_pedestrians:
        dims["dim_pedestrians"] = _frame(
            [
                {
                    "run_id": RUN_ID,
                    "episode_id": episode_id,
                    "env_id": "env_0",
                    "ped_id": "ped_0",
                    "robot_awareness": 1.0,
                }
                for episode_id in (0, 1, 2)
            ],
            dim_ped_schema,
        )
    write_benchmark_dimensions(dims, root)

    metrics = root / "episodes" / "episode_000" / "metrics"
    metrics.mkdir(parents=True)
    _frame(
        [
            {
                "run_id": RUN_ID,
                "episode_id": episode_id,
                "robot_id": ROBOT_ID,
                "result": "SUCCESS" if episode_id != 1 else "FAILURE",
                "invalid_reason": "" if episode_id in (0, 1) else "pose_teleport",
                "pose_frame": "map",
            }
            for episode_id in (0, 1, 2)
        ],
        fact_ep_schema,
    ).write_parquet(metrics / "fact_episodes.parquet")

    _frame(
        [
            {
                "run_id": RUN_ID,
                "episode_id": 0,
                "env_id": "env_0",
                "robot_id": ROBOT_ID,
                "robot_name": "jackal_0",
                "success": True,
                "time_to_goal": 10.0,
                "path_efficiency": 0.9,
                "energy_total_wh": 2.0,
            },
            {
                "run_id": RUN_ID,
                "episode_id": 1,
                "env_id": "env_0",
                "robot_id": ROBOT_ID,
                "robot_name": "jackal_0",
                "success": False,
                "time_to_goal": 20.0,
                "path_efficiency": 0.7,
                "energy_total_wh": 4.0,
            },
            {
                "run_id": RUN_ID,
                "episode_id": 2,
                "env_id": "env_0",
                "robot_id": ROBOT_ID,
                "robot_name": "jackal_0",
                "success": True,
                "time_to_goal": 30.0,
                "path_efficiency": 0.8,
                "energy_total_wh": 6.0,
            },
        ],
        fact_rob_schema,
    ).write_parquet(metrics / "fact_robots.parquet")

    _frame(
        [
            {
                "run_id": RUN_ID,
                "episode_id": 0,
                "robot_id": ROBOT_ID,
                "contact_id": "c0",
                "encounter_id": "e0",
                "collider_kind": "pedestrian",
                "obstacle_id": "ped_0",
                "t_onset_ns": 1_000,
                "t_end_ns": 2_000,
                "duration_s": 0.1,
                "v_close_robot_mps": 0.8,
                "v_close_target_mps": 0.2,
                "relative_impact_speed_mps": 1.0,
                "freeze_clearance_m": 0.4,
            },
            {
                "run_id": RUN_ID,
                "episode_id": 0,
                "robot_id": ROBOT_ID,
                "contact_id": "c1",
                "encounter_id": "e1",
                "collider_kind": "other_robot",
                "obstacle_id": "robot_1",
                "t_onset_ns": 3_000,
                "t_end_ns": 4_000,
                "duration_s": 0.1,
                "v_close_robot_mps": 0.2,
                "v_close_target_mps": 0.8,
                "relative_impact_speed_mps": 1.0,
                "freeze_clearance_m": -0.2,
            },
        ],
        fact_con_schema,
    ).write_parquet(metrics / "fact_contacts.parquet")
    return root
