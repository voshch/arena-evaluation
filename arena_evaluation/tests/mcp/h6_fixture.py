"""Noisy synthetic benchmark for the H6 proof: a known effect, a non-zero-width CI."""

from __future__ import annotations

import json
import pathlib
import random
from typing import Any

import polars as pl

from arena_evaluation.storage.registry import dimensions

from star_benchmark import write_benchmark

PLANNERS = ("planner_a", "planner_b", "planner_c", "planner_d")
WORLDS = ("world_1", "world_2")
CONDITIONS = ("blind", "visible")
SEEDS = tuple(range(1, 9))
NOISE_SEED = 20261116
NOISE_AMPLITUDE = 0.4


def _time_to_goal(seed: int, world: str, planner: str, noise: float) -> float:
    base = 12.0 + (seed % 3) * 0.2 + (0.5 if world == "world_2" else 0.0)
    offset = {"planner_a": -2.0, "planner_b": 2.0}.get(planner, 0.0)
    return base + offset + noise


def write_noisy_benchmark(directory: pathlib.Path) -> dict[str, Any]:
    """Write a noisy Star Schema run plus its truth manifest; return the truth document."""
    from arena_evaluation.processing.metrics import MetricRegistry

    MetricRegistry.discover_calculators_cls()
    rng = random.Random(NOISE_SEED)
    rows: list[dict[str, Any]] = []
    episode_id = 0
    for seed in SEEDS:
        for world in WORLDS:
            for condition in CONDITIONS:
                for planner in PLANNERS:
                    noise = rng.uniform(-NOISE_AMPLITUDE, NOISE_AMPLITUDE)
                    rows.append(
                        {
                            "run_id": "h6_noisy",
                            "episode_id": episode_id,
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
                            "time_to_goal": _time_to_goal(seed, world, planner, noise),
                            "path_efficiency": (0.9 if planner == "planner_a" else 0.75),
                            "energy_total_wh": 2.0 if planner == "planner_a" else 3.0,
                            "collision_amount_wall": 0,
                            "collision_amount_static": 0,
                            "ci_max": None,
                            "null_reason": "ci_max:not_recorded",
                            "robot_blind": condition == "blind",
                        }
                    )
                    episode_id += 1

    write_benchmark(directory, rows)

    ped_schema = dimensions.DimPedestrians.polars_schema()
    pl.DataFrame(
        [
            {
                **{name: None for name in ped_schema.names()},
                "run_id": "h6_noisy",
                "episode_id": row["episode_id"],
                "env_id": "env_0",
                "ped_id": "ped_0",
                "robot_blind": row["robot_blind"],
            }
            for row in rows
        ],
        schema=ped_schema,
    ).write_parquet(directory / dimensions.DimPedestrians.relative_path())

    dm_schema = dimensions.DimMetrics.polars_schema()
    pl.DataFrame(
        [
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
        ],
        schema=dm_schema,
    ).write_parquet(directory / dimensions.DimMetrics.relative_path())

    truth: dict[str, Any] = {
        "run_id": "h6_noisy",
        "planted_diff_metric": "time_to_goal",
        "planted_diff_a": "planner_a",
        "planted_diff_b": "planner_b",
        "planted_delta": -4.0,
        "noise_seed": NOISE_SEED,
        "noise_amplitude": NOISE_AMPLITUDE,
        "hypotheses": [
            {
                "id": "H_diff",
                "claim": "planner_a achieves lower time_to_goal than planner_b",
                "metric": "time_to_goal",
                "planner_a": "planner_a",
                "planner_b": "planner_b",
                "direction": "lower",
                "expected_verdict": "verified",
            },
            {
                "id": "H_null",
                "claim": "planner_c differs from planner_d on time_to_goal",
                "metric": "time_to_goal",
                "planner_a": "planner_c",
                "planner_b": "planner_d",
                "direction": "different",
                "expected_verdict": "refuted",
            },
            {
                "id": "H_missing",
                "claim": "planner_a achieves lower ci_max than planner_b",
                "metric": "ci_max",
                "planner_a": "planner_a",
                "planner_b": "planner_b",
                "direction": "lower",
                "expected_verdict": "needs_work",
            },
        ],
    }
    (directory / "truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return truth
