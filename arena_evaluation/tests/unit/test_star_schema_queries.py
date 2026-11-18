"""Star schema queries, canonical views, and absence of legacy aliases."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest

from arena_evaluation.analytics import catalog as catalog_module
from arena_evaluation.analytics.catalog import CANONICAL_VIEW_NAMES, BenchmarkCatalog
from arena_evaluation.analytics.sql_engine import PolarsSQLEngine
from arena_evaluation.storage.registry import (
    FACT_TABLES,
    FactTable,
    NullReason,
    dimensions,
)
from arena_evaluation.storage.star_schema import write_benchmark_dimensions


def _frame(rows: list[dict[str, object]], schema: pl.Schema) -> pl.DataFrame:
    return pl.DataFrame(
        [{name: row.get(name) for name in schema.names()} for row in rows],
        schema=schema,
    )


def _write_multi_condition_benchmark(root: pathlib.Path) -> pathlib.Path:
    root.mkdir(parents=True)
    dims = {
        name: pl.DataFrame(schema=tbl.polars_schema())
        for name, tbl in dimensions.DIMENSION_TABLES.items()
    }
    dims["dim_episodes"] = _frame(
        [
            {
                "run_id": "r1",
                "episode_id": ep_id,
                "env_id": "env_0",
                "robot_id": "robot_0",
                "world": "hospital",
                "scenario": "corridor",
                "planner": planner,
                "dispatcher": "none",
                "seed": 1,
                "is_reference": False,
            }
            for ep_id, planner in ((0, "dwb"), (1, "teb"), (2, "mppi"), (3, "dwb"))
        ],
        dimensions.DimEpisodes.polars_schema(),
    )
    dims["dim_pedestrians"] = _frame(
        [
            {
                "run_id": "r1",
                "episode_id": 0,
                "env_id": "env_0",
                "ped_id": "p0",
                "robot_awareness": 0.0,
            },
            {
                "run_id": "r1",
                "episode_id": 1,
                "env_id": "env_0",
                "ped_id": "p0",
                "robot_awareness": 1.0,
            },
            {
                "run_id": "r1",
                "episode_id": 2,
                "env_id": "env_0",
                "ped_id": "p0",
                "robot_awareness": 0.0,
            },
            {
                "run_id": "r1",
                "episode_id": 2,
                "env_id": "env_0",
                "ped_id": "p1",
                "robot_awareness": 1.0,
            },
            {
                "run_id": "r1",
                "episode_id": 3,
                "env_id": "env_0",
                "ped_id": "p0",
                "robot_awareness": None,
            },
        ],
        dimensions.DimPedestrians.polars_schema(),
    )
    dims["dim_metrics"] = _frame(
        [
            {
                "metric": "time_to_goal",
                "scope": "robots",
                "fact_table": "fact_robots",
                "dtype": "Float64",
                "unit": "s",
                "description": "Time to reach goal.",
                "direction": "lower",
                "producer": "motion_metrics",
            }
        ],
        dimensions.DimMetrics.polars_schema(),
    )
    write_benchmark_dimensions(dims, root)

    metrics = root / "episodes" / "episode_000" / "metrics"
    metrics.mkdir(parents=True)
    _frame(
        [
            {
                "run_id": "r1",
                "episode_id": ep_id,
                "robot_id": "robot_0",
                "result": "SUCCESS",
                "invalid_reason": "",
            }
            for ep_id in range(4)
        ],
        FACT_TABLES[FactTable.EPISODES].polars_schema(),
    ).write_parquet(metrics / "fact_episodes.parquet")
    _frame(
        [
            {
                "run_id": "r1",
                "episode_id": ep_id,
                "env_id": "env_0",
                "robot_id": "robot_0",
                "robot_name": "robot_0",
                "success": True,
                "time_to_goal": 10.0 + ep_id,
                "path_efficiency": 0.9,
                "energy_total_wh": 2.0,
            }
            for ep_id in range(4)
        ],
        FACT_TABLES[FactTable.ROBOTS].polars_schema(),
    ).write_parquet(metrics / "fact_robots.parquet")
    _frame(
        [
            {
                "run_id": "r1",
                "episode_id": 0,
                "robot_id": "robot_0",
                "encounter_id": "enc_0",
                "actor_1": "robot_0",
                "actor_2": "p0",
                "actor_2_kind": "pedestrian",
                "t_start_ns": 0,
                "t_end_ns": 1_000_000_000,
                "contact_id": "c0",
                "approach_type": "head_on",
                "min_clearance_m": 0.8,
                "duration_s": 1.0,
                "energy_total_wh": 0.5,
                "energy_surge_ratio": 1.8,
                "probst_R_robot": 0.6,
                "probst_R_actor2": 0.4,
                "probst_R_time": 0.5,
                "probst_E_robot": 0.7,
                "probst_E_actor2": 0.3,
                "robot_path_deviation_m": 0.25,
                "detour_time_s": 0.4,
            }
        ],
        FACT_TABLES[FactTable.ENCOUNTERS].polars_schema(),
    ).write_parquet(metrics / "fact_encounters.parquet")
    return root


def test_crowd_condition_view_derives_blind_visible_mixed_and_null_reason(
    tmp_path: pathlib.Path,
) -> None:
    catalog = BenchmarkCatalog.open(
        _write_multi_condition_benchmark(tmp_path / "bench")
    )
    rows = (
        catalog.get_table("v_episode_summary")
        .select("episode_id", "crowd_condition", "crowd_condition_null_reason")
        .sort("episode_id")
        .collect()
        .to_dicts()
    )
    assert rows == [
        {
            "episode_id": 0,
            "crowd_condition": "blind",
            "crowd_condition_null_reason": None,
        },
        {
            "episode_id": 1,
            "crowd_condition": "visible",
            "crowd_condition_null_reason": None,
        },
        {
            "episode_id": 2,
            "crowd_condition": "mixed",
            "crowd_condition_null_reason": None,
        },
        {
            "episode_id": 3,
            "crowd_condition": None,
            "crowd_condition_null_reason": str(NullReason.NOT_RECORDED),
        },
    ]


def test_no_legacy_aliases_exist_in_catalog_module() -> None:
    assert not hasattr(catalog_module, "StarCatalog")
    assert not hasattr(catalog_module, "TELEMETRY_TABLE_SCHEMAS")
    assert not hasattr(catalog_module, "TELEMETRY_TABLE_NAMES")
    assert "v_contact_blame_summary" not in CANONICAL_VIEW_NAMES
    assert "v_contact_summary" in CANONICAL_VIEW_NAMES


def test_sql_engine_queries_canonical_views_and_dim_metrics(
    tmp_path: pathlib.Path,
) -> None:
    catalog = BenchmarkCatalog.open(
        _write_multi_condition_benchmark(tmp_path / "bench")
    )
    engine = PolarsSQLEngine(catalog)

    enc = engine.query(
        "SELECT crowd_condition, encounter_count, contact_encounters FROM v_encounter_summary"
    )
    assert enc.to_dicts() == [
        {"crowd_condition": "blind", "encounter_count": 1, "contact_encounters": 1}
    ]

    metrics_df = engine.query(
        "SELECT metric, direction FROM dim_metrics WHERE metric = 'time_to_goal'"
    )
    assert metrics_df.to_dicts() == [{"metric": "time_to_goal", "direction": "lower"}]


@pytest.mark.parametrize(
    "bad_alias",
    [
        "time_to_goal_s",
        "total_energy_wh",
        "path_deviation_m",
        "closer_label",
        "fault_attributed_to",
    ],
)
def test_v_episode_summary_and_v_encounter_summary_omit_retired_source_names(
    tmp_path: pathlib.Path, bad_alias: str
) -> None:
    catalog = BenchmarkCatalog.open(
        _write_multi_condition_benchmark(tmp_path / "bench")
    )
    assert (
        bad_alias not in catalog.get_table("v_episode_summary").collect_schema().names()
    )
    assert (
        bad_alias
        not in catalog.get_table("v_encounter_summary").collect_schema().names()
    )
