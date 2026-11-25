"""Validity, exclusion and normalization contracts of the registered computations."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest

from arena_evaluation.analytics import BenchmarkCatalog, run_computation
from arena_evaluation.storage.registry import FactTable
from star_benchmark import write_benchmark


def _row(
    episode_id: int,
    planner: str,
    seed: int,
    *,
    time_to_goal: float | None,
    invalid_reason: str = "",
) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "planner": planner,
        "seed": seed,
        "time_to_goal": time_to_goal,
        "path_length_m": time_to_goal,
        "is_reference": False,
        "invalid_reason": invalid_reason,
    }


def _write_null_reason(bench: pathlib.Path, episode_id: int, metric: str, reason: str) -> None:
    """Mark the NULL cells of ``metric`` in one episode's fact table with their recorded reason."""
    path = bench / "episodes" / f"episode_{episode_id:03d}" / "metrics" / f"fact_{FactTable.ROBOTS.value}.parquet"
    frame = pl.read_parquet(path)
    frame.with_columns(pl.when(pl.col(metric).is_null()).then(pl.lit(f"{metric}:{reason}")).otherwise(pl.col("null_reason")).alias("null_reason")).write_parquet(path)


def test_computation_drops_invalid_episodes_and_counts_them(
    tmp_path: pathlib.Path,
) -> None:
    bench = write_benchmark(
        tmp_path / "bench",
        [
            _row(0, "dwb", 1, time_to_goal=10.0),
            _row(1, "dwb", 2, time_to_goal=1000.0, invalid_reason="pose_teleport"),
        ],
    )
    catalog = BenchmarkCatalog.open(bench)

    result = run_computation(
        "group_summary",
        catalog,
        {"group_cols": ["planner"], "metrics": ["time_to_goal"]},
    )
    assert result.summary["mean"].to_list() == pytest.approx([10.0])
    assert result.summary["n"].to_list() == [1]
    assert result.excluded_rows["invalid_for_ranking"] == 1
    assert result.seeds_used == (1,)

    explicit = run_computation(
        "group_summary",
        catalog,
        {
            "group_cols": ["planner"],
            "metrics": ["time_to_goal"],
            "valid_only": False,
        },
    )
    assert explicit.summary["mean"].to_list() == pytest.approx([505.0])
    assert "invalid_for_ranking" not in explicit.excluded_rows


def test_computation_reports_each_null_reason_per_metric(
    tmp_path: pathlib.Path,
) -> None:
    bench = write_benchmark(
        tmp_path / "bench",
        [
            _row(0, "dwb", 1, time_to_goal=10.0),
            _row(1, "dwb", 2, time_to_goal=None),
        ],
    )
    _write_null_reason(bench, 1, "time_to_goal", "no_samples")
    catalog = BenchmarkCatalog.open(bench)

    result = run_computation(
        "group_summary",
        catalog,
        {"group_cols": ["planner"], "metrics": ["time_to_goal"]},
    )
    assert result.summary["n"].to_list() == [1]
    assert result.excluded_rows["time_to_goal:no_samples"] == 1


def test_radar_profile_nulls_untransformable_columns_and_reports_them(
    tmp_path: pathlib.Path,
) -> None:
    bench = write_benchmark(
        tmp_path / "bench",
        [
            {
                "episode_id": 0,
                "planner": "p1",
                "seed": 1,
                "mean_velocity_mps": -1.0,
                "is_reference": False,
                "invalid_reason": "",
            },
            {
                "episode_id": 1,
                "planner": "p2",
                "seed": 1,
                "mean_velocity_mps": -2.0,
                "is_reference": False,
                "invalid_reason": "",
            },
        ],
    )
    catalog = BenchmarkCatalog.open(bench)

    result = run_computation(
        "radar_profile",
        catalog,
        {
            "metrics": ["mean_velocity_mps"],
            "directions": {"mean_velocity_mps": "higher"},
        },
    )
    assert result.summary["normalized_score"].to_list() == [None, None]
    assert "radar_untransformable:mean_velocity_mps=2" in result.diagnostics
    assert "directions_from_param=1:mean_velocity_mps" in result.diagnostics
