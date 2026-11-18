"""Unit tests for query catalog and computation binding through DataResolver on real Parquet."""

from __future__ import annotations

import pathlib

import pytest
from pydantic import ValidationError

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.presentation.resolver import DataResolver
from arena_evaluation.presentation.viz_manifest import (
    PlotChannels,
    PlotDataSpec,
    PlotSpec,
)
from star_benchmark import write_benchmark

_ROWS: list[dict[str, object]] = [
    {
        "episode_id": 0,
        "planner": "dwb",
        "world": "hospital",
        "scenario": "s1",
        "success": True,
        "time_to_goal": 10.0,
        "path_efficiency": 0.9,
        "energy_total_wh": 5.0,
    },
    {
        "episode_id": 1,
        "planner": "dwb",
        "world": "hospital",
        "scenario": "s1",
        "success": True,
        "time_to_goal": 12.0,
        "path_efficiency": 0.85,
        "energy_total_wh": 6.0,
    },
    {
        "episode_id": 2,
        "planner": "mppi",
        "world": "hospital",
        "scenario": "s1",
        "success": False,
        "time_to_goal": 20.0,
        "path_efficiency": 0.7,
        "energy_total_wh": 8.0,
    },
]


def test_data_resolver_stored_query_execution(tmp_path: pathlib.Path) -> None:
    write_benchmark(tmp_path, _ROWS)
    resolver = DataResolver(benchmark_dir=tmp_path)
    spec = PlotSpec(
        id="scorecard",
        type="table",
        data=PlotDataSpec(stored_query="planner_scorecard_overview"),
        channels=PlotChannels(x="planner", y=["success_rate", "mean_time_to_goal"]),
    )
    df, diag = resolver.resolve(spec)
    assert df.height == 2
    assert set(df["planner"].to_list()) == {"dwb", "mppi"}
    assert diag.status == "PASSED"


def test_data_resolver_run_local_stored_query_in_analysis_dir(
    tmp_path: pathlib.Path,
) -> None:
    write_benchmark(tmp_path, _ROWS)
    qdir = tmp_path / "analysis" / "queries" / "local_ep_count"
    qdir.mkdir(parents=True)
    (qdir / "query.yaml").write_text(
        "name: local_ep_count\ndescription: count episodes\nsql: SELECT planner, COUNT(*) AS n FROM v_episode_summary GROUP BY planner ORDER BY planner\n",
        encoding="utf-8",
    )
    resolver = DataResolver(benchmark_dir=tmp_path)
    spec = PlotSpec(
        id="local_q",
        type="bar",
        data=PlotDataSpec(stored_query="local_ep_count"),
        channels=PlotChannels(x="planner", y="n"),
    )
    df, diag = resolver.resolve(spec)
    assert df.height == 2
    assert diag.status == "PASSED"


def test_data_resolver_missing_table_raises_without_fallback(
    tmp_path: pathlib.Path,
) -> None:
    write_benchmark(tmp_path, _ROWS)
    catalog = BenchmarkCatalog.open(tmp_path)
    resolver = DataResolver(benchmark_dir=tmp_path, catalog=catalog)
    spec = PlotSpec(
        id="bad_source",
        type="bar",
        data=PlotDataSpec(source="robot_scan_rays"),
        channels=PlotChannels(x="planner", y="time_to_goal"),
    )
    with pytest.raises((KeyError, ValueError)):
        resolver.resolve(spec)


def test_unambiguous_channels_reject_legacy_fields() -> None:
    with pytest.raises(ValidationError, match="data_key"):
        PlotSpec.model_validate(
            {"id": "legacy", "type": "bar", "data_key": "time_to_goal"}
        )
    with pytest.raises(ValidationError, match="hue"):
        PlotChannels.model_validate({"hue": "planner"})
