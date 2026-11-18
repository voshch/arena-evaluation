"""Unit tests for DataResolver, recipe merging, computation/evidence binding, and slicing."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.analytics.evidence import EvidenceStore
from arena_evaluation.presentation.resolver import (
    DataResolver,
    deep_merge,
    find_recipe_path,
    recipe_search_dirs,
)
from arena_evaluation.presentation.viz_manifest import (
    PlotChannels,
    PlotDataSpec,
    PlotSliceSpec,
    PlotSpec,
)
from star_benchmark import write_benchmark

_ROWS: list[dict[str, object]] = [
    {
        "episode_id": 0,
        "planner": "dwb",
        "success": True,
        "time_to_goal": 10.0,
        "path_length_m": 12.0,
        "is_reference": False,
    },
    {
        "episode_id": 1,
        "planner": "dwb",
        "success": True,
        "time_to_goal": 14.0,
        "path_length_m": 16.0,
        "is_reference": False,
    },
    {
        "episode_id": 2,
        "planner": "mppi",
        "success": False,
        "time_to_goal": 22.0,
        "path_length_m": 25.0,
        "is_reference": False,
    },
    {
        "episode_id": 3,
        "planner": "mppi",
        "success": True,
        "time_to_goal": 18.0,
        "path_length_m": 20.0,
        "is_reference": True,
    },
]


def test_data_resolver_catalog_view_with_where_and_default_source(
    tmp_path: pathlib.Path,
) -> None:
    write_benchmark(tmp_path, _ROWS)
    resolver = DataResolver(benchmark_dir=tmp_path)

    spec = PlotSpec(
        id="p_view",
        type="box",
        data=PlotDataSpec(source="v_episode_summary", where="is_reference = false"),
        channels=PlotChannels(x="planner", y="time_to_goal"),
    )
    df, diag = resolver.resolve(spec)
    assert df.height == 3
    assert diag.status == "PASSED"
    assert diag.snapshot_id.startswith("snap-")

    default_spec = PlotSpec(
        id="p_default", type="box", channels=PlotChannels(x="planner", y="time_to_goal")
    )
    df_default, _ = resolver.resolve(default_spec, default_source="v_episode_summary")
    assert df_default.height == 4


def test_data_resolver_polars_sql_query(tmp_path: pathlib.Path) -> None:
    write_benchmark(tmp_path, _ROWS)
    resolver = DataResolver(benchmark_dir=tmp_path)
    spec = PlotSpec(
        id="p_sql",
        type="bar",
        data=PlotDataSpec(
            query="SELECT planner, COUNT(*) AS n FROM v_episode_summary GROUP BY planner ORDER BY planner"
        ),
        channels=PlotChannels(x="planner", y="n"),
    )
    df, diag = resolver.resolve(spec)
    assert df.height == 2
    assert diag.status == "PASSED"


def test_data_resolver_computation_records_evidence_and_snapshot(
    tmp_path: pathlib.Path,
) -> None:
    write_benchmark(tmp_path, _ROWS)
    resolver = DataResolver(benchmark_dir=tmp_path)
    spec = PlotSpec(
        id="p_comp",
        type="bar",
        data=PlotDataSpec(
            computation="group_summary",
            params={"metrics": ["time_to_goal"], "group_cols": ["planner"]},
        ),
        channels=PlotChannels(x="planner", y="mean"),
    )
    df, diag = resolver.resolve(spec)
    assert df.height == 2
    assert set(df["planner"].to_list()) == {"dwb", "mppi"}
    assert diag.evidence_id.startswith("ev-")
    assert diag.snapshot_id.startswith("snap-")


def test_data_resolver_evidence_store_roundtrip(tmp_path: pathlib.Path) -> None:
    write_benchmark(tmp_path, _ROWS)
    catalog = BenchmarkCatalog.open(tmp_path)
    store = EvidenceStore(tmp_path)
    sample = pl.DataFrame({"planner": ["dwb", "mppi"], "score": [0.9, 0.7]})
    rec = store.record(
        sample, kind="sql", operation="manual", snapshot_id=catalog.snapshot_id
    )

    resolver = DataResolver(
        benchmark_dir=tmp_path, catalog=catalog, evidence_store=store
    )
    spec = PlotSpec(
        id="p_ev",
        type="bar",
        data=PlotDataSpec(evidence_id=rec.evidence_id),
        channels=PlotChannels(x="planner", y="score"),
    )
    df, diag = resolver.resolve(spec)
    assert df.height == 2
    assert diag.evidence_id == rec.evidence_id


def test_data_resolver_spatiotemporal_slicing() -> None:
    df = pl.DataFrame(
        {
            "episode_id": [0, 0, 1],
            "encounter_id": ["e1", "e2", "e3"],
            "contact_id": ["c1", "c2", "c3"],
            "val": [1.0, 2.0, 3.0],
        }
    )
    spec = PlotSpec(
        id="p_slice",
        type="line",
        slice=PlotSliceSpec(episode_id=0, encounter_id="e2"),
        channels=PlotChannels(x="encounter_id", y="val"),
    )
    sliced = DataResolver._apply_slice(spec, df)
    assert sliced.height == 1
    assert sliced["contact_id"].to_list() == ["c2"]

    with pytest.raises(ValueError, match="encounter_id"):
        DataResolver._apply_slice(spec, df.drop("encounter_id"))


def test_recipe_resolution_in_analysis_dir_and_deep_merge(
    tmp_path: pathlib.Path,
) -> None:
    recipe_dir = tmp_path / "analysis" / "plots" / "custom_bar"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "plot.yaml").write_text(
        "id: custom_bar\ntype: bar\ntitle: Default\ndata:\n  source: v_episode_summary\nchannels:\n  x: planner\n  y: time_to_goal\n",
        encoding="utf-8",
    )
    resolver = DataResolver(benchmark_dir=tmp_path)
    merged = resolver.resolve_spec(PlotSpec(uses="custom_bar", title="Override"))
    assert merged.id == "custom_bar"
    assert merged.title == "Override"
    assert merged.channels.y == "time_to_goal"

    assert recipe_search_dirs()
    assert deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}}) == {
        "a": {"x": 1, "y": 3}
    }
    assert find_recipe_path("missing_recipe_xyz") is None
    with pytest.raises(FileNotFoundError):
        resolver.resolve_spec(PlotSpec(uses="missing_recipe_xyz"))
