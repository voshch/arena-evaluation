"""Unit tests for derived-table versioning/locking and the merged-benchmark catalog (L3-060..L3-063)."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest
from star_benchmark import write_benchmark

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.analytics.derived import (
    derived_dir,
    list_derived_tables,
    save_derived_table,
)
from arena_evaluation.analytics.merge import (
    MERGE_MANIFEST_FILENAME,
    create_merged_benchmark,
    load_merge_manifest,
)

_ROWS = [
    {
        "run_id": "r1",
        "episode_id": 0,
        "planner": "dwb",
        "world": "hospital",
        "scenario": "corridor",
        "stage": "s1",
        "robot_model": "jackal",
        "seed": 1,
        "is_reference": False,
        "invalid_reason": "",
        "success": True,
        "time_to_goal": 10.0,
    },
    {
        "run_id": "r1",
        "episode_id": 1,
        "planner": "teb",
        "world": "hospital",
        "scenario": "corridor",
        "stage": "s1",
        "robot_model": "jackal",
        "seed": 2,
        "is_reference": False,
        "invalid_reason": "",
        "success": False,
        "time_to_goal": 20.0,
    },
]


def _save(bench: pathlib.Path, frame: pl.DataFrame, version_label: str) -> dict[str, object]:
    return save_derived_table(
        bench,
        name="lead",
        frame=frame,
        definition={"kind": "sql", "sql": f"SELECT 1 AS v -- {version_label}"},
        snapshot_id="snap-aaaaaaaaaaaa",
        author="test",
        purpose="check",
    )


def test_derived_table_archives_each_replaced_version(tmp_path: pathlib.Path) -> None:
    """Every superseded table keeps its own archived file; the sidecar counter names them (L3-063)."""
    bench = write_benchmark(tmp_path / "bench", _ROWS)
    first = pl.DataFrame({"v": [1, 2]})
    second = pl.DataFrame({"v": [3, 4]})
    third = pl.DataFrame({"v": [5, 6]})

    assert _save(bench, first, "one")["version"] == 1
    assert _save(bench, second, "two")["version"] == 2
    info = _save(bench, third, "three")
    assert info["version"] == 3

    root = derived_dir(bench)
    assert pl.read_parquet(root / "lead.v1.parquet").equals(first)
    assert pl.read_parquet(root / "lead.v2.parquet").equals(second)
    assert pl.read_parquet(root / "lead.parquet").equals(third)

    entries = list_derived_tables(bench, current_snapshot_id="snap-aaaaaaaaaaaa")
    assert [(entry["name"], entry["version"], entry["status"]) for entry in entries] == [("lead", 3, "fresh")]
    stale = list_derived_tables(bench, current_snapshot_id="snap-bbbbbbbbbbbb")
    assert [entry["status"] for entry in stale] == ["stale"]


def test_derived_table_refuses_a_nested_frame(tmp_path: pathlib.Path) -> None:
    """A List column cannot be persisted as a derived table; the error says how to flatten it (L3-059)."""
    bench = write_benchmark(tmp_path / "bench", _ROWS)
    nested = pl.DataFrame({"metric": ["ttg"], "seeds": [[1, 2, 3]]})
    with pytest.raises(ValueError, match="must be flat"):
        _save(bench, nested, "nested")
    assert not (derived_dir(bench) / "lead.parquet").exists()


def _two_runs(root: pathlib.Path) -> pathlib.Path:
    benchmarks = root / "benchmarks"
    write_benchmark(
        benchmarks / "r1",
        _ROWS,
        robot_tracks=[
            {
                "episode_id": 0,
                "t_ns": 0,
                "env_id": "env_0",
                "robot_id": "robot_0",
                "x": 0.0,
                "y": 0.0,
            }
        ],
    )
    write_benchmark(
        benchmarks / "r2",
        [
            {
                **_ROWS[0],
                "run_id": "r2",
                "planner": "mppi",
                "world": "office",
                "robot_model": "burger",
                "time_to_goal": 12.0,
            }
        ],
        robot_tracks=[
            {
                "episode_id": 0,
                "t_ns": 0,
                "env_id": "env_0",
                "robot_id": "robot_0",
                "x": 1.0,
                "y": 1.0,
            }
        ],
    )
    return create_merged_benchmark(
        benchmarks,
        run_ids=["r1", "r2"],
        name="merged",
        labels={"r1": {"corpus": "sweep"}, "r2": {"corpus": "paper"}},
    )


def test_merged_benchmark_stores_run_ids_and_keeps_labels_and_run_key(
    tmp_path: pathlib.Path,
) -> None:
    """The manifest is root-relative, and the merged views keep the source-run key and its labels (L3-060)."""
    merged = _two_runs(tmp_path)
    manifest = load_merge_manifest(merged)
    assert manifest is not None
    on_disk = (merged / MERGE_MANIFEST_FILENAME).read_text(encoding="utf-8")
    assert str(merged.parent) not in on_disk
    assert "run_dir: r1" in on_disk
    assert [source.run_dir for source in manifest.sources] == [
        str((merged.parent / "r1").resolve()),
        str((merged.parent / "r2").resolve()),
    ]

    catalog = BenchmarkCatalog.open(merged)
    scorecard = catalog.get_table("v_planner_scorecard").collect()
    assert {"source_run_id", "corpus"} <= set(scorecard.columns)
    assert set(scorecard["source_run_id"].to_list()) == {"r1", "r2"}
    assert set(scorecard["corpus"].to_list()) == {"sweep", "paper"}

    tracks = catalog.get_table("robot_tracks").collect()
    assert "run_id" in tracks.columns
    assert set(tracks["run_id"].to_list()) == {"r1", "r2"}


def test_merged_catalog_raises_for_a_missing_source(tmp_path: pathlib.Path) -> None:
    """A merge that lost a source run fails loudly instead of publishing a smaller benchmark (L3-060)."""
    merged = _two_runs(tmp_path)
    import shutil

    shutil.rmtree(merged.parent / "r2")
    with pytest.raises(FileNotFoundError, match="r2"):
        BenchmarkCatalog.open(merged)
