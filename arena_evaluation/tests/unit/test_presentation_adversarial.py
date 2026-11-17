"""Adversarial presentation grammar and resolver tests."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest
from pydantic import ValidationError

from arena_evaluation.presentation.predicate import compile_where_predicate
from arena_evaluation.presentation.resolver import DataResolver
from arena_evaluation.presentation.viz_manifest import (
    PlotChannels,
    PlotDataSpec,
    PlotSpec,
    VizManifest,
)
from star_benchmark import write_benchmark


def test_resolver_empty_benchmark_reports_failed_diagnostics(
    tmp_path: pathlib.Path,
) -> None:
    write_benchmark(tmp_path, [])
    resolver = DataResolver(benchmark_dir=tmp_path)
    spec = PlotSpec(
        id="empty_bar",
        type="bar",
        data=PlotDataSpec(source="v_episode_summary"),
        channels=PlotChannels(x="planner", y="path_length_m"),
    )
    df, diag = resolver.resolve(spec)
    assert df.height == 0
    assert diag.status == "FAILED"
    assert "0 rows" in diag.issues[0]


def test_manifest_rejects_non_mapping_yaml(tmp_path: pathlib.Path) -> None:
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        VizManifest.load(bad_yaml)


def test_predicate_rejects_non_boolean_bare_column() -> None:
    schema = pl.Schema({"planner": pl.String, "time_to_goal": pl.Float64})
    with pytest.raises(ValueError, match="boolean column"):
        compile_where_predicate("planner", schema)
    with pytest.raises(ValidationError, match="data_key"):
        PlotSpec.model_validate(
            {"id": "bad", "type": "bar", "data_key": "time_to_goal"}
        )
