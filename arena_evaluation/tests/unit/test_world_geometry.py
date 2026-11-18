"""The world-geometry dimension and the rows it publishes for real Arena worlds."""

from __future__ import annotations

import logging

import polars as pl
import pytest
from polars.testing import assert_frame_equal

pytest.importorskip("arena_simulation_setup")

from arena_evaluation.processing.world_geometry import (
    benchmark_world_geometry,
    world_geometry_rows,
)
from arena_evaluation.storage.registry.dimensions import (
    DIMENSION_TABLES,
    DimWorldGeometry,
)
from arena_evaluation.storage.registry.table import TableKind

EXPECTED_SCHEMA = pl.Schema(
    {
        "world": pl.String,
        "element_kind": pl.String,
        "element_name": pl.String,
        "part_idx": pl.Int64,
        "vertex_idx": pl.Int64,
        "x": pl.Float64,
        "y": pl.Float64,
        "null_reason": pl.String,
    }
)
ELEMENT_KINDS = {"wall", "door", "elevator", "zone", "object"}
KEY_COLUMNS = (
    "world",
    "element_kind",
    "element_name",
    "part_idx",
    "vertex_idx",
)
WALL_VERTEX_COUNT = 2
POLYGON_MIN_VERTEX_COUNT = 3


def test_dim_world_geometry_is_a_benchmark_dimension_keyed_by_vertex() -> None:
    assert DimWorldGeometry.table_name == "dim_world_geometry"
    assert DimWorldGeometry.table_kind is TableKind.DIMENSION
    assert DimWorldGeometry.key == KEY_COLUMNS
    assert DimWorldGeometry.relative_path().as_posix() == "dim_world_geometry.parquet"
    assert DIMENSION_TABLES["dim_world_geometry"] is DimWorldGeometry


def test_dim_world_geometry_has_the_pinned_flat_schema_and_required_columns() -> None:
    schema = DimWorldGeometry.polars_schema()

    assert schema == EXPECTED_SCHEMA
    assert list(schema.names()) == list(EXPECTED_SCHEMA.names())
    for dtype in schema.values():
        assert not isinstance(dtype, (pl.List, pl.Struct, pl.Array, pl.Object))
    for name in (*KEY_COLUMNS, "x", "y"):
        assert DimWorldGeometry.column(name).required is True
    assert DimWorldGeometry.null_reason_columns() == ()


def _element_count(frame: pl.DataFrame, element_kind: str) -> int:
    return (
        frame.filter(pl.col("element_kind") == element_kind)
        .select(pl.col("element_name").n_unique())
        .item()
    )


def _polygon_count(frame: pl.DataFrame, element_kind: str) -> int:
    return (
        frame.filter(pl.col("element_kind") == element_kind)
        .select("element_name", "part_idx")
        .unique()
        .height
    )


def _vertices_per_element(frame: pl.DataFrame, element_kind: str) -> pl.DataFrame:
    return (
        frame.filter(pl.col("element_kind") == element_kind)
        .group_by("element_name", "part_idx")
        .agg(pl.len().alias("vertex_count"))
    )


def _assert_unique_keys(frame: pl.DataFrame) -> None:
    assert frame.unique(subset=list(KEY_COLUMNS)).height == frame.height


def _assert_required_columns_are_set(frame: pl.DataFrame) -> None:
    required = (*KEY_COLUMNS, "x", "y")
    nulls = frame.select([pl.col(name).is_null().sum() for name in required]).row(0)
    assert nulls == (0,) * len(required)


def _assert_coordinates_are_finite(frame: pl.DataFrame) -> None:
    finite = frame.select(
        pl.col("x").is_finite().all(), pl.col("y").is_finite().all()
    ).row(0)
    assert finite == (True, True)


def _assert_wall_vertices(frame: pl.DataFrame) -> None:
    walls = _vertices_per_element(frame, "wall")
    assert walls.height > 0
    assert walls["vertex_count"].eq(WALL_VERTEX_COUNT).all()


def _assert_polygon_vertices(frame: pl.DataFrame) -> None:
    polygons = (
        frame.filter(pl.col("element_kind") != "wall")
        .group_by("element_name", "part_idx")
        .agg(pl.len().alias("vertex_count"))
    )
    assert polygons["vertex_count"].ge(POLYGON_MIN_VERTEX_COUNT).all()


def test_map_empty_geometry_publishes_the_world_file_counts() -> None:
    # Counts of arena_simulation_setup worlds/map_empty/world.yaml at jazzy bad76c0.
    frame = world_geometry_rows("map_empty", None)

    assert frame.schema == DimWorldGeometry.polars_schema()
    assert frame.columns == list(EXPECTED_SCHEMA.names())
    assert frame.height == 54
    assert set(frame["world"].unique().to_list()) == {"map_empty"}
    assert _element_count(frame, "wall") == 9
    assert _polygon_count(frame, "door") == 3
    assert _element_count(frame, "elevator") == 2
    assert _polygon_count(frame, "elevator") == 4
    assert _element_count(frame, "zone") == 1
    assert _element_count(frame, "object") == 1
    assert set(frame["element_kind"].unique().to_list()) == ELEMENT_KINDS
    assert_frame_equal(
        frame.filter(pl.col("element_kind") == "wall")
        .select("element_name")
        .unique()
        .sort("element_name"),
        pl.DataFrame({"element_name": [f"wall_{index:03d}" for index in range(9)]}),
        check_dtypes=True,
    )
    _assert_unique_keys(frame)
    _assert_required_columns_are_set(frame)
    _assert_coordinates_are_finite(frame)
    _assert_wall_vertices(frame)
    _assert_polygon_vertices(frame)


def test_hospital_1_bare_geometry_keeps_the_world_invariants() -> None:
    frame = world_geometry_rows("hospital_1_bare", None)

    assert frame.height > 0
    kinds = set(frame["element_kind"].unique().to_list())
    assert kinds <= ELEMENT_KINDS
    assert {"wall", "door", "zone"} <= kinds
    assert _element_count(frame, "wall") > 0
    assert _polygon_count(frame, "door") > 0
    assert _element_count(frame, "zone") > 0
    _assert_unique_keys(frame)
    _assert_required_columns_are_set(frame)
    _assert_coordinates_are_finite(frame)
    _assert_wall_vertices(frame)
    _assert_polygon_vertices(frame)


def test_a_world_that_does_not_resolve_raises() -> None:
    with pytest.raises(FileNotFoundError):
        world_geometry_rows("hospital_1", None)


def test_a_benchmark_publishes_the_worlds_that_resolve_and_names_the_others(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        frame = benchmark_world_geometry(["hospital_1", "map_empty"], None)

    assert frame.schema == DimWorldGeometry.polars_schema()
    assert frame["world"].unique().to_list() == ["map_empty"]
    assert "world 'hospital_1' has no published geometry" in caplog.text


def test_a_benchmark_without_a_world_publishes_an_empty_table() -> None:
    frame = benchmark_world_geometry([], None)

    assert frame.height == 0
    assert frame.schema == DimWorldGeometry.polars_schema()
