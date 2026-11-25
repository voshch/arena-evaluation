"""The world-geometry dimension and the rows it publishes for real Arena worlds."""

from __future__ import annotations


import polars as pl
import pytest

pytest.importorskip("arena_simulation_setup")

from arena_evaluation.storage.registry.dimensions import (
    DimWorldGeometry,
)

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
    return frame.filter(pl.col("element_kind") == element_kind).select(pl.col("element_name").n_unique()).item()


def _polygon_count(frame: pl.DataFrame, element_kind: str) -> int:
    return frame.filter(pl.col("element_kind") == element_kind).select("element_name", "part_idx").unique().height


def _vertices_per_element(frame: pl.DataFrame, element_kind: str) -> pl.DataFrame:
    return frame.filter(pl.col("element_kind") == element_kind).group_by("element_name", "part_idx").agg(pl.len().alias("vertex_count"))


def _assert_unique_keys(frame: pl.DataFrame) -> None:
    assert frame.unique(subset=list(KEY_COLUMNS)).height == frame.height


def _assert_required_columns_are_set(frame: pl.DataFrame) -> None:
    required = (*KEY_COLUMNS, "x", "y")
    nulls = frame.select([pl.col(name).is_null().sum() for name in required]).row(0)
    assert nulls == (0,) * len(required)


def _assert_coordinates_are_finite(frame: pl.DataFrame) -> None:
    finite = frame.select(pl.col("x").is_finite().all(), pl.col("y").is_finite().all()).row(0)
    assert finite == (True, True)


def _assert_wall_vertices(frame: pl.DataFrame) -> None:
    walls = _vertices_per_element(frame, "wall")
    assert walls.height > 0
    assert walls["vertex_count"].eq(WALL_VERTEX_COUNT).all()


def _assert_polygon_vertices(frame: pl.DataFrame) -> None:
    polygons = frame.filter(pl.col("element_kind") != "wall").group_by("element_name", "part_idx").agg(pl.len().alias("vertex_count"))
    assert polygons["vertex_count"].ge(POLYGON_MIN_VERTEX_COUNT).all()
