"""Pin tests for analytics/sql_engine.py before refactoring validate_readonly_sql."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.analytics.sql_engine import (
    PolarsSQLEngine,
    SQLGuardError,
    normalize_sql_scientific_notation,
    validate_readonly_sql,
)


def test_normalize_sql_scientific_notation_preserves_literals_and_identifiers() -> None:
    sql = "SELECT 1e-6 AS tiny, 2.5E2 AS big, '1e-6' AS lit, \"2e-5\" AS qlit, col1e2 FROM t"
    out = normalize_sql_scientific_notation(sql)
    assert "0.000001" in out
    assert "250.0" in out
    assert "'1e-6'" in out
    assert '"2e-5"' in out
    assert "col1e2" in out


@pytest.mark.parametrize(
    ("bad_sql", "match"),
    [
        ("", "non-empty string"),
        ("   ", "non-empty string"),
        ("-- only comment\n/* block */", "only comments or whitespace"),
        ("SELECT 1; SELECT 2", "Multi-statement"),
        ("()", "no valid statement tokens"),
        ("DROP TABLE foo", "Forbidden SQL keyword"),
        ("SELECT * FROM read_parquet('x.parquet')", "External table function"),
        ("SELECT * FROM scan_csv('x.csv')", "External table function"),
        ("EXPLAIN SELECT 1", "Only SELECT or WITH"),
        ("WITH cte AS (VALUES (1)) SHOW TABLES", "must contain a SELECT clause"),
    ],
)
def test_validate_readonly_sql_rejects_disallowed_queries(bad_sql: str, match: str) -> None:
    with pytest.raises(SQLGuardError, match=match):
        validate_readonly_sql(bad_sql)


def test_validate_readonly_sql_accepts_comments_literals_and_trailing_semicolon() -> None:
    sql = "/* head */ SELECT 'DROP TABLE x; 1;2' AS s FROM t -- tail\n;"
    cleaned = validate_readonly_sql(sql)
    assert cleaned.startswith("SELECT 'DROP TABLE x; 1;2' AS s FROM t")
    assert not cleaned.endswith(";")

    cte = validate_readonly_sql("WITH c AS (SELECT 1 AS a) SELECT a FROM c")
    assert cte == "WITH c AS (SELECT 1 AS a) SELECT a FROM c"


def test_polars_sql_engine_init_register_and_query_options(
    tmp_path: pathlib.Path,
) -> None:
    engine_from_path = PolarsSQLEngine(tmp_path)
    assert isinstance(engine_from_path.catalog, BenchmarkCatalog)

    catalog = BenchmarkCatalog()
    engine = PolarsSQLEngine(catalog)
    df = pl.DataFrame({"x": [1, 2, 3]})
    engine.register_table("custom_tbl", df)

    limited = engine.query(
        "SELECT x FROM custom_tbl ORDER BY x",
        limit=2,
        save_as="top_two",
    )
    assert limited["x"].to_list() == [1, 2]
    assert engine.query("SELECT x FROM top_two ORDER BY x")["x"].to_list() == [1, 2]

    with pytest.raises(ValueError, match="limit must be non-negative"):
        engine.query("SELECT x FROM custom_tbl", limit=-1)

    with pytest.raises(SQLGuardError, match="Cannot overwrite protected"):
        engine.register_table("v_episode_summary", df)

    with pytest.raises(SQLGuardError, match="Cannot overwrite protected"):
        engine.query("SELECT x FROM custom_tbl", save_as="v_episode_summary")
