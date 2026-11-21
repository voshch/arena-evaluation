"""The native benchmark layout: stale outputs are refused, stamps gate completeness, writes are atomic."""

from __future__ import annotations

import hashlib
import pathlib

import polars as pl
import pytest
import yaml

from arena_evaluation.storage.contracts import enforce_benchmark_contract
from arena_evaluation.storage.exceptions import (
    ContractViolationError,
)
from arena_evaluation.storage.layout import (
    check_benchmark_layout,
    retire_legacy_outputs,
    write_benchmark_stamp,
    write_episode_stamp,
)
from arena_evaluation.storage.registry.dimensions import DIMENSION_TABLES
from arena_evaluation.storage.registry.facts import FACT_TABLES
from arena_evaluation.storage.registry.table import Table, TableKind, registered_tables
from arena_evaluation.storage.star_schema import write_benchmark_dimensions


def _native_tables() -> tuple[type[Table], ...]:
    """The same native table set the layout hash covers."""
    return (
        *FACT_TABLES.values(),
        *DIMENSION_TABLES.values(),
        *(table for table in registered_tables() if table.table_kind is TableKind.TELEMETRY),
    )


def _digest(entries: list[tuple[str, str]]) -> str:
    """A copy of the layout digest, over an explicit (name, schema text) list."""
    payload = "\n".join(f"{name}\0{schema}" for name, schema in entries)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stale_episode(root: pathlib.Path, name: str = "episode_000") -> pathlib.Path:
    """A 23-column fact file of the old layout, with no completion stamp."""
    metrics = root / "episodes" / name / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    path = metrics / "fact_episodes.parquet"
    pl.DataFrame({f"old_{index}": [index] for index in range(23)}).write_parquet(path)
    return path


def _dimension_frames() -> dict[str, pl.DataFrame]:
    """Every declared dimension as an empty frame of its registry schema."""
    return {name: pl.DataFrame(schema=table.polars_schema()) for name, table in DIMENSION_TABLES.items()}


def test_retiring_moves_stale_outputs_to_legacy_and_never_deletes(
    tmp_path: pathlib.Path,
) -> None:
    root = tmp_path / "benchmarks" / "stale"
    stale = _stale_episode(root)
    original = stale.read_bytes()
    benchmark_metrics = root / "metrics.parquet"
    benchmark_metrics.write_bytes(b"benchmark-metrics")
    combined = root / "combined_metrics.parquet"
    combined.write_bytes(b"combined-metrics")

    complete_metrics = root / "episodes" / "episode_001" / "metrics"
    complete_metrics.mkdir(parents=True)
    complete_fact = complete_metrics / "fact_episodes.parquet"
    pl.DataFrame({"value": [1]}).write_parquet(complete_fact)
    write_episode_stamp(complete_metrics)

    target = root / "legacy" / "episodes" / "episode_000" / "metrics" / "fact_episodes.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"resident")

    moved = retire_legacy_outputs(root)

    assert target in moved
    assert target.read_bytes() == original
    backups = sorted(target.parent.glob(f"{target.name}.*"))
    assert [path.read_bytes() for path in backups] == [b"resident"]
    assert (root / "legacy" / "metrics.parquet").read_bytes() == b"benchmark-metrics"
    assert (root / "legacy" / "combined_metrics.parquet").read_bytes() == b"combined-metrics"
    assert complete_fact.is_file()
    assert (complete_metrics / ".complete").is_file()
    assert retire_legacy_outputs(root) == ()


def test_enforce_writes_a_readable_report_for_a_dimension_violation(
    tmp_path: pathlib.Path,
) -> None:
    root = tmp_path / "benchmarks" / "clean"
    root.mkdir(parents=True)
    dimensions = _dimension_frames()
    dimensions["dim_metrics"] = dimensions["dim_metrics"].with_columns(pl.col("metric").cast(pl.Int64))

    with pytest.raises(ContractViolationError) as error:
        enforce_benchmark_contract(root, dimensions=dimensions)

    assert "dim_metrics" in str(error.value)
    payload = yaml.safe_load((root / "contract_report.yaml").read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert any(violation["table"] == "dim_metrics" for violation in payload["violations"])


def test_an_interrupted_dimension_write_keeps_the_previous_file(
    tmp_path: pathlib.Path,
) -> None:
    dimensions = _dimension_frames()
    write_benchmark_dimensions(dimensions, tmp_path)
    previous = pl.read_parquet(tmp_path / "dim_episodes.parquet")

    broken = dict(dimensions)
    broken["dim_robots"] = pl.DataFrame({"metric": pl.Series([object()], dtype=pl.Object)})
    with pytest.raises(pl.exceptions.PolarsError):
        write_benchmark_dimensions(broken, tmp_path)

    assert pl.read_parquet(tmp_path / "dim_episodes.parquet").equals(previous)
    assert [path for path in tmp_path.iterdir() if path.name.endswith(".tmp")] == []


def test_a_forced_process_retires_a_stale_benchmark_stamp(
    tmp_path: pathlib.Path,
) -> None:
    """The stamp of an older layout moves to legacy/ with the stale outputs, so the rewritten benchmark passes the check."""
    root = tmp_path / "benchmarks" / "stamped"
    root.mkdir(parents=True)
    stale_stamp = root / "benchmark_layout.yaml"
    stale_stamp.write_text("layout_version: 1\nschema_hash: older\n", encoding="utf-8")

    retire_legacy_outputs(root)

    assert not stale_stamp.exists()
    assert (root / "legacy" / "benchmark_layout.yaml").is_file()
    check_benchmark_layout(root)


def test_retiring_keeps_a_current_benchmark_stamp(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "benchmarks" / "current"
    root.mkdir(parents=True)
    write_benchmark_stamp(root)

    retire_legacy_outputs(root)

    assert (root / "benchmark_layout.yaml").is_file()
