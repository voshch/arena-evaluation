"""Publish-time contract checks: the registry schema, keys, orphans, windows and a reason for every NULL."""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Mapping
from typing import Final

import polars as pl
import yaml

from arena_evaluation.storage.exceptions import ContractViolationError
from arena_evaluation.storage.registry.dimensions import DIMENSION_TABLES
from arena_evaluation.storage.registry.facts import FACT_TABLES
from arena_evaluation.storage.registry.null_reasons import (
    NULL_REASON_MEANINGS,
    NULL_REASON_PARAMETERISED,
)
from arena_evaluation.storage.registry.scopes import FactTable
from arena_evaluation.storage.registry.table import NULL_REASON_COLUMN, Table

__all__ = [
    "ContractReport",
    "ContractViolation",
    "check_benchmark",
    "enforce_benchmark_contract",
    "enforce_dimension_contract",
    "is_known_null_reason",
    "write_contract_report",
]

#: Fixed-size ``Array`` is nested too; the 1NF rule bans it alongside List/Struct/Object.
_NESTED_DTYPES: Final[tuple[type[pl.DataType], ...]] = (
    pl.List,
    pl.Struct,
    pl.Object,
    pl.Array,
)
_IDENTITY_KEYS: Final[tuple[str, ...]] = ("run_id", "episode_id", "robot_id")

#: The dimension a fact row must resolve against, and the columns the join uses.
_FACT_PARENT: Final[dict[str, tuple[str, tuple[str, ...]]]] = {
    FactTable.EPISODES.value: ("dim_episodes", _IDENTITY_KEYS),
    FactTable.ENCOUNTERS.value: ("dim_episodes", _IDENTITY_KEYS),
    FactTable.CONTACTS.value: ("dim_episodes", _IDENTITY_KEYS),
    FactTable.PEDESTRIANS.value: (
        "dim_pedestrians",
        ("run_id", "episode_id", "env_id", "ped_id"),
    ),
    FactTable.ROBOTS.value: (
        "dim_robots",
        ("run_id", "episode_id", "env_id", "robot_id"),
    ),
    FactTable.EDGE_CASES.value: ("dim_episodes", _IDENTITY_KEYS),
    FactTable.FORMATIONS.value: ("dim_episodes", _IDENTITY_KEYS),
    FactTable.PHASE_OUTCOMES.value: ("dim_episodes", _IDENTITY_KEYS),
    FactTable.JUDGE_DISAGREEMENTS.value: ("dim_episodes", _IDENTITY_KEYS),
    FactTable.MECHANISM_STATES.value: ("dim_episodes", _IDENTITY_KEYS),
    FactTable.COLLISIONS.value: ("dim_episodes", _IDENTITY_KEYS),
}

#: (fact table, start column, end column) of every windowed fact.
_WINDOW_COLUMNS: Final[tuple[tuple[str, str, str], ...]] = (
    (FactTable.ENCOUNTERS.value, "t_start_ns", "t_end_ns"),
    (FactTable.CONTACTS.value, "t_onset_ns", "t_end_ns"),
)


@dataclasses.dataclass(frozen=True, slots=True)
class ContractViolation:
    """One broken contract rule, named by table and column."""

    table: str
    column: str
    check: str
    detail: str

    def message(self) -> str:
        """``table.column: check: detail``, the form the raised error and the report use."""
        return f"{self.table}.{self.column}: {self.check}: {self.detail}"


@dataclasses.dataclass(frozen=True, slots=True)
class ContractReport:
    """Every contract violation found in one benchmark, empty when the benchmark is clean."""

    benchmark_dir: pathlib.Path
    violations: tuple[ContractViolation, ...]
    checked_tables: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True when no contract rule was broken."""
        return not self.violations

    def summary(self) -> str:
        """One line per violation, prefixed by the benchmark path."""
        head = f"benchmark contract failed for {self.benchmark_dir}: {len(self.violations)} violation(s)"
        return "\n".join(
            [head, *(f"- {violation.message()}" for violation in self.violations)]
        )

    def as_dict(self) -> dict[str, object]:
        """The report as plain data, for the YAML artifact."""
        return {
            "benchmark_dir": str(self.benchmark_dir),
            "ok": self.ok,
            "checked_tables": list(self.checked_tables),
            "violations": [
                {
                    "table": violation.table,
                    "column": violation.column,
                    "check": violation.check,
                    "detail": violation.detail,
                }
                for violation in self.violations
            ],
        }


def enforce_dimension_contract(dimensions: Mapping[str, pl.DataFrame]) -> None:
    """Raise ContractViolationError unless every dimension has the registry schema, unique non-NULL keys and a reason for each NULL."""
    for name, table in DIMENSION_TABLES.items():
        if name not in dimensions:
            raise ContractViolationError(f"{name}: the benchmark has no such dimension")
        frame = dimensions[name]
        expected = table.polars_schema()
        if frame.schema != expected:
            raise ContractViolationError(
                f"{name}: columns or dtypes differ from the registry: {dict(frame.schema)} != {dict(expected)}"
            )
        _check_dimension_keys(name, frame, table)
        _check_dimension_null_reasons(name, frame, table)


def _check_dimension_keys(name: str, frame: pl.DataFrame, table: type[Table]) -> None:
    keys = list(table.key)
    if (
        frame.height
        and frame.select(pl.any_horizontal(pl.col(keys).is_null()).any()).item()
    ):
        raise ContractViolationError(f"{name}: a key column is NULL")
    duplicated = frame.group_by(keys).len().filter(pl.col("len") > 1)
    if duplicated.height:
        raise ContractViolationError(
            f"{name}: {duplicated.height} duplicated key(s) on {keys}"
        )


def _check_dimension_null_reasons(
    name: str, frame: pl.DataFrame, table: type[Table]
) -> None:
    for column in table.null_reason_columns():
        reasoned = (
            pl.col(NULL_REASON_COLUMN)
            .fill_null("")
            .str.contains(column + ":", literal=True)
        )
        unexplained = frame.filter(pl.col(column).is_null() & ~reasoned)
        if unexplained.height:
            raise ContractViolationError(
                f"{name}.{column}: {unexplained.height} NULL value(s) without a reason"
            )


def _episode_directories(benchmark_dir: pathlib.Path) -> tuple[pathlib.Path, ...]:
    """The ``episodes/episode_*`` folders of a benchmark, sorted."""
    episodes = benchmark_dir / "episodes"
    if not episodes.is_dir():
        return ()
    return tuple(sorted(path for path in episodes.glob("episode_*") if path.is_dir()))


def _empty(table: type[Table]) -> pl.DataFrame:
    """A frame without rows, with the tables's declared columns and dtypes."""
    return pl.DataFrame(schema=table.polars_schema())


def _fact_views(
    benchmark_dir: pathlib.Path, episodes: tuple[pathlib.Path, ...]
) -> dict[str, pl.LazyFrame]:
    """Every fact table of the benchmark, stacked over its episodes; an absent table is empty."""
    views: dict[str, pl.LazyFrame] = {}
    for table in FACT_TABLES.values():
        paths = [
            episode / "metrics" / table.file_name()
            for episode in episodes
            if (episode / "metrics" / table.file_name()).is_file()
        ]
        views[_fact_key(table)] = (
            pl.concat([pl.scan_parquet(path) for path in paths])
            if paths
            else _empty(table).lazy()
        )
    return views


def _dimension_views(
    benchmark_dir: pathlib.Path,
    dimensions: Mapping[str, pl.DataFrame] | None,
) -> dict[str, pl.LazyFrame]:
    """The benchmark dimensions: the in-memory frames when given, else the stored files, else empty."""
    views: dict[str, pl.LazyFrame] = {}
    for name, table in DIMENSION_TABLES.items():
        supplied = None if dimensions is None else dimensions.get(name)
        path = benchmark_dir / table.relative_path()
        if supplied is not None:
            views[name] = supplied.lazy()
        elif path.is_file():
            views[name] = pl.scan_parquet(path)
        else:
            views[name] = _empty(table).lazy()
    return views


def _nested_columns(schema: Mapping[str, pl.DataType]) -> tuple[str, ...]:
    """Columns whose dtype is List, Struct, Object or Array."""
    return tuple(
        name for name, dtype in schema.items() if isinstance(dtype, _NESTED_DTYPES)
    )


def _schema_violations(
    table: str, actual: pl.Schema, expected: pl.Schema
) -> list[ContractViolation]:
    """Columns the stored schema adds, drops or types differently from its declaration."""
    checked = [
        (name, "schema_missing") for name in expected.names() if name not in actual
    ]
    checked += [
        (name, "schema_unexpected") for name in actual.names() if name not in expected
    ]
    checked += [
        (name, "schema_dtype")
        for name in expected.names()
        if name in actual and actual[name] != expected[name]
    ]
    return [
        ContractViolation(
            table=table,
            column=column,
            check=kind,
            detail=(
                f"expected {expected[column]!r}, stored {actual[column]!r}"
                if kind == "schema_dtype"
                else f"declared columns: {list(expected.names())}"
            ),
        )
        for column, kind in checked
    ]


def _file_schema_violations(
    table: str, path: pathlib.Path, expected: pl.Schema
) -> list[ContractViolation]:
    """The schema and flatness checks of one stored Parquet file."""
    actual = pl.scan_parquet(path).collect_schema()
    violations = _schema_violations(table, actual, expected)
    violations.extend(
        ContractViolation(
            table=table,
            column=column,
            check="nested_dtype",
            detail=f"stored {actual[column]!r} in {path.name}",
        )
        for column in _nested_columns(actual)
    )
    return violations


def _check_stored_schemas(
    benchmark_dir: pathlib.Path,
    episodes: tuple[pathlib.Path, ...],
    *,
    check_dimension_files: bool,
) -> list[ContractViolation]:
    """Every stored fact file, and every stored dimension when read from disk, is declared and flat."""
    violations: list[ContractViolation] = []
    if check_dimension_files:
        for name, table in DIMENSION_TABLES.items():
            path = benchmark_dir / table.relative_path()
            if path.is_file():
                violations.extend(
                    _file_schema_violations(name, path, table.polars_schema())
                )
    known = {table.file_name(): table for table in FACT_TABLES.values()}
    for episode in episodes:
        metrics = episode / "metrics"
        if not metrics.is_dir():
            continue
        for path in sorted(metrics.glob("fact_*.parquet")):
            table = known.get(path.name)
            if table is None:
                violations.append(
                    ContractViolation(
                        table=path.stem,
                        column="",
                        check="table_unknown",
                        detail=f"{path} is not a declared fact table",
                    )
                )
                continue
            violations.extend(
                _file_schema_violations(_fact_key(table), path, table.polars_schema())
            )
    return violations


def _check_dimension_frames(
    dimensions: Mapping[str, pl.DataFrame],
) -> list[ContractViolation]:
    """In-memory dimensions are written only after they pass their declared flat schema."""
    violations: list[ContractViolation] = []
    for name, frame in dimensions.items():
        table = DIMENSION_TABLES.get(name)
        if table is None:
            continue
        schema = frame.schema
        violations.extend(_schema_violations(name, schema, table.polars_schema()))
        violations.extend(
            ContractViolation(
                table=name,
                column=column,
                check="nested_dtype",
                detail=f"in-memory {schema[column]!r}",
            )
            for column in _nested_columns(schema)
        )
    return violations


def _check_missing_dimensions(
    benchmark_dir: pathlib.Path, views: Mapping[str, pl.LazyFrame]
) -> list[ContractViolation]:
    """A benchmark that stores facts must also store the dimensions they resolve against."""
    has_facts = any(
        views[_fact_key(table)].select(pl.len()).collect().item() > 0
        for table in FACT_TABLES.values()
    )
    if not has_facts:
        return []
    return [
        ContractViolation(
            table=name,
            column="run_id",
            check="dimension_missing",
            detail=f"{benchmark_dir / table.relative_path()} is absent",
        )
        for name, table in DIMENSION_TABLES.items()
        if not (benchmark_dir / table.relative_path()).is_file()
    ]


def _fact_key(table: type[Table]) -> str:
    """The name a fact table is stored and reported under: its file stem."""
    return f"fact_{table.table_name}"


def _check_keys(views: Mapping[str, pl.LazyFrame]) -> list[ContractViolation]:
    """Every fact row key is non-null and, where the table has one, unique."""
    violations: list[ContractViolation] = []
    for table in FACT_TABLES.values():
        table_name = _fact_key(table)
        frame = views[table_name]
        keys = list(table.key)
        null_counts = (
            frame.select([pl.col(key).is_null().sum().alias(key) for key in keys])
            .collect()
            .row(0, named=True)
        )
        for key, count in null_counts.items():
            if count:
                violations.append(
                    ContractViolation(
                        table=table_name,
                        column=key,
                        check="key_null",
                        detail=f"{count} rows",
                    )
                )
        if not table.unique_key:
            continue
        duplicates = (
            frame.group_by(keys)
            .agg(pl.len().alias("rows"))
            .filter(pl.col("rows") > 1)
            .collect()
        )
        if duplicates.height:
            violations.append(
                ContractViolation(
                    table=table_name,
                    column=",".join(keys),
                    check="key_duplicate",
                    detail=(
                        f"{duplicates.height} duplicated keys, first "
                        f"{duplicates.row(0, named=True)}"
                    ),
                )
            )
    return violations


def _check_orphans(views: Mapping[str, pl.LazyFrame]) -> list[ContractViolation]:
    """Every fact row resolves to a row of its dimension."""
    violations: list[ContractViolation] = []
    for name, (dimension, keys) in _FACT_PARENT.items():
        table_name = f"fact_{name}"
        dimension_keys = views[dimension].select(list(keys)).unique()
        orphan_count = (
            views[table_name]
            .select(list(keys))
            .join(dimension_keys, on=list(keys), how="anti")
            .select(pl.len())
            .collect()
            .item()
        )
        if orphan_count:
            violations.append(
                ContractViolation(
                    table=table_name,
                    column=",".join(keys),
                    check="orphan_fact",
                    detail=f"{orphan_count} rows have no {dimension} row",
                )
            )
    return violations


def _check_windows(views: Mapping[str, pl.LazyFrame]) -> list[ContractViolation]:
    """A windowed fact row starts strictly before it ends."""
    violations: list[ContractViolation] = []
    for name, start_column, end_column in _WINDOW_COLUMNS:
        table_name = f"fact_{name}"
        invalid = views[table_name].filter(
            pl.col(start_column).is_not_null()
            & pl.col(end_column).is_not_null()
            & (pl.col(start_column) >= pl.col(end_column))
        )
        count = invalid.select(pl.len()).collect().item()
        if count:
            violations.append(
                ContractViolation(
                    table=table_name,
                    column=f"{start_column},{end_column}",
                    check="window_order",
                    detail=f"{count} rows with {start_column} >= {end_column}",
                )
            )
    return violations


def _check_actor_pairs(views: Mapping[str, pl.LazyFrame]) -> list[ContractViolation]:
    """An encounter never pairs an actor with itself."""
    invalid = views[f"fact_{FactTable.ENCOUNTERS.value}"].filter(
        pl.col("actor_1").is_not_null() & (pl.col("actor_1") == pl.col("actor_2"))
    )
    count = invalid.select(pl.len()).collect().item()
    if not count:
        return []
    return [
        ContractViolation(
            table="fact_encounters",
            column="actor_1,actor_2",
            check="actor_self_pair",
            detail=f"{count} encounters pair an actor with itself",
        )
    ]


def _check_null_reasons(views: Mapping[str, pl.LazyFrame]) -> list[ContractViolation]:
    """Every NULL metric cell on a fact row is explained by that row's ``null_reason``."""
    violations: list[ContractViolation] = []
    for table in FACT_TABLES.values():
        columns = table.null_reason_columns()
        if not columns:
            continue
        frame = views[_fact_key(table)]
        unexplained = frame.filter(
            pl.any_horizontal([pl.col(column).is_null() for column in columns])
            & (
                pl.col(NULL_REASON_COLUMN).is_null()
                | (pl.col(NULL_REASON_COLUMN) == "")
            )
        )
        count = unexplained.select(pl.len()).collect().item()
        if not count:
            continue
        null_flags = (
            unexplained.select(
                [pl.col(column).is_null().any().alias(column) for column in columns]
            )
            .collect()
            .row(0, named=True)
        )
        null_columns = [column for column, is_null in null_flags.items() if is_null]
        violations.append(
            ContractViolation(
                table=_fact_key(table),
                column=",".join(null_columns),
                check="null_reason_missing",
                detail=f"{count} rows with NULL metric cells and no null_reason",
            )
        )
    return violations


def split_null_reason_item(item: str) -> tuple[str, str, str]:
    """``(column, code, parameter)`` of one ``;``-separated provenance item; ``column`` is empty for a bare code."""
    first, separator, rest = item.partition(":")
    if not separator:
        return "", item, ""
    if first in NULL_REASON_PARAMETERISED:
        return "", first, rest
    code, _, parameter = rest.partition(":")
    return first, code, parameter


def is_known_null_reason(text: str) -> bool:
    """Whether every item of a published ``null_reason`` is a vocabulary code with its parameter."""
    return all(_is_known_item(item) for item in text.split(";") if item)


def _is_known_item(item: str) -> bool:
    """One provenance item is a known code, and a parameterised one carries its parameter."""
    _, code, parameter = split_null_reason_item(item)
    if code in NULL_REASON_PARAMETERISED:
        return bool(parameter)
    return code in NULL_REASON_MEANINGS


def _check_null_reason_vocabulary(
    views: Mapping[str, pl.LazyFrame],
) -> list[ContractViolation]:
    """Every emitted ``null_reason`` is a code of the closed vocabulary, parameter included."""
    violations: list[ContractViolation] = []
    for table in FACT_TABLES.values():
        emitted = (
            views[_fact_key(table)]
            .select(pl.col(NULL_REASON_COLUMN).drop_nulls().unique())
            .collect()
            .get_column(NULL_REASON_COLUMN)
        )
        unknown = [value for value in emitted if not is_known_null_reason(value)]
        if unknown:
            violations.append(
                ContractViolation(
                    table=_fact_key(table),
                    column=NULL_REASON_COLUMN,
                    check="null_reason_unknown",
                    detail=(
                        f"{len(unknown)} code(s) outside the vocabulary, "
                        f"first {unknown[0]!r}"
                    ),
                )
            )
    return violations


def _check_robot_descriptors(
    views: Mapping[str, pl.LazyFrame],
) -> list[ContractViolation]:
    """Every robot a row names has a recorded descriptor: its model is not NULL."""
    descriptors = views["dim_robots"].filter(pl.col("robot_model").is_null())
    count = descriptors.select(pl.len()).collect().item()
    if not count:
        return []
    return [
        ContractViolation(
            table="dim_robots",
            column="robot_model",
            check="descriptor_missing",
            detail=f"{count} robot(s) have no recorded robot_descriptor row",
        )
    ]


def check_benchmark(
    path: pathlib.Path,
    *,
    dimensions: Mapping[str, pl.DataFrame] | None = None,
) -> ContractReport:
    """Run every publish-time check over the benchmark stored at ``path``.

    ``dimensions`` lets the publish boundary validate the dimensions it is about to write,
    before any benchmark-level file exists; without it the stored ``dim_*.parquet`` are read.
    """
    benchmark_dir = pathlib.Path(path)
    episode_dirs = _episode_directories(benchmark_dir)
    views = _fact_views(benchmark_dir, episode_dirs)
    views.update(_dimension_views(benchmark_dir, dimensions))
    violations = [
        *_check_stored_schemas(
            benchmark_dir, episode_dirs, check_dimension_files=dimensions is None
        ),
        *(_check_dimension_frames(dimensions) if dimensions is not None else []),
        *(
            _check_missing_dimensions(benchmark_dir, views)
            if dimensions is None
            else []
        ),
        *_check_keys(views),
        *_check_orphans(views),
        *_check_windows(views),
        *_check_actor_pairs(views),
        *_check_null_reasons(views),
        *_check_null_reason_vocabulary(views),
        *_check_robot_descriptors(views),
    ]
    return ContractReport(
        benchmark_dir=benchmark_dir,
        violations=tuple(violations),
        checked_tables=tuple(views),
    )


def write_contract_report(report: ContractReport, path: pathlib.Path) -> pathlib.Path:
    """Write the report as YAML, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(report.as_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def enforce_benchmark_contract(
    benchmark_dir: pathlib.Path,
    *,
    dimensions: Mapping[str, pl.DataFrame] | None = None,
) -> ContractReport:
    """Check the benchmark, write its report and raise when a rule is broken.

    Called at the publish boundary with the dimensions still in memory: a failed check stops
    processing before any benchmark-level table is published, and the YAML report keeps every
    table and column that broke.
    """
    report = check_benchmark(benchmark_dir, dimensions=dimensions)
    if report.ok:
        return report
    write_contract_report(report, benchmark_dir / "contract_report.yaml")
    raise ContractViolationError(report.summary())
