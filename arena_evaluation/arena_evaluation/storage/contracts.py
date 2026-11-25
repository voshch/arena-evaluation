"""Publish-time contract checks: the registry schema, keys, orphans, windows and a reason for every NULL."""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Callable, Mapping
from typing import Final

import polars as pl
import yaml

from arena_evaluation.storage.atomic import staged_temp_path
from arena_evaluation.storage.exceptions import ContractViolationError
from arena_evaluation.storage.layout import check_benchmark_layout
from arena_evaluation.storage.registry.dimensions import DIMENSION_TABLES
from arena_evaluation.storage.registry.facts import FACT_TABLES
from arena_evaluation.storage.registry.null_reasons import (
    NULL_REASON_MEANINGS,
    NULL_REASON_PARAMETERISED,
)
from arena_evaluation.storage.registry.scopes import FactTable
from arena_evaluation.storage.registry.table import (
    NULL_REASON_COLUMN,
    Table,
    TableKind,
    registered_tables,
)
from arena_evaluation.storage.registry.telemetry import (
    duplicate_row_keys,
    null_row_keys,
)

__all__ = [
    "ContractReport",
    "ContractViolation",
    "check_benchmark",
    "enforce_benchmark_contract",
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

#: The recorded robot parameters a metric needs; a NULL one carries its own null reason.
_REQUIRED_ROBOT_PARAMETERS: Final[tuple[str, ...]] = ("robot_radius", "robot_mass_kg")

#: ``FactEpisodes`` stores its outcome in ``result``; the table declares no separate status column.
_EPISODE_STATUS_COLUMN: Final[str] = "result"

#: The terminal verdicts a recorded outcome can carry; any other value is a status token.
_EVALUATED_EPISODE_RESULTS: Final[tuple[str, ...]] = (
    "GOAL_REACHED",
    "FAILED",
    "CANCELLED",
    "FATAL",
    "COLLISION",
    "UNRESOLVED",
)

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
    FactTable.TASK_OUTCOMES.value: ("dim_episodes", _IDENTITY_KEYS),
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
        return "\n".join([head, *(f"- {violation.message()}" for violation in self.violations)])

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


def _episode_directories(benchmark_dir: pathlib.Path) -> tuple[pathlib.Path, ...]:
    """The ``episodes/episode_*`` folders of a benchmark, sorted."""
    episodes = benchmark_dir / "episodes"
    if not episodes.is_dir():
        return ()
    return tuple(sorted(path for path in episodes.glob("episode_*") if path.is_dir()))


def _empty(table: type[Table]) -> pl.DataFrame:
    """A frame without rows, with the tables's declared columns and dtypes."""
    return pl.DataFrame(schema=table.polars_schema())


def _named_first(values: pl.Series) -> str:
    """The first three values of a scalar column, joined for a violation detail."""
    return values.cast(pl.String).head(3).str.join(", ").item()


def _fact_key(table: type[Table]) -> str:
    """The name a fact table is stored and reported under: its file stem."""
    return f"fact_{table.table_name}"


def _views(
    benchmark_dir: pathlib.Path,
    episodes: tuple[pathlib.Path, ...],
    dimensions: Mapping[str, pl.DataFrame] | None,
) -> dict[str, pl.LazyFrame]:
    """Every fact table stacked over the episodes, and every dimension from memory, disk or empty."""
    views: dict[str, pl.LazyFrame] = {}
    for table in FACT_TABLES.values():
        paths = [episode / "metrics" / table.file_name() for episode in episodes if (episode / "metrics" / table.file_name()).is_file()]
        views[_fact_key(table)] = pl.concat([pl.scan_parquet(path) for path in paths]) if paths else _empty(table).lazy()
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
    return tuple(name for name, dtype in schema.items() if isinstance(dtype, _NESTED_DTYPES))


def _schema_violations(
    table: str,
    actual: pl.Schema,
    expected: pl.Schema,
    describe: Callable[[pl.DataType], str],
) -> list[ContractViolation]:
    """Columns the stored schema adds, drops or types differently, plus every nested column."""
    checked = [(name, "schema_missing") for name in expected.names() if name not in actual]
    checked += [(name, "schema_unexpected") for name in actual.names() if name not in expected]
    checked += [(name, "schema_dtype") for name in expected.names() if name in actual and actual[name] != expected[name]]
    violations = [
        ContractViolation(
            table=table,
            column=column,
            check=kind,
            detail=(f"expected {expected[column]!r}, stored {actual[column]!r}" if kind == "schema_dtype" else f"declared columns: {list(expected.names())}"),
        )
        for column, kind in checked
    ]
    violations.extend(
        ContractViolation(
            table=table,
            column=column,
            check="nested_dtype",
            detail=describe(actual[column]),
        )
        for column in _nested_columns(actual)
    )
    return violations


def _file_schema_violations(table: str, path: pathlib.Path, expected: pl.Schema) -> list[ContractViolation]:
    """The schema and flatness checks of one stored Parquet file."""
    actual = pl.scan_parquet(path).collect_schema()
    return _schema_violations(table, actual, expected, lambda dtype: f"stored {dtype!r} in {path.name}")


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
                violations.extend(_file_schema_violations(name, path, table.polars_schema()))
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
            violations.extend(_file_schema_violations(_fact_key(table), path, table.polars_schema()))
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
        violations.extend(
            _schema_violations(
                name,
                frame.schema,
                table.polars_schema(),
                lambda dtype: f"in-memory {dtype!r}",
            )
        )
    return violations


def _check_missing_dimensions(benchmark_dir: pathlib.Path, views: Mapping[str, pl.LazyFrame]) -> list[ContractViolation]:
    """A benchmark that stores facts must also store the dimensions they resolve against."""
    has_facts = any(views[_fact_key(table)].select(pl.len()).collect().item() > 0 for table in FACT_TABLES.values())
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


def _check_episode_coverage(
    views: Mapping[str, pl.LazyFrame],
) -> list[ContractViolation]:
    """Every dimension episode has a fact row, and every fact episode has an evaluated result."""
    missing = (
        views["dim_episodes"]
        .select("run_id", "episode_id")
        .unique()
        .join(
            views["fact_episodes"].select("run_id", "episode_id").unique(),
            on=["run_id", "episode_id"],
            how="anti",
        )
        .select("episode_id")
        .unique()
        .collect()
    )
    unevaluated = (
        views["fact_episodes"]
        .filter(pl.col(_EPISODE_STATUS_COLUMN).is_not_null() & ~pl.col(_EPISODE_STATUS_COLUMN).is_in(_EVALUATED_EPISODE_RESULTS))
        .select(
            pl.concat_str(
                [
                    pl.lit("episode "),
                    pl.col("episode_id").cast(pl.String),
                    pl.lit(" ("),
                    pl.col(_EPISODE_STATUS_COLUMN),
                    pl.lit(")"),
                ]
            ).alias("episode")
        )
        .unique()
        .collect()
    )
    violations: list[ContractViolation] = []
    if not missing.is_empty():
        violations.append(
            ContractViolation(
                table="dim_episodes",
                column="episode_id",
                check="episode_uncovered",
                detail=(f"{missing.height} episode(s) have no fact_episodes row; first {_named_first(missing.get_column('episode_id'))}"),
            )
        )
    if not unevaluated.is_empty():
        violations.append(
            ContractViolation(
                table="fact_episodes",
                column=_EPISODE_STATUS_COLUMN,
                check="episode_not_evaluated",
                detail=(f"{unevaluated.height} episode(s) have no evaluated result; first {_named_first(unevaluated.get_column('episode'))}"),
            )
        )
    return violations


def _check_row_keys(views: Mapping[str, pl.LazyFrame]) -> list[ContractViolation]:
    """Fact keys are non-null and unique; dimension keys too, and every NULL dimension value carries a reason."""
    violations: list[ContractViolation] = []
    for table in FACT_TABLES.values():
        table_name = _fact_key(table)
        frame = views[table_name]
        keys = list(table.key)
        null_counts = frame.select([pl.col(key).is_null().sum().alias(key) for key in keys]).collect().row(0, named=True)
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
        duplicates = frame.group_by(keys).agg(pl.len().alias("rows")).filter(pl.col("rows") > 1).collect()
        if duplicates.height:
            violations.append(
                ContractViolation(
                    table=table_name,
                    column=",".join(keys),
                    check="key_duplicate",
                    detail=(f"{duplicates.height} duplicated keys, first {duplicates.row(0, named=True)}"),
                )
            )
    for name, table in DIMENSION_TABLES.items():
        frame = views[name].collect()
        if frame.height == 0:
            continue
        keys = list(table.key)
        null_keys = frame.select(pl.any_horizontal(pl.col(keys).is_null()).sum()).item()
        if null_keys:
            violations.append(
                ContractViolation(
                    table=name,
                    column=",".join(keys),
                    check="key_null",
                    detail=f"{null_keys} row(s) with a NULL key",
                )
            )
        duplicates = frame.group_by(keys).len().filter(pl.col("len") > 1)
        if duplicates.height:
            violations.append(
                ContractViolation(
                    table=name,
                    column=",".join(keys),
                    check="key_duplicate",
                    detail=f"{duplicates.height} duplicated keys on {keys}",
                )
            )
        for column in table.null_reason_columns():
            reasoned = pl.col(NULL_REASON_COLUMN).fill_null("").str.contains(column + ":", literal=True)
            unexplained = frame.filter(pl.col(column).is_null() & ~reasoned).height
            if unexplained:
                violations.append(
                    ContractViolation(
                        table=name,
                        column=column,
                        check="null_reason_missing",
                        detail=f"{unexplained} NULL value(s) without a reason",
                    )
                )
    return violations


def _check_orphans(views: Mapping[str, pl.LazyFrame]) -> list[ContractViolation]:
    """Every fact row resolves to a row of its dimension."""
    violations: list[ContractViolation] = []
    for name, (dimension, keys) in _FACT_PARENT.items():
        table_name = f"fact_{name}"
        dimension_keys = views[dimension].select(list(keys)).unique()
        orphan_count = views[table_name].select(list(keys)).join(dimension_keys, on=list(keys), how="anti").select(pl.len()).collect().item()
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


def _check_fact_windows(views: Mapping[str, pl.LazyFrame]) -> list[ContractViolation]:
    """A windowed fact row starts strictly before it ends, and an encounter never pairs an actor with itself."""
    violations: list[ContractViolation] = []
    for name, start_column, end_column in _WINDOW_COLUMNS:
        table_name = f"fact_{name}"
        invalid = views[table_name].filter(pl.col(start_column).is_not_null() & pl.col(end_column).is_not_null() & (pl.col(start_column) >= pl.col(end_column)))
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
    invalid = views[f"fact_{FactTable.ENCOUNTERS.value}"].filter(pl.col("actor_1").is_not_null() & (pl.col("actor_1") == pl.col("actor_2")))
    count = invalid.select(pl.len()).collect().item()
    if count:
        violations.append(
            ContractViolation(
                table="fact_encounters",
                column="actor_1,actor_2",
                check="actor_self_pair",
                detail=f"{count} encounters pair an actor with itself",
            )
        )
    return violations


def _check_null_reasons(views: Mapping[str, pl.LazyFrame]) -> list[ContractViolation]:
    """Every NULL metric cell on a fact row is explained, and every emitted reason is a vocabulary code."""
    violations: list[ContractViolation] = []
    for table in FACT_TABLES.values():
        columns = table.null_reason_columns()
        if not columns:
            continue
        frame = views[_fact_key(table)]
        unexplained = frame.filter(pl.any_horizontal([pl.col(column).is_null() for column in columns]) & (pl.col(NULL_REASON_COLUMN).is_null() | (pl.col(NULL_REASON_COLUMN) == "")))
        count = unexplained.select(pl.len()).collect().item()
        if not count:
            continue
        null_flags = unexplained.select([pl.col(column).is_null().any().alias(column) for column in columns]).collect().row(0, named=True)
        null_columns = [column for column, is_null in null_flags.items() if is_null]
        violations.append(
            ContractViolation(
                table=_fact_key(table),
                column=",".join(null_columns),
                check="null_reason_missing",
                detail=f"{count} rows with NULL metric cells and no null_reason",
            )
        )
    for table in FACT_TABLES.values():
        emitted = views[_fact_key(table)].select(pl.col(NULL_REASON_COLUMN).drop_nulls().unique()).collect().get_column(NULL_REASON_COLUMN)
        unknown = [value for value in emitted if not is_known_null_reason(value)]
        if unknown:
            violations.append(
                ContractViolation(
                    table=_fact_key(table),
                    column=NULL_REASON_COLUMN,
                    check="null_reason_unknown",
                    detail=(f"{len(unknown)} code(s) outside the vocabulary, first {unknown[0]!r}"),
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


def _telemetry_robot_ids(
    episodes: tuple[pathlib.Path, ...],
) -> pl.LazyFrame | None:
    """The robot id column of every stored robot telemetry file, None when none is stored."""
    paths = [path for episode in episodes for path in sorted((episode / "topics").glob("robot_*.parquet"))]
    if not paths:
        return None
    return pl.concat([pl.scan_parquet(path).select("robot_id").filter(pl.col("robot_id").is_not_null()) for path in paths]).unique()


def _check_robot_descriptors(
    views: Mapping[str, pl.LazyFrame],
) -> list[ContractViolation]:
    """Every dim_robots row records its model and the required recorded parameters."""
    violations: list[ContractViolation] = []
    descriptors = views["dim_robots"].filter(pl.col("robot_model").is_null())
    count = descriptors.select(pl.len()).collect().item()
    if count:
        violations.append(
            ContractViolation(
                table="dim_robots",
                column="robot_model",
                check="descriptor_missing",
                detail=f"{count} robot(s) have no recorded robot_descriptor row",
            )
        )
    robots = views["dim_robots"].select("robot_id", NULL_REASON_COLUMN, *_REQUIRED_ROBOT_PARAMETERS)
    reason = pl.col(NULL_REASON_COLUMN).fill_null("")
    for parameter in _REQUIRED_ROBOT_PARAMETERS:
        unexplained = robots.filter(pl.col(parameter).is_null() & ~reason.str.contains(f"{parameter}:", literal=True))
        robots_missing = unexplained.select(pl.col("robot_id").n_unique()).collect().item()
        if not robots_missing:
            continue
        names = _named_first(unexplained.select("robot_id").unique().collect().get_column("robot_id"))
        violations.append(
            ContractViolation(
                table="dim_robots",
                column=parameter,
                check="descriptor_param_missing",
                detail=(f"{robots_missing} robot(s) miss {parameter} without a null reason; first {names}"),
            )
        )
    return violations


def _check_telemetry_descriptors(
    views: Mapping[str, pl.LazyFrame],
    *,
    telemetry_robot_ids: pl.LazyFrame | None,
) -> list[ContractViolation]:
    """A robot id with robot telemetry also has a dim_robots row."""
    if telemetry_robot_ids is None:
        return []
    undescribed = (
        telemetry_robot_ids.join(
            views["dim_robots"].select("robot_id").unique(),
            on="robot_id",
            how="anti",
        )
        .select("robot_id")
        .collect()
    )
    if undescribed.is_empty():
        return []
    names = _named_first(undescribed.get_column("robot_id"))
    return [
        ContractViolation(
            table="dim_robots",
            column="robot_id",
            check="descriptor_missing_telemetry",
            detail=(f"{undescribed.height} robot id(s) have telemetry but no dim_robots row; first {names}"),
        )
    ]


def _telemetry_table(file_name: str) -> type[Table] | None:
    """The declared telemetry table a stored ``topics/<file>`` holds, None for a foreign file."""
    for table in registered_tables():
        if table.table_kind is TableKind.TELEMETRY and table.file_name() == file_name:
            return table
    return None


def _check_telemetry_row_keys(
    episodes: tuple[pathlib.Path, ...],
) -> list[ContractViolation]:
    """Every stored telemetry file has a non-null, unique row key; the detail names the episode and a sample."""
    violations: list[ContractViolation] = []
    for episode in episodes:
        topics = episode / "topics"
        if not topics.is_dir():
            continue
        for path in sorted(topics.glob("*.parquet")):
            table = _telemetry_table(path.name)
            if table is None:
                continue
            keys = list(table.row_key)
            try:
                frame = pl.read_parquet(path, columns=keys)
            except pl.exceptions.ColumnNotFoundError as error:
                violations.append(
                    ContractViolation(
                        table=path.stem,
                        column=",".join(keys),
                        check="row_key_missing",
                        detail=f"{path} does not carry its declared row key: {error}",
                    )
                )
                continue
            nulls = null_row_keys(table, frame)
            if nulls.height:
                violations.append(
                    ContractViolation(
                        table=path.stem,
                        column=",".join(keys),
                        check="row_key_null",
                        detail=(f"{nulls.height} row(s) in {episode.name} with a NULL key"),
                    )
                )
            duplicates = duplicate_row_keys(table, frame)
            if duplicates.height:
                sample = ", ".join(f"{key}={value!r}" for key, value in duplicates.row(0, named=True).items())
                violations.append(
                    ContractViolation(
                        table=path.stem,
                        column=",".join(keys),
                        check="row_key_duplicate",
                        detail=(f"{duplicates.height} duplicated row key(s) in {episode.name}, first {sample}"),
                    )
                )
    return violations


def check_benchmark(
    path: pathlib.Path,
    *,
    dimensions: Mapping[str, pl.DataFrame] | None = None,
) -> ContractReport:
    """Run every publish-time check over the benchmark stored at ``path``, refusing one whose outputs predate the native layout."""
    benchmark_dir = pathlib.Path(path)
    check_benchmark_layout(benchmark_dir)
    episode_dirs = _episode_directories(benchmark_dir)
    views = _views(benchmark_dir, episode_dirs, dimensions)
    telemetry_robot_ids = _telemetry_robot_ids(episode_dirs)
    violations = [
        *_check_stored_schemas(benchmark_dir, episode_dirs, check_dimension_files=dimensions is None),
        *(_check_dimension_frames(dimensions) if dimensions is not None else []),
        *(_check_missing_dimensions(benchmark_dir, views) if dimensions is None else []),
        *_check_row_keys(views),
        *_check_orphans(views),
        *(_check_episode_coverage(views) if dimensions is None else []),
        *_check_fact_windows(views),
        *_check_null_reasons(views),
        *(_check_robot_descriptors(views) if dimensions is None else []),
        *_check_telemetry_descriptors(views, telemetry_robot_ids=telemetry_robot_ids),
        *_check_telemetry_row_keys(episode_dirs),
    ]
    return ContractReport(
        benchmark_dir=benchmark_dir,
        violations=tuple(violations),
        checked_tables=tuple(views),
    )


def write_contract_report(report: ContractReport, path: pathlib.Path) -> pathlib.Path:
    """Write the report as YAML in one rename, creating its parent directory."""
    with staged_temp_path(path) as temporary:
        temporary.write_text(
            yaml.safe_dump(report.as_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    return path


def enforce_benchmark_contract(
    benchmark_dir: pathlib.Path,
    *,
    dimensions: Mapping[str, pl.DataFrame] | None = None,
) -> ContractReport:
    """Check the benchmark, write its report and raise when a rule is broken."""
    check_benchmark_layout(benchmark_dir)
    report = check_benchmark(benchmark_dir, dimensions=dimensions)
    if report.ok:
        return report
    write_contract_report(report, benchmark_dir / "contract_report.yaml")
    raise ContractViolationError(report.summary())
