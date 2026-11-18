"""One declaration per stored table: the registry base class, its validation and the views it derives."""

from __future__ import annotations

import dataclasses
import inspect
import pathlib
import re
from collections.abc import Iterable
from enum import StrEnum
from typing import Any, ClassVar, Final

import polars as pl
import pyarrow as pa

from arena_evaluation.storage.exceptions import SchemaViolationError
from arena_evaluation.storage.registry.scopes import FactTable, MetricScope

__all__ = [
    "METRICS",
    "Column",
    "Metric",
    "MetricStore",
    "Table",
    "TableKind",
    "define_metric",
    "defined_metrics",
    "registered_tables",
]

TIME_COLUMN: Final[str] = "t_ns"
NULL_REASON_COLUMN: Final[str] = "null_reason"
DIMENSION_PREFIX: Final[str] = "dim_"
SERIES_PREFIX: Final[str] = "series_"
_SERIES_NAME: Final[re.Pattern[str]] = re.compile(r"series_[a-z][a-z0-9_]*")
GLOBAL_TELEMETRY_NAMES: Final[frozenset[str]] = frozenset(
    {"tf", "tf_static", "world_semantics"}
)
_TELEMETRY_NAME: Final[re.Pattern[str]] = re.compile(
    r"(robot|pedestrian|episode)_[a-z][a-z0-9_]*"
)
_NESTED_DTYPES: Final[tuple[type[pl.DataType], ...]] = (
    pl.List,
    pl.Array,
    pl.Struct,
    pl.Object,
)
_FLOAT_DTYPES: Final[tuple[type[pl.DataType], ...]] = (pl.Float32, pl.Float64)
_ARROW_TYPES: Final[dict[type[pl.DataType], pa.DataType]] = {
    pl.Boolean: pa.bool_(),
    pl.Int64: pa.int64(),
    pl.Float32: pa.float32(),
    pl.Float64: pa.float64(),
    pl.String: pa.string(),
}
_REGISTERED: list[type[Table]] = []
_TABLE_ATTRIBUTES: Final[frozenset[str]] = frozenset(
    {"table_name", "table_kind", "table_scope", "key", "unique_key", "required_by_default", "_columns"}
)


class TableKind(StrEnum):
    """Recorded telemetry, a benchmark dimension, a per-episode fact table, or a per-episode series."""

    TELEMETRY = "telemetry"
    DIMENSION = "dimension"
    FACT = "fact"
    SERIES = "series"


#: The kinds whose tables end with a null_reason column; telemetry and series hold measured samples without one.
_KINDS_WITH_NULL_REASON: Final[frozenset[TableKind]] = frozenset({TableKind.FACT, TableKind.DIMENSION})


@dataclasses.dataclass(frozen=True, slots=True)
class Column:
    """One typed column of a stored table, with the metadata the dictionary and the contract use."""

    dtype: pl.DataType | type[pl.DataType]
    _: dataclasses.KW_ONLY
    description: str = ""
    unit: str = ""
    required: bool | None = None
    null_reason: bool = False

    def __post_init__(self) -> None:
        if self.null_reason and self.required:
            raise SchemaViolationError(
                f"a column cannot be required and carry a null_reason: {self.dtype!r}"
            )


NULL_REASON: Final[Column] = Column(
    pl.String,
    required=False,
    description="Semicolon-separated <column>:<code> reasons of the NULL cells of this row, empty when there are none.",
)


@dataclasses.dataclass(frozen=True, slots=True)
class Metric(Column):
    """A metric, defined once in the metric store, stored at each scope it lists and nowhere else."""

    name: str = dataclasses.field(kw_only=True)
    scopes: tuple[MetricScope, ...] = dataclasses.field(kw_only=True)

    def __post_init__(self) -> None:
        Column.__post_init__(self)
        if not self.scopes:
            raise SchemaViolationError(f"metric {self.name!r} lists no scope")
        if len(set(self.scopes)) != len(self.scopes):
            raise SchemaViolationError(f"metric {self.name!r} lists a scope twice")


class MetricStore:
    """The metrics declared so far: each defined once, with the scopes it is stored at."""

    def __init__(self) -> None:
        self._metrics: dict[str, Metric] = {}

    def define(
        self,
        name: str,
        dtype: pl.DataType | type[pl.DataType],
        *,
        scopes: Iterable[MetricScope],
        unit: str = "",
        description: str = "",
        null_reason: bool = False,
        required: bool | None = None,
    ) -> Metric:
        """Define a metric once; a second definition, or a name a fact table of its scopes already declares, is refused."""
        if name in self._metrics:
            raise SchemaViolationError(f"metric {name!r} is defined twice")
        metric = Metric(
            dtype,
            description=description,
            unit=unit,
            required=required,
            null_reason=null_reason,
            name=name,
            scopes=tuple(scopes),
        )
        _check_dtype(f"metric {name!r}", metric)
        _check_documentation(f"metric {name!r}", metric)
        for table in _REGISTERED:
            if (
                table.table_kind is TableKind.FACT
                and table.table_scope in metric.scopes
                and name in table._columns
            ):
                raise SchemaViolationError(
                    f"metric {name!r} is already a declared column of {table.table_name!r}"
                )
        self._metrics[name] = metric
        return metric

    def all(self) -> tuple[Metric, ...]:
        """Every metric defined so far, in definition order."""
        return tuple(self._metrics.values())

    def at(self, scope: MetricScope) -> tuple[Metric, ...]:
        """The metrics stored at ``scope``, in definition order."""
        return tuple(
            metric for metric in self._metrics.values() if scope in metric.scopes
        )


METRICS: Final[MetricStore] = MetricStore()


def define_metric(
    name: str,
    dtype: pl.DataType | type[pl.DataType],
    *,
    scopes: Iterable[MetricScope],
    unit: str = "",
    description: str = "",
    null_reason: bool = False,
    required: bool | None = None,
) -> Metric:
    """Define a metric in the shared store; each fact table of a listed scope stores it under its name."""
    return METRICS.define(
        name,
        dtype,
        scopes=scopes,
        unit=unit,
        description=description,
        null_reason=null_reason,
        required=required,
    )


def defined_metrics() -> tuple[Metric, ...]:
    """Every metric of the shared store, in definition order."""
    return METRICS.all()


class Table:
    """A stored table declared once: its name, kind, scope, row key and typed columns.

    The table's own metadata lives under ``table_*`` names so that no column name can shadow it.
    A fact table also stores the metrics of the shared store that list its scope.
    """

    table_name: ClassVar[str]
    table_kind: ClassVar[TableKind]
    table_scope: ClassVar[MetricScope | None]
    key: ClassVar[tuple[str, ...]] = ()
    unique_key: ClassVar[bool] = True
    required_by_default: ClassVar[bool] = False
    _columns: ClassVar[dict[str, Column]]

    def __init_subclass__(
        cls,
        *,
        name: str = "",
        kind: TableKind | None = None,
        scope: MetricScope | None = None,
        abstract: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        _refuse_table_attribute_columns(cls)
        if abstract:
            return
        if kind is None:
            raise SchemaViolationError(f"{cls.__name__}: a table declares its kind")
        cls.table_name = name
        cls.table_kind = kind
        cls.table_scope = scope
        cls._columns = _collect_columns(cls)
        if cls.table_kind in _KINDS_WITH_NULL_REASON and NULL_REASON_COLUMN not in cls._columns:
            cls._columns[NULL_REASON_COLUMN] = NULL_REASON
        _validate(cls)
        _REGISTERED.append(cls)


    @classmethod
    def column(cls, name: str) -> Column:
        """The column of that name; a name the table does not store is refused."""
        columns = cls._all_columns()
        if name not in columns:
            raise SchemaViolationError(
                f"{cls.table_name}: column {name!r} is not declared"
            )
        return columns[name]

    @classmethod
    def polars_schema(cls) -> pl.Schema:
        """The Polars schema of the table, columns in schema order."""
        return pl.Schema(
            {name: column.dtype for name, column in cls._all_columns().items()}
        )

    @classmethod
    def arrow_schema(cls) -> pa.Schema:
        """The Arrow schema the writer uses, with each field's type and nullability."""
        return pa.schema(
            [
                pa.field(
                    name,
                    _arrow_type(column.dtype),
                    nullable=not _is_required(cls, column),
                )
                for name, column in cls._all_columns().items()
            ]
        )

    @classmethod
    def null_reason_columns(cls) -> tuple[str, ...]:
        """The columns whose NULL cells must carry a null_reason."""
        return tuple(
            name for name, column in cls._all_columns().items() if column.null_reason
        )

    @classmethod
    def file_name(cls) -> str:
        """The Parquet file name of the table."""
        if cls.table_kind is TableKind.FACT:
            return f"fact_{cls.table_name}.parquet"
        return f"{cls.table_name}.parquet"

    @classmethod
    def relative_path(cls) -> pathlib.PurePosixPath:
        """Where the table sits: under ``topics/`` or ``metrics/`` of an episode, or at the benchmark root."""
        match cls.table_kind:
            case TableKind.TELEMETRY:
                return pathlib.PurePosixPath("topics", cls.file_name())
            case TableKind.FACT | TableKind.SERIES:
                return pathlib.PurePosixPath("metrics", cls.file_name())
            case TableKind.DIMENSION:
                return pathlib.PurePosixPath(cls.file_name())

    @classmethod
    def scan(cls, directory: pathlib.Path) -> pl.LazyFrame:
        """Scan the table lazily under ``directory``: an episode folder, or the benchmark root for a dimension."""
        return pl.scan_parquet(directory / cls.relative_path())


    @classmethod
    def _all_columns(cls) -> dict[str, Column]:
        """The declared columns; a fact table adds the metrics of its scope before its trailing null-reason column."""
        if cls.table_kind is not TableKind.FACT or cls.table_scope is None:
            return dict(cls._columns)
        columns = {
            name: column
            for name, column in cls._columns.items()
            if name != NULL_REASON_COLUMN
        }
        for metric in METRICS.at(cls.table_scope):
            columns[metric.name] = metric
        if NULL_REASON_COLUMN in cls._columns:
            columns[NULL_REASON_COLUMN] = cls._columns[NULL_REASON_COLUMN]
        return columns


def registered_tables() -> tuple[type[Table], ...]:
    """Every concrete table class, in the order the classes were declared."""
    return tuple(_REGISTERED)


def _refuse_table_attribute_columns(table: type[Table]) -> None:
    """Refuse a column named like a table attribute: the class assignment would overwrite it and the column would vanish."""
    shadowing = sorted(name for name, value in vars(table).items() if isinstance(value, Column) and name in _TABLE_ATTRIBUTES)
    if shadowing:
        raise SchemaViolationError(f"{table.__name__}: column(s) {shadowing} take the name of a table attribute")


def _collect_columns(table: type[Table]) -> dict[str, Column]:
    """The columns of a table and its bases, base columns first; a name declared twice is refused."""
    columns: dict[str, Column] = {}
    for base in reversed(table.__mro__):
        for attribute, value in vars(base).items():
            if not isinstance(value, Column):
                continue
            if attribute in columns:
                raise SchemaViolationError(
                    f"{table.__name__}: column {attribute!r} is declared twice"
                )
            columns[attribute] = value
    return columns


def _validate(table: type[Table]) -> None:
    """Refuse a table that breaks a declaration rule; the class is then not registered."""
    _validate_name(table)
    _validate_keys(table)
    _validate_dtypes(table)
    _validate_no_metric_declarations(table)
    if table.table_kind is TableKind.TELEMETRY:
        _validate_time_column(table)
    elif table.table_kind is TableKind.SERIES:
        _validate_time_column(table)
        _validate_documentation(table)
    else:
        _validate_documentation(table)


def _validate_name(table: type[Table]) -> None:
    """Unique names, and names that carry their kind's prefix or are a fact table's name."""
    name = table.table_name
    if not name:
        raise SchemaViolationError(f"{table.__name__}: a table needs a name")
    if any(registered.table_name == name for registered in _REGISTERED):
        raise SchemaViolationError(
            f"duplicate table name {name!r}: {table.__name__} and an earlier class"
        )
    if table.table_kind is TableKind.TELEMETRY and not (
        _TELEMETRY_NAME.fullmatch(name) or name in GLOBAL_TELEMETRY_NAMES
    ):
        raise SchemaViolationError(
            f"misnamed table {name!r}: telemetry names carry a scope prefix (robot_, pedestrian_, episode_) or are global"
        )
    if table.table_kind is TableKind.DIMENSION and not name.startswith(
        DIMENSION_PREFIX
    ):
        raise SchemaViolationError(
            f"misnamed table {name!r}: dimension names start with {DIMENSION_PREFIX}"
        )
    if table.table_kind is TableKind.FACT and name not in {
        member.value for member in FactTable
    }:
        raise SchemaViolationError(
            f"misnamed table {name!r}: a fact table carries the name of a FactTable member"
        )
    if table.table_kind is TableKind.SERIES and not _SERIES_NAME.fullmatch(name):
        raise SchemaViolationError(
            f"misnamed table {name!r}: series names carry the {SERIES_PREFIX} prefix and a stem"
        )


def _validate_keys(table: type[Table]) -> None:
    """Every row-key column is declared by the table."""
    for column in table.key:
        if column not in table._columns:
            raise SchemaViolationError(
                f"{table.table_name}: key column {column!r} is not declared"
            )


def _validate_dtypes(table: type[Table]) -> None:
    """No nested dtype, and every dtype has an Arrow type the writer can store."""
    for name, column in table._columns.items():
        _check_dtype(f"{table.table_name}.{name}", column)


def _validate_no_metric_declarations(table: type[Table]) -> None:
    """A metric is defined in the metric store and reaches a fact table through its scopes, never as a table attribute."""
    for name, column in table._columns.items():
        if isinstance(column, Metric):
            raise SchemaViolationError(
                f"{table.table_name}.{name}: metric {column.name!r} is defined with define_metric, not declared in a table"
            )


def _validate_time_column(table: type[Table]) -> None:
    """Telemetry is keyed by its native timestamp, a required Int64 column."""
    time_column = table._columns.get(TIME_COLUMN)
    if time_column is None:
        raise SchemaViolationError(
            f"{table.table_name}: {TIME_COLUMN} is not declared; telemetry is keyed by its native time"
        )
    if _base_class(time_column.dtype) is not pl.Int64 or not _is_required(
        table, time_column
    ):
        raise SchemaViolationError(
            f"{table.table_name}.{TIME_COLUMN}: must be a required Int64 column"
        )


def _validate_documentation(table: type[Table]) -> None:
    """Every fact or dimension column is described, and every float column has a unit."""
    for name, column in table._columns.items():
        _check_documentation(f"{table.table_name}.{name}", column)


def _check_dtype(owner: str, column: Column) -> None:
    """No nested dtype, and every dtype has an Arrow type the writer can store."""
    if issubclass(_base_class(column.dtype), _NESTED_DTYPES):
        raise SchemaViolationError(
            f"{owner}: nested dtype {column.dtype!r} is not stored; declare a child table"
        )
    _arrow_type(column.dtype)


def _check_documentation(owner: str, column: Column) -> None:
    """A column is described, and a float column has a unit."""
    if not column.description:
        raise SchemaViolationError(f"{owner}: a column needs a description")
    if _base_class(column.dtype) in _FLOAT_DTYPES and not column.unit:
        raise SchemaViolationError(f"{owner}: a float column needs a unit")


def _is_required(table: type[Table], column: Column) -> bool:
    """Whether a column is never NULL: a null_reason column is not, and otherwise the table default decides."""
    if column.null_reason:
        return False
    if column.required is not None:
        return column.required
    return table.required_by_default


def _base_class(dtype: pl.DataType | type[pl.DataType]) -> type[pl.DataType]:
    """The dtype class of a Polars dtype class or instance."""
    return dtype if inspect.isclass(dtype) else type(dtype)


def _arrow_type(dtype: pl.DataType | type[pl.DataType]) -> pa.DataType:
    """The Arrow type the writer stores a dtype as; a dtype without one is refused."""
    base = _base_class(dtype)
    if base not in _ARROW_TYPES:
        raise SchemaViolationError(f"no Arrow type is declared for {dtype!r}")
    return _ARROW_TYPES[base]
