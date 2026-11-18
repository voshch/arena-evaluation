"""SQL Query Store and Declarations for Benchmark OLAP Analytics (Layer 3).

Provides declarative, parameterized analytical SQL queries, catalog lookup,
parameter type enforcement, safe literal substitution, and execution over
:class:`PolarsSQLEngine`.
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
from collections.abc import Mapping, Sequence
from typing import Final

import polars as pl
import yaml

from arena_evaluation.analytics.sql_engine import PolarsSQLEngine, validate_readonly_sql

__all__ = [
    "QueryCatalog",
    "QueryDeclaration",
    "format_sql_literal",
]

_PARAM_COLON_RE: Final[re.Pattern[str]] = re.compile(r":([A-Za-z_][A-Za-z0-9_]*)")
_PARAM_BRACE_RE: Final[re.Pattern[str]] = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_VALID_PARAM_TYPES: Final[frozenset[str]] = frozenset(
    {"string", "integer", "float", "boolean"}
)

_PACKAGE_QUERY_DIR: Final[pathlib.Path] = (
    pathlib.Path(__file__).resolve().parents[2]
    / "configs"
    / "declarations"
    / "query_declarations"
)


def format_sql_literal(value: object) -> str:
    """Format a Python value as a safe, canonical SQL literal string."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    if isinstance(value, (list, tuple, set)):
        items = [format_sql_literal(v) for v in value]
        return f"({', '.join(items)})"
    escaped_obj = str(value).replace("'", "''")
    return f"'{escaped_obj}'"


def _validate_param_value(name: str, value: object, expected_type: str) -> None:
    """Validate that ``value`` matches the declared parameter ``expected_type``."""
    if expected_type not in _VALID_PARAM_TYPES:
        raise ValueError(
            f"Unsupported parameter type {expected_type!r} for '{name}'. Expected one of {sorted(_VALID_PARAM_TYPES)}."
        )
    if value is None:
        return
    match expected_type:
        case "string":
            valid = isinstance(value, str)
        case "integer":
            valid = isinstance(value, int) and not isinstance(value, bool)
        case "float":
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        case "boolean":
            valid = isinstance(value, bool)
    if not valid:
        raise TypeError(
            f"Parameter '{name}' expects type {expected_type!r}, got {type(value).__name__} ({value!r})."
        )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class QueryDeclaration:
    """Immutable declaration of a parameterized analytical SQL query."""

    name: str
    description: str
    sql: str
    parameters: Mapping[str, Mapping[str, object] | object] = dataclasses.field(
        default_factory=dict
    )
    title: str = ""

    def default_parameters(self) -> dict[str, object]:
        """Extract default parameter values from the declaration."""
        defaults: dict[str, object] = {}
        for key, spec in self.parameters.items():
            if isinstance(spec, Mapping) and "default" in spec:
                defaults[key] = spec["default"]
            elif not isinstance(spec, Mapping):
                defaults[key] = spec
        return defaults

    def validate_parameters(self, params: Mapping[str, object]) -> None:
        """Enforce declared parameter ``type`` constraints on ``params``."""
        for key, spec in self.parameters.items():
            if not isinstance(spec, Mapping) or "type" not in spec:
                continue
            if key in params:
                _validate_param_value(key, params[key], str(spec["type"]))

    def render_sql(self, params: Mapping[str, object] | None = None) -> str:
        """Substitute parameters into the query SQL and validate read-only compliance."""
        combined_params = self.default_parameters()
        if params is not None:
            combined_params.update(params)
        self.validate_parameters(combined_params)

        def replace_param(match: re.Match[str]) -> str:
            param_name = match.group(1)
            if param_name not in combined_params:
                raise ValueError(
                    f"Missing required parameter '{param_name}' for query declaration '{self.name}'."
                )
            return format_sql_literal(combined_params[param_name])

        rendered = _PARAM_COLON_RE.sub(replace_param, self.sql)
        rendered = _PARAM_BRACE_RE.sub(replace_param, rendered)
        return validate_readonly_sql(rendered)

    def to_dict(self) -> dict[str, object]:
        """Serialize query declaration to a dictionary."""
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "parameters": dict(self.parameters),
            "sql": self.sql,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> QueryDeclaration:
        """Deserialize a QueryDeclaration from a dictionary."""
        raw_params = payload.get("parameters") or {}
        if not isinstance(raw_params, Mapping):
            raise ValueError("Query 'parameters' must be a mapping.")
        return cls(
            name=str(payload["name"]),
            description=str(payload.get("description") or ""),
            sql=str(payload["sql"]),
            parameters=dict(raw_params),
            title=str(payload.get("title") or ""),
        )


def _package_share_query_dir() -> pathlib.Path | None:
    try:
        from ament_index_python.packages import get_package_share_directory
    except ImportError:
        return None
    try:
        return (
            pathlib.Path(get_package_share_directory("arena_evaluation"))
            / "configs"
            / "declarations"
            / "query_declarations"
        )
    except Exception:
        return None


class QueryCatalog:
    """Catalog of stored reusable SQL query declarations with discovery and execution."""

    def __init__(
        self,
        search_dirs: Sequence[pathlib.Path] | None = None,
        *,
        run_dir: pathlib.Path | None = None,
    ) -> None:
        candidates: list[pathlib.Path] = []
        if run_dir is not None:
            candidates.append(run_dir / "analysis" / "queries")
        if search_dirs:
            candidates.extend(search_dirs)
        candidates.append(_PACKAGE_QUERY_DIR)
        share_dir = _package_share_query_dir()
        if share_dir is not None and share_dir not in candidates:
            candidates.append(share_dir)

        self._search_dirs = tuple(candidates)
        self._cache: dict[str, QueryDeclaration] = {}
        self._load_all()

    @property
    def search_dirs(self) -> tuple[pathlib.Path, ...]:
        """Configured directory search paths for query declarations."""
        return self._search_dirs

    def _load_all(self) -> None:
        """Scan search directories and populate cached query declarations, failing on invalid YAML."""
        for directory in self._search_dirs:
            if not directory.is_dir():
                continue
            for file_path in sorted(directory.glob("*.yaml")) + sorted(
                directory.glob("*.yml")
            ):
                name = file_path.stem
                if name in self._cache:
                    continue
                payload = yaml.safe_load(file_path.read_text(encoding="utf-8"))
                if not isinstance(payload, Mapping) or "sql" not in payload:
                    raise ValueError(
                        f"Invalid query declaration in {file_path}: expected a YAML mapping with a 'sql' field."
                    )
                raw_params = payload.get("parameters") or {}
                if not isinstance(raw_params, Mapping):
                    raise ValueError(
                        f"Invalid query declaration in {file_path}: 'parameters' must be a mapping."
                    )
                decl = QueryDeclaration(
                    name=str(payload.get("name") or name),
                    description=str(payload.get("description") or ""),
                    sql=str(payload["sql"]),
                    parameters=dict(raw_params),
                    title=str(payload.get("title") or name),
                )
                decl.validate_parameters(decl.default_parameters())
                self._cache[name] = decl
                self._cache[decl.name] = decl

    def list_queries(self) -> list[QueryDeclaration]:
        """Return all discovered query declarations sorted by name."""
        unique = {decl.name: decl for decl in self._cache.values()}
        return [unique[k] for k in sorted(unique.keys())]

    def has_query(self, name: str) -> bool:
        """Check whether a query declaration with ``name`` is available."""
        stem = name.removesuffix(".yaml").removesuffix(".yml")
        return stem in self._cache or name in self._cache

    def get_query(self, name: str) -> QueryDeclaration:
        """Retrieve a ``QueryDeclaration`` by name or stem."""
        stem = name.removesuffix(".yaml").removesuffix(".yml")
        if stem in self._cache:
            return self._cache[stem]
        if name in self._cache:
            return self._cache[name]
        raise KeyError(
            f"Query declaration '{name}' not found. Available: {sorted(set(self._cache))}"
        )

    def save_query(
        self,
        *,
        name: str,
        description: str,
        sql: str,
        parameters: Mapping[str, Mapping[str, object] | object] | None = None,
        title: str | None = None,
        directory: pathlib.Path | None = None,
    ) -> QueryDeclaration:
        """Validate SQL and parameters, then persist a query declaration to YAML."""
        clean_name = name.removesuffix(".yaml").removesuffix(".yml").strip()
        if not clean_name:
            raise ValueError("Query name must be a non-empty string.")

        decl = QueryDeclaration(
            name=clean_name,
            description=description.strip(),
            sql=sql.strip(),
            parameters=dict(parameters or {}),
            title=(title or clean_name).strip(),
        )
        decl.render_sql()

        target_dir = directory
        if target_dir is None:
            for d in self._search_dirs:
                if d.is_dir():
                    target_dir = d
                    break
            if target_dir is None:
                target_dir = self._search_dirs[0]

        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{clean_name}.yaml"
        file_path.write_text(
            yaml.safe_dump(decl.to_dict(), sort_keys=False), encoding="utf-8"
        )

        self._cache[clean_name] = decl
        return decl

    def execute(
        self,
        name_or_declaration: str | QueryDeclaration,
        engine: PolarsSQLEngine,
        *,
        params: Mapping[str, object] | None = None,
        limit: int | None = None,
    ) -> pl.DataFrame:
        """Render and execute a stored query over :class:`PolarsSQLEngine`."""
        decl = (
            name_or_declaration
            if isinstance(name_or_declaration, QueryDeclaration)
            else self.get_query(name_or_declaration)
        )
        rendered_sql = decl.render_sql(params)
        return engine.query(rendered_sql, limit=limit)
