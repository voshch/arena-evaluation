"""SQL query store and declarations for benchmark OLAP analytics (Layer 3)."""

from __future__ import annotations

import dataclasses
import pathlib
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Final

import polars as pl
import yaml

from arena_evaluation.analytics.query_identifier import QueryIdentifier
from arena_evaluation.analytics.sql_engine import PolarsSQLEngine, validate_readonly_sql
from arena_evaluation.storage.config_identifiers import bucket_cache_dirs, config_dirs
from arena_evaluation.storage.config_layout import (
    CONFIG_KINDS,
    bundle_path,
    run_analysis_root,
)

__all__ = ["QueryCatalog", "QueryDeclaration", "format_sql_literal"]

_PARAM_COLON_RE: Final[re.Pattern[str]] = re.compile(r":([A-Za-z_][A-Za-z0-9_]*)")
_PARAM_BRACE_RE: Final[re.Pattern[str]] = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_PARAM_VALIDATORS: Final[Mapping[str, Callable[[object], bool]]] = {
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "float": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
}


def format_sql_literal(value: object) -> str:
    """Format a Python value as a safe, canonical SQL literal string."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return f"({', '.join(format_sql_literal(item) for item in value)})"
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class QueryDeclaration:
    """Immutable declaration of a parameterized analytical SQL query."""

    name: str
    description: str
    sql: str
    parameters: Mapping[str, Mapping[str, object] | object] = dataclasses.field(default_factory=dict)
    title: str = ""

    def default_parameters(self) -> dict[str, object]:
        """Extract default parameter values from the declaration."""
        return {key: spec["default"] if isinstance(spec, Mapping) and "default" in spec else spec for key, spec in self.parameters.items() if not isinstance(spec, Mapping) or "default" in spec}

    def validate_parameters(self, params: Mapping[str, object]) -> None:
        """Enforce declared parameter ``type`` constraints on ``params``."""
        for key, spec in self.parameters.items():
            if not isinstance(spec, Mapping) or "type" not in spec or key not in params:
                continue
            expected, value = str(spec["type"]), params[key]
            if expected not in _PARAM_VALIDATORS:
                raise ValueError(f"Unsupported parameter type {expected!r} for '{key}'. Expected one of {sorted(_PARAM_VALIDATORS)}.")
            if value is not None and not _PARAM_VALIDATORS[expected](value):
                raise TypeError(f"Parameter '{key}' expects type {expected!r}, got {type(value).__name__} ({value!r}).")

    def render_sql(self, params: Mapping[str, object] | None = None) -> str:
        """Substitute parameters into the query SQL and validate read-only compliance."""
        combined = {**self.default_parameters(), **(params or {})}
        self.validate_parameters(combined)

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in combined:
                raise ValueError(f"Missing required parameter '{name}' for query declaration '{self.name}'.")
            return format_sql_literal(combined[name])

        rendered = _PARAM_COLON_RE.sub(replace, self.sql)
        return validate_readonly_sql(_PARAM_BRACE_RE.sub(replace, rendered))

    def to_dict(self) -> dict[str, object]:
        """Serialize query declaration to a dictionary."""
        data = dataclasses.asdict(self)
        return {key: data[key] for key in ("name", "title", "description", "parameters", "sql")}

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


def declaration_from_payload(payload: object, *, folder_name: str, source: str) -> QueryDeclaration:
    """Validate one parsed ``query.yaml``; a declaration that names no query takes its folder's name."""
    if not isinstance(payload, Mapping) or "sql" not in payload:
        raise ValueError(f"Invalid query declaration in {source}: expected a YAML mapping with a 'sql' field.")
    raw_params = payload.get("parameters") or {}
    if not isinstance(raw_params, Mapping):
        raise ValueError(f"Invalid query declaration in {source}: 'parameters' must be a mapping.")
    declaration = QueryDeclaration.from_dict(
        {
            **payload,
            "name": payload.get("name") or folder_name,
            "title": payload.get("title") or folder_name,
        }
    )
    declaration.validate_parameters(declaration.default_parameters())
    return declaration


def read_query_declaration(file_path: pathlib.Path) -> QueryDeclaration:
    """Parse one ``query.yaml`` bundle file into a validated declaration named after its folder."""
    payload = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    return declaration_from_payload(payload, folder_name=file_path.parent.name, source=str(file_path))


class QueryCatalog:
    """Catalog of stored reusable SQL query declarations with discovery and execution."""

    def __init__(
        self,
        search_dirs: Sequence[pathlib.Path] | None = None,
        *,
        run_dir: pathlib.Path | None = None,
    ) -> None:
        candidates = [run_analysis_root(run_dir)] if run_dir is not None else []
        candidates.extend(search_dirs or ())
        candidates.extend(root for root in (*config_dirs(), *bucket_cache_dirs()) if root not in candidates)
        self._search_dirs = tuple(candidates)
        self._cache: dict[str, QueryDeclaration] = {}
        self._load_all()

    @property
    def search_dirs(self) -> tuple[pathlib.Path, ...]:
        """Configured config-root search paths for query declarations."""
        return self._search_dirs

    def _load_all(self) -> None:
        """Scan search roots for ``queries/<name>/query.yaml`` bundles."""
        query_kind = CONFIG_KINDS["query"]
        for root in self._search_dirs:
            queries_dir = root / query_kind.folder
            if not queries_dir.is_dir():
                continue
            for entry in sorted(queries_dir.iterdir()):
                file_path = entry / query_kind.filename
                if entry.name in self._cache or not file_path.is_file():
                    continue
                declaration = read_query_declaration(file_path)
                self._cache[entry.name] = self._cache[declaration.name] = declaration

    def list_queries(self) -> list[QueryDeclaration]:
        """Return all discovered query declarations sorted by name."""
        unique = {declaration.name: declaration for declaration in self._cache.values()}
        return [unique[name] for name in sorted(unique)]

    def _lookup(self, name: str) -> QueryDeclaration | None:
        stem = name.removesuffix(".yaml").removesuffix(".yml")
        if (declaration := self._cache.get(stem) or self._cache.get(name)) is not None:
            return declaration
        return self._fetch(stem)

    def _fetch(self, stem: str) -> QueryDeclaration | None:
        """Resolve a query that no local root holds from the benchmark buckets, caching what it loads."""
        try:
            payload = QueryIdentifier(name=stem).resolve_sync()
        except FileNotFoundError:
            return None
        declaration = declaration_from_payload(payload, folder_name=stem, source=f"benchmark bucket query '{stem}'")
        self._cache[stem] = self._cache[declaration.name] = declaration
        return declaration

    def has_query(self, name: str) -> bool:
        """Check whether a query declaration with ``name`` is available."""
        return self._lookup(name) is not None

    def get_query(self, name: str) -> QueryDeclaration:
        """Retrieve a ``QueryDeclaration`` by name or stem."""
        if (declaration := self._lookup(name)) is None:
            raise KeyError(f"Query declaration '{name}' not found. Available: {sorted(set(self._cache))}")
        return declaration

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
        declaration = QueryDeclaration(
            name=clean_name,
            description=description.strip(),
            sql=sql.strip(),
            parameters=dict(parameters or {}),
            title=(title or clean_name).strip(),
        )
        declaration.render_sql()
        target_dir = directory or next((d for d in self._search_dirs if d.is_dir()), self._search_dirs[0])
        file_path = bundle_path(target_dir, "query", clean_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(yaml.safe_dump(declaration.to_dict(), sort_keys=False), encoding="utf-8")
        self._cache[clean_name] = declaration
        return declaration

    def execute(
        self,
        name_or_declaration: str | QueryDeclaration,
        engine: PolarsSQLEngine,
        *,
        params: Mapping[str, object] | None = None,
        limit: int | None = None,
    ) -> pl.DataFrame:
        """Render and execute a stored query over :class:`PolarsSQLEngine`."""
        declaration = name_or_declaration if isinstance(name_or_declaration, QueryDeclaration) else self.get_query(name_or_declaration)
        return engine.query(declaration.render_sql(params), limit=limit)
