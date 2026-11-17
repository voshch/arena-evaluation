"""Polars OLAP SQL engine with read-only query guard (Layer 3)."""

from __future__ import annotations

import pathlib
import re
from typing import Final

import polars as pl

from arena_evaluation.analytics.catalog import BenchmarkCatalog

__all__ = ["FORBIDDEN_SQL_KEYWORDS", "PolarsSQLEngine", "SQLGuardError", "normalize_sql_scientific_notation", "validate_readonly_sql"]

FORBIDDEN_SQL_KEYWORDS: Final[frozenset[str]] = frozenset({
    "DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE",
    "TRUNCATE", "COPY", "ATTACH", "DETACH", "PRAGMA", "VACUUM",
})
_ALLOWED_START_KEYWORDS: Final[frozenset[str]] = frozenset({"SELECT", "WITH"})
_LEXER_RE: Final[re.Pattern[str]] = re.compile(
    r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|--[^\r\n]*|/\*.*?\*/",
    re.DOTALL,
)
_STRING_LITERAL_RE: Final[re.Pattern[str]] = re.compile(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"")
_WORD_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_SCIENTIFIC_NOTATION_RE: Final[re.Pattern[str]] = re.compile(r"(?<![a-zA-Z0-9_])([0-9]+(?:\.[0-9]+)?[eE][+-]?[0-9]+)(?![a-zA-Z0-9_])")
_EXTERNAL_READER_RE: Final[re.Pattern[str]] = re.compile(r"\b((?:READ|SCAN)_[A-Z0-9_]*)\s*\(", re.IGNORECASE)


def _format_scientific(match: re.Match[str]) -> str:
    value = float(match.group(1))
    text = str(value)
    if "e" in text or "E" in text:
        text = f"{value:.12f}".rstrip("0")
        if text.endswith("."):
            text += "0"
    return text


def normalize_sql_scientific_notation(sql: str) -> str:
    """Replace scientific notation numbers outside string literals with standard float strings."""
    parts: list[str] = []
    last_end = 0
    for match in _STRING_LITERAL_RE.finditer(sql):
        parts.append(_SCIENTIFIC_NOTATION_RE.sub(_format_scientific, sql[last_end : match.start()]))
        parts.append(match.group(0))
        last_end = match.end()
    return "".join([*parts, _SCIENTIFIC_NOTATION_RE.sub(_format_scientific, sql[last_end:])])


class SQLGuardError(ValueError):
    """Raised when a SQL query violates the read-only analytical query guard."""


def _scrub_sql(sql: str) -> tuple[str, str]:
    """Strip comments, mask string literals, and remove one trailing semicolon."""
    cleaned_parts: list[str] = []
    masked_parts: list[str] = []
    last_end = 0
    for match in _LEXER_RE.finditer(sql):
        code = sql[last_end : match.start()]
        cleaned_parts.append(code)
        masked_parts.append(code)
        token = match.group(0)
        if token.startswith("'"):
            cleaned_parts.append(token)
            masked_parts.append("''")
        elif token.startswith(('"', "`")):
            cleaned_parts.append(token)
            masked_parts.append(f" {token[1:-1]} ")
        else:
            cleaned_parts.append(" ")
            masked_parts.append(" ")
        last_end = match.end()
    tail = sql[last_end:]
    cleaned = "".join([*cleaned_parts, tail]).strip()
    masked = "".join([*masked_parts, tail]).strip()
    if not cleaned:
        raise SQLGuardError("SQL query contains only comments or whitespace.")
    if masked.endswith(";"):
        masked = masked[:-1].strip()
        cleaned = cleaned[:-1].rstrip() if cleaned.endswith(";") else cleaned
    if ";" in masked:
        raise SQLGuardError("Multi-statement SQL queries (';') are forbidden.")
    return cleaned, masked


def validate_readonly_sql(sql: str) -> str:
    """Validate that ``sql`` is a single read-only ``SELECT`` or ``WITH ... SELECT`` statement."""
    if not isinstance(sql, str) or not sql.strip():
        raise SQLGuardError("SQL query must be a non-empty string.")
    cleaned, masked = _scrub_sql(sql)
    tokens = [token.upper() for token in _WORD_TOKEN_RE.findall(masked)]
    if not tokens:
        raise SQLGuardError("SQL query contains no valid statement tokens.")
    forbidden = [token for token in tokens if token in FORBIDDEN_SQL_KEYWORDS]
    if forbidden:
        raise SQLGuardError(
            f"Forbidden SQL keyword(s) detected: {', '.join(dict.fromkeys(forbidden))}"
        )
    readers = [match.group(1).upper() for match in _EXTERNAL_READER_RE.finditer(masked)]
    if readers:
        raise SQLGuardError(
            f"External table function(s) forbidden: {', '.join(dict.fromkeys(readers))}. Query registered catalog tables only."
        )
    first_token = tokens[0]
    if first_token not in _ALLOWED_START_KEYWORDS:
        raise SQLGuardError(
            f"Only SELECT or WITH ... SELECT queries are permitted; got '{first_token}'."
        )
    if first_token == "WITH" and "SELECT" not in tokens:
        raise SQLGuardError("CTE (WITH) query must contain a SELECT clause.")
    return cleaned


class PolarsSQLEngine:
    """Guarded read-only OLAP SQL engine backed by ``polars.SQLContext`` and ``BenchmarkCatalog``."""

    def __init__(self, catalog: BenchmarkCatalog | pathlib.Path) -> None:
        self.catalog = catalog if isinstance(catalog, BenchmarkCatalog) else BenchmarkCatalog(catalog)

    def register_table(self, name: str, frame: pl.LazyFrame | pl.DataFrame) -> None:
        """Register a derived table or view in the underlying catalog."""
        if self.catalog.is_protected_table(name):
            raise SQLGuardError(f"Cannot overwrite protected catalog table or view '{name}'.")
        self.catalog.register_table(name, frame)

    def query_lazy(self, sql: str) -> pl.LazyFrame:
        """Validate ``sql`` and return a lazy Polars execution plan."""
        validated = normalize_sql_scientific_notation(validate_readonly_sql(sql))
        context = pl.SQLContext(frames=self.catalog.tables())
        return context.execute(validated, eager=False)

    def query(
        self,
        sql: str,
        *,
        limit: int | None = None,
        save_as: str | None = None,
    ) -> pl.DataFrame:
        """Execute a validated read-only SQL query and materialize the resulting ``DataFrame``."""
        if limit is not None and limit < 0:
            raise ValueError(f"limit must be non-negative, got {limit}")
        if save_as is not None and self.catalog.is_protected_table(save_as):
            raise SQLGuardError(f"Cannot overwrite protected catalog table or view '{save_as}'.")
        lazy_frame = self.query_lazy(sql)
        if limit is not None:
            lazy_frame = lazy_frame.limit(limit)
        if save_as is not None:
            self.catalog.register_table(save_as, lazy_frame)
        return lazy_frame.collect()
