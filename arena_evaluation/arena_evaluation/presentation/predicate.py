"""Compile a manifest ``where`` string into a Polars predicate over one catalog frame."""

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import Final, NoReturn

import polars as pl

__all__ = ["compile_where_predicate"]

type ScalarLiteral = bool | int | float | str | None

_HINT: Final[str] = (
    "Hint: write comparisons between a column of the bound table and a literal, combine them with and/or/not and parentheses, "
    "e.g. \"is_reference is null and planner = 'dwb'\". Equality accepts = or ==, inequality <> or !=, and in (...) filters a list."
)

_BOOLEAN_LITERALS: Final[dict[str, bool]] = {"true": True, "false": False}
_NULL_LITERALS: Final[frozenset[str]] = frozenset({"null", "none"})
_SET_LITERALS: Final[tuple[type[ast.expr], ...]] = (ast.Tuple, ast.List, ast.Set)

_COMPARE: Final[
    dict[
        type[ast.cmpop],
        Callable[[pl.Expr | ScalarLiteral, pl.Expr | ScalarLiteral], pl.Expr],
    ]
] = {
    ast.Eq: lambda left, right: left == right,
    ast.NotEq: lambda left, right: left != right,
    ast.Lt: lambda left, right: left < right,
    ast.LtE: lambda left, right: left <= right,
    ast.Gt: lambda left, right: left > right,
    ast.GtE: lambda left, right: left >= right,
}


def _reject(where: str, reason: str) -> NoReturn:
    raise ValueError(f"unsupported where predicate {where!r}: {reason}. {_HINT}")


def _sql_equality_to_python(where: str) -> str:
    """Rewrite SQL ``=`` and ``<>`` to Python tokens while leaving quoted strings intact."""
    out: list[str] = []
    quote: str | None = None
    for index, char in enumerate(where):
        if quote is not None:
            out.append(char)
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            out.append(char)
            continue
        following = where[index + 1 : index + 2]
        preceding = where[index - 1 : index]
        if char == "=" and preceding not in ("=", "!", "<", ">") and following != "=":
            out.append("==")
        elif char == "<" and following == ">":
            out.append("!=")
        elif char == ">" and preceding == "<":
            continue
        else:
            out.append(char)
    return "".join(out)


def _column_name(where: str, node: ast.expr, schema: pl.Schema) -> str:
    if not isinstance(node, ast.Name):
        _reject(where, f"{ast.unparse(node)!r} is not a bare column name")
    if node.id not in schema.names():
        available = ", ".join(sorted(schema.names()))
        _reject(
            where,
            f"{node.id!r} is not a column of the bound table; columns: {available}",
        )
    return node.id


def _literal(where: str, node: ast.expr) -> ScalarLiteral:
    if isinstance(node, ast.Name):
        lowered = node.id.lower()
        if lowered in _NULL_LITERALS:
            return None
        if lowered in _BOOLEAN_LITERALS:
            return _BOOLEAN_LITERALS[lowered]
        _reject(where, f"{node.id!r} is a column, not a literal; quote a text value")
    if isinstance(node, ast.Constant) and isinstance(
        node.value, (bool, int, float, str, type(None))
    ):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
        and not isinstance(node.operand.value, bool)
    ):
        return -node.operand.value
    _reject(where, f"{ast.unparse(node)!r} is not a literal")


def _ensure_fits(where: str, dtype: pl.DataType, value: ScalarLiteral) -> ScalarLiteral:
    if value is None:
        _reject(where, "compare against null with is null or is not null")
    if isinstance(value, bool):
        fits = dtype == pl.Boolean
    elif isinstance(value, (int, float)):
        fits = dtype.is_numeric()
    elif isinstance(value, str):
        fits = (
            dtype == pl.String or dtype == pl.Categorical or isinstance(dtype, pl.Enum)
        )
    else:
        fits = False
    if not fits:
        _reject(where, f"{value!r} does not fit column dtype {dtype}")
    return value


def _membership(where: str, literal_node: ast.expr) -> list[ScalarLiteral]:
    elements = (
        literal_node.elts if isinstance(literal_node, _SET_LITERALS) else [literal_node]
    )
    if not elements:
        _reject(where, "in needs at least one literal")
    return [_literal(where, element) for element in elements]


def _comparison(
    where: str, left: ast.expr, op: ast.cmpop, right: ast.expr, schema: pl.Schema
) -> pl.Expr:
    column_on_left = isinstance(left, ast.Name)
    column_node, literal_node = (left, right) if column_on_left else (right, left)

    if isinstance(op, (ast.Is, ast.IsNot)):
        column = pl.col(_column_name(where, column_node, schema))
        if _literal(where, literal_node) is not None:
            _reject(
                where, "is / is not compares against null only, use = or != for a value"
            )
        is_null = column.is_null()
        return is_null if isinstance(op, ast.Is) else is_null.not_()

    name = _column_name(where, column_node, schema)
    dtype = schema[name]

    if isinstance(op, (ast.In, ast.NotIn)):
        if not column_on_left:
            _reject(
                where,
                "in needs the column on the left and a list of literals on the right",
            )
        values = [
            _ensure_fits(where, dtype, value)
            for value in _membership(where, literal_node)
        ]
        membership = pl.col(name).is_in(values)
        return membership if isinstance(op, ast.In) else membership.not_()

    op_fn = _COMPARE.get(type(op))
    if op_fn is None:
        _reject(where, f"unsupported comparison operator {type(op).__name__}")
    value = _ensure_fits(where, dtype, _literal(where, literal_node))
    return op_fn(pl.col(name), value) if column_on_left else op_fn(value, pl.col(name))


def _call(where: str, node: ast.Call, schema: pl.Schema) -> pl.Expr:
    if not isinstance(node.func, ast.Attribute) or node.args or node.keywords:
        _reject(
            where,
            f"only column.is_null() and column.is_not_null() are callable, got {ast.unparse(node)!r}",
        )
    if node.func.attr not in ("is_null", "is_not_null"):
        _reject(
            where,
            f"{node.func.attr!r} is not a supported column method; use is_null() or is_not_null()",
        )
    column = pl.col(_column_name(where, node.func.value, schema)).is_null()
    return column if node.func.attr == "is_null" else column.not_()


def _column_predicate(where: str, node: ast.Name, schema: pl.Schema) -> pl.Expr:
    name = _column_name(where, node, schema)
    if schema[name] != pl.Boolean:
        _reject(
            where,
            f"{name!r} is a predicate only as a boolean column; compare it against a literal",
        )
    return pl.col(name)


def _combine(expressions: list[pl.Expr], *, conjunction: bool) -> pl.Expr:
    combined = expressions[0]
    for expression in expressions[1:]:
        combined = (combined & expression) if conjunction else (combined | expression)
    return combined


def _expression(where: str, node: ast.expr, schema: pl.Schema) -> pl.Expr:
    match node:
        case ast.BoolOp(op=ast.And(), values=values):
            return _combine(
                [_expression(where, value, schema) for value in values],
                conjunction=True,
            )
        case ast.BoolOp(op=ast.Or(), values=values):
            return _combine(
                [_expression(where, value, schema) for value in values],
                conjunction=False,
            )
        case ast.BoolOp():
            _reject(where, "only and / or combine predicates")
        case ast.UnaryOp(op=ast.Not(), operand=operand):
            return _expression(where, operand, schema).not_()
        case ast.UnaryOp():
            _reject(where, f"unary {ast.unparse(node)!r} is not a predicate")
        case ast.Compare(left=left, ops=[op], comparators=[right]):
            return _comparison(where, left, op, right, schema)
        case ast.Compare():
            _reject(where, "chained comparisons are ambiguous, combine them with and")
        case ast.Call():
            return _call(where, node, schema)
        case ast.Name():
            return _column_predicate(where, node, schema)
        case _:
            _reject(
                where,
                f"{ast.unparse(node)!r} is not a comparison, membership test or null check",
            )


def compile_where_predicate(where: str, schema: pl.Schema) -> pl.Expr:
    """Compile a manifest ``where`` string into a Polars predicate over ``schema``."""
    if not where.strip():
        raise ValueError(f"empty where predicate. {_HINT}")
    try:
        tree = ast.parse(_sql_equality_to_python(where), mode="eval")
    except SyntaxError as error:
        _reject(where, f"it does not parse ({error.msg})")
    return _expression(where, tree.body, schema)
