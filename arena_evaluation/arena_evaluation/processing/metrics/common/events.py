from __future__ import annotations

import typing

import polars as pl

_RECONSTRUCTED_EVENTS_SCHEMA: typing.Final[dict[str, pl.DataType]] = {
    "t_ns": pl.Int64,
    "entity": pl.Utf8,
    "kind": pl.Utf8,
    "field": pl.Utf8,
    "previous": pl.Utf8,
    "current": pl.Utf8,
}


def reconstruct_events(snapshot: pl.DataFrame | None) -> pl.DataFrame:
    """Event-equivalent (t_ns, entity, kind, field, previous, current) rows, derived by
    keeping only the change points of each entity+field's stepwise snapshot series. The first
    kept row per series is the seed value (from the episode's seed snapshot), `previous` empty.
    Members rows are kept as field `members` (see `_stringify_snapshot_value`), so membership
    changes are change points like any other field.
    """
    empty = pl.DataFrame(schema=_RECONSTRUCTED_EVENTS_SCHEMA)
    if snapshot is None or len(snapshot) == 0:
        return empty
    if not {"t_ns", "entity_name", "kind", "field_name"} <= set(snapshot.columns):
        return empty

    str_val = pl.when(pl.col("field_kind") == "predicate").then(pl.when(pl.col("value_bool").is_null()).then(pl.lit(None, dtype=pl.String)).when(pl.col("value_bool").cast(pl.Boolean, strict=False)).then(pl.lit("true")).otherwise(pl.lit("false"))).when(pl.col("field_kind") == "continuous").then(pl.col("value_num").cast(pl.String)).otherwise(pl.col("value_str").fill_null(""))

    group_keys = ["entity_name", "kind", "field_name"]
    return (
        snapshot.sort([*group_keys, "t_ns"])
        .with_columns(current=str_val)
        .with_columns(
            prev_raw=pl.col("current").shift(1).over(group_keys),
        )
        .filter(pl.col("prev_raw").is_null() | pl.col("current").ne_missing(pl.col("prev_raw")))
        .select(
            t_ns=pl.col("t_ns"),
            entity=pl.col("entity_name"),
            kind=pl.col("kind"),
            field=pl.col("field_name"),
            previous=pl.col("prev_raw").fill_null(""),
            current=pl.col("current"),
        )
        .sort("t_ns")
    )
