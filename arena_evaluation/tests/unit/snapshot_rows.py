import polars as pl

_WORLD = "w"


def _members_value(members) -> str:
    """Members of an occupancy or door, encoded as the recorder encodes them: sorted, comma-joined."""
    return members if isinstance(members, str) else ",".join(sorted(members))


def snapshot_frame(rows):
    """Flat world_semantics rows: (t_ns, entity_name, kind, field_name, field_kind, value)."""
    t_ns, entity_name, kind, field_name, field_kind = [], [], [], [], []
    value_str, value_num, value_bool = [], [], []
    for t, e, k, f, fk, v in rows:
        t_ns.append(t)
        entity_name.append(e)
        kind.append(k)
        field_name.append(f)
        field_kind.append(fk)
        value_str.append(_members_value(v) if fk == "members" else (v if fk == "discrete" else None))
        value_num.append(float(v) if fk == "continuous" else None)
        value_bool.append(bool(v) if fk == "predicate" else None)
    return pl.DataFrame(
        {
            "t_ns": t_ns,
            "env_id": ["env_0"] * len(rows),
            "world": [_WORLD] * len(rows),
            "entity_name": entity_name,
            "kind": kind,
            "field_name": field_name,
            "field_kind": field_kind,
            "value_str": value_str,
            "value_num": value_num,
            "value_bool": value_bool,
        },
        schema_overrides={"value_num": pl.Float64, "value_bool": pl.Boolean, "value_str": pl.String},
    )
