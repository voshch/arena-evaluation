"""Reported point estimates lie inside their own bootstrap intervals, including degenerate bootstraps whose replicates differ only by rounding."""

import polars as pl


def _pair_frame(values_by_seed: list[list[float]]) -> pl.DataFrame:
    rows: list[dict[str, str | int | float]] = []
    for seed, values in enumerate(values_by_seed):
        for pair_id, value in enumerate(values):
            rows.append(
                {"planner": "A", "seed": seed, "pair_id": pair_id, "metric": value}
            )
            rows.append(
                {"planner": "B", "seed": seed, "pair_id": pair_id, "metric": 0.0}
            )
    return pl.DataFrame(rows)


