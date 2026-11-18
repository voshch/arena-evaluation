"""Series tables declared in the registry are exposed as joined ``v_series_*`` views in the catalog."""

from __future__ import annotations

import pathlib

import polars as pl

from arena_evaluation.storage.registry import dimensions, series
from arena_evaluation.storage.star_schema import write_benchmark_dimensions


def _write_minimal_benchmark(
    root: pathlib.Path, series_stems: tuple[str, ...]
) -> pathlib.Path:
    dims = {
        name: pl.DataFrame(schema=tbl.polars_schema())
        for name, tbl in dimensions.DIMENSION_TABLES.items()
    }
    dims["dim_episodes"] = pl.DataFrame(
        [
            {
                **{col: None for col in dimensions.DimEpisodes.polars_schema().names()},
                "run_id": "run-1",
                "episode_id": 0,
                "env_id": "env_0",
                "robot_id": "robot_0",
                "planner": "dwb",
                "world": "map_empty",
                "stage": "s1",
                "is_reference": False,
            }
        ],
        schema=dimensions.DimEpisodes.polars_schema(),
    )
    write_benchmark_dimensions(dims, root)
    metrics_dir = root / "episodes" / "episode_000" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for stem in series_stems:
        table = series.SERIES_TABLES[stem]
        pl.DataFrame(schema=table.polars_schema()).write_parquet(
            root / "episodes" / "episode_000" / table.relative_path()
        )
    return root


