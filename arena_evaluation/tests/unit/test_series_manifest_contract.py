"""Series tables declared in the registry are exposed as joined ``v_series_*`` views in the catalog."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest

from arena_evaluation.analytics.catalog import BenchmarkCatalog
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


def test_every_series_table_has_a_joined_view(tmp_path: pathlib.Path) -> None:
    stems = tuple(series.SERIES_TABLES.keys())
    catalog = BenchmarkCatalog.open(_write_minimal_benchmark(tmp_path, stems))
    for stem in stems:
        frame = catalog.get_table(f"v_series_{stem}").collect()
        assert "planner" in frame.columns
        assert "world" in frame.columns
        assert "local_planner" in frame.columns
        assert "crowd_condition" in frame.columns


@pytest.mark.parametrize(
    "table", ["v_series_characterization", "v_series_energy", "v_series_clearance"]
)
def test_an_absent_series_table_has_no_joined_view(
    tmp_path: pathlib.Path, table: str
) -> None:
    catalog = BenchmarkCatalog.open(_write_minimal_benchmark(tmp_path, ()))
    assert table not in catalog.table_names()
