"""Star schema scan functions and lazy catalog views."""

from __future__ import annotations

import pathlib

import polars as pl

from arena_evaluation.storage.registry import FACT_TABLES, FactTable, dimensions, series
from arena_evaluation.storage.registry.telemetry import RobotTracks
from arena_evaluation.storage.star_schema import (
    episode_directories,
    episode_directory_id,
    write_benchmark_dimensions,
)


def _write_two_episode_benchmark(root: pathlib.Path) -> pathlib.Path:
    dims = {name: pl.DataFrame(schema=tbl.polars_schema()) for name, tbl in dimensions.DIMENSION_TABLES.items()}
    dims["dim_episodes"] = pl.DataFrame(
        [
            {
                **{col: None for col in dimensions.DimEpisodes.polars_schema().names()},
                "run_id": "run-1",
                "episode_id": ep_id,
                "env_id": "env_0",
                "robot_id": "robot_0",
                "planner": "dwb",
                "world": "map_empty",
                "stage": "s0",
                "is_reference": False,
            }
            for ep_id in (0, 1)
        ],
        schema=dimensions.DimEpisodes.polars_schema(),
    )
    write_benchmark_dimensions(dims, root)

    for ep_id in (0, 1):
        ep_dir = root / "episodes" / f"episode_{ep_id:03d}"
        (ep_dir / "metrics").mkdir(parents=True)
        (ep_dir / "topics").mkdir(parents=True)
        fact_ep = pl.DataFrame(
            [
                {
                    **{col: None for col in FACT_TABLES[FactTable.EPISODES].polars_schema().names()},
                    "run_id": "run-1",
                    "episode_id": ep_id,
                    "robot_id": "robot_0",
                    "result": "SUCCESS",
                    "invalid_reason": "",
                    "pose_frame": "map",
                }
            ],
            schema=FACT_TABLES[FactTable.EPISODES].polars_schema(),
        )
        fact_ep.write_parquet(ep_dir / FACT_TABLES[FactTable.EPISODES].relative_path())
        pl.DataFrame(
            [
                {
                    **{col: None for col in series.SeriesClearance.polars_schema().names()},
                    "run_id": "run-1",
                    "episode_id": ep_id,
                    "robot_id": "robot_0",
                    "t_ns": ep_id * 1_000_000,
                    "obstacle_clearance_m": 1.5,
                    "pedestrian_clearance_m": 2.0,
                    "combined_clearance_m": 1.5,
                }
            ],
            schema=series.SeriesClearance.polars_schema(),
        ).write_parquet(ep_dir / series.SeriesClearance.relative_path())
        pl.DataFrame(
            [
                {
                    **{col: None for col in RobotTracks.polars_schema().names()},
                    "t_ns": ep_id * 1_000_000,
                    "env_id": "env_0",
                    "robot_id": "robot_0",
                    "speed_mps": 0.5 + ep_id,
                }
            ],
            schema=RobotTracks.polars_schema(),
        ).write_parquet(ep_dir / RobotTracks.relative_path())
    return root


def test_episode_directories_and_id_parsing(tmp_path: pathlib.Path) -> None:
    root = _write_two_episode_benchmark(tmp_path / "bench")
    dirs = episode_directories(root)
    assert [d.name for d in dirs] == ["episode_000", "episode_001"]
    assert [episode_directory_id(d) for d in dirs] == [0, 1]
    assert episode_directory_id(pathlib.Path("not_an_episode")) is None
