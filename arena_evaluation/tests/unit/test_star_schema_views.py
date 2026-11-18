"""Star schema scan functions and lazy catalog views."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest

from arena_evaluation.analytics.catalog import CANONICAL_VIEW_NAMES, BenchmarkCatalog
from arena_evaluation.storage.registry import FACT_TABLES, FactTable, dimensions, series
from arena_evaluation.storage.registry.telemetry import RobotTracks
from arena_evaluation.storage.star_schema import (
    episode_directories,
    episode_directory_id,
    scan_series_tables,
    scan_star_schema,
    scan_telemetry_tables,
    write_benchmark_dimensions,
)


def _write_two_episode_benchmark(root: pathlib.Path) -> pathlib.Path:
    dims = {
        name: pl.DataFrame(schema=tbl.polars_schema())
        for name, tbl in dimensions.DIMENSION_TABLES.items()
    }
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
                    **{
                        col: None
                        for col in FACT_TABLES[FactTable.EPISODES]
                        .polars_schema()
                        .names()
                    },
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
                    **{
                        col: None
                        for col in series.SeriesClearance.polars_schema().names()
                    },
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


def test_scan_star_schema_series_and_telemetry_across_episodes(
    tmp_path: pathlib.Path,
) -> None:
    root = _write_two_episode_benchmark(tmp_path / "bench")
    star = scan_star_schema(root)
    assert set(dimensions.DIMENSION_TABLES) <= set(star)
    assert {f"fact_{t.table_name}" for t in FACT_TABLES.values()} <= set(star)
    assert star["fact_episodes"].collect().height == 2

    ser = scan_series_tables(root)
    assert set(ser) == {"series_clearance"}
    assert ser["series_clearance"].collect().height == 2

    tel = scan_telemetry_tables(root)
    assert set(tel) == {"robot_tracks"}
    assert sorted(tel["robot_tracks"].collect()["episode_id"].to_list()) == [0, 1]


def test_catalog_exposes_canonical_views_and_dim_metrics(
    tmp_path: pathlib.Path,
) -> None:
    root = _write_two_episode_benchmark(tmp_path / "bench")
    catalog = BenchmarkCatalog.open(root)
    for view_name in CANONICAL_VIEW_NAMES:
        assert view_name in catalog.table_names()
    assert "dim_metrics" in catalog.table_names()
    assert "v_series_clearance" in catalog.table_names()
    assert catalog.get_table("v_series_clearance").collect().height == 2


def test_catalog_get_table_raises_for_unknown_table() -> None:
    catalog = BenchmarkCatalog()
    with pytest.raises(KeyError, match="Unknown table or view"):
        catalog.get_table("does_not_exist")


def test_v_contacts_closing_shares_and_null_planner(
    tmp_path: pathlib.Path,
) -> None:
    root = _write_two_episode_benchmark(tmp_path / "bench")
    contact_schema = FACT_TABLES[FactTable.CONTACTS].polars_schema()
    contacts = pl.DataFrame(
        [
            {
                **{col: None for col in contact_schema.names()},
                "run_id": "run-1",
                "episode_id": 0,
                "contact_id": idx,
                "v_close_robot_mps": r_spd,
                "v_close_target_mps": t_spd,
            }
            for idx, (r_spd, t_spd) in enumerate([(0.75, 0.25), (0.0, 0.0)])
        ],
        schema=contact_schema,
    )

    ep_dim_path = root / dimensions.DIMENSION_TABLES["dim_episodes"].relative_path()
    ep_dim = pl.read_parquet(ep_dim_path)
    ep_dim = ep_dim.with_columns(
        pl.when(pl.col("episode_id") == 0)
        .then(pl.lit(""))
        .otherwise(pl.col("planner"))
        .alias("planner")
    )
    ep_dim.write_parquet(ep_dim_path)

    contacts_path = (
        root
        / "episodes"
        / "episode_000"
        / FACT_TABLES[FactTable.CONTACTS].relative_path()
    )
    contacts_path.parent.mkdir(parents=True, exist_ok=True)
    contacts.write_parquet(contacts_path)

    catalog = BenchmarkCatalog.open(root)
    v_contacts = catalog.get_table("v_contacts").collect().sort("contact_id")
    r_shares = v_contacts["robot_closing_share"].to_list()
    t_shares = v_contacts["target_closing_share"].to_list()
    assert r_shares[0] == pytest.approx(0.75) and r_shares[1] is None
    assert t_shares[0] == pytest.approx(0.25) and t_shares[1] is None

    v_ep = catalog.get_table("v_episode_summary").collect().sort("episode_id")
    assert v_ep["local_planner"].to_list()[0] is None
