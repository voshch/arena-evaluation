"""Contract tests for the flat telemetry tables and their lazy catalog scans."""

from __future__ import annotations

import pathlib

import polars as pl

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.storage.registry import TableKind, dimensions, registered_tables
from arena_evaluation.storage.registry.telemetry import (
    EpisodeIntegrity,
    EpisodeLifecycle,
    PedestrianAgentStates,
    PedestrianArenaPeds,
    RobotCollisionContacts,
    RobotCollisionEvents,
    RobotTracks,
)
from arena_evaluation.storage.star_schema import write_benchmark_dimensions


def test_collision_contacts_carry_the_header_stamp_and_obstacle_geometry() -> None:
    names = set(RobotCollisionContacts.polars_schema().names())
    assert {
        "t_ns",
        "env_id",
        "robot_id",
        "kind",
        "obstacle_id",
        "obstacle_x",
        "obstacle_y",
        "obstacle_z",
        "distance",
    } <= names


def test_collision_events_keep_the_per_message_counts() -> None:
    assert {"t_ns", "env_id", "robot_id", "event_count"} <= set(
        RobotCollisionEvents.polars_schema().names()
    )


def test_pedestrian_rows_carry_the_crowd_engine_attributes() -> None:
    assert {
        "radius_m",
        "desired_speed",
        "policy",
        "agent_type",
        "gait_phase",
        "gait_cadence",
    } <= set(PedestrianArenaPeds.polars_schema().names())
    assert {
        "robot_awareness",
        "robot_fov_gain",
        "robot_repulsion_gain",
        "robot_reaction_gain",
    } <= set(PedestrianAgentStates.polars_schema().names())


def test_episode_lifecycle_carries_world_seed_and_accepted_human_params() -> None:
    assert {"world", "seed", "human_params", "episode_id", "outcome_state"} <= set(
        EpisodeLifecycle.polars_schema().names()
    )


def test_lifecycle_flags_have_their_own_flat_table() -> None:
    assert {"t_ns", "env_id", "flag_idx", "flag"} <= set(
        EpisodeIntegrity.polars_schema().names()
    )


def test_every_robot_table_uses_the_canonical_column_names() -> None:
    robot_tables = [
        table
        for table in registered_tables()
        if table.table_kind is TableKind.TELEMETRY
        and table.table_name.startswith("robot_")
    ]
    assert robot_tables
    for table in robot_tables:
        names = set(table.polars_schema().names())
        assert {"t_ns", "env_id"} <= names, table.table_name


def test_catalog_scans_telemetry_without_alias_prefix(tmp_path: pathlib.Path) -> None:
    dims = {
        name: pl.DataFrame(schema=tbl.polars_schema())
        for name, tbl in dimensions.DIMENSION_TABLES.items()
    }
    write_benchmark_dimensions(dims, tmp_path)
    ep_dir = tmp_path / "episodes" / "episode_000"
    (ep_dir / "topics").mkdir(parents=True)
    pl.DataFrame(schema=RobotTracks.polars_schema()).write_parquet(
        ep_dir / RobotTracks.relative_path()
    )

    catalog = BenchmarkCatalog.open(tmp_path)

    assert "robot_tracks" in catalog.table_names()
    assert "telemetry_robot_tracks" not in catalog.table_names()
    assert "episode_id" in catalog.get_table("robot_tracks").collect_schema().names()
