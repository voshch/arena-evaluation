"""The benchmark dimensions: an episode's rows, the reason of every NULL, the stacked contract and the write at the benchmark root."""

import pathlib

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from arena_evaluation.processing.dimension_rows import episode_dimensions
from arena_evaluation.storage.contracts import enforce_benchmark_contract
from arena_evaluation.storage.exceptions import ContractViolationError
from arena_evaluation.storage.registry.dimensions import (
    DIMENSION_TABLES,
    DimWorldGeometry,
)
from arena_evaluation.storage.schemas import RunMetadata, TopicBundle
from arena_evaluation.storage.star_schema import stack, write_benchmark_dimensions

SEC = 1_000_000_000


def _metadata(world: str = "barn") -> RunMetadata:
    return RunMetadata(
        benchmark_id="bench",
        planner="nav2",
        local_planner="dwb",
        map=world,
        stage="s1",
        recording_started_at="2026-01-01T00:00:00",
        python_version="3.12",
        ros_distro="jazzy",
        modes={},
        params={},
    )


def _track(stamps_s: list[float], env: str, robot: str) -> pl.DataFrame:
    stamps = [round(stamp * SEC) for stamp in stamps_s]
    count = len(stamps)
    return pl.DataFrame(
        {
            "t_ns": stamps,
            "env_id": [env] * count,
            "robot_id": [robot] * count,
            "x": [float(i) for i in range(count)],
            "y": [0.0] * count,
            "yaw": [0.0] * count,
        }
    )


def _pedestrians(blind: list[bool | None], env: str) -> pl.DataFrame:
    count = len(blind)
    names = [f"ped_{index}" for index in range(count)]
    return pl.DataFrame(
        {
            "t_ns": [0] * count,
            "env_id": [env] * count,
            "ped_id": names,
            "ped_name": names,
            "x": [1.0] * count,
            "y": [2.0] * count,
            "yaw": [0.0] * count,
            "agent_type": ["adult"] * count,
            "policy": ["orca"] * count,
            "desired_speed": [1.2] * count,
            "radius_m": [0.3] * count,
            "robot_range_offset": [0.0] * count,
            "robot_fov_offset": [0.0] * count,
            "robot_repulsion_gain": [1.0] * count,
            "robot_reaction_gain": [1.0] * count,
            "robot_blind": blind,
        },
        schema_overrides={"robot_blind": pl.Boolean},
    )


def _entities(env: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "env_id": [env, env],
            "entity_name": ["door_1", "door_1"],
            "kind": ["door", "door"],
            "field_name": ["zone", "activation_dist"],
            "value_str": ["lobby", None],
            "value_num": [None, 2.5],
            "value_bool": [None, None],
        }
    )


def _bundle(
    *,
    env: str = "env_0",
    robot: str = "jackal",
    stamps: list[float] | None = None,
    params: dict[str, str] | None = None,
    blind: list[bool | None] | None = None,
) -> TopicBundle:
    stamps = [0.0, 0.1, 0.2] if stamps is None else stamps
    descriptor = pl.DataFrame({"env_id": [env], "robot_id": [robot], "name": [robot], "model": ["turtlebot3"]})
    bundle = TopicBundle(
        robot_tracks=_track(stamps, env, robot),
        robot_descriptor=descriptor,
        world_semantics=_entities(env),
    )
    if params:
        bundle.robot_params = pl.DataFrame(
            {
                "env_id": [env] * len(params),
                "robot_id": [robot] * len(params),
                "param_key": list(params),
                "param_value": list(params.values()),
            }
        )
    if blind is not None:
        bundle.pedestrian_agent_states = _pedestrians(blind, env)
    return bundle


def _episode(episode_id: int, bundles: dict[str, TopicBundle]) -> dict[str, pl.DataFrame]:
    return episode_dimensions(run_id="bench", episode_id=episode_id, metadata=_metadata(), bundles=bundles)


def test_the_seed_is_the_episodes_own_not_the_latched_or_the_next_queued_one():
    lifecycle = pl.DataFrame(
        {
            "t_ns": [50, 60, 100, 900, 900],
            "outcome_state": [3, 1, 1, 2, 0],
            "seed": [6, 7, 7, 7, 8],
            "start_t_ns": [None, 60, 60, 60, None],
        }
    )
    bundle = _bundle()
    bundle.episode_lifecycle = lifecycle
    assert _episode(0, {"jackal": bundle})["dim_episodes"]["seed"].to_list() == [7]
    bundle.episode_lifecycle = lifecycle.drop("seed").with_columns(seed=pl.lit(None, dtype=pl.Int64))
    assert _episode(0, {"jackal": bundle})["dim_episodes"]["seed"].to_list() == [None]


def test_stacked_episodes_satisfy_the_contract_and_round_trip_through_parquet(
    tmp_path: pathlib.Path,
):
    stacked = stack(
        [
            _episode(0, {"jackal": _bundle(blind=[False])}),
            _episode(1, {"jackal": _bundle(blind=[True])}),
        ]
    )

    enforce_benchmark_contract(tmp_path, dimensions=stacked)
    write_benchmark_dimensions(stacked, tmp_path)
    back = {name: pl.read_parquet(tmp_path / table.relative_path()) for name, table in DIMENSION_TABLES.items()}
    assert {name: frame.schema for name, frame in back.items()} == {name: table.polars_schema() for name, table in DIMENSION_TABLES.items()}
    assert back["dim_episodes"].height == 2


def test_a_duplicated_key_breaks_the_contract(tmp_path):
    episode = _episode(0, {"jackal": _bundle()})
    stacked = stack([episode, episode])

    with pytest.raises(ContractViolationError):
        enforce_benchmark_contract(tmp_path, dimensions=stacked)
    assert "key_duplicate" in (tmp_path / "contract_report.yaml").read_text()


def test_a_null_without_a_reason_breaks_the_contract(tmp_path):
    stacked = stack([_episode(0, {"jackal": _bundle()})])
    stacked["dim_robots"] = stacked["dim_robots"].with_columns(null_reason=pl.lit(None, dtype=pl.String))

    with pytest.raises(ContractViolationError):
        enforce_benchmark_contract(tmp_path, dimensions=stacked)
    assert "null_reason_missing" in (tmp_path / "contract_report.yaml").read_text()


def test_extracting_a_benchmark_publishes_the_geometry_of_its_world(
    tmp_path: pathlib.Path,
):
    pytest.importorskip("mcap_ros2.writer")
    pytest.importorskip("arena_simulation_setup")
    from test_pipeline_extract_cache import _episode

    from arena_evaluation.processing.world_geometry import world_geometry_rows
    from arena_evaluation.storage.manifest import MetadataWriter

    pipeline, episode, _ = _episode(tmp_path)
    MetadataWriter.write(_metadata("map_empty"), pathlib.Path(episode.episode_dir) / "episode_000.yaml")
    pipeline.extract_benchmark("bench")

    stored = DimWorldGeometry.scan(tmp_path / "bench").collect()
    assert_frame_equal(stored, world_geometry_rows("map_empty", tmp_path / "bench"))


def test_episode_dimensions_rejects_bundle_without_identity():
    with pytest.raises(ValueError, match="names its environment and robot"):
        episode_dimensions(
            run_id="bench",
            episode_id=0,
            metadata=_metadata(),
            bundles={"jackal": TopicBundle()},
        )


def test_non_numeric_robot_parameter_is_refused():
    with pytest.raises(ContractViolationError, match="robot_radius"):
        _episode(0, {"jackal": _bundle(params={"robot_radius": "not_a_number"})})
