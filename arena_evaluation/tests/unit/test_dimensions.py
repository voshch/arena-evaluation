"""The benchmark dimensions: an episode's rows, the reason of every NULL, the stacked contract and the write at the benchmark root."""

import logging
import pathlib

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from arena_evaluation.processing.dimension_rows import episode_dimensions
from arena_evaluation.processing.metrics.dictionary import metric_dictionary
from arena_evaluation.storage.contracts import enforce_dimension_contract
from arena_evaluation.storage.exceptions import ContractViolationError
from arena_evaluation.storage.registry.dimensions import DIMENSION_TABLES, DimMetrics
from arena_evaluation.storage.schemas import RunMetadata, TopicBundle
from arena_evaluation.storage.star_schema import stack, write_benchmark_dimensions

SEC = 1_000_000_000


def _metadata() -> RunMetadata:
    return RunMetadata(
        benchmark_id="bench",
        planner="dwb",
        map="barn",
        stage="s1",
        recording_started_at="2026-01-01T00:00:00",
        python_version="3.12",
        ros_distro="jazzy",
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


def _pedestrians(awareness: list[float | None], env: str) -> pl.DataFrame:
    count = len(awareness)
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
            "robot_awareness": awareness,
            "robot_fov_gain": [1.0] * count,
            "robot_repulsion_gain": [1.0] * count,
            "robot_reaction_gain": [1.0] * count,
        },
        schema_overrides={"robot_awareness": pl.Float64},
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


def _bundle(*, env: str = "env_0", robot: str = "jackal", stamps: list[float] | None = None, params: dict[str, str] | None = None, awareness: list[float | None] | None = None) -> TopicBundle:
    stamps = [0.0, 0.1, 0.2] if stamps is None else stamps
    descriptor = pl.DataFrame({"env_id": [env], "robot_id": [robot], "name": [robot], "model": ["turtlebot3"]})
    bundle = TopicBundle(robot_tracks=_track(stamps, env, robot), robot_descriptor=descriptor, world_semantics=_entities(env))
    if params:
        bundle.robot_params = pl.DataFrame(
            {
                "env_id": [env] * len(params),
                "robot_id": [robot] * len(params),
                "param_key": list(params),
                "param_value": list(params.values()),
            }
        )
    if awareness is not None:
        bundle.pedestrian_agent_states = _pedestrians(awareness, env)
    return bundle


def _episode(episode_id: int, bundles: dict[str, TopicBundle]) -> dict[str, pl.DataFrame]:
    return episode_dimensions(run_id="bench", episode_id=episode_id, metadata=_metadata(), bundles=bundles)


def test_an_episode_has_a_row_per_robot_and_one_pedestrian_and_entity_set_per_environment():
    bundles = {
        "jackal": _bundle(robot="jackal", awareness=[1.0, 0.0]),
        "turtle": _bundle(robot="turtle", awareness=[1.0, 0.0]),
    }
    dims = _episode(3, bundles)

    assert dims["dim_episodes"].height == 2
    assert sorted(dims["dim_robots"]["robot_id"].to_list()) == ["jackal", "turtle"]
    assert dims["dim_pedestrians"].height == 2
    assert dims["dim_world_entities"].height == 1
    assert dims["dim_world_entities"]["activation_dist"].to_list() == [2.5]


def test_the_pose_stream_of_each_robot_is_recorded_with_its_gap():
    dense = _episode(0, {"jackal": _bundle(stamps=[0.0, 0.1, 0.2])})["dim_episodes"]
    sparse = _episode(1, {"jackal": _bundle(stamps=[0.0, 1.0, 2.0])})["dim_episodes"]

    assert dense["pose_source"].to_list() == ["tf"]
    assert dense["pose_gap_s"][0] == pytest.approx(0.1)
    assert sparse["pose_source"].to_list() == ["tf_sparse"]
    assert sparse["pose_gap_s"][0] == pytest.approx(1.0)


def test_recorded_parameters_are_parsed_and_a_missing_one_carries_its_reason():
    robots = _episode(0, {"jackal": _bundle(params={"robot_radius": "0.25"})})["dim_robots"]

    assert robots["robot_radius"].to_list() == [0.25]


def test_stacked_episodes_satisfy_the_contract_and_round_trip_through_parquet(tmp_path: pathlib.Path):
    stacked = stack([_episode(0, {"jackal": _bundle(awareness=[1.0])}), _episode(1, {"jackal": _bundle(awareness=[0.0])})])

    enforce_dimension_contract(stacked)
    write_benchmark_dimensions(stacked, tmp_path)
    back = {name: pl.read_parquet(tmp_path / table.relative_path()) for name, table in DIMENSION_TABLES.items()}
    assert {name: frame.schema for name, frame in back.items()} == {name: table.polars_schema() for name, table in DIMENSION_TABLES.items()}
    assert back["dim_episodes"].height == 2


def test_a_duplicated_key_breaks_the_contract():
    episode = _episode(0, {"jackal": _bundle()})
    stacked = stack([episode, episode])

    with pytest.raises(ContractViolationError, match="duplicated key"):
        enforce_dimension_contract(stacked)


def test_a_null_without_a_reason_breaks_the_contract():
    stacked = stack([_episode(0, {"jackal": _bundle()})])
    stacked["dim_robots"] = stacked["dim_robots"].with_columns(null_reason=pl.lit(None, dtype=pl.String))

    with pytest.raises(ContractViolationError, match="without a reason"):
        enforce_dimension_contract(stacked)


def test_extracting_a_benchmark_writes_its_dimensions_once_at_the_root(tmp_path: pathlib.Path):
    pytest.importorskip("mcap_ros2.writer")
    from test_pipeline_extract_cache import _episode

    from arena_evaluation.storage.manifest import MetadataWriter

    pipeline, episode, _ = _episode(tmp_path)
    MetadataWriter.write(_metadata(), pathlib.Path(episode.episode_dir) / "episode_000.yaml")
    pipeline.extract_benchmark("bench")

    stored = {name: pl.read_parquet(tmp_path / "bench" / table.relative_path()) for name, table in DIMENSION_TABLES.items()}
    assert stored["dim_episodes"].height == 1
    assert stored["dim_episodes"]["env_id"].to_list() == ["env_0"]
    assert stored["dim_episodes"]["pose_source"].to_list() == ["absent"]


def test_extracting_a_benchmark_publishes_the_metric_dictionary_beside_its_dimensions(tmp_path: pathlib.Path):
    pytest.importorskip("mcap_ros2.writer")
    from test_pipeline_extract_cache import _episode

    from arena_evaluation.storage.manifest import MetadataWriter

    pipeline, episode, _ = _episode(tmp_path)
    MetadataWriter.write(_metadata(), pathlib.Path(episode.episode_dir) / "episode_000.yaml")
    pipeline.extract_benchmark("bench")

    assert (tmp_path / "bench" / "dim_metrics.parquet").is_file()
    stored = DimMetrics.scan(tmp_path / "bench").collect()
    assert stored.schema == DimMetrics.polars_schema()
    assert_frame_equal(stored, metric_dictionary())


def test_an_episode_without_a_recording_warns_and_the_benchmark_still_writes_its_dimensions(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
):
    pytest.importorskip("mcap_ros2.writer")
    from test_pipeline_extract_cache import _episode

    from arena_evaluation.storage.manifest import MetadataWriter

    pipeline, episode, recording = _episode(tmp_path)
    MetadataWriter.write(_metadata(), pathlib.Path(episode.episode_dir) / "episode_000.yaml")
    recording.unlink()

    with caplog.at_level(logging.WARNING):
        pipeline.extract_benchmark("bench")

    assert "no MCAP found" in caplog.text
    assert "no extracted topics" in caplog.text
    stored = {name: pl.read_parquet(tmp_path / "bench" / table.relative_path()) for name, table in DIMENSION_TABLES.items()}
    assert stored["dim_episodes"].height == 0
