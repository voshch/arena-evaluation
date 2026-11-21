"""Legacy generations: field-tolerant extraction, descriptor synthesis, benchmark-root migration and its report."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest
import yaml

from arena_evaluation.ingestion.migrate import (
    MigrationError,
    migrate_benchmark,
    migrate_episode,
    migrate_run,
)
from arena_evaluation.processing import legacy_schemas
from test_mcap_reader_topics import (
    AGENT_STATES,
    EPISODE_RECORD,
    HEADER,
    KEY_VALUE,
    PARAM,
    PARAM_VALUE,
    SEP,
    TIME,
    _header,
    _msgdef,
    _write_bag,
)

SEC = 1_000_000_000
STRING = ("std_msgs/msg/String", "string data")
SEPT_EPISODE = (
    "task_generator_msgs/msg/EpisodeRecord",
    SEP.join(
        [
            "uint32 episode_id\nstring world\nint64 seed\nstring tm_robots\n"
            "string tm_obstacles\nstring[] tm_modules\nstring[] robots\n"
            "uint8 RUNNING = 1\nuint8 outcome_state\nstring outcome_info\n"
            "string goal_uuid\nfloat32 goal_dist_start\nfloat32 goal_dist_min\n"
            "float32 path_length\nbuiltin_interfaces/Time start_time\n"
            "string[] integrity\nrcl_interfaces/Parameter[] obstacles_params\n"
            "rcl_interfaces/Parameter[] robots_params\n"
            "rcl_interfaces/Parameter[] human_params\nstring conditions\n"
            "string phases",
            TIME,
            PARAM,
            PARAM_VALUE,
            KEY_VALUE,
        ]
    ),
)
OLD_AGENT_META = (
    "arena_humansim_msgs/msg/AgentMeta",
    _msgdef(
        "std_msgs/Header header\nstring[] policies\nint32[] agent_id\nstring[] name\nstring[] handedness\nstring[] agent_type\nstring[] source_name\nint32[] source_seq",
        HEADER,
        TIME,
    ),
)

_METADATA = {
    "benchmark_id": "bench",
    "planner": "dwa",
    "robot_model": ["jackal"],
    "map": "barn",
    "stage": "s0",
    "recording_started_at": "2026-01-01T00:00:00",
    "python_version": "3.12",
    "ros_distro": "jazzy",
    "tm_robots": "jackal",
    "tm_obstacles": "static",
    "tm_modules": [],
    "robots_params": {"speed": 1.0},
    "obstacles_params": {"count": 3},
}


def _sept_records(
    robot_phase: bool = True,
) -> list[tuple[str, tuple[str, str], dict, int]]:
    records: list[tuple[str, tuple[str, str], dict, int]] = [
        (
            "/env_0/state/episode",
            SEPT_EPISODE,
            {
                "episode_id": 1,
                "world": "barn",
                "seed": 7,
                "tm_robots": "jackal",
                "tm_obstacles": "static",
                "tm_modules": [],
                "robots": ["jackal"],
                "outcome_state": 2,
                "outcome_info": "",
                "goal_uuid": "",
                "goal_dist_start": 1.0,
                "goal_dist_min": 0.5,
                "path_length": 2.0,
                "start_time": {"sec": 1, "nanosec": 0},
                "integrity": [],
                "obstacles_params": [{"name": "count", "value": {"type": 2, "integer_value": 3}}],
                "robots_params": [{"name": "speed", "value": {"type": 3, "double_value": 1.0}}],
                "human_params": [],
                "conditions": "",
                "phases": "",
            },
            SEC,
        )
    ]
    if robot_phase:
        records.append(("/env_0/jackal/characterization_phase", STRING, {"data": "ramp"}, 2 * SEC))
    return records


def _episode(
    directory: pathlib.Path,
    records: list[tuple[str, tuple[str, str], dict, int]],
    metadata: dict | None = None,
) -> pathlib.Path:
    directory.mkdir(parents=True)
    _write_bag(directory / f"{directory.name}.mcap", records)
    (directory / f"{directory.name}.yaml").write_text(yaml.safe_dump(_METADATA if metadata is None else metadata), encoding="utf-8")
    return directory


def test_the_september_generation_is_detected_and_mapped_without_a_native_field(
    tmp_path: pathlib.Path,
) -> None:
    episode = _episode(tmp_path / "episode_000", _sept_records())

    assert legacy_schemas.detect(episode / "episode_000.mcap").name == "legacy_2026_09"

    metadata = migrate_episode(episode)

    assert metadata.modes == {
        "robots": ["jackal"],
        "obstacles": ["static"],
        "modules": [],
    }
    assert metadata.params == {"task.jackal.speed": 1.0, "task.static.count": 3}
    lifecycle = pl.read_parquet(episode / "topics" / "episode_lifecycle.parquet")
    row = lifecycle.row(0, named=True)
    assert row["task_layer"] is None
    assert "jackal" in row["modes_json"]
    assert "task.jackal.speed" in row["params_yaml"]
    assert row["seed"] == 7


def test_an_old_agent_meta_leaves_the_crowd_settings_null(
    tmp_path: pathlib.Path,
) -> None:
    records: list[tuple[str, tuple[str, str], dict, int]] = [
        (
            "/env_0/state/episode",
            EPISODE_RECORD,
            {
                "episode_id": 1,
                "world": "barn",
                "seed": 7,
                "task_layer": "task_generator",
                "modes": [{"key": "robots", "value": "jackal"}],
                "robots": ["jackal"],
                "outcome_state": 2,
                "outcome_info": "",
                "goal_uuid": "",
                "goal_dist_start": 1.0,
                "goal_dist_min": 0.5,
                "path_length": 2.0,
                "start_time": {"sec": 1, "nanosec": 0},
                "integrity": [],
                "params": [{"name": "x", "value": {"type": 3, "double_value": 1.0}}],
                "human_params": [],
                "conditions": "",
                "phases": "",
            },
            SEC,
        ),
        (
            "/env_0/agent_meta",
            OLD_AGENT_META,
            {
                "header": _header(0, 0, "env_0/map"),
                "policies": ["orca"],
                "agent_id": [1],
                "name": ["ped_1"],
                "handedness": ["r"],
                "agent_type": ["adult"],
                "source_name": [""],
                "source_seq": [0],
            },
            2 * SEC,
        ),
        (
            "/env_0/agent_states",
            AGENT_STATES,
            {
                "header": _header(2, 0, "env_0/map"),
                "agents": [
                    {
                        "agent_id": 1,
                        "name": "ped_1",
                        "kind": 0,
                        "pose": {"x": 1.0, "y": 2.0, "theta": 0.0},
                        "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "desired_velocity": 1.2,
                        "radius": 0.3,
                        "agent_type": "adult",
                        "animation_state": 1,
                        "policy": "orca",
                    }
                ],
            },
            2 * SEC,
        ),
        (
            "/env_0/jackal/characterization_phase",
            STRING,
            {"data": "ramp"},
            3 * SEC,
        ),
    ]
    episode = _episode(tmp_path / "episode_000", records)

    assert legacy_schemas.detect(episode / "episode_000.mcap").name == "legacy_2026_11"

    migrate_episode(episode)

    states = pl.read_parquet(episode / "topics" / "pedestrian_agent_states.parquet")
    assert states.height == 1
    assert states["robot_awareness"].to_list() == [None]
    generation = legacy_schemas.detect(episode / "episode_000.mcap")
    assert any("AgentMeta has no robot_* fields" in note for note in generation.unsupported)


def test_a_missing_robot_definition_is_a_loud_migration_error(
    tmp_path: pathlib.Path,
) -> None:
    metadata = {**_METADATA, "robot_model": ["definitely_not_a_robot"]}
    episode = _episode(tmp_path / "episode_000", _sept_records(), metadata)

    with pytest.raises(MigrationError, match="definitely_not_a_robot"):
        migrate_episode(episode)


def test_a_benchmark_root_migrates_only_its_episodes_and_writes_the_report(
    tmp_path: pathlib.Path,
) -> None:
    root = tmp_path / "bench"
    (root / "episodes").mkdir(parents=True)
    _episode(root / "episodes" / "episode_000", _sept_records())
    (root / "manifest.yaml").write_text("run_id: bench\n", encoding="utf-8")
    (root / "report_manifest.yaml").write_text("id: r\n", encoding="utf-8")

    migrated = migrate_run(root)

    assert migrated == [root / "episodes" / "episode_000"]
    report = yaml.safe_load((root / "migration_report.yaml").read_text(encoding="utf-8"))
    assert report["episodes"][0]["episode"] == "episode_000"
    assert report["episodes"][0]["generation"] == "legacy_2026_09"
    assert report["episodes"][0]["migrated"] is True
    assert (root / "episodes" / "episode_000" / "topics" / "episode_lifecycle.parquet").exists()


def test_a_dry_run_migrates_nothing(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "bench"
    (root / "episodes").mkdir(parents=True)
    episode = _episode(root / "episodes" / "episode_000", _sept_records())

    report = migrate_benchmark(root, dry_run=True)

    assert report.episodes[0].generation == "legacy_2026_09"
    assert report.episodes[0].migrated is False
    assert not (episode / "topics").exists()
    assert not (root / "migration_report.yaml").exists()


def test_a_second_run_is_idempotent(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "bench"
    (root / "episodes").mkdir(parents=True)
    episode = _episode(root / "episodes" / "episode_000", _sept_records())
    first = migrate_benchmark(root)
    descriptor = pl.read_parquet(episode / "topics" / "robot_descriptor.parquet")

    second = migrate_benchmark(root)

    assert first.episodes[0].migrated is True
    assert second.episodes[0].migrated is False
    assert pl.read_parquet(episode / "topics" / "robot_descriptor.parquet").height == 1
    assert descriptor.height == 1


def test_an_unknown_generation_fails_loudly_but_the_other_episodes_continue(
    tmp_path: pathlib.Path,
) -> None:
    root = tmp_path / "bench"
    (root / "episodes").mkdir(parents=True)
    _episode(root / "episodes" / "episode_000", _sept_records())
    mystery = _episode(root / "episodes" / "episode_001", _sept_records(robot_phase=False))
    _write_bag(
        mystery / "episode_001.mcap",
        [
            (
                "/env_0/state/episode",
                ("some_pkg/msg/MysteryRecord", "uint32 episode_id"),
                {"episode_id": 1},
                SEC,
            )
        ],
    )

    with pytest.raises(MigrationError, match="episode_001"):
        migrate_run(root)

    report = yaml.safe_load((root / "migration_report.yaml").read_text(encoding="utf-8"))
    by_name = {row["episode"]: row for row in report["episodes"]}
    assert by_name["episode_000"]["migrated"] is True
    assert by_name["episode_001"]["error"]
    assert by_name["episode_001"]["generation"] == "unknown"


SEPT_REAL_EPISODE = (
    "task_generator_msgs/msg/EpisodeRecord",
    SEP.join(
        [
            "uint32 episode_id\nstring world\nint64 seed\nstring tm_robots\n"
            "string tm_obstacles\nstring[] tm_modules\nstring[] robots\n"
            "uint8 RUNNING = 1\nuint8 outcome_state\nstring outcome_info\n"
            "string goal_uuid\nfloat32 goal_dist_start\nfloat32 goal_dist_min\n"
            "float32 path_length\nbuiltin_interfaces/Time start_time\n"
            "bool integrity\nrcl_interfaces/Parameter[] obstacles_params\n"
            "rcl_interfaces/Parameter[] robots_params\nstring conditions",
            TIME,
            PARAM,
            PARAM_VALUE,
            KEY_VALUE,
        ]
    ),
)
SEPT_MYSTERY_EPISODE = (
    "task_generator_msgs/msg/EpisodeRecord",
    SEP.join(
        [
            "uint32 episode_id\nstring world\nint64 seed\nstring tm_robots\n"
            "string tm_obstacles\nstring[] tm_modules\nstring[] robots\n"
            "uint8 RUNNING = 1\nuint8 outcome_state\nstring outcome_info\n"
            "string goal_uuid\nfloat32 goal_dist_start\nfloat32 goal_dist_min\n"
            "float32 path_length\nbuiltin_interfaces/Time start_time\n"
            "bool integrity\nrcl_interfaces/Parameter[] obstacles_params\n"
            "rcl_interfaces/Parameter[] robots_params\nstring conditions\n"
            "string mystery",
            TIME,
            PARAM,
            PARAM_VALUE,
            KEY_VALUE,
        ]
    ),
)


def _sept_real_records() -> list[tuple[str, tuple[str, str], dict, int]]:
    return [
        (
            "/env_0/state/episode",
            SEPT_REAL_EPISODE,
            {
                "episode_id": 433,
                "world": "library_acoustic_partition",
                "seed": 101,
                "tm_robots": "jackal",
                "tm_obstacles": "static",
                "tm_modules": [],
                "robots": ["jackal"],
                "outcome_state": 2,
                "outcome_info": "finished",
                "goal_uuid": "goal",
                "goal_dist_start": 1.0,
                "goal_dist_min": 0.5,
                "path_length": 2.0,
                "start_time": {"sec": 1, "nanosec": 0},
                "integrity": False,
                "obstacles_params": [],
                "robots_params": [],
                "conditions": '[{"op": "never", "p": "robot in restricted"}]',
            },
            SEC,
        )
    ]


def test_the_real_september_shape_migrates_with_the_absent_fields_null(
    tmp_path: pathlib.Path,
) -> None:
    episode = _episode(tmp_path / "episode_000", _sept_real_records())

    assert legacy_schemas.detect(episode / "episode_000.mcap").name == "legacy_2026_09"

    migrate_episode(episode)

    lifecycle = pl.read_parquet(episode / "topics" / "episode_lifecycle.parquet")
    row = lifecycle.row(0, named=True)
    assert row["human_params"] is None
    assert row["phases_yaml"] is None
    assert row["conditions_yaml"] == '[{"op": "never", "p": "robot in restricted"}]'
    assert row["task_layer"] is None
    generation = legacy_schemas.detect(episode / "episode_000.mcap")
    assert any("human_params" in note for note in generation.unsupported)
    assert any("phases" in note for note in generation.unsupported)


def test_an_unknown_episode_field_is_a_loud_migration_error(
    tmp_path: pathlib.Path,
) -> None:
    message = {
        **_sept_real_records()[0][2],
        "mystery": "x",
    }
    episode = _episode(
        tmp_path / "episode_000",
        [("/env_0/state/episode", SEPT_MYSTERY_EPISODE, message, SEC)],
    )

    with pytest.raises(MigrationError, match="mystery"):
        migrate_episode(episode)


_SMOKE_ROOT = pathlib.Path("/home/nelson/arena_ws/data/benchmarks")


@pytest.mark.skipif(not _SMOKE_ROOT.is_dir(), reason="no recorded benchmark corpus on this machine")
def test_a_real_recorded_benchmark_has_a_detectable_generation() -> None:
    recording = next(_SMOKE_ROOT.rglob("*.mcap"), None)
    if recording is None:
        pytest.skip("the benchmark corpus holds no recording")
    generation = legacy_schemas.detect(recording)
    assert generation.name in {"native", "legacy_2026_09", "legacy_2026_11"}
