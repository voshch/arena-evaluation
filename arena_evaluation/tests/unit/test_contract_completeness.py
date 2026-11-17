"""Contract completeness: dim-to-fact coverage, evaluated episodes and robot descriptors."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest

from arena_evaluation.storage.contracts import (
    check_benchmark,
    enforce_benchmark_contract,
)
from arena_evaluation.storage.exceptions import ContractViolationError
from arena_evaluation.storage.layout import write_benchmark_stamp, write_episode_stamp
from arena_evaluation.storage.registry.dimensions import DIMENSION_TABLES
from arena_evaluation.storage.registry.facts import FactEpisodes
from arena_evaluation.storage.registry.table import Table
from arena_evaluation.storage.registry.telemetry import RobotTracks

RUN_ID = "bench-completeness"
ROBOT_ID = "env_0_jackal_0"
ENV_ID = "env_0"


def _frame(rows: list[dict[str, object]], table: type[Table]) -> pl.DataFrame:
    """The rows as the table's declared schema, absent cells NULL."""
    schema = table.polars_schema()
    return pl.DataFrame(
        [{name: row.get(name) for name in schema.names()} for row in rows],
        schema=schema,
    )


def _episode_row(**overrides: object) -> dict[str, object]:
    """One fact_episodes row of an evaluated episode with a recorded pose frame."""
    return {
        "run_id": RUN_ID,
        "episode_id": 0,
        "robot_id": ROBOT_ID,
        "result": "GOAL_REACHED",
        "invalid_reason": "",
        "pose_frame": "map",
        "null_reason": None,
        **overrides,
    }


def _robot_row(**overrides: object) -> dict[str, object]:
    """One dim_robots row with a model and the required recorded parameters."""
    return {
        "run_id": RUN_ID,
        "episode_id": 0,
        "env_id": ENV_ID,
        "robot_id": ROBOT_ID,
        "robot_model": "turtlebot3",
        "robot_radius": 0.22,
        "robot_mass_kg": 1.5,
        "null_reason": None,
        **overrides,
    }


def _dimension_frames(
    robots: list[dict[str, object]] | None = None,
) -> dict[str, pl.DataFrame]:
    """The benchmark dimensions: one episode and robot row, and empty sibling tables."""
    robot_rows = [_robot_row()] if robots is None else robots
    return {
        "dim_episodes": _frame(
            [
                {
                    "run_id": RUN_ID,
                    "episode_id": 0,
                    "env_id": ENV_ID,
                    "robot_id": ROBOT_ID,
                }
            ],
            DIMENSION_TABLES["dim_episodes"],
        ),
        "dim_robots": _frame(robot_rows, DIMENSION_TABLES["dim_robots"]),
        "dim_pedestrians": pl.DataFrame(
            schema=DIMENSION_TABLES["dim_pedestrians"].polars_schema()
        ),
        "dim_world_entities": pl.DataFrame(
            schema=DIMENSION_TABLES["dim_world_entities"].polars_schema()
        ),
        "dim_world_geometry": pl.DataFrame(
            schema=DIMENSION_TABLES["dim_world_geometry"].polars_schema()
        ),
        "dim_metrics": pl.DataFrame(
            schema=DIMENSION_TABLES["dim_metrics"].polars_schema()
        ),
    }


def _write_benchmark(
    root: pathlib.Path,
    *,
    fact_episodes: pl.DataFrame,
    robots: list[dict[str, object]] | None = None,
    topics: dict[str, pl.DataFrame] | None = None,
) -> pathlib.Path:
    """Write the dimension root, one episode's fact file, its stamp and optional topics."""
    root.mkdir(parents=True, exist_ok=True)
    for name, frame in _dimension_frames(robots).items():
        frame.write_parquet(root / f"{name}.parquet")
    write_benchmark_stamp(root)

    episode = root / "episodes" / "episode_000"
    metrics = episode / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    fact_episodes.write_parquet(metrics / FactEpisodes.file_name())
    write_episode_stamp(metrics)

    if topics:
        topics_dir = episode / "topics"
        topics_dir.mkdir(parents=True, exist_ok=True)
        for file_name, frame in topics.items():
            frame.write_parquet(topics_dir / file_name)
    return root


def test_a_dimension_episode_without_a_fact_row_fails(
    tmp_path: pathlib.Path,
) -> None:
    root = _write_benchmark(
        tmp_path / "benchmarks" / RUN_ID,
        fact_episodes=_frame([], FactEpisodes),
    )
    report = check_benchmark(root)
    uncovered = [
        violation
        for violation in report.violations
        if violation.check == "episode_uncovered"
    ]
    assert uncovered
    assert uncovered[0].table == "dim_episodes"
    assert uncovered[0].column == "episode_id"
    assert "0" in uncovered[0].detail


def test_a_fact_episode_with_an_error_status_is_not_evaluated(
    tmp_path: pathlib.Path,
) -> None:
    root = _write_benchmark(
        tmp_path / "benchmarks" / RUN_ID,
        fact_episodes=_frame([_episode_row(result="error")], FactEpisodes),
    )
    report = check_benchmark(root)
    unevaluated = [
        violation
        for violation in report.violations
        if violation.check == "episode_not_evaluated"
    ]
    assert unevaluated
    assert "0" in unevaluated[0].detail
    assert "error" in unevaluated[0].detail
    with pytest.raises(ContractViolationError):
        enforce_benchmark_contract(root)
    assert (root / "contract_report.yaml").is_file()


def test_a_robot_missing_a_required_parameter_fails(
    tmp_path: pathlib.Path,
) -> None:
    root = _write_benchmark(
        tmp_path / "benchmarks" / RUN_ID,
        fact_episodes=_frame([_episode_row()], FactEpisodes),
        robots=[_robot_row(robot_radius=None)],
    )
    report = check_benchmark(root)
    missing = [
        violation
        for violation in report.violations
        if violation.check == "descriptor_param_missing"
    ]
    assert missing
    assert missing[0].table == "dim_robots"
    assert missing[0].column == "robot_radius"
    assert ROBOT_ID in missing[0].detail


def test_a_telemetry_robot_without_a_dimension_row_fails(
    tmp_path: pathlib.Path,
) -> None:
    ghost = "env_0_ghost_0"
    root = _write_benchmark(
        tmp_path / "benchmarks" / RUN_ID,
        fact_episodes=_frame([_episode_row()], FactEpisodes),
        topics={
            "robot_tracks.parquet": _frame(
                [{"env_id": ENV_ID, "robot_id": ghost, "t_ns": 0}],
                RobotTracks,
            )
        },
    )
    report = check_benchmark(root)
    undescribed = [
        violation
        for violation in report.violations
        if violation.check == "descriptor_missing_telemetry"
    ]
    assert undescribed
    assert undescribed[0].table == "dim_robots"
    assert undescribed[0].column == "robot_id"
    assert ghost in undescribed[0].detail
