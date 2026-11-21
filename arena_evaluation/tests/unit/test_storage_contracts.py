"""Publish-time contract checks: planted failures name the table and column."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest
import yaml

from arena_evaluation.storage.contracts import (
    check_benchmark,
    enforce_benchmark_contract,
)
from arena_evaluation.storage.exceptions import ContractViolationError
from arena_evaluation.storage.layout import write_benchmark_stamp, write_episode_stamp
from arena_evaluation.storage.registry.dimensions import DIMENSION_TABLES
from arena_evaluation.storage.registry.table import NULL_REASON_COLUMN, Table
from arena_evaluation.storage.registry.facts import (
    FactEncounters,
    FactEpisodes,
    FactPhaseOutcomes,
    FactRobots,
)

RUN_ID = "bench-contract"
ROBOT_ID = "env_0_jackal_0"
ENV_ID = "env_0"


def _frame(rows: list[dict[str, object]], table: type[Table]) -> pl.DataFrame:
    """The rows as the table's declared schema, absent cells NULL."""
    schema = table.polars_schema()
    return pl.DataFrame(
        [{name: row.get(name) for name in schema.names()} for row in rows],
        schema=schema,
    )


def _episode_row(episode_id: int = 0, **overrides: object) -> dict[str, object]:
    return {
        "run_id": RUN_ID,
        "episode_id": episode_id,
        "robot_id": ROBOT_ID,
        "result": None,
        "null_reason": "result:not_recorded",
        **overrides,
    }


def _encounter_row(**overrides: object) -> dict[str, object]:
    return {
        "run_id": RUN_ID,
        "episode_id": 0,
        "robot_id": ROBOT_ID,
        "encounter_id": "enc-000-001",
        "actor_1": ROBOT_ID,
        "actor_2": "Pedestrian_1",
        "actor_2_kind": "pedestrian",
        "t_start_ns": 0,
        "t_end_ns": 1_000_000_000,
        "contact_id": "",
        "null_reason": "detour_time_s:no_samples",
        **overrides,
    }


def _robot_row(robot_id: str = ROBOT_ID, **overrides: object) -> dict[str, object]:
    return {
        "run_id": RUN_ID,
        "episode_id": 0,
        "env_id": ENV_ID,
        "robot_id": robot_id,
        "robot_name": robot_id,
        "null_reason": None,
        **overrides,
    }


def _protagonist_robot_row(**overrides: object) -> dict[str, object]:
    return {
        "run_id": RUN_ID,
        "episode_id": 0,
        "env_id": ENV_ID,
        "robot_id": ROBOT_ID,
        "robot_model": "turtlebot3",
        "planner": "dwb",
        "robot_radius": 0.22,
        "robot_mass_kg": 1.5,
        "null_reason": None,
        **overrides,
    }


def _raw_dimension_frames() -> dict[str, pl.DataFrame]:
    return {
        "dim_episodes": _frame(
            [
                {
                    "run_id": RUN_ID,
                    "episode_id": 0,
                    "env_id": ENV_ID,
                    "robot_id": ROBOT_ID,
                    "benchmark_id": RUN_ID,
                    "planner": "dwb",
                    "stage": "s0",
                    "world": "map_empty",
                    "null_reason": None,
                }
            ],
            DIMENSION_TABLES["dim_episodes"],
        ),
        "dim_robots": _frame(
            [_protagonist_robot_row()],
            DIMENSION_TABLES["dim_robots"],
        ),
        "dim_pedestrians": pl.DataFrame(schema=DIMENSION_TABLES["dim_pedestrians"].polars_schema()),
        "dim_world_entities": pl.DataFrame(schema=DIMENSION_TABLES["dim_world_entities"].polars_schema()),
        "dim_world_geometry": pl.DataFrame(schema=DIMENSION_TABLES["dim_world_geometry"].polars_schema()),
        "dim_metrics": pl.DataFrame(schema=DIMENSION_TABLES["dim_metrics"].polars_schema()),
    }


def _explained(frame: pl.DataFrame, table: type[Table]) -> pl.DataFrame:
    """The frame with every NULL in a reason-carrying column explained as not recorded."""
    rows = []
    for row in frame.iter_rows(named=True):
        reasons = [f"{column}:not_recorded" for column in table.null_reason_columns() if row[column] is None]
        joined = "; ".join(filter(None, [row[NULL_REASON_COLUMN], *reasons]))
        rows.append({**row, NULL_REASON_COLUMN: joined or None})
    return pl.DataFrame(rows, schema=frame.schema)


def _dimension_frames() -> dict[str, pl.DataFrame]:
    """The clean dimension frames, every NULL explained by its reason."""
    return {name: _explained(frame, DIMENSION_TABLES[name]) for name, frame in _raw_dimension_frames().items()}


def _write_benchmark(
    root: pathlib.Path,
    *,
    fact_episodes: pl.DataFrame,
    fact_encounters: pl.DataFrame | None = None,
    fact_robots: pl.DataFrame | None = None,
    fact_phase_outcomes: pl.DataFrame | None = None,
    write_dimensions: bool = True,
) -> pathlib.Path:
    """Write the canonical dimension root and the per-episode fact files, stamped for the layout."""
    root.mkdir(parents=True, exist_ok=True)
    if write_dimensions:
        for name, frame in _dimension_frames().items():
            frame.write_parquet(root / f"{name}.parquet")
    write_benchmark_stamp(root)

    metrics = root / "episodes" / "episode_000" / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    fact_episodes.write_parquet(metrics / "fact_episodes.parquet")
    for table, frame in (
        (FactEncounters, fact_encounters),
        (FactRobots, fact_robots),
        (FactPhaseOutcomes, fact_phase_outcomes),
    ):
        if frame is not None:
            frame.write_parquet(metrics / table.file_name())
    write_episode_stamp(metrics)
    return root


def _clean_benchmark(tmp_path: pathlib.Path) -> pathlib.Path:
    return _write_benchmark(
        tmp_path / "benchmarks" / RUN_ID,
        fact_episodes=_frame([_episode_row()], FactEpisodes),
        fact_encounters=_frame([_encounter_row()], FactEncounters),
    )


def test_a_nested_column_fails_the_flat_schema_contract(
    tmp_path: pathlib.Path,
) -> None:
    nested = _frame([_episode_row()], FactEpisodes).with_columns(pl.Series("tags", [["tag"]], dtype=pl.List(pl.String)))
    root = _write_benchmark(tmp_path / "benchmarks" / RUN_ID, fact_episodes=nested)
    report = check_benchmark(root)
    nested_violations = [violation for violation in report.violations if violation.check == "nested_dtype"]
    assert nested_violations
    assert nested_violations[0].table == "fact_episodes"
    assert nested_violations[0].column == "tags"
    assert "tags" in nested_violations[0].message()


def test_an_orphan_fact_row_fails(tmp_path: pathlib.Path) -> None:
    root = _write_benchmark(
        tmp_path / "benchmarks" / RUN_ID,
        fact_episodes=_frame([_episode_row()], FactEpisodes),
        fact_robots=_frame([_robot_row("ghost")], FactRobots),
    )
    report = check_benchmark(root)
    orphans = [violation for violation in report.violations if violation.check == "orphan_fact"]
    assert orphans and orphans[0].table == "fact_robots"
    assert "robot_id" in orphans[0].column


def test_an_unknown_null_reason_code_fails_and_names_the_table(
    tmp_path: pathlib.Path,
) -> None:
    root = _write_benchmark(
        tmp_path / "benchmarks" / RUN_ID,
        fact_episodes=_frame([_episode_row(null_reason="result:made_up")], FactEpisodes),
    )
    report = check_benchmark(root)
    unknown = [violation for violation in report.violations if violation.check == "null_reason_unknown"]
    assert unknown and unknown[0].table == "fact_episodes"
    assert unknown[0].column == "null_reason"
    assert "made_up" in unknown[0].message()


def test_a_window_that_does_not_move_forward_fails(tmp_path: pathlib.Path) -> None:
    root = _write_benchmark(
        tmp_path / "benchmarks" / RUN_ID,
        fact_episodes=_frame([_episode_row()], FactEpisodes),
        fact_encounters=_frame(
            [_encounter_row(t_start_ns=1_000_000_000, t_end_ns=1_000_000_000)],
            FactEncounters,
        ),
    )
    report = check_benchmark(root)
    assert any(violation.check == "window_order" and violation.table == "fact_encounters" for violation in report.violations)


def test_an_encounter_actor_paired_with_itself_fails(
    tmp_path: pathlib.Path,
) -> None:
    root = _write_benchmark(
        tmp_path / "benchmarks" / RUN_ID,
        fact_episodes=_frame([_episode_row()], FactEpisodes),
        fact_encounters=_frame([_encounter_row(actor_2=ROBOT_ID)], FactEncounters),
    )
    report = check_benchmark(root)
    assert any(violation.check == "actor_self_pair" and violation.table == "fact_encounters" for violation in report.violations)


def test_a_robot_without_a_recorded_descriptor_fails(
    tmp_path: pathlib.Path,
) -> None:
    root = _write_benchmark(
        tmp_path / "benchmarks" / RUN_ID,
        fact_episodes=_frame([_episode_row()], FactEpisodes),
        write_dimensions=False,
    )
    dimensions = _dimension_frames()
    dimensions["dim_robots"] = _frame(
        [_protagonist_robot_row(robot_model=None, null_reason="robot_model:not_recorded")],
        DIMENSION_TABLES["dim_robots"],
    )
    for name, frame in dimensions.items():
        frame.write_parquet(root / f"{name}.parquet")
    report = check_benchmark(root)
    missing = [violation for violation in report.violations if violation.check == "descriptor_missing"]
    assert missing and missing[0].table == "dim_robots"
    assert missing[0].column == "robot_model"


def test_enforce_writes_the_report_and_raises(tmp_path: pathlib.Path) -> None:
    root = _write_benchmark(
        tmp_path / "benchmarks" / RUN_ID,
        fact_episodes=_frame([_episode_row(), _episode_row()], FactEpisodes),
    )
    with pytest.raises(ContractViolationError) as error:
        enforce_benchmark_contract(root)
    assert "fact_episodes" in str(error.value)
    assert "episode_id" in str(error.value)
    report_path = root / "contract_report.yaml"
    assert report_path.is_file()
    payload = yaml.safe_load(report_path.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["violations"]
    assert any("fact_episodes" in violation["table"] for violation in payload["violations"])
