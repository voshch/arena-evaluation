"""Golden schemas of the twelve fact tables' identity, recorded columns, row keys, and null reasons."""

from __future__ import annotations

import polars as pl
import pytest

from arena_evaluation.storage.registry import (
    FACT_KEY_COLUMNS,
    FACT_TABLES,
    FactTable,
)

_EXPECTED_SCHEMAS: dict[FactTable, pl.Schema] = {
    FactTable.EPISODES: pl.Schema(
        {
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "robot_id": pl.String,
            "result": pl.String,
            "invalid_reason": pl.String,
            "pose_frame": pl.String,
            "null_reason": pl.String,
        }
    ),
    FactTable.ENCOUNTERS: pl.Schema(
        {
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "robot_id": pl.String,
            "encounter_id": pl.String,
            "actor_1": pl.String,
            "actor_2": pl.String,
            "actor_2_kind": pl.String,
            "t_start_ns": pl.Int64,
            "t_end_ns": pl.Int64,
            "contact_id": pl.String,
            "approach_type": pl.String,
            "min_clearance_m": pl.Float64,
            "duration_s": pl.Float64,
            "detector_version": pl.String,
            "detector_params_hash": pl.String,
            "null_reason": pl.String,
        }
    ),
    FactTable.CONTACTS: pl.Schema(
        {
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "robot_id": pl.String,
            "contact_id": pl.String,
            "encounter_id": pl.String,
            "collider_kind": pl.String,
            "obstacle_id": pl.String,
            "t_onset_ns": pl.Int64,
            "t_end_ns": pl.Int64,
            "duration_s": pl.Float64,
            "null_reason": pl.String,
        }
    ),
    FactTable.PEDESTRIANS: pl.Schema(
        {
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "env_id": pl.String,
            "ped_id": pl.String,
            "ped_name": pl.String,
            "ped_policy": pl.String,
            "desired_speed": pl.Float64,
            "radius_m": pl.Float64,
            "spawn_x": pl.Float64,
            "spawn_y": pl.Float64,
            "spawn_yaw": pl.Float64,
            "null_reason": pl.String,
        }
    ),
    FactTable.ROBOTS: pl.Schema(
        {
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "env_id": pl.String,
            "robot_id": pl.String,
            "robot_name": pl.String,
            "null_reason": pl.String,
        }
    ),
    FactTable.EDGE_CASES: pl.Schema(
        {
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "robot_id": pl.String,
            "case_idx": pl.Int64,
            "kind": pl.String,
            "ordinal": pl.Int64,
            "element": pl.String,
            "null_reason": pl.String,
        }
    ),
    FactTable.FORMATIONS: pl.Schema(
        {
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "robot_id": pl.String,
            "formation_idx": pl.Int64,
            "role": pl.String,
            "element": pl.String,
            "null_reason": pl.String,
        }
    ),
    FactTable.PHASE_OUTCOMES: pl.Schema(
        {
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "robot_id": pl.String,
            "phase_idx": pl.Int64,
            "outcome": pl.String,
            "met": pl.Boolean,
            "null_reason": pl.String,
        }
    ),
    FactTable.TASK_OUTCOMES: pl.Schema(
        {
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "robot_id": pl.String,
            "task_idx": pl.Int64,
            "task_id": pl.String,
            "outcome": pl.String,
            "duration_s": pl.Float64,
            "null_reason": pl.String,
        }
    ),
    FactTable.JUDGE_DISAGREEMENTS: pl.Schema(
        {
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "robot_id": pl.String,
            "disagreement_idx": pl.Int64,
            "disagreement": pl.String,
            "null_reason": pl.String,
        }
    ),
    FactTable.MECHANISM_STATES: pl.Schema(
        {
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "robot_id": pl.String,
            "entity": pl.String,
            "level": pl.String,
            "kind": pl.String,
            "field": pl.String,
            "t_ns": pl.Int64,
            "value": pl.Float64,
            "null_reason": pl.String,
        }
    ),
    FactTable.COLLISIONS: pl.Schema(
        {
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "robot_id": pl.String,
            "collision_idx": pl.Int64,
            "obstacle_id": pl.String,
            "obstacle_kind": pl.String,
            "t_ns": pl.Int64,
            "x_m": pl.Float64,
            "y_m": pl.Float64,
            "null_reason": pl.String,
        }
    ),
}

_EXPECTED_KEYS: dict[str, tuple[str, ...]] = {
    "episodes": ("run_id", "episode_id", "robot_id"),
    "encounters": ("run_id", "episode_id", "robot_id", "encounter_id"),
    "contacts": ("run_id", "episode_id", "robot_id", "contact_id"),
    "pedestrians": ("run_id", "episode_id", "env_id", "ped_id"),
    "robots": ("run_id", "episode_id", "env_id", "robot_id"),
    "edge_cases": ("run_id", "episode_id", "robot_id", "case_idx", "kind", "ordinal"),
    "formations": ("run_id", "episode_id", "robot_id"),
    "phase_outcomes": ("run_id", "episode_id", "robot_id", "phase_idx"),
    "task_outcomes": ("run_id", "episode_id", "robot_id", "task_idx"),
    "judge_disagreements": ("run_id", "episode_id", "robot_id", "disagreement_idx"),
    "mechanism_states": ("run_id", "episode_id", "robot_id"),
    "collisions": ("run_id", "episode_id", "robot_id", "collision_idx"),
}


def _declared_schema(table: FactTable) -> pl.Schema:
    """The table's declared base columns; ``polars_schema`` merges its scope's metrics."""
    return pl.Schema({name: column.dtype for name, column in FACT_TABLES[table]._columns.items()})


def test_the_registry_holds_the_twelve_fact_tables() -> None:
    assert set(FACT_TABLES) == set(FactTable)


@pytest.mark.parametrize("table", list(FactTable))
def test_fact_schema_is_pinned_in_name_and_order(table: FactTable) -> None:
    assert list(_declared_schema(table).items()) == list(_EXPECTED_SCHEMAS[table].items())


def test_row_keys_are_pinned() -> None:
    assert dict(FACT_KEY_COLUMNS) == _EXPECTED_KEYS
