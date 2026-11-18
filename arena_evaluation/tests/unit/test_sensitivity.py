"""The sensitivity rerun reads the published layout through the storage API and reruns each robot.

Every fixture is written to disk in the published layout by these tests; no mocks.
"""

from __future__ import annotations

import ast
import pathlib
from collections.abc import Sequence

import polars as pl
import pytest

import arena_evaluation.processing.sensitivity as sensitivity
from arena_evaluation.processing.sensitivity import rerun_sensitivity
from arena_evaluation.storage.exceptions import MetricInputError
from arena_evaluation.storage.registry.dimensions import DIMENSION_TABLES
from arena_evaluation.storage.registry.facts import FACT_TABLES
from arena_evaluation.storage.registry.telemetry import (
    PedestrianArenaPeds,
    RobotTracks,
)

SEC = 1_000_000_000
TIMES_NS = [index * SEC for index in range(11)]

ROBOT_SCHEMA = {
    "t_ns": pl.Int64,
    "env_id": pl.String,
    "robot_id": pl.String,
    "x": pl.Float64,
    "y": pl.Float64,
    "yaw": pl.Float64,
    "vx": pl.Float64,
    "vy": pl.Float64,
}
PED_SCHEMA = {
    "t_ns": pl.Int64,
    "ped_id": pl.String,
    "ped_name": pl.String,
    "x": pl.Float64,
    "y": pl.Float64,
    "vx": pl.Float64,
    "vy": pl.Float64,
    "radius_m": pl.Float64,
}
DIM_SCHEMA = {
    "run_id": pl.String,
    "episode_id": pl.Int64,
    "env_id": pl.String,
    "robot_id": pl.String,
    "robot_radius": pl.Float64,
}


def _write(path: pathlib.Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def _robot_frame(robots: Sequence[tuple[str, str, float, float]]) -> pl.DataFrame:
    rows = [
        (stamp, env_id, robot_id, x, y)
        for env_id, robot_id, x, y in robots
        for stamp in TIMES_NS
    ]
    count = len(rows)
    return pl.DataFrame(
        {
            "t_ns": [row[0] for row in rows],
            "env_id": [row[1] for row in rows],
            "robot_id": [row[2] for row in rows],
            "x": [row[3] for row in rows],
            "y": [row[4] for row in rows],
            "yaw": [0.0] * count,
            "vx": [0.0] * count,
            "vy": [0.0] * count,
        },
        schema=ROBOT_SCHEMA,
    )


def _ped_frame(peds: Sequence[tuple[str, float, float, float]]) -> pl.DataFrame:
    rows = [
        (stamp, name, x, y, radius) for name, x, y, radius in peds for stamp in TIMES_NS
    ]
    count = len(rows)
    return pl.DataFrame(
        {
            "t_ns": [row[0] for row in rows],
            "ped_id": [row[1] for row in rows],
            "ped_name": [row[1] for row in rows],
            "x": [row[2] for row in rows],
            "y": [row[3] for row in rows],
            "vx": [0.0] * count,
            "vy": [0.0] * count,
            "radius_m": [row[4] for row in rows],
        },
        schema=PED_SCHEMA,
    )


def _write_episode(
    root: pathlib.Path,
    episode_id: int,
    *,
    robots: Sequence[tuple[str, str, float, float]] | None = None,
    peds: Sequence[tuple[str, float, float, float]] | None = None,
    ped_table: type = PedestrianArenaPeds,
) -> None:
    episode_dir = root / "episodes" / f"episode_{episode_id:03d}"
    if robots is not None:
        _write(episode_dir / RobotTracks.relative_path(), _robot_frame(robots))
    if peds is not None:
        _write(episode_dir / ped_table.relative_path(), _ped_frame(peds))


def _write_robot_radii(
    root: pathlib.Path, rows: Sequence[tuple[int, str, str, float]]
) -> None:
    _write(
        root / DIMENSION_TABLES["dim_robots"].relative_path(),
        pl.DataFrame(
            {
                "run_id": ["run"] * len(rows),
                "episode_id": [row[0] for row in rows],
                "env_id": [row[1] for row in rows],
                "robot_id": [row[2] for row in rows],
                "robot_radius": [row[3] for row in rows],
            },
            schema=DIM_SCHEMA,
        ),
    )


def _episode_row(result: pl.DataFrame, episode_id: int) -> dict:
    return result.filter(pl.col("episode_id") == episode_id).row(0, named=True)


def test_each_robot_of_an_episode_is_rerun_with_its_own_radius(
    tmp_path: pathlib.Path,
) -> None:
    _write_robot_radii(
        tmp_path,
        [(0, "env_0", "env_0_jackal_0", 0.3), (0, "env_0", "env_0_jackal_1", 0.6)],
    )
    _write_episode(
        tmp_path,
        0,
        robots=[
            ("env_0", "env_0_jackal_0", 0.0, 0.0),
            ("env_0", "env_0_jackal_1", 5.0, 0.0),
        ],
        peds=[("Pedestrian_0", 6.5, 0.0, 0.3)],
    )

    row = _episode_row(rerun_sensitivity(tmp_path), 0)

    assert row["rerun_n_encounters"] == 1
    assert row["rerun_min_clearance_m"] == pytest.approx(0.6)


def test_the_detector_parameters_change_the_rerun(tmp_path: pathlib.Path) -> None:
    _write_robot_radii(tmp_path, [(0, "env_0", "env_0_jackal_0", 0.3)])
    _write_episode(
        tmp_path,
        0,
        robots=[("env_0", "env_0_jackal_0", 0.0, 0.0)],
        peds=[("Pedestrian_0", 1.0, 0.0, 0.3)],
    )

    tight = _episode_row(
        rerun_sensitivity(tmp_path, detector_params={"encounter_edge_distance_m": 0.1}),
        0,
    )

    assert tight["rerun_n_encounters"] == 0


def test_a_missing_robot_stream_is_named_with_the_registry_vocabulary(
    tmp_path: pathlib.Path,
) -> None:
    _write_robot_radii(tmp_path, [(0, "env_0", "env_0_jackal_0", 0.3)])
    _write_episode(tmp_path, 0, peds=[("Pedestrian_0", 1.0, 0.0, 0.3)])

    row = _episode_row(rerun_sensitivity(tmp_path), 0)

    assert row["null_reason"] == "topic_missing:robot_tracks"


def test_a_robot_without_a_recorded_radius_is_refused_not_guessed(
    tmp_path: pathlib.Path,
) -> None:
    _write_robot_radii(tmp_path, [])
    _write_episode(
        tmp_path,
        0,
        robots=[("env_0", "env_0_jackal_0", 0.0, 0.0)],
        peds=[("Pedestrian_0", 1.0, 0.0, 0.3)],
    )

    row = _episode_row(rerun_sensitivity(tmp_path), 0)

    assert row["null_reason"] == "not_recorded:robot_radius"
    assert row["rerun_n_encounters"] is None


def test_an_unknown_detector_parameter_is_refused(tmp_path: pathlib.Path) -> None:
    with pytest.raises(MetricInputError, match="unknown detector parameter"):
        rerun_sensitivity(tmp_path, detector_params={"encounter_distance_m": 1.0})


def test_the_published_side_counts_the_pedestrian_encounters_of_the_episode(
    tmp_path: pathlib.Path,
) -> None:
    _write_robot_radii(tmp_path, [(0, "env_0", "env_0_jackal_0", 0.3)])
    _write_episode(
        tmp_path,
        0,
        robots=[("env_0", "env_0_jackal_0", 0.0, 0.0)],
        peds=[("Pedestrian_0", 1.0, 0.0, 0.3)],
    )
    _write(
        tmp_path
        / "episodes"
        / "episode_000"
        / FACT_TABLES["encounters"].relative_path(),
        pl.DataFrame(
            {
                "episode_id": [0, 0, 0],
                "actor_2_kind": ["pedestrian", "pedestrian", "robot"],
                "min_clearance_m": [0.5, 0.9, 0.1],
                "duration_s": [1.0, 2.0, 3.0],
            }
        ),
    )

    row = _episode_row(rerun_sensitivity(tmp_path), 0)

    assert row["published_n_encounters"] == 2
    assert row["published_min_clearance_m"] == pytest.approx(0.5)
    assert row["published_total_duration_s"] == pytest.approx(3.0)


def test_the_rerun_reads_the_recorded_layout_only_through_the_storage_api() -> None:
    tree = ast.parse(pathlib.Path(sensitivity.__file__).read_text(encoding="utf-8"))
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    assert not {"read_parquet", "scan_parquet"} & attributes
