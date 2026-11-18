from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import os
import pathlib
import sys
import time

import polars as pl
import pytest

sourced = pytest.mark.skipif(
    "BASH_FUNC__arena_cli%%" not in os.environ,
    reason="needs a shell that sourced arena",
)

RUN_ID = "20261019-120000-basic-basic"
EMPTY_RUN_ID = "20261019-110000-empty-empty"
STEPS = ("dwb/s1", "dwb_social/s1", "unhindered_peds/s1", "dwb/s2")


def _record(world: str, outcome: int) -> object:
    from arena_runtime.env.task_layer import encode_modes
    from arena_runtime_msgs.msg import EpisodeRecord

    return EpisodeRecord(
        world=world,
        seed=7,
        modes=encode_modes({"robots": ["random"], "obstacles": ["random"]}),
        robots=["jackal"],
        outcome_state=outcome,
        outcome_info="",
    )


def manifest(
    run_id: str,
    steps: tuple[str, ...] = STEPS,
    *,
    created_at: str = "2026-10-19T12:00:00+00:00",
) -> object:
    from arena_evaluation.benchmark.state import Manifest

    return Manifest(
        run_id=run_id,
        created_at=created_at,
        arena_git_sha=None,
        arena_git_dirty=False,
        cli_args=["--suite", "basic", "--contest", "basic"],
        env_n=1,
        headless=True,
        config_hash="fixture",
        simulator="dummy",
        scale_episodes=1.0,
        suite_name="basic",
        contest_name="basic",
        suite={"stages": []},
        contest=[],
        steps=[{"key": key, "episodes_planned": 2} for key in steps],
    )


def metric_rows() -> list[dict]:
    """Canonical episode-summary rows: two contestants over two stages plus two reference rows."""
    rows = []
    for planner, success, ttg, collisions, path_efficiency, energy in (
        (
            "dwb",
            [True, True, True, False],
            [40.0, 42.0, 44.0, 90.0],
            [0, 0, 1, 2],
            [0.9, 0.8, 0.7, 0.5],
            [10.0, 11.0, 12.0, 20.0],
        ),
        (
            "dwb_social",
            [True, False, False, False],
            [60.0, 90.0, 90.0, 90.0],
            [1, 2, 3, 3],
            [0.6, 0.5, 0.4, 0.3],
            [15.0, 18.0, 20.0, 25.0],
        ),
    ):
        for index in range(4):
            rows.append(
                {
                    "episode_id": len(rows),
                    "planner": planner,
                    "stage": "s1" if index < 2 else "s2",
                    "world": "map_empty",
                    "is_reference": False,
                    "success": success[index],
                    "time_to_goal": ttg[index],
                    "contact_count": collisions[index],
                    "collision_amount_static": 0,
                    "path_efficiency": path_efficiency[index],
                    "energy_total_wh": energy[index],
                    "min_ped_clearance_m": 0.5,
                    "mar_vs_planned_route": 0.4,
                    "pfi_vs_planned_route": 0.2,
                }
            )
    for _ in range(2):
        rows.append(
            {
                "episode_id": len(rows),
                "planner": "unhindered_peds",
                "stage": "s1",
                "world": "map_empty",
                "is_reference": True,
                "success": None,
                "time_to_goal": None,
                "contact_count": None,
                "collision_amount_static": None,
                "path_efficiency": None,
                "energy_total_wh": None,
                "min_ped_clearance_m": None,
                "mar_vs_planned_route": None,
                "pfi_vs_planned_route": None,
            }
        )
    return rows


def _ensure_calculators_loaded() -> None:
    """Register all calculator-contributed metrics before writing Star Schema fixtures."""
    from arena_evaluation.processing.metrics import MetricRegistry

    MetricRegistry.discover_calculators_cls()


def write_star_schema(benchmark_dir: pathlib.Path, rows: list[dict]) -> None:
    """Write the canonical benchmark dimension and per-episode fact tables the catalog scans."""
    from arena_evaluation.storage.registry.dimensions import DIMENSION_TABLES
    from arena_evaluation.storage.registry.facts import FACT_TABLES
    from arena_evaluation.storage.registry.telemetry import RobotTracks

    _ensure_calculators_loaded()

    benchmark_dir.mkdir(parents=True, exist_ok=True)
    resolved = [
        {"run_id": "run", "robot_id": "robot_0", "env_id": "env_0", **row}
        for row in rows
    ]
    pl.DataFrame(
        [
            {
                name: row.get(name)
                for name in DIMENSION_TABLES["dim_episodes"].polars_schema().names()
            }
            for row in resolved
        ],
        schema=DIMENSION_TABLES["dim_episodes"].polars_schema(),
    ).write_parquet(benchmark_dir / "dim_episodes.parquet")
    for episode_id in sorted({int(row["episode_id"]) for row in resolved}):
        metrics = benchmark_dir / "episodes" / f"episode_{episode_id:03d}" / "metrics"
        metrics.mkdir(parents=True, exist_ok=True)
        episode_rows = [row for row in resolved if int(row["episode_id"]) == episode_id]
        pl.DataFrame(
            [
                {
                    **{
                        name: row.get(name)
                        for name in FACT_TABLES["episodes"].polars_schema().names()
                    },
                    # A written fact_episodes row is evaluated: the empty reason marks it
                    # valid for ranking (L3-002), never NULL.
                    "invalid_reason": row.get("invalid_reason") or "",
                }
                for row in episode_rows
            ],
            schema=FACT_TABLES["episodes"].polars_schema(),
        ).write_parquet(metrics / "fact_episodes.parquet")
        robots_schema = FACT_TABLES["robots"].polars_schema()
        robot_rows = [
            {name: row.get(name) for name in robots_schema.names()}
            for row in episode_rows
        ]
        pl.DataFrame(robot_rows, schema=robots_schema).write_parquet(
            metrics / "fact_robots.parquet"
        )
        encounters_schema = FACT_TABLES["encounters"].polars_schema()
        encounter_rows = [
            {
                **{name: None for name in encounters_schema.names()},
                "run_id": row["run_id"],
                "episode_id": episode_id,
                "robot_id": row["robot_id"],
                "env_id": row["env_id"],
                "encounter_id": 0,
                "actor_1": row["robot_id"],
                "actor_2": "ped_0",
                "actor_2_kind": "pedestrian",
                "min_clearance_m": 0.5,
                "duration_s": 1.0,
            }
            for row in episode_rows
        ]
        pl.DataFrame(encounter_rows, schema=encounters_schema).write_parquet(
            metrics / "fact_encounters.parquet"
        )
        tracks_schema = RobotTracks.polars_schema()
        tracks_path = (
            benchmark_dir
            / "episodes"
            / f"episode_{episode_id:03d}"
            / "topics"
            / "robot_tracks.parquet"
        )
        tracks_path.parent.mkdir(parents=True, exist_ok=True)
        track_rows = [
            {
                **{name: None for name in tracks_schema.names()},
                "run_id": episode_rows[0]["run_id"],
                "robot_id": episode_rows[0]["robot_id"],
                "env_id": episode_rows[0]["env_id"],
                "t_ns": step * 500_000_000,
                "x": float(step) * 0.5,
                "y": 0.0,
                "z": 0.0,
                "yaw": 0.0,
                "vx": 0.5,
                "vy": 0.0,
                "vz": 0.0,
                "omega_z": 0.0,
                "speed_mps": 0.5,
                "pose_source": "truth",
            }
            for step in range(2)
        ]
        pl.DataFrame(track_rows, schema=tracks_schema).write_parquet(tracks_path)
    dim_metric_rows = [
        (
            "success",
            "robots",
            "fact_robots",
            "Boolean",
            "",
            "Episode success",
            "higher",
            "core",
            None,
        ),
        (
            "time_to_goal",
            "robots",
            "fact_robots",
            "Float64",
            "s",
            "Time to reach goal",
            "lower",
            "core",
            None,
        ),
        (
            "path_length_m",
            "episodes",
            "fact_episodes",
            "Float64",
            "m",
            "Total path length",
            "lower",
            "core",
            None,
        ),
        (
            "path_efficiency",
            "robots",
            "fact_robots",
            "Float64",
            "",
            "Path efficiency ratio",
            "higher",
            "core",
            None,
        ),
        (
            "roughness_mean",
            "robots",
            "fact_robots",
            "Float64",
            "rad/m",
            "Mean path roughness",
            "lower",
            "motion",
            None,
        ),
        (
            "energy_total_wh",
            "robots",
            "fact_robots",
            "Float64",
            "Wh",
            "Total energy consumed",
            "lower",
            "energy",
            None,
        ),
        (
            "mar_vs_planned_route",
            "encounters",
            "fact_encounters",
            "Float64",
            "",
            "Mutual accommodation ratio against the planned route",
            "higher",
            "social",
            None,
        ),
        (
            "min_clearance_m",
            "encounters",
            "fact_encounters",
            "Float64",
            "m",
            "Minimum clearance",
            "higher",
            "proxemics",
            None,
        ),
        (
            "duration_s",
            "encounters",
            "fact_encounters",
            "Float64",
            "s",
            "Encounter duration",
            "lower",
            "proxemics",
            None,
        ),
        (
            "phases_total",
            "episodes",
            "fact_episodes",
            "Int64",
            "",
            "Total phases",
            None,
            "core",
            None,
        ),
        (
            "contact_count",
            "episodes",
            "fact_episodes",
            "Int64",
            "",
            "Total contacts",
            "lower",
            "contacts",
            None,
        ),
        (
            "num_pedestrians",
            "episodes",
            "fact_episodes",
            "Int64",
            "",
            "Pedestrian count",
            None,
            "core",
            None,
        ),
    ]
    dm_schema = DIMENSION_TABLES["dim_metrics"].polars_schema()
    pl.DataFrame(
        [
            {
                "metric": m,
                "scope": s,
                "fact_table": ft,
                "dtype": dt,
                "unit": u,
                "description": desc,
                "direction": d,
                "producer": prod,
                "null_reason": nr,
            }
            for m, s, ft, dt, u, desc, d, prod, nr in dim_metric_rows
        ],
        schema=dm_schema,
    ).write_parquet(benchmark_dir / "dim_metrics.parquet")


def _iso(timestamp: float) -> str:
    return datetime.datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")


def build_run(
    root: pathlib.Path,
    folder: str,
    *,
    manifest_run_id: str | None = None,
    created_at: str = "2026-10-19T12:00:00+00:00",
    steps: tuple[str, ...] = STEPS,
) -> pathlib.Path:
    """One run folder with a manifest and its empty state and progress writers."""
    from arena_evaluation.benchmark.state import RunDir

    run = RunDir.create(
        root, folder, manifest(manifest_run_id or folder, steps, created_at=created_at)
    )
    run.progress.close()
    return run.path


def build_runs(root: pathlib.Path) -> pathlib.Path:
    """A benchmark run written by the package's own writers: manifest, state and progress plus a second lane, an episode, logs and the Star Schema."""
    from arena_evaluation.benchmark.state import RunDir
    from arena_evaluation.benchmark.step import StepErrorKind, StepResult
    from arena_evaluation.storage.manifest import MetadataWriter
    from arena_evaluation.storage.schemas import RunMetadata

    root.mkdir(parents=True, exist_ok=True)
    now = time.time()
    manifest_record = manifest(RUN_ID, STEPS)
    run = RunDir.create(root, RUN_ID, manifest_record)
    run.state.write(
        {
            "dwb/s1": StepResult(
                "dwb/s1", "ok", 0, now - 60, now - 30, None, None, episodes_run=2
            ),
            "unhindered_peds/s1": StepResult(
                "unhindered_peds/s1", "in_progress", 0, now - 10, None, None, None
            ),
        }
    )
    for episode in range(2):
        run.progress.append(
            ts_iso=_iso(now - 55 + episode),
            run_id=RUN_ID,
            step_key="dwb/s1",
            contestant="dwb",
            stage="s1",
            env_id=0,
            episode_id=episode,
            episode_record=_record("map_empty", 2),
            started_at=now - 60,
            ended_at=now - 45,
        )
    run.progress.close()
    lane = RunDir.create(root, RUN_ID, manifest_record, lane="p2")
    lane.state.write(
        {
            "dwb_social/s1": StepResult(
                "dwb_social/s1",
                "failed",
                1,
                now - 50,
                now - 20,
                StepErrorKind.SIM_DEAD,
                "gz died",
            )
        }
    )
    lane.progress.append(
        ts_iso=_iso(now - 40),
        run_id=RUN_ID,
        step_key="dwb_social/s1",
        contestant="dwb_social",
        stage="s1",
        env_id=1,
        episode_id=2,
        episode_record=_record("map_empty", 3),
        started_at=now - 50,
        ended_at=now - 20,
    )
    lane.progress.close()
    (run.path / "runner.log").write_text(
        "".join(f"runner line {index}\n" for index in range(300))
    )
    (run.path / "runner.p2.log").write_text("lane p2 started\nlane p2 sim died\n")
    episode_dir = run.path / "episodes" / "episode_000"
    episode_dir.mkdir(parents=True, exist_ok=True)
    MetadataWriter.write(
        RunMetadata(
            benchmark_id=RUN_ID,
            planner="dwb",
            map="map_empty",
            stage="s1",
            recording_started_at="2026-10-19T12:00:00",
            python_version="3.12",
            ros_distro="jazzy",
            modes={},
            params={},
        ),
        episode_dir / "episode_000.yaml",
    )
    write_star_schema(run.path, metric_rows())
    empty = RunDir.create(root, EMPTY_RUN_ID, manifest(EMPTY_RUN_ID, ("dwb/s1",)))
    empty.progress.close()
    return run.path


@contextlib.contextmanager
def arena_data(tmp_path: pathlib.Path) -> object:
    """ARENA_DATA_DIR pointing at tmp_path/data for direct tool calls, restored afterwards."""
    previous = os.environ.get("ARENA_DATA_DIR")
    os.environ["ARENA_DATA_DIR"] = str(tmp_path / "data")
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("ARENA_DATA_DIR", None)
        else:
            os.environ["ARENA_DATA_DIR"] = previous


def build_query_run(
    root: pathlib.Path, run_id: str = "query-run", rows: int = 10_000
) -> pathlib.Path:
    """A run directory whose dim_episodes has rows episodes, the table a capped SELECT reads."""
    from arena_evaluation.storage.registry.dimensions import DIMENSION_TABLES

    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "run_id": run_id,
            "episode_id": episode,
            "env_id": "env_0",
            "robot_id": "robot_0",
            "benchmark_id": run_id,
            "suite_name": "basic",
            "contest_name": "basic",
            "stage": "s1",
            "world": "map_empty",
            "scenario": "",
            "planner": "dwb",
            "inter_planner": "",
            "dispatcher": "",
            "crowd_condition": "visible",
            "ped_policy_default": "sfm",
            "crowd_size": 0,
            "seed": episode,
            "is_reference": False,
            "reference_type": "",
            "duration_s": 1.0,
            "outcome_state": 2,
            "outcome_info": "",
        }
        for episode in range(rows)
    ]
    pl.DataFrame(
        records, schema=DIMENSION_TABLES["dim_episodes"].polars_schema()
    ).write_parquet(run_dir / "dim_episodes.parquet")
    return run_dir


def server_env(tmp_path: pathlib.Path, **extra: str) -> dict[str, str]:
    return {**os.environ, "ARENA_DATA_DIR": str(tmp_path / "data"), **extra}


async def session(
    body: object, env: dict[str, str], domains: tuple[str, ...] = ("evaluation", "jobs")
) -> object:
    from mcp import ClientSession, StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "arena_mcp", "serve", *domains], env=env
    )
    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as client,
    ):
        await client.initialize()
        return await body(client)


def run_session(
    body: object, env: dict[str, str], domains: tuple[str, ...] = ("evaluation", "jobs")
) -> object:
    return asyncio.run(session(body, env, domains))


def payload(result: object) -> dict:
    """The JSON object a tool returned."""
    assert not result.is_error, result.content[0].text if result.content else result
    assert len(result.content) == 1, result.content
    return json.loads(result.content[0].text)


def error_text(result: object) -> str:
    assert result.is_error, result
    return result.content[0].text
