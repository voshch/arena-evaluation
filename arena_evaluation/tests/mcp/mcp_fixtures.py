from __future__ import annotations

import asyncio
import json
import os
import pathlib
import shutil
import sys
import time

import pytest

sourced = pytest.mark.skipif(not os.environ.get("ARENA_WS_DIR"), reason="needs a sourced Arena workspace")

RUN_ID = "20261019-120000-basic-basic"
EMPTY_RUN_ID = "20261019-110000-empty-empty"
STEPS = ("dwb/s1", "dwb_social/s1", "unhindered_peds/s1", "dwb/s2")


def _record(world: str, outcome: int) -> object:
    from task_generator_msgs.msg import EpisodeRecord

    return EpisodeRecord(world=world, seed=7, tm_robots="random", tm_obstacles="random", robots=["jackal"], outcome_state=outcome, outcome_info="")


def _manifest(run_id: str, steps: tuple[str, ...]) -> object:
    from arena_evaluation.benchmark.state import Manifest

    return Manifest(
        run_id=run_id,
        created_at="2026-10-19T12:00:00+00:00",
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
    rows = []
    for planner, local, success, ttg, collisions in (("dwb", "dwb", [True, True, True, False], [40.0, 42.0, 44.0, 90.0], [0, 0, 1, 2]), ("dwb_social", "dwb", [True, False, False, False], [60.0, 90.0, 90.0, 90.0], [1, 2, 3, 3])):
        for index in range(4):
            rows.append(
                {
                    "episode": len(rows),
                    "planner": planner,
                    "local_planner": local,
                    "inter_planner": "social" if planner == "dwb_social" else "plain",
                    "stage": "s1" if index < 2 else "s2",
                    "map": "map_empty",
                    "is_reference": False,
                    "success": success[index],
                    "time_to_goal": ttg[index],
                    "collision_amount": collisions[index],
                    "judge_agrees": success[index],
                    "phases_total": 1,
                    "path_speeds": [0.5, 0.6 + index * 0.1],
                }
            )
    for index in range(2):
        rows.append({"episode": len(rows), "planner": "unhindered_peds", "local_planner": "unhindered_peds", "inter_planner": "", "stage": "s1", "map": "map_empty", "is_reference": True, "success": None, "time_to_goal": None, "collision_amount": None, "judge_agrees": None, "phases_total": None, "path_speeds": []})
    return rows


def build_runs(root: pathlib.Path) -> pathlib.Path:
    """A benchmark run written by the package's own writers: manifest, state and progress plus a second lane, an episode, logs and combined metrics."""
    from arena_evaluation.benchmark.state import RunDir
    from arena_evaluation.benchmark.step import StepErrorKind, StepResult
    from arena_evaluation.processing.parquet_store import ParquetStore
    from arena_evaluation.storage.manifest import MetadataWriter
    from arena_evaluation.storage.schemas import RunMetadata

    root.mkdir(parents=True, exist_ok=True)
    now = time.time()
    manifest = _manifest(RUN_ID, STEPS)
    run = RunDir.create(root, RUN_ID, manifest)
    run.state.write({"dwb/s1": StepResult("dwb/s1", "ok", 0, now - 60, now - 30, None, None, episodes_run=2), "unhindered_peds/s1": StepResult("unhindered_peds/s1", "in_progress", 0, now - 10, None, None, None)})
    for episode in range(2):
        run.progress.append(ts_iso=f"2026-10-19T12:0{episode}:00", run_id=RUN_ID, step_key="dwb/s1", contestant="dwb", stage="s1", env_id=0, episode_id=episode, episode_record=_record("map_empty", 2), started_at=now - 60, ended_at=now - 45)
    run.progress.close()
    lane = RunDir.create(root, RUN_ID, manifest, lane="p2")
    lane.state.write({"dwb_social/s1": StepResult("dwb_social/s1", "failed", 1, now - 50, now - 20, StepErrorKind.SIM_DEAD, "gz died")})
    lane.progress.append(ts_iso="2026-10-19T12:05:00", run_id=RUN_ID, step_key="dwb_social/s1", contestant="dwb_social", stage="s1", env_id=1, episode_id=2, episode_record=_record("map_empty", 3), started_at=now - 50, ended_at=now - 20)
    lane.progress.close()
    (run.path / "runner.log").write_text("".join(f"runner line {index}\n" for index in range(300)))
    (run.path / "runner.p2.log").write_text("lane p2 started\nlane p2 sim died\n")
    episode_dir = run.path / "episodes" / "episode_000"
    episode_dir.mkdir(parents=True, exist_ok=True)
    MetadataWriter.write(RunMetadata(benchmark_id=RUN_ID, planner="dwb", map="map_empty", stage="s1", recording_started_at="2026-10-19T12:00:00", python_version="3.12", ros_distro="jazzy"), episode_dir / "episode_000.yaml")
    ParquetStore.write_rows(metric_rows(), run.path / "combined_metrics.parquet")
    empty = RunDir.create(root, EMPTY_RUN_ID, _manifest(EMPTY_RUN_ID, ("dwb/s1",)))
    empty.progress.close()
    return run.path


def build_world_root(root: pathlib.Path) -> pathlib.Path:
    """A writable world root holding a copy of the installed map_empty world, for ARENA_WORLD_PATH."""
    from arena_simulation_setup.tree.World import WorldIdentifier

    source = WorldIdentifier("map_empty").resolve_sync().path
    target = root / "map_empty"
    shutil.copytree(source, target, symlinks=False, ignore=shutil.ignore_patterns("scenarios"), ignore_dangling_symlinks=True)
    (target / "scenarios").mkdir()
    return root


def server_env(tmp_path: pathlib.Path, **extra: str) -> dict[str, str]:
    return {**os.environ, "ARENA_DATA_DIR": str(tmp_path / "data"), **extra}


async def session(body: object, env: dict[str, str], domains: tuple[str, ...] = ("evaluation", "jobs")) -> object:
    from mcp import ClientSession, StdioServerParameters, stdio_client

    params = StdioServerParameters(command=sys.executable, args=["-m", "arena_mcp", "serve", *domains], env=env)
    async with stdio_client(params) as (read, write), ClientSession(read, write) as client:
        await client.initialize()
        return await body(client)


def run_session(body: object, env: dict[str, str], domains: tuple[str, ...] = ("evaluation", "jobs")) -> object:
    return asyncio.run(session(body, env, domains))


def payload(result: object) -> dict:
    """The JSON object a tool returned."""
    assert not result.is_error, result.content[0].text if result.content else result
    assert len(result.content) == 1, result.content
    return json.loads(result.content[0].text)


def error_text(result: object) -> str:
    assert result.is_error, result
    return result.content[0].text
