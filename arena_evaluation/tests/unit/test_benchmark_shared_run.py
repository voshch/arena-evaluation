"""Shared runs of the benchmark run dir: lane files, union reads, fold, claims, episode ids, shared manifest."""

from __future__ import annotations

import csv
import json
import os
import pathlib
import subprocess
import sys
import threading
import types

import pytest

import arena_evaluation.benchmark.state as state_module
from arena_evaluation.benchmark.config import Contest, Suite
from arena_evaluation.benchmark.runner import block_claim_key, group_pending
from arena_evaluation.benchmark.state import (
    Manifest,
    ManifestMismatchError,
    ProgressLog,
    RunDir,
    StateFile,
    fold_run_dir,
)
from arena_evaluation.benchmark.step import Step, StepErrorKind, StepResult
from task_generator.constants import Constants

_PACKAGE_ROOT = pathlib.Path(state_module.__file__).resolve().parents[2]


def _make_manifest(**overrides: object) -> Manifest:
    base: dict[str, object] = {
        "run_id": "shared-run",
        "created_at": "2026-10-03T10:00:00+00:00",
        "arena_git_sha": "abc123",
        "arena_git_dirty": False,
        "cli_args": ["--suite", "basic", "--contest", "basic"],
        "env_n": 1,
        "headless": True,
        "config_hash": "0" * 40,
        "simulator": "gazebo",
        "scale_episodes": 1.0,
        "suite_name": "basic",
        "contest_name": "basic",
        "suite": {"stages": [{"name": "s1", "episodes": 3}]},
        "contest": [{"name": "c1"}],
        "steps": [{"key": "c1/s1", "episodes_planned": 3}],
    }
    base.update(overrides)
    return Manifest(**base)  # type: ignore[arg-type]


def _result(key: str, status: str, started_at: float, **overrides: object) -> StepResult:
    fields: dict[str, object] = dict(
        key=key,
        status=status,
        env_id=0,
        started_at=started_at,
        ended_at=started_at + 1.0,
        error_kind=None,
        error_detail=None,
    )
    fields.update(overrides)
    return StepResult(**fields)  # type: ignore[arg-type]


def _record(info: str = "") -> types.SimpleNamespace:
    return types.SimpleNamespace(
        world="map1",
        seed=1,
        tm_robots="random",
        tm_obstacles="random",
        tm_modules=[],
        robots=["jackal"],
        outcome_state=2,
        outcome_info=info,
        robots_params=[],
        obstacles_params=[],
        goal_dist_start=0.0,
        goal_dist_min=0.0,
        path_length=0.0,
    )


def _append(log: ProgressLog, ts: str, step_key: str, episode_id: int, info: str = "") -> None:
    log.append(
        ts_iso=ts,
        run_id="shared-run",
        step_key=step_key,
        contestant=step_key.split("/")[0],
        stage=step_key.split("/")[1],
        env_id=0,
        episode_id=episode_id,
        episode_record=_record(info),
        started_at=0.0,
        ended_at=1.0,
    )


def _rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _lanes(tmp_path: pathlib.Path, *lanes: str) -> list[RunDir]:
    return [RunDir.create(tmp_path, "shared-run", _make_manifest(), lane=lane) for lane in lanes]


def _make_step(contestant: str, stage: str, world: str) -> Step:
    return Step(
        contestant=Contest.Contestant(name=contestant, args={}),
        stage=Suite.Stage(
            name=stage,
            episodes=2,
            robot="jackal",
            map=world,
            tm_robots=Constants.TaskMode.TM_Robots.RANDOM,
            tm_obstacles=Constants.TaskMode.TM_Obstacles.RANDOM,
            config={},
            seed=0,
            timeout=60.0,
        ),
        episodes=2,
    )


def test_lane_state_writes_land_in_lane_file_only(tmp_path: pathlib.Path):
    p1, p2 = _lanes(tmp_path, "p1", "p2")
    p1.state.write({"a/s": _result("a/s", "ok", 1.0)})
    p2.state.write({"b/s": _result("b/s", "failed", 2.0, error_kind=StepErrorKind.ENV_SETUP)})

    run = tmp_path / "shared-run"
    lane1 = json.loads((run / ".benchmark_state.p1.json").read_text())["steps"]
    lane2 = json.loads((run / ".benchmark_state.p2.json").read_text())["steps"]
    assert set(lane1) == {"a/s"}
    assert set(lane2) == {"b/s"}
    assert json.loads((run / ".benchmark_state.json").read_text())["steps"] == {}


def test_union_prefers_ok_then_latest_started_at(tmp_path: pathlib.Path):
    p1, p2 = _lanes(tmp_path, "p1", "p2")
    StateFile(tmp_path / "shared-run", {}).write({"a/s": _result("a/s", "failed", 5.0)})
    p1.state.write({"a/s": _result("a/s", "ok", 1.0), "b/s": _result("b/s", "failed", 1.0), "c/s": _result("c/s", "ok", 2.0)})
    p2.state.write({"b/s": _result("b/s", "partial", 3.0), "c/s": _result("c/s", "ok", 4.0, env_id=7)})

    steps = StateFile.open(tmp_path / "shared-run", "p1").steps
    assert steps["a/s"].status == "ok"
    assert steps["b/s"].status == "partial"
    assert steps["c/s"].env_id == 7


def test_state_open_without_lane_sees_union(tmp_path: pathlib.Path):
    p1, p2 = _lanes(tmp_path, "p1", "p2")
    p1.state.write({"a/s": _result("a/s", "ok", 1.0)})
    p2.state.write({"b/s": _result("b/s", "in_progress", 2.0)})

    steps = StateFile.open(tmp_path / "shared-run").steps
    assert {k: v.status for k, v in steps.items()} == {"a/s": "ok", "b/s": "in_progress"}
    reopened = RunDir.open(tmp_path, "shared-run")
    assert set(reopened.state.steps) == {"a/s", "b/s"}
    assert reopened.state.filename == ".benchmark_state.json"


def test_fold_merges_state_and_dedupes_progress(tmp_path: pathlib.Path):
    p1, p2 = _lanes(tmp_path, "p1", "p2")
    run = tmp_path / "shared-run"
    p1.state.write({"a/s": _result("a/s", "ok", 1.0)})
    p2.state.write({"b/s": _result("b/s", "partial", 2.0)})
    p1.progress.write_comment("resumed at 2026-10-03T10:00:00+00:00")
    _append(p1.progress, "2026-10-03T10:00:03+00:00", "a/s", 0, "p1-old")
    _append(p2.progress, "2026-10-03T10:00:05+00:00", "a/s", 0, "p2-new")
    _append(p2.progress, "2026-10-03T10:00:01+00:00", "b/s", 1)
    _append(p1.progress, "2026-10-03T10:00:02+00:00", "a/s", 2)

    fold_run_dir(run)

    state = json.loads((run / ".benchmark_state.json").read_text())["steps"]
    assert {k: v["status"] for k, v in state.items()} == {"a/s": "ok", "b/s": "partial"}
    text = (run / "progress.csv").read_text()
    assert "#" not in text
    assert text.splitlines()[0] == ProgressLog._HEADER
    rows = _rows(run / "progress.csv")
    assert [(r["step_key"], r["episode_id"]) for r in rows] == [("b/s", "1"), ("a/s", "2"), ("a/s", "0")]
    assert rows[-1]["outcome_info"] == "p2-new"
    assert (run / "progress.p1.csv").exists()
    assert (run / ".benchmark_state.p2.json").exists()


def test_fold_twice_gives_identical_files(tmp_path: pathlib.Path):
    p1, p2 = _lanes(tmp_path, "p1", "p2")
    run = tmp_path / "shared-run"
    p1.state.write({"a/s": _result("a/s", "ok", 1.0)})
    p2.state.write({"b/s": _result("b/s", "failed", 2.0)})
    _append(p1.progress, "2026-10-03T10:00:01+00:00", "a/s", 0)
    _append(p2.progress, "2026-10-03T10:00:01+00:00", "b/s", 1)
    _append(p2.progress, "2026-10-03T10:00:02+00:00", "b/s", 2)

    fold_run_dir(run)
    first = ((run / "progress.csv").read_bytes(), (run / ".benchmark_state.json").read_bytes())
    fold_run_dir(run)
    second = ((run / "progress.csv").read_bytes(), (run / ".benchmark_state.json").read_bytes())
    assert first == second
    assert len(_rows(run / "progress.csv")) == 3


def test_fold_keeps_rows_of_a_prior_progress_csv(tmp_path: pathlib.Path):
    run = tmp_path / "shared-run"
    run.mkdir()
    (run / "manifest.yaml").write_text(_make_manifest().to_yaml())
    legacy = ProgressLog(run / "progress.csv")
    _append(legacy, "2026-10-03T09:00:00+00:00", "a/s", 0, "legacy")
    legacy.close()

    p1 = RunDir.open(tmp_path, "shared-run", lane="p1")
    _append(p1.progress, "2026-10-03T10:00:00+00:00", "a/s", 1)
    p1.fold()

    rows = _rows(run / "progress.csv")
    assert [(r["episode_id"], r["outcome_info"]) for r in rows] == [("0", "legacy"), ("1", "")]


def test_fold_drops_partial_trailing_row(tmp_path: pathlib.Path):
    (p1,) = _lanes(tmp_path, "p1")
    run = tmp_path / "shared-run"
    _append(p1.progress, "2026-10-03T10:00:01+00:00", "a/s", 0)
    with (run / "progress.p1.csv").open("a") as fh:
        fh.write("2026-10-03T10:00:02+00:00,shared-run,a/s")

    fold_run_dir(run)

    assert [r["episode_id"] for r in _rows(run / "progress.csv")] == ["0"]


def test_claim_is_exclusive_across_lanes(tmp_path: pathlib.Path):
    p1, p2 = _lanes(tmp_path, "p1", "p2")
    assert not p2.claimed("tok", "abc")
    assert p1.claim("tok", "abc")
    assert not p2.claim("tok", "abc")
    assert not p1.claim("tok", "abc")
    assert p2.claimed("tok", "abc")
    assert (tmp_path / "shared-run" / "claims" / "tok" / "abc").read_text().strip() == "p1"
    assert p2.claim("other-token", "abc")


def test_concurrent_claimants_have_one_winner(tmp_path: pathlib.Path):
    lanes = _lanes(tmp_path, *(f"p{i}" for i in range(8)))
    barrier = threading.Barrier(len(lanes))
    wins: list[str] = []
    lock = threading.Lock()

    def _try(run_dir: RunDir) -> None:
        barrier.wait()
        if run_dir.claim("tok", "block"):
            with lock:
                wins.append(run_dir.lane or "")

    threads = [threading.Thread(target=_try, args=(rd,)) for rd in lanes]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(wins) == 1
    assert (tmp_path / "shared-run" / "claims" / "tok" / "block").read_text().strip() == wins[0]


def test_episode_ids_start_after_existing_episode_dirs(tmp_path: pathlib.Path):
    p1, p2 = _lanes(tmp_path, "p1", "p2")
    episodes = tmp_path / "shared-run" / "episodes"
    (episodes / "episode_4").mkdir(parents=True)
    assert p1.reserve_episode_ids(3) == 5
    assert p2.reserve_episode_ids(2) == 8
    assert p1.reserve_episode_ids(1) == 10
    assert (episodes / ".next_episode_id").read_text().strip() == "11"


def test_episode_ids_start_at_zero_in_fresh_run(tmp_path: pathlib.Path):
    (p1,) = _lanes(tmp_path, "p1")
    assert p1.reserve_episode_ids(4) == 0
    assert p1.reserve_episode_ids(1) == 4


def test_episode_id_ranges_from_concurrent_processes_are_disjoint(tmp_path: pathlib.Path):
    _lanes(tmp_path, "p0")
    script = "import pathlib, sys\nfrom arena_evaluation.benchmark.state import RunDir\nrd = RunDir.open(pathlib.Path(sys.argv[1]), 'shared-run', lane=sys.argv[2])\nprint(' '.join(str(rd.reserve_episode_ids(3)) for _ in range(20)))\n"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(_PACKAGE_ROOT), os.environ.get("PYTHONPATH")]))}
    procs = [subprocess.Popen([sys.executable, "-c", script, str(tmp_path), f"p{i}"], stdout=subprocess.PIPE, text=True, env=env) for i in range(1, 5)]
    starts: list[int] = []
    for proc in procs:
        out, _ = proc.communicate(timeout=120)
        assert proc.returncode == 0
        starts.extend(int(x) for x in out.split())
    assert sorted(starts) == list(range(0, 3 * len(starts), 3))


def test_shared_create_joins_existing_manifest(tmp_path: pathlib.Path):
    first = RunDir.create(tmp_path, "shared-run", _make_manifest(created_at="first"), lane="p1")
    second = RunDir.create(tmp_path, "shared-run", _make_manifest(created_at="second"), lane="p2")
    assert first.manifest.created_at == "first"
    assert second.manifest.created_at == "first"
    assert Manifest.from_yaml((tmp_path / "shared-run" / "manifest.yaml").read_text()).created_at == "first"
    assert not list((tmp_path / "shared-run").glob(".manifest.*.tmp"))


def test_shared_create_rejects_config_hash_mismatch(tmp_path: pathlib.Path):
    RunDir.create(tmp_path, "shared-run", _make_manifest(), lane="p1")
    with pytest.raises(ManifestMismatchError, match="config_hash"):
        RunDir.create(tmp_path, "shared-run", _make_manifest(config_hash="1" * 40), lane="p2")


def test_create_without_lane_still_refuses_existing_dir(tmp_path: pathlib.Path):
    RunDir.create(tmp_path, "shared-run", _make_manifest(), lane="p1")
    with pytest.raises(FileExistsError):
        RunDir.create(tmp_path, "shared-run", _make_manifest())


def test_lane_file_names(tmp_path: pathlib.Path):
    (p1,) = _lanes(tmp_path, "p1")
    run = tmp_path / "shared-run"
    assert p1.log_path == run / "runner.p1.log"
    assert p1.pid_path == run / "runner.p1.pid"
    assert (run / "progress.p1.csv").exists()
    plain = RunDir.create(tmp_path, "plain-run", _make_manifest())
    assert plain.log_path == tmp_path / "plain-run" / "runner.log"
    assert not list((tmp_path / "plain-run").glob("progress.*.csv"))


def test_block_claim_key_is_stable_and_distinct_per_block():
    steps = [_make_step(c, s, w) for c in ("dwb", "teb") for s, w in (("s1", "map1"), ("s2", "map1"), ("s3", "map2"))]
    world_steps = [s for s in steps if s.stage.map == "map1"]
    keys_a = [block_claim_key(b, "map1", "gazebo") for b in group_pending(world_steps, "gazebo")]
    keys_b = [block_claim_key(b, "map1", "gazebo") for b in group_pending(list(world_steps), "gazebo")]
    assert keys_a == keys_b
    assert len(set(keys_a)) == len(keys_a) == 2
    late = [block_claim_key(b, "map1", "gazebo") for b in group_pending(world_steps[1:], "gazebo")]
    assert set(late) == set(keys_a)
    assert all(len(k) == 40 for k in keys_a)
    other_world = [block_claim_key(b, "map2", "gazebo") for b in group_pending([s for s in steps if s.stage.map == "map2"], "gazebo")]
    assert not set(other_world) & set(keys_a)


def _cli(args: list[str], lane: str | None) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "ARENA_LANE"}
    if lane is not None:
        env["ARENA_LANE"] = lane
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_PACKAGE_ROOT), os.environ.get("PYTHONPATH")]))
    script = "import sys\nfrom arena_evaluation.benchmark.runner import cli_main\nraise SystemExit(cli_main(sys.argv[1:]))\n"
    return subprocess.run([sys.executable, "-c", script, *args], capture_output=True, text=True, env=env, timeout=120)


@pytest.mark.parametrize(
    ("args", "lane", "needle"),
    [
        (["--shared", "tok", "--run-id", "r1"], None, "ARENA_LANE"),
        (["--shared", "tok", "--resume"], "p1", "bare --resume"),
        (["--shared", "tok"], "p1", "--run-id"),
        (["--shared", "../x", "--run-id", "r1"], "p1", "plain file name"),
    ],
)
def test_shared_flag_rejects_incomplete_arguments(tmp_path: pathlib.Path, args: list[str], lane: str | None, needle: str):
    proc = _cli([*args, "--data-root", str(tmp_path)], lane)
    assert proc.returncode == 2
    assert needle in proc.stderr
    assert not list(tmp_path.iterdir())
