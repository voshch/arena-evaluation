"""Tests for the benchmark state layer: Manifest, StateFile, ProgressLog, RunDir, and resumability."""

from __future__ import annotations

import csv
import json
import logging
import pathlib
import subprocess
import types

from arena_runtime.env.task_layer import encode_modes
from rcl_interfaces.msg import Parameter as MsgParameter
from rcl_interfaces.msg import ParameterType, ParameterValue

from arena_evaluation.benchmark.state import (
    Manifest,
    ProgressLog,
    RunDir,
    _params_to_json,
    find_most_recent_resumable,
)
from arena_evaluation.benchmark.step import StepResult


# helpers


def _make_manifest(**overrides: object) -> Manifest:
    base: dict[str, object] = {
        "run_id": "20260623-202509-basic-basic",
        "created_at": "2026-06-23T20:25:09+00:00",
        "arena_git_sha": "abc123def",
        "arena_git_dirty": True,
        "cli_args": ["--suite", "basic", "--contest", "basic"],
        "env_n": 2,
        "headless": False,
        "config_hash": "0123456789abcdef0123456789abcdef01234567",
        "simulator": "gazebo",
        "scale_episodes": 1.0,
        "suite_name": "basic",
        "contest_name": "basic",
        "suite": {"stages": [{"name": "s1", "episodes": 3}]},
        "contest": [{"name": "c1", "mobile.local_planner": "dwa"}],
        "steps": [{"key": "c1/s1", "episodes_planned": 3}],
    }
    base.update(overrides)
    return Manifest(**base)  # type: ignore[arg-type]


def _make_episode_record(*, episode_id: int = 1) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        episode_id=episode_id,
        world="map1",
        seed=42,
        modes=encode_modes(
            {"robots": ["random"], "obstacles": ["random"], "modules": ["benchmark"]}
        ),
        robots=["turtlebot3_burger"],
        outcome_state=1,
        outcome_info="",
        params=[],
        goal_dist_start=0.0,
        goal_dist_min=0.0,
        path_length=0.0,
        integrity=[],
    )


def _ok_step(key: str, **overrides: object) -> StepResult:
    fields: dict[str, object] = dict(
        key=key,
        status="ok",
        env_id=1,
        started_at=1.0,
        ended_at=2.0,
        error_kind=None,
        error_detail=None,
    )
    fields.update(overrides)
    return StepResult(**fields)  # type: ignore[arg-type]


# compute_config_hash


# find_most_recent_resumable


def _make_resumable_run(
    root: pathlib.Path,
    name: str,
    *,
    state_steps: object | None = None,
    manifest_text: str | None = None,
    with_state: bool = True,
) -> pathlib.Path:
    """Create a run dir that is structurally resumable (manifest + state)."""
    run_path = root / name
    run_path.mkdir(parents=True)
    (run_path / "manifest.yaml").write_text(
        manifest_text
        if manifest_text is not None
        else _make_manifest(run_id=name).to_yaml()
    )
    if with_state:
        (run_path / ".benchmark_state.json").write_text(
            json.dumps({"steps": state_steps or {}})
        )
    return run_path


def test_find_resumable_missing_data_root(tmp_path: pathlib.Path):
    assert find_most_recent_resumable(tmp_path / "nope") is None


def test_find_resumable_skips_non_dirs(tmp_path: pathlib.Path):
    (tmp_path / "file").write_text("x")
    assert find_most_recent_resumable(tmp_path) is None


def test_find_resumable_skips_missing_manifest(tmp_path: pathlib.Path):
    (tmp_path / "r1").mkdir()
    assert find_most_recent_resumable(tmp_path) is None


def test_find_resumable_skips_corrupt_manifest(tmp_path: pathlib.Path):
    _make_resumable_run(
        tmp_path, "r1", manifest_text="not: [valid yaml\n", with_state=False
    )
    _make_resumable_run(
        tmp_path, "r2", manifest_text="run_id: only_one_field\n", with_state=False
    )
    assert find_most_recent_resumable(tmp_path) is None


def test_find_resumable_skips_all_ok_run(tmp_path: pathlib.Path):
    _make_resumable_run(tmp_path, "r1", state_steps={"a/s": {"status": "ok"}})
    assert find_most_recent_resumable(tmp_path) is None


def test_find_resumable_empty_steps_is_candidate(tmp_path: pathlib.Path):
    _make_resumable_run(tmp_path, "r1", state_steps={})
    assert find_most_recent_resumable(tmp_path) == "r1"


def test_find_resumable_corrupt_state_is_candidate(tmp_path: pathlib.Path):
    run_path = tmp_path / "r1"
    run_path.mkdir()
    (run_path / "manifest.yaml").write_text(_make_manifest(run_id="r1").to_yaml())
    (run_path / ".benchmark_state.json").write_text("{not json")
    assert find_most_recent_resumable(tmp_path) == "r1"


# capture_git_sha


def _init_git_repo(repo: pathlib.Path) -> str:
    subprocess.run(
        ["git", "init", str(repo)], check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test"],
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = repo / "file.txt"
    tracked.write_text("v1\n")
    subprocess.run(
        ["git", "-C", str(repo), "add", "file.txt"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "init"],
        check=True,
        capture_output=True,
        text=True,
    )
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


# Manifest round trip


# StateFile


# _params_to_json


def _msg_param(name: str, ptype: int, value) -> MsgParameter:
    p = MsgParameter()
    p.name = name
    pv = ParameterValue()
    pv.type = ptype
    if ptype == ParameterType.PARAMETER_INTEGER:
        pv.integer_value = value
    elif ptype == ParameterType.PARAMETER_INTEGER_ARRAY:
        pv.integer_array_value = value
    elif ptype == ParameterType.PARAMETER_STRING:
        pv.string_value = value
    p.value = pv
    return p


def test_params_to_json_array_tolist():
    """numpy array values are converted with .tolist() before serialization."""
    out = _params_to_json(
        [_msg_param("a_arr", ParameterType.PARAMETER_INTEGER_ARRAY, [1, 2, 3])]
    )
    assert json.loads(out) == [{"name": "a_arr", "value": [1, 2, 3]}]


def test_params_to_json_fallback_on_bad_msg():
    """If Parameter.from_parameter_msg raises, the raw .value is stringified."""
    bad = types.SimpleNamespace(name="x", value="raw")
    assert json.loads(_params_to_json([bad])) == [{"name": "x", "value": "raw"}]


# ProgressLog

_HEADERS = [
    "ts_iso",
    "run_id",
    "step_key",
    "contestant",
    "stage",
    "env_id",
    "episode_id",
    "parent_episode_id",
    "is_reference",
    "reference_type",
    "world",
    "seed",
    "modes",
    "robots",
    "outcome_state",
    "outcome_info",
    "started_at",
    "ended_at",
    "runtime_s",
    "params_json",
    "error_kind",
    "error_detail",
    "lockstep_stalls",
    "lockstep_max_stall_s",
    "lockstep_rtf",
    "lockstep_beat_channels",
    "lockstep_ticks",
    "lockstep_beats",
    "lockstep_beats_due",
    "goal_dist_start",
    "goal_dist_min",
    "path_length",
    "integrity",
]


def test_progress_log_append_with_parameter_msg_params(tmp_path: pathlib.Path):
    """Integration: rcl_interfaces Parameter msgs flow through _params_to_json."""
    log = ProgressLog(tmp_path / "progress.csv")
    rec = types.SimpleNamespace(
        episode_id=1,
        world="w",
        seed=1,
        modes=encode_modes({"robots": ["r"], "obstacles": ["o"], "modules": ["m"]}),
        robots=["bot"],
        outcome_state=1,
        outcome_info="",
        params=[_msg_param("task.r.speed", ParameterType.PARAMETER_INTEGER, 3)],
        goal_dist_start=0.0,
        goal_dist_min=0.0,
        path_length=0.0,
        integrity=[],
    )
    log.append(
        ts_iso="t",
        run_id="r",
        step_key="k",
        contestant="c",
        stage="s",
        env_id=0,
        episode_id=1,
        episode_record=rec,
        started_at=0.0,
        ended_at=1.0,
    )
    log.close()
    with (tmp_path / "progress.csv").open(newline="") as fh:
        row = next(csv.DictReader(fh))
    assert json.loads(row["params_json"]) == [{"name": "task.r.speed", "value": 3}]
    assert json.loads(row["modes"]) == {
        "modules": ["m"],
        "obstacles": ["o"],
        "robots": ["r"],
    }


_OLD_COLUMNS = [
    "ts_iso", "run_id", "step_key", "contestant", "stage", "env_id", "episode_id", "world", "seed",
    "tm_robots", "tm_obstacles", "tm_modules", "robots", "robots_params_json", "obstacles_params_json",
]  # fmt: skip


def _write_log_lines(path: pathlib.Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def test_dedupe_comment_only_noop(tmp_path: pathlib.Path):
    path = tmp_path / "progress.csv"
    path.write_text("# only a comment\n")
    log = ProgressLog(path)
    log.dedupe_in_place({})
    log.close()
    assert path.read_text() == "# only a comment\n"


def test_dedupe_keeps_latest_ts_and_sorts(tmp_path: pathlib.Path):
    path = tmp_path / "progress.csv"
    _write_log_lines(
        path,
        [
            ",".join(_HEADERS),
            "2026-01-01T00:00:03+00:00,r,k,c,s,,3,,false,,w,1,o,o,m,b,1,,1,2,1,[],[],,",
            "2026-01-01T00:00:01+00:00,r,k,c,s,,1,,false,,w,1,o,o,m,b,1,,1,2,1,[],[],,",
            "2026-01-01T00:00:02+00:00,r,k,c,s,,2,,false,,w,1,o,o,m,b,1,,1,2,1,[],[],,",
            "2026-01-01T00:00:00+00:00,r,k,c,s,,2,,false,,w,1,o,o,m,b,1,,1,2,1,[],[],,",
            "2026-01-01T00:00:05+00:00,r,k,c,s,,1,,false,,w,1,o,o,m,b,1,,1,2,1,[],[],,",
        ],
    )
    log = ProgressLog(path)
    log.dedupe_in_place({})
    log.close()
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    # (k,1) keeps the 05 row; the stale @00 duplicate of (k,2) is dropped;
    # remaining rows sorted by ts_iso.
    assert [r["episode_id"] for r in rows] == ["2", "3", "1"]
    assert rows[0]["ts_iso"] == "2026-01-01T00:00:02+00:00"
    assert rows[-1]["ts_iso"] == "2026-01-01T00:00:05+00:00"


# RunDir


def test_run_dir_attach_log_handler(tmp_path: pathlib.Path):
    run_dir = RunDir.create(tmp_path, "r1", _make_manifest(run_id="r1"))
    logger = logging.getLogger("arena_evaluation.tests.run_dir")
    logger.setLevel(logging.INFO)
    try:
        run_dir.attach_log_handler(logger)
        logger.info("hello from runner")
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()
        text = (run_dir.path / "runner.log").read_text()
        assert "hello from runner" in text
        assert "INFO" in text
    finally:
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
                handler.close()
