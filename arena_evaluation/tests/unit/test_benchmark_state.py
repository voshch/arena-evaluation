"""Tests for the benchmark state layer: Manifest, StateFile, ProgressLog, RunDir,
config hashing, git SHA capture, and resumability scanning.

Covers :mod:`arena_evaluation.benchmark.state` (importable without a ROS graph;
only ``rclpy.parameter`` is imported for parameter serialization).
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
import pathlib
import subprocess
import types

import pytest
from rcl_interfaces.msg import Parameter as MsgParameter
from rcl_interfaces.msg import ParameterType, ParameterValue

from arena_evaluation.benchmark.state import (
    Manifest,
    ProgressLog,
    RunDir,
    StateFile,
    _params_to_json,
    capture_git_sha,
    compute_config_hash,
    find_most_recent_resumable,
)
from arena_evaluation.benchmark.step import StepErrorKind, StepResult


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


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
        tm_robots="random",
        tm_obstacles="random",
        tm_modules=["benchmark"],
        robots=["turtlebot3_burger"],
        outcome_state=1,
        outcome_info="",
        robots_params=[],
        obstacles_params=[],
        goal_dist_start=0.0,
        goal_dist_min=0.0,
        path_length=0.0,
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


# ---------------------------------------------------------------------------
# compute_config_hash
# ---------------------------------------------------------------------------


def test_compute_config_hash_deterministic():
    suite = {"stages": [{"name": "s1", "map": "m1"}]}
    contest = [{"name": "c1", "mobile.local_planner": "dwa"}]
    assert compute_config_hash(suite, contest) == compute_config_hash(suite, contest)


def test_compute_config_hash_key_order_insensitive():
    """sort_keys=True makes dict insertion order irrelevant."""
    h1 = compute_config_hash({"a": 1, "b": [1, 2]}, [{"x": 1}])
    h2 = compute_config_hash({"b": [1, 2], "a": 1}, [{"x": 1}])
    assert h1 == h2


def test_compute_config_hash_default_str_for_unserializable():
    """default=str lets non-JSON values (Path, datetime) hash instead of raising."""
    suite = {"path": pathlib.Path("/tmp/x")}
    contest = [{"when": datetime.datetime(2026, 1, 1)}]
    h = compute_config_hash(suite, contest)
    assert isinstance(h, str)
    assert len(h) == 40  # sha1 hexdigest


def test_compute_config_hash_differs_on_input():
    h1 = compute_config_hash({"a": 1}, [])
    h2 = compute_config_hash({"a": 2}, [])
    assert h1 != h2


# ---------------------------------------------------------------------------
# find_most_recent_resumable
# ---------------------------------------------------------------------------


def _make_resumable_run(root: pathlib.Path, name: str, *, state_steps: object | None = None, manifest_text: str | None = None, with_state: bool = True) -> pathlib.Path:
    """Create a run dir that is structurally resumable (manifest + state)."""
    run_path = root / name
    run_path.mkdir(parents=True)
    (run_path / "manifest.yaml").write_text(manifest_text if manifest_text is not None else _make_manifest(run_id=name).to_yaml())
    if with_state:
        (run_path / ".benchmark_state.json").write_text(json.dumps({"steps": state_steps or {}}))
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
    _make_resumable_run(tmp_path, "r1", manifest_text="not: [valid yaml\n", with_state=False)
    _make_resumable_run(tmp_path, "r2", manifest_text="run_id: only_one_field\n", with_state=False)
    assert find_most_recent_resumable(tmp_path) is None


def test_find_resumable_skips_all_ok_run(tmp_path: pathlib.Path):
    _make_resumable_run(tmp_path, "r1", state_steps={"a/s": {"status": "ok"}})
    assert find_most_recent_resumable(tmp_path) is None


def test_find_resumable_mixed_status_run_is_candidate(tmp_path: pathlib.Path):
    steps = {"a/s1": {"status": "ok"}, "a/s2": {"status": "failed", "error_kind": "internal"}}
    _make_resumable_run(tmp_path, "r1", state_steps=steps)
    assert find_most_recent_resumable(tmp_path) == "r1"


def test_find_resumable_empty_steps_is_candidate(tmp_path: pathlib.Path):
    _make_resumable_run(tmp_path, "r1", state_steps={})
    assert find_most_recent_resumable(tmp_path) == "r1"


def test_find_resumable_corrupt_state_is_candidate(tmp_path: pathlib.Path):
    run_path = tmp_path / "r1"
    run_path.mkdir()
    (run_path / "manifest.yaml").write_text(_make_manifest(run_id="r1").to_yaml())
    (run_path / ".benchmark_state.json").write_text("{not json")
    assert find_most_recent_resumable(tmp_path) == "r1"


def test_find_resumable_no_state_file_is_candidate(tmp_path: pathlib.Path):
    _make_resumable_run(tmp_path, "r1", with_state=False)
    assert find_most_recent_resumable(tmp_path) == "r1"


def test_find_resumable_returns_most_recent(tmp_path: pathlib.Path):
    _make_resumable_run(tmp_path, "20260623-100000-s-c", state_steps={"a/s": {"status": "failed"}})
    _make_resumable_run(tmp_path, "20260624-090000-s-c", state_steps={"a/s": {"status": "partial"}})
    assert find_most_recent_resumable(tmp_path) == "20260624-090000-s-c"


# ---------------------------------------------------------------------------
# capture_git_sha
# ---------------------------------------------------------------------------


def _install_fake_git(monkeypatch, results: list[subprocess.CompletedProcess]) -> list[list[str]]:
    calls: list[list[str]] = []

    def _run(cmd, capture_output, text, timeout):  # noqa: ARG001
        calls.append(list(cmd))
        return results.pop(0)

    monkeypatch.setattr("arena_evaluation.benchmark.state.subprocess.run", _run)
    return calls


def test_capture_git_sha_clean(monkeypatch):
    results = [
        subprocess.CompletedProcess([], 0, stdout="abc123\n", stderr=""),
        subprocess.CompletedProcess([], 0, stdout="", stderr=""),
    ]
    calls = _install_fake_git(monkeypatch, results)
    sha, dirty = capture_git_sha(pathlib.Path("/ws"))
    assert sha == "abc123"
    assert dirty is False
    assert calls[0][0] == "git" and calls[0][1] == "-C"


def test_capture_git_sha_dirty(monkeypatch):
    results = [
        subprocess.CompletedProcess([], 0, stdout="abc123\n", stderr=""),
        subprocess.CompletedProcess([], 0, stdout=" M src/x.py\n", stderr=""),
    ]
    _install_fake_git(monkeypatch, results)
    sha, dirty = capture_git_sha(pathlib.Path("/ws"))
    assert sha == "abc123"
    assert dirty is True


def test_capture_git_sha_nonzero_returncode(monkeypatch):
    results = [subprocess.CompletedProcess([], 1, stdout="", stderr="fatal: not a git repo")]
    _install_fake_git(monkeypatch, results)
    assert capture_git_sha(pathlib.Path("/ws")) == (None, False)


def test_capture_git_sha_raises_returns_none(monkeypatch):
    def _boom(cmd, capture_output, text, timeout):  # noqa: ARG001
        raise RuntimeError("git missing")

    monkeypatch.setattr("arena_evaluation.benchmark.state.subprocess.run", _boom)
    assert capture_git_sha(pathlib.Path("/ws")) == (None, False)


# ---------------------------------------------------------------------------
# Manifest round trip
# ---------------------------------------------------------------------------


def test_manifest_round_trip_all_fields():
    man = _make_manifest()
    man2 = Manifest.from_yaml(man.to_yaml())
    assert man2 == man
    assert man2.run_id == "20260623-202509-basic-basic"
    assert man2.suite == {"stages": [{"name": "s1", "episodes": 3}]}
    assert man2.steps == [{"key": "c1/s1", "episodes_planned": 3}]


def test_manifest_to_yaml_unicode_and_ordering():
    man = _make_manifest(suite={"stages": [{"name": "café_東京"}]})
    text = man.to_yaml()
    assert "café_東京" in text  # allow_unicode=True
    assert text.index("run_id:") < text.index("created_at:")  # sort_keys=False


def test_manifest_launch_args_round_trip():
    man = _make_manifest()
    man.launch_args = {"isaac.physics": "newton", "headless": "true"}
    man2 = Manifest.from_yaml(man.to_yaml())
    assert man2.launch_args == {"isaac.physics": "newton", "headless": "true"}


def test_manifest_missing_launch_args_defaults_empty():
    import yaml

    data = yaml.safe_load(_make_manifest().to_yaml())
    del data["launch_args"]
    assert Manifest.from_yaml(yaml.dump(data)).launch_args == {}


def test_manifest_from_yaml_missing_fields_raises():
    with pytest.raises(TypeError):
        Manifest.from_yaml("run_id: only_one_field\n")


# ---------------------------------------------------------------------------
# StateFile
# ---------------------------------------------------------------------------


def test_state_file_open_empty(tmp_path: pathlib.Path):
    sf = StateFile.open(tmp_path)
    assert sf.steps == {}
    assert sf.path == tmp_path


def test_state_file_write_round_trip(tmp_path: pathlib.Path):
    sf = StateFile.open(tmp_path)
    steps = {
        "p/s1": _ok_step("p/s1"),
        "p/s2": StepResult(
            key="p/s2",
            status="partial",
            env_id=0,
            started_at=3.0,
            ended_at=4.0,
            error_kind=StepErrorKind.EPISODE_TIMEOUT,
            error_detail="slow",
            episodes_run=2,
            episodes_failed=1,
        ),
    }
    sf.write(steps)

    assert (tmp_path / ".benchmark_state.json").exists()
    assert not (tmp_path / ".benchmark_state.json.tmp").exists()  # atomic replace

    data = json.loads((tmp_path / ".benchmark_state.json").read_text())
    entry = data["steps"]["p/s2"]
    assert entry["status"] == "partial"
    assert entry["error_kind"] == "episode_timeout"
    assert entry["error_detail"] == "slow"
    assert entry["episodes_run"] == 2 and entry["episodes_failed"] == 1

    sf2 = StateFile.open(tmp_path)
    assert set(sf2.steps) == {"p/s1", "p/s2"}
    assert sf2.steps["p/s1"].error_kind is None
    assert sf2.steps["p/s2"].error_kind == StepErrorKind.EPISODE_TIMEOUT
    assert sf2.steps["p/s2"].episodes_failed == 1


def test_state_file_write_none_error_kind_serialized_as_null(tmp_path: pathlib.Path):
    sf = StateFile.open(tmp_path)
    sf.write({"p/s": _ok_step("p/s")})
    data = json.loads((tmp_path / ".benchmark_state.json").read_text())
    assert data["steps"]["p/s"]["error_kind"] is None


def test_state_file_write_updates_member(tmp_path: pathlib.Path):
    sf = StateFile.open(tmp_path)
    sf.write({"p/s": _ok_step("p/s")})
    assert sf.steps["p/s"].status == "ok"


def test_state_file_backward_compat_error_field(tmp_path: pathlib.Path):
    (tmp_path / ".benchmark_state.json").write_text(
        json.dumps(
            {
                "steps": {
                    "p/s": {
                        "status": "failed",
                        "env_id": None,
                        "started_at": 0.0,
                        "ended_at": 1.0,
                        "error": "legacy message",
                    }
                }
            }
        )
    )
    sf = StateFile.open(tmp_path)
    r = sf.steps["p/s"]
    assert r.error_detail == "legacy message"
    assert r.error_kind is None
    assert r.episodes_run == 0  # missing fields default


def test_state_file_open_invalid_error_kind_raises(tmp_path: pathlib.Path):
    (tmp_path / ".benchmark_state.json").write_text(json.dumps({"steps": {"p/s": {"status": "failed", "error_kind": "bogus_kind"}}}))
    with pytest.raises(ValueError):
        StateFile.open(tmp_path)


def test_state_file_open_corrupt_json_raises(tmp_path: pathlib.Path):
    (tmp_path / ".benchmark_state.json").write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        StateFile.open(tmp_path)


# ---------------------------------------------------------------------------
# _params_to_json
# ---------------------------------------------------------------------------


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


def test_params_to_json_scalar():
    out = _params_to_json([_msg_param("a_int", ParameterType.PARAMETER_INTEGER, 7)])
    assert out == '[{"name": "a_int", "value": 7}]'


def test_params_to_json_array_tolist():
    """numpy array values are converted with .tolist() before serialization."""
    out = _params_to_json([_msg_param("a_arr", ParameterType.PARAMETER_INTEGER_ARRAY, [1, 2, 3])])
    assert json.loads(out) == [{"name": "a_arr", "value": [1, 2, 3]}]


def test_params_to_json_fallback_on_bad_msg():
    """If Parameter.from_parameter_msg raises, the raw .value is stringified."""
    bad = types.SimpleNamespace(name="x", value="raw")
    assert json.loads(_params_to_json([bad])) == [{"name": "x", "value": "raw"}]


def test_params_to_json_empty():
    assert _params_to_json([]) == "[]"


# ---------------------------------------------------------------------------
# ProgressLog
# ---------------------------------------------------------------------------

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
    "tm_robots",
    "tm_obstacles",
    "tm_modules",
    "robots",
    "outcome_state",
    "outcome_info",
    "started_at",
    "ended_at",
    "runtime_s",
    "robots_params_json",
    "obstacles_params_json",
    "error_kind",
    "error_detail",
    "lockstep_stalls",
    "lockstep_max_stall_s",
    "lockstep_rtf",
    "lockstep_beats",
    "goal_dist_start",
    "goal_dist_min",
    "path_length",
]


def test_progress_log_writes_header_on_new_file(tmp_path: pathlib.Path):
    log = ProgressLog(tmp_path / "progress.csv")
    log.close()
    with (tmp_path / "progress.csv").open(newline="") as fh:
        assert next(csv.reader(fh)) == _HEADERS


def test_progress_log_full_row(tmp_path: pathlib.Path):
    log = ProgressLog(tmp_path / "progress.csv")
    rec = _make_episode_record(episode_id=5)
    log.append(
        ts_iso="2026-06-23T20:25:09+00:00",
        run_id="r1",
        step_key="c1/s1",
        contestant="c1",
        stage="s1",
        env_id=3,
        episode_id=5,
        episode_record=rec,
        started_at=100.0,
        ended_at=105.5,
        is_reference=True,
        reference_type="unobstructed_robot",
    )
    log.close()

    with (tmp_path / "progress.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    row = rows[0]
    assert row["env_id"] == "3"
    assert row["is_reference"] == "true"
    assert row["reference_type"] == "unobstructed_robot"
    assert row["runtime_s"] == "5.5"
    assert row["error_kind"] == ""
    assert row["error_detail"] == ""


def test_progress_log_optional_fields_empty(tmp_path: pathlib.Path):
    log = ProgressLog(tmp_path / "progress.csv")
    rec = _make_episode_record(episode_id=1)
    log.append(
        ts_iso="t",
        run_id="r",
        step_key="k",
        contestant="c",
        stage="s",
        env_id=None,
        episode_id=1,
        episode_record=rec,
        started_at=0.0,
        ended_at=1.0,
    )
    log.close()
    with (tmp_path / "progress.csv").open(newline="") as fh:
        row = next(csv.DictReader(fh))
    assert row["env_id"] == ""
    assert row["parent_episode_id"] == ""
    assert row["is_reference"] == "false"
    assert row["reference_type"] == ""


def test_progress_log_error_fields(tmp_path: pathlib.Path):
    log = ProgressLog(tmp_path / "progress.csv")
    rec = _make_episode_record(episode_id=2)
    log.append(
        ts_iso="t",
        run_id="r",
        step_key="k",
        contestant="c",
        stage="s",
        env_id=0,
        episode_id=2,
        episode_record=rec,
        started_at=0.0,
        ended_at=1.0,
        error_kind=StepErrorKind.INTERNAL,
        error_detail="boom",
    )
    log.close()
    with (tmp_path / "progress.csv").open(newline="") as fh:
        row = next(csv.DictReader(fh))
    assert row["error_kind"] == "internal"
    assert row["error_detail"] == "boom"


def test_progress_log_reopen_no_duplicate_header(tmp_path: pathlib.Path):
    path = tmp_path / "progress.csv"
    log1 = ProgressLog(path)
    log1.close()
    log2 = ProgressLog(path)
    log2.close()
    with path.open(newline="") as fh:
        lines = fh.readlines()
    assert sum(1 for l in lines if l.startswith("ts_iso,")) == 1


def test_progress_log_write_comment(tmp_path: pathlib.Path):
    log = ProgressLog(tmp_path / "progress.csv")
    log.write_comment("resumed at 2026-01-01T00:00:00+00:00")
    log.close()
    assert "# resumed at" in (tmp_path / "progress.csv").read_text()


def test_progress_log_append_with_parameter_msg_params(tmp_path: pathlib.Path):
    """Integration: rcl_interfaces Parameter msgs flow through _params_to_json."""
    log = ProgressLog(tmp_path / "progress.csv")
    rec = types.SimpleNamespace(
        episode_id=1,
        world="w",
        seed=1,
        tm_robots="r",
        tm_obstacles="o",
        tm_modules=["m"],
        robots=["bot"],
        outcome_state=1,
        outcome_info="",
        robots_params=[_msg_param("speed", ParameterType.PARAMETER_INTEGER, 3)],
        obstacles_params=[],
        goal_dist_start=0.0,
        goal_dist_min=0.0,
        path_length=0.0,
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
    assert json.loads(row["robots_params_json"]) == [{"name": "speed", "value": 3}]


# -- dedupe_in_place --


def _write_log_lines(path: pathlib.Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def test_dedupe_empty_file_noop(tmp_path: pathlib.Path):
    """Opening an empty file writes the header; dedupe leaves just the header."""
    path = tmp_path / "progress.csv"
    path.write_text("")
    log = ProgressLog(path)
    log.dedupe_in_place()
    log.close()
    assert path.read_text().strip() == ",".join(_HEADERS)


def test_dedupe_comment_only_noop(tmp_path: pathlib.Path):
    path = tmp_path / "progress.csv"
    path.write_text("# only a comment\n")
    log = ProgressLog(path)
    log.dedupe_in_place()
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
    log.dedupe_in_place()
    log.close()
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    # (k,1) keeps the 05 row; the stale @00 duplicate of (k,2) is dropped;
    # remaining rows sorted by ts_iso.
    assert [r["episode_id"] for r in rows] == ["2", "3", "1"]
    assert rows[0]["ts_iso"] == "2026-01-01T00:00:02+00:00"
    assert rows[-1]["ts_iso"] == "2026-01-01T00:00:05+00:00"


# ---------------------------------------------------------------------------
# RunDir
# ---------------------------------------------------------------------------


def test_run_dir_create_layout(tmp_path: pathlib.Path):
    run_dir = RunDir.create(tmp_path, "r1", _make_manifest(run_id="r1"))
    assert run_dir.path == tmp_path / "r1"
    assert (run_dir.path / "manifest.yaml").exists()
    assert (run_dir.path / "progress.csv").exists()
    assert run_dir.state.steps == {}
    assert Manifest.from_yaml((run_dir.path / "manifest.yaml").read_text()).run_id == "r1"


def test_run_dir_create_existing_raises(tmp_path: pathlib.Path):
    (tmp_path / "r1").mkdir()
    with pytest.raises(FileExistsError):
        RunDir.create(tmp_path, "r1", _make_manifest(run_id="r1"))


def test_run_dir_open_round_trip(tmp_path: pathlib.Path):
    man = _make_manifest(run_id="r2")
    RunDir.create(tmp_path, "r2", man)
    opened = RunDir.open(tmp_path, "r2")
    assert opened.manifest == man
    assert opened.path == tmp_path / "r2"
    assert opened.state.steps == {}
    opened.state.write({"p/s": _ok_step("p/s")})

    reopened = RunDir.open(tmp_path, "r2")
    assert reopened.state.steps["p/s"].status == "ok"


def test_run_dir_attach_log_handler(tmp_path: pathlib.Path):
    run_dir = RunDir.create(tmp_path, "r1", _make_manifest(run_id="r1"))
    logger = logging.getLogger("arena_evaluation.tests.run_dir")
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
