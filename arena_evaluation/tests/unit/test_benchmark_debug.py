"""Tests for benchmark debug utilities: process introspection and console logs."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time

import pytest

from arena_evaluation.benchmark import debug as debug_mod

GONE = 2**30


# _classify


@pytest.mark.parametrize(
    "cmdline,expected",
    [
        (
            "python -m arena_evaluation evaluation benchmark --run-id r1",
            "benchmark_runner",
        ),
        (
            "/opt/ws/install/lib/arena_evaluation/benchmark/runner.py",
            "benchmark_runner",
        ),
        ("python -m world_generator", "world_generator"),
        ("gz sim -s --render-engine ogre", "simulation"),
        ("ign gazebo --verbose", "simulation"),
        ("gazebo --server", "simulation"),
        ("gzserver --world w", "simulation"),
        ("python -m arena_runtime_node", "arena_runtime_node"),
        ("ros2 run arena_evaluation data_recorder", "arena_runtime_node"),
        ("ros2 run arena_evaluation recorder_node", "arena_runtime_node"),
        ("arena_cli extract --run-dir x", "arena_cli"),
        ("uv run arena_cli process", "arena_cli"),
        ("sleep 100", None),
        ("python setup.py", None),
    ],
)
def test_classify(cmdline: str, expected: str | None):
    assert debug_mod._classify(cmdline) == expected


# _ps_lines


def _fake_run_stdout(stdout: str):
    def _run(cmd, capture_output, text, timeout):  # noqa: ARG001
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    return _run


def _ps(monkeypatch, *rows: tuple[int, str]) -> None:
    monkeypatch.setattr(
        debug_mod.subprocess,
        "run",
        _fake_run_stdout("".join(f"{pid} {cmdline}\n" for pid, cmdline in rows)),
    )


@pytest.fixture
def sleeper():
    procs: list[subprocess.Popen] = []

    def spawn() -> int:
        procs.append(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]))
        return procs[-1].pid

    yield spawn
    for proc in procs:
        proc.kill()
        proc.wait()


def test_ps_lines_parses_and_filters(monkeypatch):
    fake = _fake_run_stdout("  1234 /usr/bin/python evaluation benchmark --run-id r1\n  5678 gz sim -s\n  9999 garbage no split here\n  0xBAD not-a-pid\n  3333 pkill -f arena\n  4444 ps -ef\n  5555 grep arena\n 99999 arena_runtime_node --self --pid\n")
    monkeypatch.setattr(debug_mod.subprocess, "run", fake)
    monkeypatch.setattr(debug_mod.os, "getpid", lambda: 99999)

    rows = debug_mod._ps_lines()
    assert (1234, "/usr/bin/python evaluation benchmark --run-id r1") in rows
    assert (5678, "gz sim -s") in rows
    assert (9999, "garbage no split here") in rows
    assert all(pid != 99999 for pid, _ in rows)  # self excluded
    assert all(pid not in (3333, 4444, 5555) for pid, _ in rows)  # pkill/ps/grep


def test_ps_lines_run_exception_returns_empty(monkeypatch):
    def _boom(cmd, capture_output, text, timeout):  # noqa: ARG001
        raise FileNotFoundError("ps not available")

    monkeypatch.setattr(debug_mod.subprocess, "run", _boom)
    assert debug_mod._ps_lines() == []


def test_ps_lines_single_field_line_skipped(monkeypatch):
    fake = _fake_run_stdout("justapidthatwonotparse\n")
    monkeypatch.setattr(debug_mod.subprocess, "run", fake)
    assert debug_mod._ps_lines() == []


# _elapsed_s / _runner_starttime (real /proc reads against this process)


def test_runner_starttime_missing_pid_returns_zero():
    assert debug_mod._runner_starttime(2**30) == 0.0


# is_pid_alive


def test_is_pid_alive_permission_error(monkeypatch):
    def _kill(pid, sig):  # noqa: ARG001
        raise PermissionError

    monkeypatch.setattr(debug_mod.os, "kill", _kill)
    assert debug_mod.is_pid_alive(1) is False


# running_processes


def test_running_processes_sort_and_filter(monkeypatch, sleeper):
    runner = sleeper()
    _ps(
        monkeypatch,
        (GONE, "gz sim -s"),
        (runner, "evaluation benchmark --run-id r1"),
        (GONE + 1, "arena_runtime_node"),
        (GONE + 2, "completely unrelated process"),
    )

    rows = debug_mod.running_processes()
    # _KIND_ORDER: benchmark_runner(0) < simulation(2) < arena_runtime_node(3); tie-break by pid
    assert [r["pid"] for r in rows] == [runner, GONE, GONE + 1]
    assert [r["kind"] for r in rows] == [
        "benchmark_runner",
        "simulation",
        "arena_runtime_node",
    ]
    assert rows[0]["elapsed_s"] >= 0.0
    assert rows[1]["elapsed_s"] is None
    assert all("unrelated" not in r["command"] for r in rows)


# find_runner_pid


def test_find_runner_pid_most_recent(monkeypatch, sleeper):
    older = sleeper()
    newer = sleeper()
    _ps(
        monkeypatch,
        (older, "python evaluation benchmark --run-id r1"),
        (newer, "python evaluation benchmark --run-id r2"),
        (GONE, "gz sim"),
    )
    assert debug_mod.find_runner_pid() == newer
    assert debug_mod.find_runner_pid("r1") == older
    assert debug_mod.find_runner_pid("nope") is None


# running_pids_by_run_id


def test_running_pids_by_run_id_fallback_token(monkeypatch):
    _ps(
        monkeypatch,
        (11, "python evaluation benchmark 20260623-120000-suite-contest"),
        (22, "python evaluation benchmark short"),  # token too short
        (33, "python evaluation benchmark a/b-with-slash"),  # path-like token
        (44, "python evaluation benchmark --run-id real-id"),
    )
    result = debug_mod.running_pids_by_run_id()
    assert result.get("20260623-120000-suite-contest") == 11
    assert "short" not in result
    assert "a/b-with-slash" not in result
    assert result.get("real-id") == 44


# console log paths / tailing


def _make_console_dir(monkeypatch, tmp_path: pathlib.Path, run_id: str) -> pathlib.Path:
    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path))
    log_path = tmp_path / "benchmarks" / run_id / "runner.log"
    return log_path


def test_tail_console_reads_and_truncates(monkeypatch, tmp_path: pathlib.Path, sleeper):
    log_path = _make_console_dir(monkeypatch, tmp_path, "r1")
    log_path.parent.mkdir(parents=True)
    log_path.write_text("line1\nline2\nline3\n")
    runner = sleeper()
    _ps(monkeypatch, (runner, "python evaluation benchmark --run-id r1"))

    res = debug_mod.tail_console("r1", lines=2)
    assert res["exists"] is True
    assert res["lines"] == ["line2", "line3"]
    assert res["truncated"] is True
    assert res["alive"] is True
    assert res["pid"] == runner


def test_tail_console_read_error_reported(monkeypatch, tmp_path: pathlib.Path):
    log_path = _make_console_dir(monkeypatch, tmp_path, "r1")
    log_path.mkdir(parents=True)
    _ps(monkeypatch)

    res = debug_mod.tail_console("r1")
    assert res["exists"] is True
    assert res["lines"] == []
    assert "runner.log" in res["error"]


# latest_running_run_id


def test_latest_running_run_id_none_when_no_runner(monkeypatch):
    _ps(monkeypatch)
    assert debug_mod.latest_running_run_id() is None


def test_latest_running_run_id_scans_cmdline(monkeypatch, sleeper):
    older = sleeper()
    newer = sleeper()
    _ps(
        monkeypatch,
        (older, "python evaluation benchmark --run-id other-run-from-before"),
        (newer, "python evaluation benchmark --run-id 20260623-120000-suite-contest"),
    )
    assert debug_mod.latest_running_run_id() == "20260623-120000-suite-contest"


def test_latest_running_run_id_skips_flag_token(monkeypatch, sleeper):
    runner = sleeper()
    _ps(
        monkeypatch,
        (
            runner,
            "python evaluation benchmark --run-id only-flag",
        ),  # token starts with --run-id
        (GONE, "unrelated process"),
    )
    assert debug_mod.latest_running_run_id() is None


def test_kill_refuses_processes_outside_arena_and_the_caller():
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        results = {entry["pid"]: entry["status"] for entry in debug_mod.kill_processes(pids=[bystander.pid, os.getpid()])}
        assert results[bystander.pid] == "refused: not an arena process"
        assert results[os.getpid()].startswith("refused: this process")
        assert bystander.poll() is None
    finally:
        bystander.kill()
        bystander.wait()


def test_kill_stops_an_arena_process_group_with_sigint_first():
    target = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; print('ready', flush=True); time.sleep(60)",
            "world_generator",
        ],
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert target.stdout is not None and target.stdout.readline() == "ready\n"
        deadline = time.time() + 10
        while target.pid not in {proc["pid"] for proc in debug_mod.running_processes()} and time.time() < deadline:
            time.sleep(0.1)
        (result,) = debug_mod.kill_processes(pids=[target.pid], grace_s=10.0)
        assert result["kind"] == "world_generator"
        assert result["status"] == "stopped"
        assert target.wait(timeout=30) != 0
    finally:
        if target.poll() is None:
            target.kill()
            target.wait()
