"""Tests for benchmark debug utilities: process introspection and console logs.

Covers :mod:`arena_evaluation.benchmark.debug`. All process tables are
injected (monkeypatched ``_ps_lines`` / ``subprocess.run``); nothing is
executed against the live system except self-directed ``/proc`` reads.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from arena_evaluation.benchmark import debug as debug_mod


# ---------------------------------------------------------------------------
# _classify
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cmdline,expected",
    [
        ("python -m arena_evaluation evaluation benchmark --run-id r1", "benchmark_runner"),
        ("/opt/ws/install/lib/arena_evaluation/benchmark/runner.py", "benchmark_runner"),
        ("python -m world_generator", "world_generator"),
        ("gz sim -s --render-engine ogre", "simulation"),
        ("ign gazebo --verbose", "simulation"),
        ("gazebo --server", "simulation"),
        ("gzserver --world w", "simulation"),
        ("python -m arena_node", "arena_node"),
        ("ros2 run arena_evaluation data_recorder", "arena_node"),
        ("ros2 run arena_evaluation recorder_node", "arena_node"),
        ("arena_cli extract --run-dir x", "arena_cli"),
        ("uv run arena_cli process", "arena_cli"),
        ("sleep 100", None),
        ("python setup.py", None),
    ],
)
def test_classify(cmdline: str, expected: str | None):
    assert debug_mod._classify(cmdline) == expected


def test_classify_benchmark_marker_wins_over_sim_marker():
    assert debug_mod._classify("evaluation benchmark gz sim child") == "benchmark_runner"


# ---------------------------------------------------------------------------
# _ps_lines
# ---------------------------------------------------------------------------


def _fake_run_stdout(stdout: str):
    def _run(cmd, capture_output, text, timeout):  # noqa: ARG001
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    return _run


def test_ps_lines_parses_and_filters(monkeypatch):
    fake = _fake_run_stdout(f"  1234 /usr/bin/python evaluation benchmark --run-id r1\n  5678 gz sim -s\n  9999 garbage no split here\n  0xBAD not-a-pid\n  3333 pkill -f arena\n  4444 ps -ef\n  5555 grep arena\n 99999 arena_node --self --pid\n")
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


# ---------------------------------------------------------------------------
# _elapsed_s / _runner_starttime (real /proc reads against this process)
# ---------------------------------------------------------------------------


def test_elapsed_s_own_pid():
    if not pathlib.Path("/proc/self/stat").exists():
        pytest.skip("requires Linux /proc")
    elapsed = debug_mod._elapsed_s(debug_mod.os.getpid())
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0


def test_elapsed_s_missing_pid_returns_none():
    # PID 2**30 cannot exist in a real kernel; /proc open fails.
    assert debug_mod._elapsed_s(2**30) is None


def test_runner_starttime_own_pid():
    if not pathlib.Path("/proc/self/stat").exists():
        pytest.skip("requires Linux /proc")
    assert debug_mod._runner_starttime(debug_mod.os.getpid()) > 0.0


def test_runner_starttime_missing_pid_returns_zero():
    assert debug_mod._runner_starttime(2**30) == 0.0


# ---------------------------------------------------------------------------
# is_pid_alive
# ---------------------------------------------------------------------------


def test_is_pid_alive_falsy():
    assert debug_mod.is_pid_alive(None) is False
    assert debug_mod.is_pid_alive(0) is False


def test_is_pid_alive_true_for_self():
    assert debug_mod.is_pid_alive(debug_mod.os.getpid()) is True


def test_is_pid_alive_process_lookup_error(monkeypatch):
    def _kill(pid, sig):  # noqa: ARG001
        raise ProcessLookupError

    monkeypatch.setattr(debug_mod.os, "kill", _kill)
    assert debug_mod.is_pid_alive(12345) is False


def test_is_pid_alive_permission_error(monkeypatch):
    def _kill(pid, sig):  # noqa: ARG001
        raise PermissionError

    monkeypatch.setattr(debug_mod.os, "kill", _kill)
    assert debug_mod.is_pid_alive(1) is False


# ---------------------------------------------------------------------------
# running_processes
# ---------------------------------------------------------------------------


def test_running_processes_sort_and_filter(monkeypatch):
    monkeypatch.setattr(
        debug_mod,
        "_ps_lines",
        lambda: [
            (3, "gz sim -s"),
            (1, "evaluation benchmark --run-id r1"),
            (2, "arena_node"),
            (9, "completely unrelated process"),
        ],
    )
    monkeypatch.setattr(debug_mod, "_elapsed_s", lambda pid: {1: 10.0, 2: 5.0, 3: 1.0}[pid])

    rows = debug_mod.running_processes()
    # _KIND_ORDER: benchmark_runner(0) < simulation(2) < arena_node(3); tie-break by pid
    assert [r["pid"] for r in rows] == [1, 3, 2]
    assert [r["kind"] for r in rows] == ["benchmark_runner", "simulation", "arena_node"]
    assert rows[0]["elapsed_s"] == 10.0
    assert all("unrelated" not in r["command"] for r in rows)


def test_running_processes_truncates_command(monkeypatch):
    long_cmd = "evaluation benchmark --run-id " + "x" * 500
    monkeypatch.setattr(debug_mod, "_ps_lines", lambda: [(1, long_cmd)])
    monkeypatch.setattr(debug_mod, "_elapsed_s", lambda pid: None)
    rows = debug_mod.running_processes()
    assert len(rows[0]["command"]) == 200
    assert rows[0]["command"].startswith("evaluation benchmark")
    assert rows[0]["elapsed_s"] is None


# ---------------------------------------------------------------------------
# find_runner_pid
# ---------------------------------------------------------------------------


def test_find_runner_pid_none(monkeypatch):
    monkeypatch.setattr(debug_mod, "_ps_lines", lambda: [])
    assert debug_mod.find_runner_pid() is None


def test_find_runner_pid_most_recent(monkeypatch):
    monkeypatch.setattr(
        debug_mod,
        "_ps_lines",
        lambda: [
            (101, "python evaluation benchmark --run-id r1"),
            (202, "python evaluation benchmark --run-id r2"),
            (303, "gz sim"),
        ],
    )
    monkeypatch.setattr(debug_mod, "_runner_starttime", lambda pid: {101: 1.0, 202: 2.0}[pid])
    assert debug_mod.find_runner_pid() == 202
    assert debug_mod.find_runner_pid("r1") == 101
    assert debug_mod.find_runner_pid("nope") is None


# ---------------------------------------------------------------------------
# running_pids_by_run_id
# ---------------------------------------------------------------------------


def test_running_pids_by_run_id_parses_flag(monkeypatch):
    monkeypatch.setattr(
        debug_mod,
        "_ps_lines",
        lambda: [
            (11, "python evaluation benchmark --run-id abc-123"),
            (22, "python evaluation benchmark --run-id 'quoted-id'"),
            (33, "python evaluation benchmark --run-id"),  # no value
        ],
    )
    result = debug_mod.running_pids_by_run_id()
    assert result["abc-123"] == 11
    assert result["quoted-id"] == 22
    assert 33 not in result.values()


def test_running_pids_by_run_id_fallback_token(monkeypatch):
    monkeypatch.setattr(
        debug_mod,
        "_ps_lines",
        lambda: [
            (11, "python evaluation benchmark 20260623-120000-suite-contest"),
            (22, "python evaluation benchmark short"),  # token too short
            (33, "python evaluation benchmark a/b-with-slash"),  # path-like token
            (44, "python evaluation benchmark --run-id real-id"),
        ],
    )
    result = debug_mod.running_pids_by_run_id()
    assert result.get("20260623-120000-suite-contest") == 11
    assert "short" not in result
    assert "a/b-with-slash" not in result
    assert result.get("real-id") == 44


def test_running_pids_by_run_id_no_match(monkeypatch):
    monkeypatch.setattr(debug_mod, "_ps_lines", lambda: [(1, "gz sim")])
    assert debug_mod.running_pids_by_run_id() == {}


# ---------------------------------------------------------------------------
# console log paths / tailing
# ---------------------------------------------------------------------------


def test_console_log_path_uses_data_root_env(monkeypatch, tmp_path: pathlib.Path):
    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path))
    assert debug_mod.console_log_path("r1") == tmp_path / "benchmarks" / "r1" / "runner.log"


def _make_console_dir(monkeypatch, tmp_path: pathlib.Path, run_id: str) -> pathlib.Path:
    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path))
    log_path = tmp_path / "benchmarks" / run_id / "runner.log"
    return log_path


def test_tail_console_missing_log(monkeypatch, tmp_path: pathlib.Path):
    monkeypatch.setattr(debug_mod, "find_runner_pid", lambda run_id: None)
    monkeypatch.setattr(debug_mod, "is_pid_alive", lambda pid: False)
    log_path = _make_console_dir(monkeypatch, tmp_path, "r1")
    res = debug_mod.tail_console("r1")
    assert res["exists"] is False
    assert res["path"] == str(log_path)
    assert res["pid"] is None
    assert res["alive"] is False
    assert res["lines"] == []
    assert res["truncated"] is False


def test_tail_console_reads_and_truncates(monkeypatch, tmp_path: pathlib.Path):
    log_path = _make_console_dir(monkeypatch, tmp_path, "r1")
    log_path.parent.mkdir(parents=True)
    log_path.write_text("line1\nline2\nline3\n")
    monkeypatch.setattr(debug_mod, "find_runner_pid", lambda run_id: 42)
    monkeypatch.setattr(debug_mod, "is_pid_alive", lambda pid: True)

    res = debug_mod.tail_console("r1", lines=2)
    assert res["exists"] is True
    assert res["lines"] == ["line2", "line3"]
    assert res["truncated"] is True
    assert res["alive"] is True
    assert res["pid"] == 42


def test_tail_console_lines_zero_returns_all(monkeypatch, tmp_path: pathlib.Path):
    log_path = _make_console_dir(monkeypatch, tmp_path, "r1")
    log_path.parent.mkdir(parents=True)
    log_path.write_text("a\nb\n")
    monkeypatch.setattr(debug_mod, "find_runner_pid", lambda run_id: None)
    monkeypatch.setattr(debug_mod, "is_pid_alive", lambda pid: False)
    res = debug_mod.tail_console("r1", lines=0)
    assert res["lines"] == ["a", "b"]
    # truncated flags "more content than requested lines", so 0 => True even
    # though all lines are returned
    assert res["truncated"] is True


def test_tail_console_read_error_reported(monkeypatch, tmp_path: pathlib.Path):
    log_path = _make_console_dir(monkeypatch, tmp_path, "r1")
    log_path.parent.mkdir(parents=True)
    log_path.write_text("x\n")
    monkeypatch.setattr(debug_mod, "find_runner_pid", lambda run_id: None)
    monkeypatch.setattr(debug_mod, "is_pid_alive", lambda pid: False)

    def _read_bytes_boom(self):  # noqa: ARG001
        raise OSError("disk gone")

    monkeypatch.setattr(pathlib.Path, "read_bytes", _read_bytes_boom)
    res = debug_mod.tail_console("r1")
    assert res["exists"] is True
    assert res["lines"] == []
    assert "error" in res
    assert "disk gone" in res["error"]


def test_tail_console_decodes_utf8_with_replacement(monkeypatch, tmp_path: pathlib.Path):
    log_path = _make_console_dir(monkeypatch, tmp_path, "r1")
    log_path.parent.mkdir(parents=True)
    log_path.write_bytes(b"ok\n\xff\xfe\n")
    monkeypatch.setattr(debug_mod, "find_runner_pid", lambda run_id: None)
    monkeypatch.setattr(debug_mod, "is_pid_alive", lambda pid: False)
    res = debug_mod.tail_console("r1")
    assert res["lines"][1] == "��"  # replacement chars, no crash


# ---------------------------------------------------------------------------
# latest_running_run_id
# ---------------------------------------------------------------------------


def test_latest_running_run_id_none_when_no_runner(monkeypatch):
    monkeypatch.setattr(debug_mod, "find_runner_pid", lambda run_id: None)
    assert debug_mod.latest_running_run_id() is None


def test_latest_running_run_id_scans_cmdline(monkeypatch):
    monkeypatch.setattr(debug_mod, "find_runner_pid", lambda run_id: 77)
    monkeypatch.setattr(debug_mod, "running_pids_by_run_id", lambda: {"some-id": 77})
    monkeypatch.setattr(
        debug_mod,
        "_ps_lines",
        lambda: [
            (55, "python evaluation benchmark --run-id other-run"),  # wrong pid: skipped
            (77, "python evaluation benchmark --run-id 20260623-120000-suite-contest"),
        ],
    )
    assert debug_mod.latest_running_run_id() == "20260623-120000-suite-contest"


def test_latest_running_run_id_skips_flag_token(monkeypatch):
    monkeypatch.setattr(debug_mod, "find_runner_pid", lambda run_id: 77)
    monkeypatch.setattr(debug_mod, "running_pids_by_run_id", lambda: {})
    monkeypatch.setattr(
        debug_mod,
        "_ps_lines",
        lambda: [
            (77, "python evaluation benchmark --run-id only-flag"),  # token starts with --run-id
            (88, "unrelated process"),
        ],
    )
    assert debug_mod.latest_running_run_id() is None
