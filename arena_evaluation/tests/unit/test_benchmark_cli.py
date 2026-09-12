"""Tests for the benchmark management CLI (list / status / tail / ps / console).

Covers :mod:`arena_evaluation.benchmark.cli`. No subprocesses are executed:
``tail``/``subprocess.run`` are stubbed, the ``status --watch`` ROS path is
driven with in-memory stub modules (the real import stays lazy), and all
filesystem state lives under ``tmp_path``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import types

import pytest

from arena_evaluation.benchmark import cli as cli_mod
from arena_evaluation.benchmark.state import Manifest, RunDir
from arena_evaluation.benchmark.step import StepErrorKind, StepResult


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_manifest(run_id: str = "r1", **overrides: object) -> Manifest:
    base: dict[str, object] = {
        "run_id": run_id,
        "created_at": "2026-06-23T20:25:09+00:00",
        "arena_git_sha": None,
        "arena_git_dirty": False,
        "cli_args": ["--suite", "basic"],
        "env_n": 1,
        "headless": False,
        "config_hash": "h" * 40,
        "simulator": "gazebo",
        "scale_episodes": 1.0,
        "suite_name": "basic",
        "contest_name": "basic",
        "suite": {"stages": [{"name": "s1"}]},
        "contest": [{"name": "c1"}],
        "steps": [{"key": "c1/s1"}],
    }
    base.update(overrides)
    return Manifest(**base)  # type: ignore[arg-type]


def _make_run(data_root: pathlib.Path, run_id: str, manifest: Manifest | None = None, state_steps: dict[str, StepResult] | None = None, progress_csv: str | None = None) -> pathlib.Path:
    run_dir = RunDir.create(data_root, run_id, manifest or _make_manifest(run_id=run_id))
    if state_steps:
        run_dir.state.write(state_steps)
    if progress_csv is not None:
        (run_dir.path / "progress.csv").write_text(progress_csv)
    return run_dir.path


def _args(**kw: object) -> argparse.Namespace:
    return argparse.Namespace(**kw)


# ---------------------------------------------------------------------------
# _resolve_run
# ---------------------------------------------------------------------------


def test_resolve_run_explicit_id(tmp_path: pathlib.Path):
    _make_run(tmp_path, "r1")
    assert cli_mod._resolve_run(tmp_path, "r1") == tmp_path / "r1"


def test_resolve_run_explicit_missing_id_raises(tmp_path: pathlib.Path):
    with pytest.raises(SystemExit, match="no run found"):
        cli_mod._resolve_run(tmp_path, "ghost")


def test_resolve_run_missing_data_root_raises(tmp_path: pathlib.Path):
    with pytest.raises(SystemExit, match="data root does not exist"):
        cli_mod._resolve_run(tmp_path / "nope", None)


def test_resolve_run_empty_root_raises(tmp_path: pathlib.Path):
    with pytest.raises(SystemExit, match="no benchmark runs"):
        cli_mod._resolve_run(tmp_path, None)


def test_resolve_run_returns_most_recent(tmp_path: pathlib.Path):
    _make_run(tmp_path, "old-run")
    _make_run(tmp_path, "z-run")
    assert cli_mod._resolve_run(tmp_path, None).name == "z-run"


# ---------------------------------------------------------------------------
# _count_by_status
# ---------------------------------------------------------------------------


def _sr(key: str, status: str) -> StepResult:
    return StepResult(key, status, None, 0.0, 1.0, None, None)


def test_count_by_status_all_statuses():
    steps = {
        "a/s": _sr("a/s", "ok"),
        "b/s": _sr("b/s", "partial"),
        "c/s": _sr("c/s", "failed"),
        "d/s": _sr("d/s", "skipped"),
        "e/s": _sr("e/s", "in_progress"),
    }
    counts = cli_mod._count_by_status(steps)
    assert counts == {"ok": 1, "partial": 1, "failed": 1, "skipped": 1, "in_progress": 1}


def test_count_by_status_unknown_status_ignored():
    counts = cli_mod._count_by_status({"a/s": _sr("a/s", "mystery")})
    assert counts == {"ok": 0, "partial": 0, "failed": 0, "skipped": 0, "in_progress": 0}


# ---------------------------------------------------------------------------
# _cmd_list
# ---------------------------------------------------------------------------


def test_cmd_list_missing_root(tmp_path: pathlib.Path, capsys):
    rc = cli_mod._cmd_list(_args(data_root=str(tmp_path / "nope")))
    assert rc == 0
    assert "no benchmark runs in" in capsys.readouterr().out


def test_cmd_list_empty_root(tmp_path: pathlib.Path, capsys):
    rc = cli_mod._cmd_list(_args(data_root=str(tmp_path)))
    assert rc == 0
    assert "no benchmark runs in" in capsys.readouterr().out


def test_cmd_list_skips_runs_without_valid_manifest(tmp_path: pathlib.Path, capsys):
    (tmp_path / "bad-manifest").mkdir()
    (tmp_path / "bad-manifest" / "manifest.yaml").write_text("not: [valid yaml\n")
    (tmp_path / "no-manifest").mkdir()
    rc = cli_mod._cmd_list(_args(data_root=str(tmp_path)))
    assert rc == 0
    assert "no benchmark runs in" in capsys.readouterr().out


def test_cmd_list_prints_table(tmp_path: pathlib.Path, capsys):
    _make_run(
        tmp_path,
        "run-b",
        manifest=_make_manifest(run_id="run-b", suite_name="suite_b", contest_name="contest_b", created_at="2026-06-23T20:25:09+00:00"),
        state_steps={"c1/s1": _sr("c1/s1", "ok"), "c1/s2": _sr("c1/s2", "failed")},
    )
    _make_run(
        tmp_path,
        "run-a",
        manifest=_make_manifest(run_id="run-a", suite_name="suite_a", contest_name="contest_a", created_at="2026-06-23T20:25:09+00:00"),
        state_steps={"c1/s1": _sr("c1/s1", "in_progress")},
    )
    rc = cli_mod._cmd_list(_args(data_root=str(tmp_path)))
    out = capsys.readouterr().out
    assert rc == 0
    assert "RUN_ID" in out and "IN_FLIGHT" in out
    # created_at[:16] with 'T' replaced by space
    assert "2026-06-23 20:25" in out
    lines = {l.split()[0]: l.split() for l in out.splitlines() if l.split() and l.split()[0] in ("run-a", "run-b")}
    # columns: run_id suite contest total ok partial failed skipped in_flight created...
    assert lines["run-a"][3:9] == ["1", "0", "0", "0", "0", "1"]
    assert lines["run-b"][3:9] == ["1", "1", "0", "1", "0", "0"]


def test_cmd_list_empty_created_at(tmp_path: pathlib.Path, capsys):
    _make_run(tmp_path, "r1", manifest=_make_manifest(run_id="r1", created_at=""))
    cli_mod._cmd_list(_args(data_root=str(tmp_path)))
    out = capsys.readouterr().out
    assert "r1" in out  # empty created_at does not crash


# ---------------------------------------------------------------------------
# _format_status_block
# ---------------------------------------------------------------------------


def test_format_status_block_counts_and_pending():
    block = cli_mod._format_status_block(
        run_id="r1",
        suite="s",
        contest="c",
        simulator="gazebo",
        env_n=2,
        headless=False,
        created_at="when",
        steps_total=6,
        ok=2,
        partial=1,
        failed=1,
        skipped=1,
        in_flight=1,
        active=[],
        failed_steps=[],
    )
    assert "steps: 6    ok: 2  partial: 1  failed: 1  skipped: 1  in_flight: 1  pending: 0" in block


def test_format_status_block_active_with_and_without_started():
    block = cli_mod._format_status_block(
        run_id="r1",
        suite="s",
        contest="c",
        simulator="gazebo",
        env_n=1,
        headless=True,
        created_at="when",
        steps_total=2,
        ok=0,
        partial=0,
        failed=0,
        skipped=0,
        in_flight=2,
        active=[("p/s1", "30s ago"), ("p/s2", None)],
        failed_steps=[],
    )
    assert "active:" in block
    assert "p/s1 (started 30s ago)" in block
    assert "p/s2" in block


def test_format_status_block_failed_with_unknown_kind():
    block = cli_mod._format_status_block(
        run_id="r1",
        suite="s",
        contest="c",
        simulator="gazebo",
        env_n=1,
        headless=False,
        created_at="when",
        steps_total=1,
        ok=0,
        partial=0,
        failed=1,
        skipped=0,
        in_flight=0,
        active=[],
        failed_steps=[("p/s1", None, None)],
    )
    assert "failed:" in block
    assert "p/s1: unknown: " in block


def test_format_status_block_failed_with_kind_and_detail():
    block = cli_mod._format_status_block(
        run_id="r1",
        suite="s",
        contest="c",
        simulator="",
        env_n=1,
        headless=False,
        created_at="when",
        steps_total=1,
        ok=0,
        partial=0,
        failed=1,
        skipped=0,
        in_flight=0,
        active=[],
        failed_steps=[("p/s1", "internal", "boom")],
    )
    assert "p/s1: internal: boom" in block


# ---------------------------------------------------------------------------
# _ago
# ---------------------------------------------------------------------------


def test_ago_none():
    assert cli_mod._ago(None) is None


def test_ago_seconds_minutes_hours():
    now = __import__("time").time()
    assert cli_mod._ago(now - 30) == "30s ago"
    assert cli_mod._ago(now - 300) == "5m ago"
    assert cli_mod._ago(now - 7200) == "2h ago"


# ---------------------------------------------------------------------------
# _status_from_disk / _cmd_status
# ---------------------------------------------------------------------------


def test_status_from_disk_active_and_failed(tmp_path: pathlib.Path):
    _make_run(
        tmp_path,
        "r1",
        state_steps={
            "p/s0": _sr("p/s0", "ok"),  # neither active nor failed
            "p/s1": StepResult("p/s1", "in_progress", None, 100.0, None, None, None),
            "p/s2": StepResult("p/s2", "failed", 1, 0.0, 1.0, StepErrorKind.INTERNAL, "boom"),
        },
    )
    block = cli_mod._status_from_disk(tmp_path, "r1")
    assert "run: r1" in block
    assert "p/s2: internal: boom" in block
    assert "active:" in block and "p/s1" in block
    assert "pending:" in block
    assert "p/s0" not in block  # ok steps render nowhere


def test_cmd_status_no_watch(tmp_path: pathlib.Path, capsys):
    _make_run(tmp_path, "r1")
    rc = cli_mod._cmd_status(_args(data_root=str(tmp_path), run_id="r1", watch=False))
    out = capsys.readouterr().out
    assert rc == 0
    assert "run: r1" in out
    assert "suite/contest: basic/basic" in out


def test_cmd_status_no_run_id_resolves_latest(tmp_path: pathlib.Path, capsys):
    _make_run(tmp_path, "r2")
    _make_run(tmp_path, "r1")
    rc = cli_mod._cmd_status(_args(data_root=str(tmp_path), run_id=None, watch=False))
    assert "run: r2" in capsys.readouterr().out
    assert rc == 0


# -- status --watch, driven through in-memory stub modules (imports stay lazy) --


def _install_watch_stubs(monkeypatch) -> types.SimpleNamespace:
    """Stub rclpy / arena_evaluation_msgs / arena_rclpy_mixins so the --watch
    branch runs without a ROS graph. run_main() drives one state message."""
    state = types.SimpleNamespace(msgs=[])

    qos_mod = types.SimpleNamespace(
        QoSProfile=lambda **kw: kw,
        DurabilityPolicy=types.SimpleNamespace(TRANSIENT_LOCAL="transient"),
        ReliabilityPolicy=types.SimpleNamespace(RELIABLE="reliable"),
    )
    rclpy_stub = types.SimpleNamespace(qos=qos_mod)

    msg_mod = types.ModuleType("arena_evaluation_msgs.msg")
    msg_mod.BenchmarkState = type("BenchmarkState", (), {})

    mixin_mod = types.ModuleType("arena_rclpy_mixins")

    class FakeNode:
        def __init__(self, name: str):
            self.name = name
            self.subscriptions = []

        def create_subscription(self, msg_type, topic, cb, qos):  # noqa: ARG001
            self.subscriptions.append((topic, qos))

    mixin_mod.ArenaMixinNode = FakeNode

    def run_main(node_cls):
        # _WatchNode.__init__ takes no args; it passes the name to the base.
        node = node_cls()
        msg = msg_mod.BenchmarkState()
        msg.run_id = "r1"
        msg.suite = "suite_x"
        msg.contest = "contest_y"
        msg.simulator = "gazebo"
        msg.env_n = 2
        msg.headless = True
        msg.steps_total = 4
        msg.steps_done = 2
        msg.steps_partial = 0
        msg.steps_failed = 0
        msg.steps_skipped = 0
        msg.steps_in_flight = 1
        msg.active_keys = ["p/s1"]
        node._on_state(msg)
        state.msgs.append(msg)

    mixin_mod.run_main = run_main

    monkeypatch.setitem(sys.modules, "rclpy", rclpy_stub)
    monkeypatch.setitem(sys.modules, "arena_evaluation_msgs.msg", msg_mod)
    monkeypatch.setitem(sys.modules, "arena_rclpy_mixins", mixin_mod)
    return state


def test_cmd_status_watch_stubbed(tmp_path: pathlib.Path, capsys, monkeypatch):
    _install_watch_stubs(monkeypatch)
    rc = cli_mod._cmd_status(_args(data_root=str(tmp_path), run_id=None, watch=True))
    out = capsys.readouterr().out
    assert rc == 0
    assert "run: r1" in out
    assert "suite/contest: suite_x/contest_y" in out
    assert "steps: 4" in out
    assert "p/s1" in out  # active_keys rendered


def test_cmd_status_watch_keyboard_interrupt(tmp_path: pathlib.Path, capsys, monkeypatch):
    state = _install_watch_stubs(monkeypatch)

    def run_main_kb(node_cls):  # noqa: ARG001
        raise KeyboardInterrupt

    monkeypatch.setattr(sys.modules["arena_rclpy_mixins"], "run_main", run_main_kb)
    assert state is not None
    rc = cli_mod._cmd_status(_args(data_root=str(tmp_path), run_id=None, watch=True))
    assert rc == 0
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# _cmd_tail
# ---------------------------------------------------------------------------


def test_cmd_tail_waits_for_csv_then_tails(tmp_path: pathlib.Path, capsys, monkeypatch):
    run_path = _make_run(tmp_path, "r1")
    csv_path = run_path / "progress.csv"
    csv_path.unlink()  # simulate a run whose csv does not exist yet
    calls: list[tuple] = []

    def _fake_sleep(secs: float) -> None:  # noqa: ARG001
        csv_path.write_text("ts_iso,...\n")

    def _fake_run(cmd, check=False):
        calls.append((list(cmd), check))

    monkeypatch.setattr(cli_mod.time, "sleep", _fake_sleep)
    monkeypatch.setattr(cli_mod.subprocess, "run", _fake_run)

    rc = cli_mod._cmd_tail(_args(data_root=str(tmp_path), run_id="r1"))
    assert rc == 0
    assert "progress.csv not yet created" in capsys.readouterr().out
    assert calls == [(["tail", "-n", "50", "-F", str(csv_path)], False)]


def test_cmd_tail_keyboard_interrupt(tmp_path: pathlib.Path, monkeypatch):
    _make_run(tmp_path, "r1")

    def _fake_run(cmd, check=False):  # noqa: ARG001
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_mod.subprocess, "run", _fake_run)
    rc = cli_mod._cmd_tail(_args(data_root=str(tmp_path), run_id="r1"))
    assert rc == 0


# ---------------------------------------------------------------------------
# _cmd_ps
# ---------------------------------------------------------------------------


def test_cmd_ps_no_processes(capsys, monkeypatch):
    # _cmd_ps re-imports running_processes at call time from .debug
    monkeypatch.setattr("arena_evaluation.benchmark.debug.running_processes", lambda: [])
    rc = cli_mod._cmd_ps(_args(data_root=None))
    assert rc == 0
    assert "no arena processes running" in capsys.readouterr().out


def test_cmd_ps_prints_rows_and_elapsed_formats(capsys, monkeypatch):
    monkeypatch.setattr(
        "arena_evaluation.benchmark.debug.running_processes",
        lambda: [
            {"pid": 7, "kind": "benchmark_runner", "elapsed_s": None, "command": "eval benchmark r1"},
            {"pid": 12, "kind": "simulation", "elapsed_s": 12.0, "command": "gz sim"},
            {"pid": 305, "kind": "arena_node", "elapsed_s": 305.0, "command": "arena_node"},
            {"pid": 3900, "kind": "arena_cli", "elapsed_s": 3900.0, "command": "arena_cli x"},
        ],
    )
    rc = cli_mod._cmd_ps(_args(data_root=None))
    out = capsys.readouterr().out
    assert rc == 0
    assert "PID" in out and "KIND" in out and "ELAPSED" in out
    assert "?" in out  # None elapsed
    assert "12s" in out
    assert "5m05s" in out
    assert "1h05m" in out
    assert "benchmark_runner" in out


# ---------------------------------------------------------------------------
# _cmd_console
# ---------------------------------------------------------------------------


def test_cmd_console_resolves_latest_run(monkeypatch, tmp_path: pathlib.Path, capsys):
    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path))
    # _data_root() = $ARENA_DATA_DIR/benchmarks; _resolve_run sorts reverse,
    # so the lexicographically-largest run name wins
    _make_run(tmp_path / "benchmarks", "a_older")
    _make_run(tmp_path / "benchmarks", "z_newest")
    rc = cli_mod._cmd_console(_args(data_root=None, run_id=None, lines=200, follow=False))
    out = capsys.readouterr().out
    assert rc == 1
    # resolution came from the manifest of the most recent run
    assert "no console log for run 'z_newest'" in out


def test_cmd_console_log_missing(capsys, monkeypatch):
    # _cmd_console re-imports tail_console at call time from .debug
    monkeypatch.setattr(
        "arena_evaluation.benchmark.debug.tail_console",
        lambda run_id, lines: {
            "run_id": run_id,
            "path": "/logs/runner.log",
            "exists": False,
            "pid": None,
            "alive": False,
            "truncated": False,
            "lines": [],
        },
    )
    rc = cli_mod._cmd_console(_args(data_root=None, run_id="r1", lines=200, follow=False))
    out = capsys.readouterr().out
    assert rc == 1
    assert "no console log for run 'r1'" in out
    assert "hint: the benchmark writes runner.log" in out


def test_cmd_console_prints_log(capsys, monkeypatch):
    monkeypatch.setattr(
        "arena_evaluation.benchmark.debug.tail_console",
        lambda run_id, lines: {
            "run_id": run_id,
            "path": "/logs/runner.log",
            "exists": True,
            "pid": 42,
            "alive": True,
            "truncated": False,
            "lines": ["line1", "line2"],
        },
    )
    rc = cli_mod._cmd_console(_args(data_root=None, run_id="r1", lines=200, follow=False))
    out = capsys.readouterr().out
    assert rc == 0
    assert "run: r1  [running]  pid: 42  log: /logs/runner.log" in out
    assert "line1" in out and "line2" in out


def test_cmd_console_truncated_message(capsys, monkeypatch):
    """Pins current truncation notice. NOTE: it reports args.lines twice
    instead of the actual line count — suspected source bug."""
    monkeypatch.setattr(
        "arena_evaluation.benchmark.debug.tail_console",
        lambda run_id, lines: {
            "run_id": run_id,
            "path": "/logs/runner.log",
            "exists": True,
            "pid": 1,
            "alive": False,
            "truncated": True,
            "lines": ["a", "b", "c"],
        },
    )
    rc = cli_mod._cmd_console(_args(data_root=None, run_id="r1", lines=2, follow=False))
    out = capsys.readouterr().out
    assert rc == 0
    assert "(last 2 of 2 lines, use --lines to see more)" in out


def test_cmd_console_follow_until_runner_exits(capsys, monkeypatch):
    state = {"n": 0}

    def _tail(run_id, lines):  # noqa: ARG001
        state["n"] += 1
        if state["n"] == 1:
            return {"run_id": run_id, "path": "/logs/runner.log", "exists": True, "pid": 7, "alive": True, "truncated": False, "lines": ["a", "b"]}
        if state["n"] == 2:
            return {"run_id": run_id, "path": "/logs/runner.log", "exists": False, "pid": 7, "alive": True, "truncated": False, "lines": []}
        return {"run_id": run_id, "path": "/logs/runner.log", "exists": True, "pid": 7, "alive": False, "truncated": False, "lines": ["a", "b", "c"]}

    monkeypatch.setattr("arena_evaluation.benchmark.debug.tail_console", _tail)
    monkeypatch.setattr(cli_mod.time, "sleep", lambda secs: None)

    rc = cli_mod._cmd_console(_args(data_root=None, run_id="r1", lines=200, follow=True))
    out = capsys.readouterr().out
    assert rc == 0
    assert out.count("console log:") == 1  # header printed once
    assert out.count("a\n") >= 1
    assert out.count("b\n") >= 1
    assert out.count("c\n") == 1
    assert out.rstrip().endswith("runner exited")


def test_cmd_console_follow_runner_died_log_missing(capsys, monkeypatch):
    """Runner exits while the log file is gone: still reports runner exited."""
    monkeypatch.setattr(
        "arena_evaluation.benchmark.debug.tail_console",
        lambda run_id, lines: {
            "run_id": run_id,
            "path": "/logs/runner.log",
            "exists": False,
            "pid": None,
            "alive": False,
            "truncated": False,
            "lines": [],
        },
    )
    monkeypatch.setattr(cli_mod.time, "sleep", lambda secs: None)
    rc = cli_mod._cmd_console(_args(data_root=None, run_id="r1", lines=200, follow=True))
    out = capsys.readouterr().out
    assert rc == 0
    assert "console log: /logs/runner.log" in out
    assert out.rstrip().endswith("runner exited")


# ---------------------------------------------------------------------------
# main() dispatch
# ---------------------------------------------------------------------------


def test_main_list_command(tmp_path: pathlib.Path, capsys):
    _make_run(tmp_path, "r1")
    rc = cli_mod.main(["list", "--data-root", str(tmp_path)])
    assert rc == 0
    assert "r1" in capsys.readouterr().out


def test_main_status_command(tmp_path: pathlib.Path, capsys):
    _make_run(tmp_path, "r1")
    rc = cli_mod.main(["status", "--data-root", str(tmp_path), "r1"])
    assert rc == 0
    assert "run: r1" in capsys.readouterr().out


def test_main_ps_command_no_processes(capsys, monkeypatch):
    monkeypatch.setattr("arena_evaluation.benchmark.debug.running_processes", lambda: [])
    assert cli_mod.main(["ps"]) == 0
    assert "no arena processes running" in capsys.readouterr().out


def test_main_unknown_command_raises_systemexit():
    with pytest.raises(SystemExit):
        cli_mod.main(["frobnicate"])


def test_main_tail_missing_run_re_raises_systemexit(tmp_path: pathlib.Path):
    """SystemExit from inside a command handler is re-raised by main()"""
    with pytest.raises(SystemExit, match="no run found"):
        cli_mod.main(["tail", "--data-root", str(tmp_path), "ghost"])


def test_main_no_args_raises_systemexit():
    with pytest.raises(SystemExit):
        cli_mod.main([])


def test_main_without_argv_uses_sys_argv(tmp_path: pathlib.Path, capsys, monkeypatch):
    _make_run(tmp_path, "r1")
    monkeypatch.setattr(sys, "argv", ["evaluation_cli", "list", "--data-root", str(tmp_path)])
    assert cli_mod.main() == 0
    assert "r1" in capsys.readouterr().out


def test_main_errors_go_to_stderr_return_1(monkeypatch, tmp_path: pathlib.Path, capsys):
    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path))
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "benchmarks" / "no-manifest").mkdir()
    rc = cli_mod.main(["console"])
    assert rc == 1
    assert "error:" in capsys.readouterr().err


def test_main_kill_no_processes(capsys, monkeypatch):
    monkeypatch.setattr("arena_evaluation.benchmark.debug.kill_processes", lambda **kwargs: [])
    assert cli_mod.main(["kill"]) == 0
    assert "no running arena processes found" in capsys.readouterr().out


def test_main_kill_with_processes(capsys, monkeypatch):
    killed = [
        {"pid": 1234, "kind": "benchmark_runner", "command": "python benchmark ...", "status": "killed"},
        {"pid": 5678, "kind": "simulation", "command": "gz sim ...", "status": "force_killed"},
    ]
    monkeypatch.setattr("arena_evaluation.benchmark.debug.kill_processes", lambda **kwargs: killed)
    assert cli_mod.main(["kill", "1234", "5678", "-9"]) == 0
    out = capsys.readouterr().out
    assert "[killed] pid 1234 (benchmark_runner)" in out
    assert "[force_killed] pid 5678 (simulation)" in out
