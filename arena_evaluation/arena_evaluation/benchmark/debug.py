"""Process introspection and benchmark console logs.

Shared by the evaluation CLI (``benchmark ps`` / ``benchmark console``) and
the ``arena_evaluation_mcp`` server (``list_running_processes`` /
``get_benchmark_console``). The console log is the runner's own
``benchmarks/<run_id>/runner.log``, tailed in place.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import time
import typing

# Command-line markers used to classify arena-related processes.
_BENCHMARK_RUNNER_MARKERS = (
    "evaluation benchmark",
    "lib/arena_evaluation/benchmark",
)
_ARENA_CLI_MARKERS = ("arena_cli", "uv run")
_SIM_MARKERS = ("gz sim", "ign gazebo", "gazebo", "gzserver")
_NODE_MARKERS = ("arena_node", "data_recorder", "recorder_node")
_GEN_MARKERS = ("world_generator",)

_KIND_ORDER = {
    "benchmark_runner": 0,
    "arena_cli": 1,
    "simulation": 2,
    "arena_node": 3,
    "world_generator": 4,
}


def _ps_lines() -> list[tuple[int, str]]:
    """(pid, cmdline) pairs for all processes, excluding self and grep-ish."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            capture_output=True, text=True, timeout=15,
        ).stdout
    except Exception:
        return []
    rows: list[tuple[int, str]] = []
    me = os.getpid()
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid_s, args = parts
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if pid == me:
            continue
        if args.startswith(("pkill ", "ps ", "grep ")):
            continue
        rows.append((pid, args))
    return rows


def _elapsed_s(pid: int) -> float | None:
    """Seconds since process start (via /proc/<pid>/stat starttime and /proc/uptime)."""
    try:
        with open("/proc/uptime", "r") as fh:
            uptime = float(fh.read().split()[0])
        with open(f"/proc/{pid}/stat", "r") as fh:
            fields = fh.read().split()
        start_ticks = int(fields[21])
        clk = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        start_sec = start_ticks / clk
        return max(0.0, uptime - start_sec)
    except Exception:
        return None


def _classify(cmdline: str) -> str | None:
    if any(m in cmdline for m in _BENCHMARK_RUNNER_MARKERS):
        return "benchmark_runner"
    if any(m in cmdline for m in _GEN_MARKERS):
        return "world_generator"
    if any(m in cmdline for m in _SIM_MARKERS):
        return "simulation"
    if any(m in cmdline for m in _NODE_MARKERS):
        return "arena_node"
    if any(m in cmdline for m in _ARENA_CLI_MARKERS):
        return "arena_cli"
    return None


def is_pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def running_processes() -> list[dict]:
    """All arena-related processes on the system, categorized.

    Returns list of dicts: pid, kind, elapsed_s, command.
    Benchmark runners are listed first, then by start time.
    """
    rows: list[dict] = []
    for pid, cmdline in _ps_lines():
        kind = _classify(cmdline)
        if kind is None:
            continue
        rows.append({
            "pid": pid,
            "kind": kind,
            "elapsed_s": _elapsed_s(pid),
            "command": cmdline[:200],
        })
    rows.sort(key=lambda r: (_KIND_ORDER.get(r["kind"], 9), r["pid"]))
    return rows


def _runner_starttime(pid: int) -> float:
    """Start timestamp (seconds) of a process; 0 on failure."""
    try:
        with open(f"/proc/{pid}/stat", "r") as fh:
            fields = fh.read().split()
        start_ticks = int(fields[21])
        clk = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        return start_ticks / clk
    except Exception:
        return 0.0


def find_runner_pid(run_id: str | None = None) -> int | None:
    """PID of a running benchmark runner, optionally filtered by run_id.

    With ``run_id=None`` returns the most recently started runner.
    """
    found: list[tuple[float, int]] = []
    for pid, cmdline in _ps_lines():
        if _classify(cmdline) != "benchmark_runner":
            continue
        if run_id and run_id not in cmdline:
            continue
        found.append((_runner_starttime(pid), pid))
    if not found:
        return None
    found.sort(reverse=True)  # most recently started first
    return found[0][1]


def running_pids_by_run_id() -> dict[str, int]:
    """Map run_id -> runner PID for all running benchmark runners.

    The run_id is parsed from the runner's ``--run-id <id>`` argument.
    Runners launched without an explicit run id are matched by searching
    the run_id string anywhere in the command line (it appears in the
    ``--run-id`` / data-dir path arguments).
    """
    result: dict[str, int] = {}
    for pid, cmdline in _ps_lines():
        if _classify(cmdline) != "benchmark_runner":
            continue
        match = None
        idx = cmdline.find("--run-id")
        if idx != -1:
            rest = cmdline[idx + len("--run-id"):].lstrip()
            match = rest.split()[0].strip("'\"") if rest else None
        if not match:
            # Fallback: any run-id-looking token (timestamp-suite-contest)
            for token in cmdline.split():
                if len(token) > 10 and "-" in token and "/" not in token:
                    match = token
                    break
        if match:
            result[match] = pid
    return result


def console_log_path(run_id: str) -> pathlib.Path:
    """Path of the console log for a benchmark run."""
    from arena_evaluation.storage.data_root import benchmarks_root

    return benchmarks_root() / run_id / "runner.log"


def tail_console(run_id: str, lines: int = 200) -> dict:
    """Tail the console log of a benchmark run.

    Returns: run_id, path, exists, pid, alive, truncated, lines (list).
    """
    path = console_log_path(run_id)
    pid = find_runner_pid(run_id)
    alive = is_pid_alive(pid)
    if not path.exists():
        return {
            "run_id": run_id,
            "path": str(path),
            "exists": False,
            "pid": pid,
            "alive": alive,
            "truncated": False,
            "lines": [],
        }
    try:
        raw = path.read_bytes().splitlines()
    except Exception as exc:
        return {
            "run_id": run_id,
            "path": str(path),
            "exists": True,
            "pid": pid,
            "alive": alive,
            "truncated": False,
            "lines": [],
            "error": str(exc),
        }
    content = [ln.decode("utf-8", errors="replace") for ln in raw]
    truncated = len(content) > lines
    return {
        "run_id": run_id,
        "path": str(path),
        "exists": True,
        "pid": pid,
        "alive": alive,
        "truncated": truncated,
        "lines": content[-lines:] if lines > 0 else content,
    }


def latest_running_run_id() -> str | None:
    """Run id of the most recently started benchmark runner, if any."""
    pid = find_runner_pid(None)
    if pid is None:
        return None
    pids = running_pids_by_run_id()
    # The pid-level map might not match (no --run-id); fall back to a
    # cmdline scan for the newest runner's run id.
    for pid2, cmdline in _ps_lines():
        if pid2 != pid:
            continue
        for token in cmdline.split():
            if token.startswith("--run-id"):
                continue
            if len(token) > 10 and "-" in token and "/" not in token:
                return token
    return None


def kill_processes(
    pids: list[int] | None = None,
    force: bool = False,
    kind: str | None = None,
    timeout_s: float = 1.5,
) -> list[dict]:
    """Terminate running arena processes."""
    import signal

    my_pid = os.getpid()
    all_procs = running_processes()
    target_procs: list[dict] = []

    if pids:
        pid_set = set(pids)
        for p in all_procs:
            if p["pid"] in pid_set and p["pid"] != my_pid:
                target_procs.append(p)
        known_pids = {p["pid"] for p in target_procs}
        for pid in pids:
            if pid not in known_pids and pid != my_pid and is_pid_alive(pid):
                target_procs.append({"pid": pid, "kind": "unknown", "command": f"pid {pid}"})
    else:
        for p in all_procs:
            if p["pid"] != my_pid:
                if kind is None or p["kind"] == kind:
                    target_procs.append(p)

    results: list[dict] = []
    for proc in target_procs:
        pid = proc["pid"]
        if not is_pid_alive(pid):
            results.append({**proc, "status": "already_dead"})
            continue

        try:
            if force:
                os.kill(pid, signal.SIGKILL)
                results.append({**proc, "status": "force_killed"})
            else:
                os.kill(pid, signal.SIGTERM)
                start_wait = time.time()
                dead = False
                while time.time() - start_wait < timeout_s:
                    if not is_pid_alive(pid):
                        dead = True
                        break
                    time.sleep(0.1)
                if dead:
                    results.append({**proc, "status": "killed"})
                else:
                    os.kill(pid, signal.SIGKILL)
                    results.append({**proc, "status": "force_killed"})
        except (ProcessLookupError, PermissionError) as exc:
            if not is_pid_alive(pid):
                results.append({**proc, "status": "killed"})
            else:
                results.append({**proc, "status": f"failed: {exc}"})

    return results
