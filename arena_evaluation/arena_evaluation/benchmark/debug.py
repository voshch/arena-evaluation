"""Process introspection, process stopping and benchmark console logs.

Shared by the evaluation CLI (``ps`` / ``kill`` / ``console``) and the
evaluation MCP domain (``evaluation_ps`` / ``evaluation_kill`` /
``evaluation_console``). The console log is the runner's own
``benchmarks/<run_id>/runner.log``, ``runner.<lane>.log`` for a lane of a
shared run, tailed in place.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import time

# Command-line markers used to classify arena-related processes.
_BENCHMARK_RUNNER_MARKERS = (
    "evaluation benchmark",
    "lib/arena_evaluation/benchmark",
)
_ARENA_CLI_MARKERS = ("arena_cli",)
_SIM_MARKERS = ("gz sim", "ign gazebo", "gzserver")
_SIM_EXECUTABLES = ("gazebo",)
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
            ["ps", "-ww", "-eo", "pid=,args="],
            capture_output=True,
            text=True,
            timeout=15,
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
        with open("/proc/uptime") as fh:
            uptime = float(fh.read().split()[0])
        with open(f"/proc/{pid}/stat") as fh:
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
    if any(m in cmdline for m in _SIM_MARKERS) or os.path.basename(cmdline.split(" ", 1)[0]) in _SIM_EXECUTABLES:
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
    except (ProcessLookupError, PermissionError):
        return False
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


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
        rows.append(
            {
                "pid": pid,
                "kind": kind,
                "elapsed_s": _elapsed_s(pid),
                "command": cmdline[:200],
            }
        )
    rows.sort(key=lambda r: (_KIND_ORDER.get(r["kind"], 9), r["pid"]))
    return rows


def _runner_starttime(pid: int) -> float:
    """Start timestamp (seconds) of a process; 0 on failure."""
    try:
        with open(f"/proc/{pid}/stat") as fh:
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
            rest = cmdline[idx + len("--run-id") :].lstrip()
            match = rest.split()[0].strip("'\"") if rest else None
        if not match:
            # Fallback: any run-id-looking token (timestamp-suite-contest)
            for token in cmdline.split():
                if len(token) > 10 and "-" in token and "/" not in token and not token.startswith("-"):
                    match = token
                    break
        if match:
            result[match] = pid
    return result


def console_log_path(run_id: str, lane: str | None = None) -> pathlib.Path:
    """Path of the console log for a benchmark run, or for one lane of a shared run."""
    from arena_evaluation.storage.data_root import benchmarks_root

    return benchmarks_root() / run_id / ("runner.log" if lane is None else f"runner.{lane}.log")


def tail_console(run_id: str, lines: int = 200, lane: str | None = None) -> dict:
    """Tail the console log of a benchmark run.

    Returns: run_id, path, exists, pid, alive, truncated, lines (list).
    """
    path = console_log_path(run_id, lane)
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
            if token.startswith("-"):
                continue
            if len(token) > 10 and "-" in token and "/" not in token:
                return token
    return None


def kill_processes(
    pids: list[int] | None = None,
    force: bool = False,
    kind: str | None = None,
    everything: bool = False,
    grace_s: float = 15.0,
) -> list[dict]:
    """Stop the arena processes named by pid, kind or everything: SIGINT first, SIGTERM and SIGKILL only past each grace window."""
    if not pids and kind is None and not everything:
        raise ValueError("name the processes to stop: pids, a kind, or all of them")
    protected = _ancestry(os.getpid())
    arena = {proc["pid"]: proc for proc in running_processes()}
    targets: list[dict] = []
    results: list[dict] = []
    if pids:
        for pid in pids:
            if pid in protected:
                results.append({"pid": pid, "kind": "caller", "command": f"pid {pid}", "status": "refused: this process or one of its parents"})
            elif pid in arena:
                targets.append(arena[pid])
            elif is_pid_alive(pid):
                results.append({"pid": pid, "kind": "unknown", "command": f"pid {pid}", "status": "refused: not an arena process"})
            else:
                results.append({"pid": pid, "kind": "unknown", "command": f"pid {pid}", "status": "already_dead"})
    else:
        targets = [proc for proc in arena.values() if proc["pid"] not in protected and (everything or proc["kind"] == kind)]
    for proc in targets:
        results.append({**proc, "status": _stop(proc["pid"], force=force, grace_s=grace_s)})
    return results


def _ancestry(pid: int) -> set[int]:
    chain: set[int] = set()
    while pid > 1 and pid not in chain:
        chain.add(pid)
        try:
            with open(f"/proc/{pid}/stat") as fh:
                pid = int(fh.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return chain


def _stop(pid: int, *, force: bool, grace_s: float) -> str:
    """Signal the process, or its whole group when it leads one, escalating only while it survives."""
    import signal

    try:
        leads_group = os.getpgid(pid) == pid
    except ProcessLookupError:
        return "already_dead"

    def send(sig: signal.Signals) -> None:
        if leads_group:
            os.killpg(pid, sig)
        else:
            os.kill(pid, sig)

    steps = ((signal.SIGKILL, 2.0, "force_killed"),) if force else ((signal.SIGINT, grace_s, "stopped"), (signal.SIGTERM, 5.0, "terminated"), (signal.SIGKILL, 2.0, "force_killed"))
    for sig, wait, outcome in steps:
        try:
            send(sig)
        except ProcessLookupError:
            return "already_dead"
        except PermissionError as exc:
            return f"failed: {exc}"
        deadline = time.time() + wait
        while time.time() < deadline:
            if not is_pid_alive(pid):
                return outcome
            time.sleep(0.1)
    return "survived SIGKILL"
