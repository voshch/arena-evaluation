"""Live web dashboard of the benchmark runs under a data root, read from their run dirs."""

from __future__ import annotations

import argparse
import collections
import contextlib
import datetime
import errno
import http.server
import json
import os
import pathlib
import re
import socket
import subprocess
import sys
import threading
import time
import typing
import urllib.parse
import urllib.request

if typing.TYPE_CHECKING:
    from .dashboard_runtime import RuntimeBridge
    from .state import Manifest

APP = "arena-evaluation-dashboard"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
VIZ_PREFIX = "viz: "
TERMINAL_STEPS = ("ok", "failed", "partial", "skipped")
STALE_S = 10.0
EVENT_POLL_S = 1.0
KEEPALIVE_S = 15.0
RATE_WINDOW_S = 600.0
MIN_RATE_WINDOW_S = 60.0
RECENT_ROWS = 30
LOG_TAIL_BYTES = 262144
MAX_BODY_BYTES = 65536
CONTROL_HEADER = "X-Arena-Dashboard"
OUTCOMES = {"0": "queued", "1": "running", "2": "success", "3": "failed", "4": "skipped", "5": "fatal"}
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*$")
_WILDCARD_HOSTS = ("", "0.0.0.0", "::")
_PAGE = pathlib.Path(__file__).with_name("dashboard.html")
_MANIFESTS: dict[pathlib.Path, tuple[tuple[int, int], Manifest]] = {}
_MANIFESTS_LOCK = threading.Lock()


class RunStatusError(ValueError):
    """A run dir whose manifest cannot be read."""


class RunNotFoundError(LookupError):
    """A run id with no run dir under the data root."""


class DashboardError(RuntimeError):
    """The dashboard server could not start."""


class ControlError(Exception):
    """A runtime page request answered with an HTTP error status, a message and an optional next step."""

    def __init__(self, status: int, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.hint = hint


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    try:
        stat = pathlib.Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def load_manifest(path: pathlib.Path) -> Manifest:
    """A run dir's manifest, parsed once per manifest.yaml mtime and size."""
    import yaml

    from .state import Manifest

    file = path / "manifest.yaml"
    try:
        stat = file.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        with _MANIFESTS_LOCK:
            cached = _MANIFESTS.get(file)
        if cached is not None and cached[0] == key:
            return cached[1]
        manifest = Manifest.from_yaml(file.read_text())
    except (OSError, TypeError, yaml.YAMLError) as exc:
        raise RunStatusError(f"run {path.name!r} has no readable manifest.yaml: {exc}") from exc
    with _MANIFESTS_LOCK:
        _MANIFESTS[file] = (key, manifest)
    return manifest


def runner_pids(path: pathlib.Path, run_id: str, running: dict[str, int] | None = None) -> list[int]:
    """Live runner processes of a run: lane pid files and runners started with its --run-id."""
    from .debug import running_pids_by_run_id

    pids = set()
    for pid_file in path.glob("runner.*.pid"):
        with contextlib.suppress(OSError, ValueError):
            pid = int(pid_file.read_text().strip())
            if _alive(pid):
                pids.add(pid)
    found = (running_pids_by_run_id() if running is None else running).get(run_id)
    if found is not None:
        pids.add(found)
    return sorted(pids)


def run_status(path: pathlib.Path, *, running: dict[str, int] | None = None, light: bool = False) -> dict[str, typing.Any]:
    """A run's progress from its manifest, the union of its lane states and, unless light, its progress rows."""
    from .cli import _count_by_status
    from .state import StateFile, progress_rows

    manifest = load_manifest(path)
    steps = StateFile.open(path).steps
    planned = [step["key"] for step in manifest.steps]
    counts = _count_by_status(steps)
    pids = runner_pids(path, manifest.run_id, running)
    done = all(key in steps and steps[key].status in TERMINAL_STEPS for key in planned)
    lanes = sorted({name.split(".")[1] for pattern in ("progress.*.csv", ".benchmark_state.*.json", "runner.*.log") for name in (entry.name.lstrip(".") for entry in path.glob(pattern))})
    return {
        "run_id": manifest.run_id,
        "run_dir": str(path),
        "suite": manifest.suite_name,
        "contest": manifest.contest_name,
        "simulator": manifest.simulator,
        "env_n": manifest.env_n,
        "headless": manifest.headless,
        "created_at": manifest.created_at,
        "scale_episodes": manifest.scale_episodes,
        "launch_args": manifest.launch_args,
        "state": "completed" if done else "running" if pids else "incomplete",
        "steps_total": len(planned),
        "steps": counts | {"pending": len(planned) - sum(counts.values())},
        "active": [{"key": step.key, "env_id": step.env_id, "started_at": step.started_at} for step in steps.values() if step.status == "in_progress"],
        "failed": [{"key": step.key, "error_kind": None if step.error_kind is None else step.error_kind.value, "error_detail": step.error_detail} for step in steps.values() if step.status == "failed"],
        "episodes_logged": None if light else len(progress_rows(path)),
        "lanes": lanes,
        "runner_pids": pids,
        "has_combined_metrics": (path / "combined_metrics.parquet").is_file(),
        "has_report": (path / "report.html").is_file(),
    }


def live_lanes(path: pathlib.Path, now: float | None = None) -> list[dict[str, typing.Any]]:
    """The live slot tables runners write, one per lane, with their age and whether they went stale."""
    now = time.time() if now is None else now
    lanes = []
    for live in sorted(path.glob("live*.json")):
        try:
            data = json.loads(live.read_text())
        except (OSError, ValueError):
            continue
        age = max(now - data["updated_at"], 0.0)
        lanes.append(data | {"file": live.name, "age_s": round(age, 1), "stale": not data["finished"] and age > STALE_S})
    return lanes


def _wall(ts_iso: str) -> float | None:
    try:
        return datetime.datetime.fromisoformat(ts_iso).timestamp()
    except ValueError:
        return None


def run_detail(path: pathlib.Path, now: float | None = None) -> dict[str, typing.Any]:
    """Everything the run page shows: status, the step grid, live lanes, episode counts, rate and ETA, recent episodes."""
    from .state import StateFile, progress_rows

    now = time.time() if now is None else now
    status = run_status(path, light=True)
    manifest = load_manifest(path)
    steps = StateFile.open(path).steps
    live = live_lanes(path, now)
    busy = {slot["step_key"]: slot["env_id"] for lane in live if not lane["finished"] and not lane["stale"] for slot in lane["slots"]}
    grid = []
    for planned in manifest.steps:
        key = planned["key"]
        contestant, _, stage = key.partition("/")
        result = steps.get(key)
        running = key in busy and (result is None or result.status not in TERMINAL_STEPS)
        grid.append(
            {
                "key": key,
                "contestant": contestant,
                "stage": stage,
                "world": planned["stage"]["map"],
                "episodes_planned": planned["episodes_planned"],
                "status": "in_progress" if running else "pending" if result is None else result.status,
                "env_id": busy[key] if running else None if result is None else result.env_id,
                "episodes_run": 0 if result is None else result.episodes_run,
                "episodes_failed": 0 if result is None else result.episodes_failed,
                "error_kind": None if running or result is None or result.error_kind is None else result.error_kind.value,
            }
        )
    counts = collections.Counter(cell["status"] for cell in grid)
    rows = progress_rows(path)
    outcomes = collections.Counter(OUTCOMES.get(row["outcome_state"], row["outcome_state"]) for row in rows)
    created = _wall(manifest.created_at)
    window = min(max(now - created, MIN_RATE_WINDOW_S), RATE_WINDOW_S) if created is not None else RATE_WINDOW_S
    recent_walls = [wall for wall in (_wall(row["ts_iso"]) for row in rows) if wall is not None and now - wall <= window]
    rate = len(recent_walls) / (window / 60.0)
    planned_total = sum(cell["episodes_planned"] for cell in grid)
    remaining = max(planned_total - len(rows), 0)
    return status | {
        "now": now,
        "steps": {name: counts[name] for name in status["steps"]},
        "grid": grid,
        "live": live,
        "episodes": {
            "logged": len(rows),
            "planned": planned_total,
            "outcomes": dict(outcomes),
            "per_min": round(rate, 2),
            "rate_window_s": round(window),
            "eta_s": round(remaining / rate * 60.0) if rate > 0 and status["state"] == "running" else None,
        },
        "recent": [
            {
                "ts_iso": row["ts_iso"],
                "step_key": row["step_key"],
                "env_id": row["env_id"],
                "episode_id": row["episode_id"],
                "world": row["world"],
                "outcome": OUTCOMES.get(row["outcome_state"], row["outcome_state"]),
                "outcome_info": row["outcome_info"],
                "runtime_s": row["runtime_s"],
                "error_kind": row["error_kind"],
            }
            for row in rows[-RECENT_ROWS:][::-1]
        ],
    }


def list_runs(data_root: pathlib.Path, limit: int = 100) -> list[dict[str, typing.Any]]:
    """Light statuses of the newest runs under data_root, newest first, unreadable run dirs left out."""
    from .debug import running_pids_by_run_id

    if not data_root.is_dir():
        return []
    running = running_pids_by_run_id()
    runs = []
    manifests = [p / "manifest.yaml" for p in data_root.iterdir() if (p / "manifest.yaml").is_file()]
    for path in [m.parent for m in sorted(manifests, key=lambda m: m.stat().st_mtime, reverse=True)[:limit]]:
        try:
            status = run_status(path, running=running, light=True)
        except RunStatusError:
            continue
        runs.append({key: status[key] for key in ("run_id", "suite", "contest", "simulator", "created_at", "state", "steps_total", "steps", "lanes")} | {"live": [{"lane": lane["lane"], "stale": lane["stale"], "finished": lane["finished"]} for lane in live_lanes(path)]})
    return runs


def log_tail(path: pathlib.Path, lines: int) -> dict[str, list[str]]:
    """The last lines of runner.log and of every lane's runner.<lane>.log."""
    tails = {}
    for log in sorted(path.glob("runner*.log")):
        with log.open("rb") as fh:
            fh.seek(max(log.stat().st_size - LOG_TAIL_BYTES, 0))
            text = fh.read().decode(errors="replace")
        tails[log.name] = text.splitlines()[-lines:]
    return tails


def _signature(path: pathlib.Path) -> tuple[tuple[tuple[str, int, int], ...], tuple[tuple[str, int, int], ...]]:
    live, run = [], []
    with contextlib.suppress(FileNotFoundError):
        for entry in os.scandir(path):
            if not entry.is_file() or entry.name.endswith(".tmp"):
                continue
            stat = entry.stat()
            (live if entry.name.startswith("live") and entry.name.endswith(".json") else run).append((entry.name, stat.st_mtime_ns, stat.st_size))
    return tuple(sorted(live)), tuple(sorted(run))


def public_host(host: str) -> str:
    return socket.gethostname() if host in _WILDCARD_HOSTS else host


def url(host: str, port: int, run_id: str | None = None) -> str:
    return f"http://{public_host(host)}:{port}/" + (f"runs/{urllib.parse.quote(run_id)}" if run_id else "")


class DashboardServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], data_root: pathlib.Path, runtime: RuntimeBridge | None = None) -> None:
        self.data_root = data_root.resolve()
        self.runtime = runtime
        self.stopping = threading.Event()
        super().__init__(address, _Handler)

    def run_dir(self, run_id: str) -> pathlib.Path:
        path = self.data_root / run_id
        if not _RUN_ID_RE.match(run_id) or not (path / "manifest.yaml").is_file():
            raise RunNotFoundError(f"no run {run_id!r} in {self.data_root}")
        return path

    def identity(self) -> dict[str, typing.Any]:
        return {"app": APP, "data_root": str(self.data_root), "pid": os.getpid(), "runtime": self.runtime is not None}

    def shutdown(self) -> None:
        self.stopping.set()
        super().shutdown()


class _Handler(http.server.BaseHTTPRequestHandler):
    @property
    def dashboard(self) -> DashboardServer:
        return typing.cast(DashboardServer, self.server)

    def log_message(self, format: str, *args: typing.Any) -> None:
        pass

    def do_GET(self) -> None:
        target = urllib.parse.urlsplit(self.path)
        parts = [urllib.parse.unquote(part) for part in target.path.split("/") if part]
        query = urllib.parse.parse_qs(target.query)
        try:
            if not parts or parts[0] in ("runs", "runtime"):
                self._send(200, "text/html; charset=utf-8", _PAGE.read_bytes())
            elif parts == ["api", "identity"]:
                self._json(self.dashboard.identity())
            elif parts == ["api", "runs"]:
                self._json({"data_root": str(self.dashboard.data_root), "runs": list_runs(self.dashboard.data_root)})
            elif len(parts) >= 3 and parts[:2] == ["api", "runs"]:
                self._run(self.dashboard.run_dir(parts[2]), parts[3:], query)
            elif parts[:2] == ["api", "runtime"]:
                self._runtime_get(parts[2:], query)
            else:
                self._json({"error": f"no endpoint {target.path}"}, 404)
        except ControlError as exc:
            self._json({"error": exc.message, "hint": exc.hint}, exc.status)
        except RunNotFoundError as exc:
            self._json({"error": str(exc)}, 404)
        except RunStatusError as exc:
            self._json({"error": str(exc)}, 409)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self) -> None:
        parts = [urllib.parse.unquote(part) for part in urllib.parse.urlsplit(self.path).path.split("/") if part]
        try:
            if self.headers.get(CONTROL_HEADER) != "1":
                raise ControlError(403, f"control requests need the {CONTROL_HEADER}: 1 header", "the dashboard page sends it, other origins cannot")
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                raise ControlError(413, f"request body over {MAX_BODY_BYTES} bytes")
            body = json.loads(self.rfile.read(length) or b"{}") if length else {}
            if not isinstance(body, dict):
                raise ControlError(400, "the request body must be a JSON object")
            bridge = self._bridge()
            if parts == ["api", "runs"]:
                self._json(bridge.start_benchmark(self.dashboard.data_root, body))
            elif len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "stop":
                self._json(bridge.stop_run(self.dashboard.run_dir(parts[2]).name))
            elif parts == ["api", "runtime", "shutdown"]:
                self._json(shutdown_runtime())
            elif len(parts) == 3 and parts[:2] == ["api", "runtime"]:
                self._json(bridge.act(parts[2], None, body))
            elif len(parts) == 5 and parts[:3] == ["api", "runtime", "envs"]:
                self._json(bridge.act(parts[4], _env_id(parts[3]), body))
            else:
                raise ControlError(404, f"no control endpoint /{'/'.join(parts)}")
        except ControlError as exc:
            self._json({"error": exc.message, "hint": exc.hint}, exc.status)
        except RunNotFoundError as exc:
            self._json({"error": str(exc)}, 404)
        except ValueError as exc:
            self._json({"error": f"the request body is not JSON: {exc}"}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _bridge(self) -> RuntimeBridge:
        if self.dashboard.runtime is None:
            raise ControlError(404, "this dashboard serves no runtime page", "start it with arena evaluation serve --runtime")
        return self.dashboard.runtime

    def _runtime_get(self, rest: list[str], query: dict[str, list[str]]) -> None:
        bridge = self._bridge()
        if not rest:
            self._json(bridge.snapshot())
        elif rest == ["options"]:
            self._json(bridge.options())
        elif rest == ["catalog"]:
            self._json(bridge.catalog(query.get("kind", [""])[0], query.get("world", [""])[0]))
        elif len(rest) == 3 and rest[0] == "envs" and rest[2] == "camera.png":
            self._send(200, "image/png", bridge.camera(_env_id(rest[1])))
        elif len(rest) == 3 and rest[0] == "envs" and rest[2] == "task":
            self._json(bridge.env_task(_env_id(rest[1])))
        else:
            raise ControlError(404, f"no runtime endpoint {'/'.join(rest)}")

    def _run(self, path: pathlib.Path, rest: list[str], query: dict[str, list[str]]) -> None:
        if not rest:
            self._json(run_detail(path))
        elif rest == ["live"]:
            self._json({"now": time.time(), "live": live_lanes(path)})
        elif rest == ["progress"]:
            from .state import progress_rows

            rows = progress_rows(path)
            if "step" in query:
                rows = [row for row in rows if row["step_key"] == query["step"][0]]
            last = _int(query, "last", 200)
            self._json({"n_rows": len(rows), "rows": rows[-last:]})
        elif rest == ["log"]:
            self._json({"logs": log_tail(path, _int(query, "tail", 200))})
        elif rest == ["events"]:
            self._events(path)
        else:
            self._json({"error": f"no endpoint {'/'.join(rest)} on a run"}, 404)

    def _events(self, path: pathlib.Path) -> None:
        last_live, last_run = _signature(path)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        sent = time.monotonic()
        while not self.dashboard.stopping.wait(EVENT_POLL_S):
            live, run = _signature(path)
            events = [name for name, changed in (("run", run != last_run), ("live", live != last_live)) if changed]
            last_live, last_run = live, run
            if events:
                self.wfile.write("".join(f"event: {name}\ndata: {{}}\n\n" for name in events).encode())
            elif time.monotonic() - sent >= KEEPALIVE_S:
                self.wfile.write(b": keepalive\n\n")
            else:
                continue
            self.wfile.flush()
            sent = time.monotonic()

    def _json(self, body: object, code: int = 200) -> None:
        self._send(code, "application/json", json.dumps(body, default=str).encode())

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _env_id(text: str) -> int:
    if not text.isdigit():
        raise ControlError(404, f"env id {text!r} is not a number")
    return int(text)


def _int(query: dict[str, list[str]], name: str, default: int) -> int:
    try:
        return max(int(query[name][0]), 1)
    except (KeyError, ValueError):
        return default


def probe(host: str, port: int) -> dict[str, typing.Any] | None:
    """The identity of the dashboard listening on host:port, None when the port holds something else."""
    target = "127.0.0.1" if host in _WILDCARD_HOSTS else host
    try:
        with urllib.request.urlopen(f"http://{target}:{port}/api/identity", timeout=2.0) as response:
            identity = json.loads(response.read())
    except (OSError, ValueError):
        return None
    return identity if isinstance(identity, dict) and identity.get("app") == APP else None


def shutdown_runtime() -> dict[str, typing.Any]:
    """Stop this container's runtime with the arena CLI, which owns it; this dashboard is not part of that process tree."""
    try:
        done = subprocess.run(["arena", "runtime", "down"], capture_output=True, text=True, timeout=180, check=False)
    except OSError as exc:
        raise ControlError(502, f"cannot run arena runtime down: {exc}", "run the dashboard inside an arena container") from exc
    if done.returncode != 0:
        raise ControlError(409, f"arena runtime down failed: {(done.stderr or done.stdout).strip()[:200]}", "the runtime may already be down")
    return {"shutdown": True}


def _held_problem(host: str, port: int, data_root: pathlib.Path) -> str | None:
    held = probe(host, port)
    if held is None:
        return f"port {port} is held by another program, pick a free one with --port (--viz-port on benchmark)"
    if held["data_root"] != str(data_root):
        return f"port {port} serves another data root ({held['data_root']}), pick a free one with --port (--viz-port on benchmark)"
    return None


def port_problem(host: str, port: int, data_root: pathlib.Path) -> str | None:
    """Why the dashboard cannot serve data_root on host:port, None when the port is free or a dashboard of data_root holds it."""
    with socket.socket(socket.AF_INET) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                return f"cannot listen on {host}:{port}: {exc.strerror}"
            return _held_problem(host, port, data_root.resolve())
    return None


def _exit_with(server: DashboardServer, pid: int) -> None:
    while _alive(pid):
        if server.stopping.wait(1.0):
            return
    server.shutdown()


def _exit_on_eof(server: DashboardServer) -> None:
    for _ in sys.stdin:
        pass
    server.shutdown()


def serve(host: str, port: int, data_root: pathlib.Path, run_id: str | None = None, exit_with: int | None = None, *, runtime: bool = False, exit_on_eof: bool = False) -> int:
    """Serve until interrupted, exit_with is gone or stdin closes (exit_on_eof), printing `viz: <url>` with the bound port once listening; reuse a dashboard of the same data root already on the port."""
    data_root = data_root.resolve()
    try:
        server = DashboardServer((host, port), data_root)
    except OSError as exc:
        problem = f"cannot listen on {host}:{port}: {exc.strerror}" if exc.errno != errno.EADDRINUSE else _held_problem(host, port, data_root)
        if problem is not None:
            print(f"dashboard: {problem}", file=sys.stderr)
            return 2
        print(f"dashboard: reusing the dashboard already serving {data_root} on port {port}", file=sys.stderr)
        print(VIZ_PREFIX + url(host, port, run_id), flush=True)
        return 0
    if runtime:
        from .dashboard_runtime import RuntimeBridge

        server.runtime = RuntimeBridge()
    bound = server.server_address[1]
    print(VIZ_PREFIX + (url(host, bound, run_id) if run_id or not runtime else f"{url(host, bound)}runtime"), flush=True)
    if exit_with is not None:
        threading.Thread(target=_exit_with, args=(server, exit_with), name="dashboard-exit-with", daemon=True).start()
    if exit_on_eof:
        threading.Thread(target=_exit_on_eof, args=(server,), name="dashboard-exit-on-eof", daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.stopping.set()
        server.server_close()
        if server.runtime is not None:
            server.runtime.close()
    return 0


def start_child(host: str, port: int, data_root: pathlib.Path, run_id: str | None = None) -> tuple[str, subprocess.Popen[str]]:
    """Start the dashboard as a child process and return its URL once it listens or reuses a running one."""
    argv = [sys.executable, "-m", "arena_evaluation.benchmark.cli", "serve", "--host", host, "--port", str(port), "--data-root", str(data_root), "--exit-with", str(os.getpid()), *(["--run-id", run_id] if run_id else [])]
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    line = proc.stdout.readline()
    proc.stdout.close()
    if not line.startswith(VIZ_PREFIX):
        raise DashboardError(f"dashboard exited rc={proc.wait()} before it listened on port {port}")
    return line[len(VIZ_PREFIX) :].strip(), proc


def stop_child(proc: subprocess.Popen[str], grace_s: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(grace_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"address to listen on (default {DEFAULT_HOST}, 0.0.0.0 for every interface)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port to listen on (default {DEFAULT_PORT}, 0 picks a free one and prints it); a dashboard of the same data root already there is reused, anything else exits 2")
    parser.add_argument("--data-root", default=None, metavar="PATH", help="benchmarks root to serve (default $ARENA_DATA_DIR/benchmarks)")
    parser.add_argument("--run-id", default=None, help="run whose page the printed URL opens (default the run list)")
    parser.add_argument("--runtime", action="store_true", help="also serve the runtime page (/runtime): the envs of the runtime on this ROS domain, their episodes and camera, spawn, despawn, reset, pause, step and lockstep controls")
    parser.add_argument("--exit-with", type=int, default=None, metavar="PID", help="stop serving once process PID is gone (benchmark --viz passes its own pid)")
    parser.add_argument("--exit-on-eof", action="store_true", help="stop serving once stdin closes (arena web runs each target's dashboard this way, so it ends with the hub's exec session)")


def serve_args(args: argparse.Namespace) -> int:
    from arena_evaluation.storage.data_root import benchmarks_root

    return serve(args.host, args.port, pathlib.Path(args.data_root) if args.data_root else benchmarks_root(), args.run_id, args.exit_with, runtime=args.runtime, exit_on_eof=args.exit_on_eof)
