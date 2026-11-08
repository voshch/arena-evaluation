"""Live benchmark dashboard: live.json from the progress display, run detail, HTTP API, events, port reuse and child lifetime."""

from __future__ import annotations

import datetime
import json
import logging
import os
import pathlib
import socket
import subprocess
import sys
import threading
import time
import types
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator

import pytest
from arena_runtime.env.task_layer import encode_modes

from arena_evaluation.benchmark import dashboard, runner
from arena_evaluation.benchmark.progress_display import BenchmarkProgressDisplay
from arena_evaluation.benchmark.runner import cli_main
from arena_evaluation.benchmark.state import Manifest, RunDir
from arena_evaluation.benchmark.step import StepErrorKind, StepResult

SUITE = "{stages: [{name: s1, episodes: 1, robot: jackal, map: map_empty, tm_robots: random, tm_obstacles: random}]}"
RUN_ID = "20261106-120000-basic-basic"


def _manifest(created_at: str) -> Manifest:
    stage = {"name": "s1", "episodes": 2, "robot": "jackal", "map": "map_empty"}
    return Manifest(
        run_id=RUN_ID,
        created_at=created_at,
        arena_git_sha=None,
        arena_git_dirty=False,
        cli_args=[],
        env_n=2,
        headless=True,
        config_hash="0" * 40,
        simulator="flatsim",
        scale_episodes=1.0,
        suite_name="basic",
        contest_name="basic",
        suite={"stages": [stage]},
        contest=[{"name": "dwb"}, {"name": "mppi"}, {"name": "teb"}],
        steps=[{"key": f"{name}/s1", "contestant": {"name": name}, "stage": stage, "episodes_planned": 2} for name in ("dwb", "mppi", "teb")],
    )


def _record(outcome: int, info: str = "") -> types.SimpleNamespace:
    return types.SimpleNamespace(
        world="map_empty",
        seed=1,
        modes=encode_modes({"robots": ["random"], "obstacles": ["random"], "modules": []}),
        robots=["jackal"],
        outcome_state=outcome,
        outcome_info=info,
        params=[],
        goal_dist_start=5.0,
        goal_dist_min=0.2,
        path_length=5.5,
        integrity=[],
    )


@pytest.fixture
def run_dir(tmp_path: pathlib.Path) -> RunDir:
    run = RunDir.create(tmp_path, RUN_ID, _manifest((datetime.datetime.now(tz=datetime.UTC) - datetime.timedelta(hours=1)).isoformat()))
    run.state.write(
        {
            "dwb/s1": StepResult(key="dwb/s1", status="ok", env_id=0, started_at=1.0, ended_at=2.0, error_kind=None, error_detail=None, episodes_run=2, episodes_total=2),
            "mppi/s1": StepResult(key="mppi/s1", status="failed", env_id=1, started_at=1.0, ended_at=2.0, error_kind=StepErrorKind.ENV_SETUP, error_detail="spawn timed out", episodes_total=2),
        }
    )
    now = datetime.datetime.now(tz=datetime.UTC)
    for episode_id, outcome in ((0, 2), (1, 3)):
        run.progress.append(
            ts_iso=(now - datetime.timedelta(seconds=30 - episode_id)).isoformat(),
            run_id=RUN_ID,
            step_key="dwb/s1",
            contestant="dwb",
            stage="s1",
            env_id=0,
            episode_id=episode_id,
            episode_record=_record(outcome, "collision" if outcome == 3 else ""),
            started_at=0.0,
            ended_at=4.0,
        )
    run.progress.close()
    run.log_path.write_text("line one\nline two\nline three\n")
    return run


def _live(run: RunDir, updated_at: float, finished: bool = False) -> None:
    run.live_path.write_text(json.dumps({"run_id": RUN_ID, "lane": None, "pid": os.getpid(), "updated_at": updated_at, "started_at": updated_at - 60, "finished": finished, "sim_time": 12.5, "env_n": 2, "total_steps": 3, "steps": {"completed": 2}, "slots": []}))


@pytest.fixture
def server(run_dir: RunDir) -> Iterator[dashboard.DashboardServer]:
    srv = dashboard.DashboardServer(("127.0.0.1", 0), run_dir.path.parent)
    thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join()


def _get(srv: dashboard.DashboardServer, path: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{srv.server_address[1]}{path}", timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_progress_display_writes_live_slots_and_a_finished_snapshot(tmp_path: pathlib.Path):
    live = tmp_path / "live.p1.json"
    display = BenchmarkProgressDisplay("t", total_steps=4, env_n=2, run_id=RUN_ID, sim_now=lambda: 42.0, live_path=live, lane="p1")
    with display:
        display.update_slot(0, env_id=3, contestant="dwb", stage="s1", step_key="dwb/s1", ep_idx=0, ep_total=2, sim_start=40.0)
        running = json.loads(live.read_text())
        display.log_step_completed("dwb/s1", "ok", "dwb", "s1", 2, 2, 0, 1.0)
        display.clear_slot(0)
        cleared = json.loads(live.read_text())
    finished = json.loads(live.read_text())

    assert running["lane"] == "p1" and running["pid"] == os.getpid() and running["sim_time"] == 42.0 and not running["finished"]
    assert running["slots"] == [{"slot": 0, "env_id": 3, "contestant": "dwb", "stage": "s1", "step_key": "dwb/s1", "ep_idx": 0, "ep_total": 2, "state": "RUNNING", "started_at": running["slots"][0]["started_at"], "sim_start": 40.0}]
    assert cleared["slots"] == [] and cleared["steps"]["ok"] == 1
    assert finished["finished"] and finished["slots"] == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["live.p1.json"]


def test_progress_display_heartbeat_refreshes_live_json(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("arena_evaluation.benchmark.progress_display.LIVE_HEARTBEAT_S", 0.05)
    live = tmp_path / "live.json"
    with BenchmarkProgressDisplay("t", total_steps=1, env_n=1, run_id=RUN_ID, live_path=live):
        first = json.loads(live.read_text())["updated_at"]
        deadline = time.time() + 5
        while json.loads(live.read_text())["updated_at"] == first and time.time() < deadline:
            time.sleep(0.02)
        assert json.loads(live.read_text())["updated_at"] > first


def test_run_detail_has_the_grid_failures_episodes_and_stale_lanes(run_dir: RunDir):
    _live(run_dir, time.time() - 60)
    detail = dashboard.run_detail(run_dir.path)

    assert [(c["contestant"], c["stage"], c["world"], c["status"], c["episodes_run"]) for c in detail["grid"]] == [("dwb", "s1", "map_empty", "ok", 2), ("mppi", "s1", "map_empty", "failed", 0), ("teb", "s1", "map_empty", "pending", 0)]
    assert detail["state"] == "incomplete"
    assert detail["failed"] == [{"key": "mppi/s1", "error_kind": "env_setup", "error_detail": "spawn timed out"}]
    assert detail["episodes"] == {"logged": 2, "planned": 6, "outcomes": {"success": 1, "failed": 1}, "per_min": 0.2, "rate_window_s": 600, "eta_s": None}
    assert [(r["episode_id"], r["outcome"], r["outcome_info"]) for r in detail["recent"]] == [("1", "failed", "collision"), ("0", "success", "")]
    assert [(lane["file"], lane["stale"]) for lane in detail["live"]] == [("live.json", True)]


def test_a_step_busy_in_a_live_lane_shows_in_progress_until_the_state_records_it(run_dir: RunDir):
    slot = {"slot": 0, "env_id": 4, "contestant": "teb", "stage": "s1", "step_key": "teb/s1", "ep_idx": 0, "ep_total": 2, "state": "RUNNING", "started_at": time.time(), "sim_start": 1.0}
    done = slot | {"slot": 1, "contestant": "dwb", "step_key": "dwb/s1"}
    _live(run_dir, time.time())
    run_dir.live_path.write_text(json.dumps(json.loads(run_dir.live_path.read_text()) | {"slots": [slot, done]}))
    detail = dashboard.run_detail(run_dir.path)

    assert [(c["key"], c["status"], c["env_id"]) for c in detail["grid"]] == [("dwb/s1", "ok", 0), ("mppi/s1", "failed", 1), ("teb/s1", "in_progress", 4)]
    assert detail["steps"] == {"ok": 1, "partial": 0, "failed": 1, "skipped": 0, "in_progress": 1, "pending": 0}


def test_rate_window_shrinks_to_a_young_runs_age_and_floors_at_a_minute(run_dir: RunDir):
    created = datetime.datetime.fromisoformat(run_dir.manifest.created_at).timestamp()

    young = dashboard.run_detail(run_dir.path, now=created + 120)["episodes"]
    newborn = dashboard.run_detail(run_dir.path, now=created + 5)["episodes"]

    assert young["rate_window_s"] == 120
    assert newborn["rate_window_s"] == 60


def test_manifest_is_parsed_once_until_the_file_changes(run_dir: RunDir):
    first = dashboard.load_manifest(run_dir.path)
    assert dashboard.load_manifest(run_dir.path) is first

    manifest_file = run_dir.path / "manifest.yaml"
    manifest_file.write_text(manifest_file.read_text().replace("suite_name: basic", "suite_name: renamed"))
    assert dashboard.load_manifest(run_dir.path).suite_name == "renamed"


def test_unreadable_manifest_is_a_run_status_error(tmp_path: pathlib.Path):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "manifest.yaml").write_text("run_id: [unclosed\n")
    with pytest.raises(dashboard.RunStatusError, match="has no readable manifest.yaml"):
        dashboard.run_status(tmp_path / "broken")


def test_finished_lane_is_never_stale(run_dir: RunDir):
    _live(run_dir, time.time() - 3600, finished=True)
    assert [lane["stale"] for lane in dashboard.live_lanes(run_dir.path)] == [False]


def test_progress_filters_to_one_step(server: dashboard.DashboardServer, run_dir: RunDir):
    rows = json.loads(_get(server, f"/api/runs/{RUN_ID}/progress?step=dwb/s1")[1])
    none = json.loads(_get(server, f"/api/runs/{RUN_ID}/progress?step=mppi/s1")[1])
    assert rows["n_rows"] == 2 and {row["step_key"] for row in rows["rows"]} == {"dwb/s1"}
    assert none == {"n_rows": 0, "rows": []}


def test_api_serves_identity_runs_detail_live_progress_log_and_page(server: dashboard.DashboardServer, run_dir: RunDir):
    _live(run_dir, time.time())
    base = f"/api/runs/{RUN_ID}"

    identity = json.loads(_get(server, "/api/identity")[1])
    runs = json.loads(_get(server, "/api/runs")[1])["runs"]
    detail = json.loads(_get(server, base)[1])
    live = json.loads(_get(server, f"{base}/live")[1])["live"]
    progress = json.loads(_get(server, f"{base}/progress?last=1")[1])
    logs = json.loads(_get(server, f"{base}/log?tail=2")[1])["logs"]
    status, page = _get(server, f"/runs/{RUN_ID}")

    assert identity["app"] == dashboard.APP and identity["data_root"] == str(run_dir.path.parent.resolve())
    assert [(r["run_id"], r["steps_total"], r["live"]) for r in runs] == [(RUN_ID, 3, [{"lane": None, "stale": False, "finished": False}])]
    assert detail["run_id"] == RUN_ID and len(detail["grid"]) == 3
    assert [lane["sim_time"] for lane in live] == [12.5]
    assert progress["n_rows"] == 2 and [row["episode_id"] for row in progress["rows"]] == ["1"]
    assert logs == {"runner.log": ["line two", "line three"]}
    assert status == 200 and b"<title>Arena Benchmarks</title>" in page


def test_runtime_endpoints_need_the_runtime_flag_and_the_control_header(server: dashboard.DashboardServer):
    url = f"http://127.0.0.1:{server.server_address[1]}/api/runtime/hold"
    forged = urllib.request.Request(url, data=b"{}", method="POST")
    allowed = urllib.request.Request(url, data=b"{}", headers={"X-Arena-Dashboard": "1"}, method="POST")
    answers = []
    for request in (forged, allowed):
        try:
            urllib.request.urlopen(request, timeout=5)
        except urllib.error.HTTPError as exc:
            answers.append((exc.code, json.loads(exc.read())))
    status, body = _get(server, "/api/runtime")

    assert answers[0][0] == 403 and "X-Arena-Dashboard: 1" in answers[0][1]["error"]
    assert answers[1][0] == 404 and answers[1][1]["hint"] == "start it with arena evaluation serve --runtime"
    assert status == 404 and json.loads(body)["error"] == "this dashboard serves no runtime page"
    assert json.loads(_get(server, "/api/identity")[1])["runtime"] is False


def test_stopping_a_run_needs_the_control_header_and_the_runtime_flag(server: dashboard.DashboardServer, run_dir: RunDir):
    url = f"http://127.0.0.1:{server.server_address[1]}/api/runs/{RUN_ID}/stop"
    forged = urllib.request.Request(url, data=b"{}", method="POST")
    allowed = urllib.request.Request(url, data=b"{}", headers={"X-Arena-Dashboard": "1"}, method="POST")
    answers = []
    for request in (forged, allowed):
        try:
            urllib.request.urlopen(request, timeout=5)
        except urllib.error.HTTPError as exc:
            answers.append((exc.code, json.loads(exc.read())))

    assert answers[0][0] == 403 and "X-Arena-Dashboard: 1" in answers[0][1]["error"]
    assert answers[1][0] == 404 and answers[1][1]["hint"] == "start it with arena evaluation serve --runtime"


@pytest.mark.parametrize("path", ["/api/runs/nope", "/api/runs/..", "/api/runs/%2E%2E", f"/api/runs/{RUN_ID}/nope", "/api/nope"])
def test_api_answers_unknown_runs_and_endpoints_with_404(server: dashboard.DashboardServer, path: str):
    status, body = _get(server, path)
    assert status == 404 and "error" in json.loads(body)


def test_events_name_live_and_run_changes(server: dashboard.DashboardServer, run_dir: RunDir, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dashboard, "EVENT_POLL_S", 0.05)
    with urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}/api/runs/{RUN_ID}/events", timeout=5) as stream:
        _live(run_dir, time.time())
        assert stream.readline() == b"event: live\n"
        assert stream.readline() == b"data: {}\n"
        assert stream.readline() == b"\n"
        with run_dir.log_path.open("a") as fh:
            fh.write("line four\n")
        assert stream.readline() == b"event: run\n"


def test_port_problem_reuses_our_dashboard_and_refuses_anything_else(server: dashboard.DashboardServer, run_dir: RunDir, tmp_path: pathlib.Path):
    port = server.server_address[1]
    other = tmp_path / "other"
    other.mkdir()
    with socket.socket() as foreign:
        foreign.bind(("127.0.0.1", 0))
        foreign.listen()
        foreign_port = foreign.getsockname()[1]
        foreign_problem = dashboard.port_problem("127.0.0.1", foreign_port, run_dir.path.parent)

    assert dashboard.port_problem("127.0.0.1", port, run_dir.path.parent) is None
    assert dashboard.port_problem("127.0.0.1", _free_port(), run_dir.path.parent) is None
    assert f"port {port} serves another data root ({run_dir.path.parent.resolve()})" in (dashboard.port_problem("127.0.0.1", port, other) or "")
    assert foreign_problem == f"port {foreign_port} is held by another program, pick a free one with --port (--viz-port on benchmark)"


def test_serve_on_our_dashboards_port_prints_the_url_and_returns(server: dashboard.DashboardServer, run_dir: RunDir, capsys: pytest.CaptureFixture[str]):
    port = server.server_address[1]
    assert dashboard.serve("127.0.0.1", port, run_dir.path.parent, RUN_ID) == 0
    captured = capsys.readouterr()
    assert captured.out == f"viz: http://127.0.0.1:{port}/runs/{RUN_ID}\n"
    assert "reusing the dashboard already serving" in captured.err


def test_child_serves_until_stopped_and_until_its_watched_process_exits(run_dir: RunDir):
    url, child = dashboard.start_child("127.0.0.1", _free_port(), run_dir.path.parent, RUN_ID)
    try:
        assert url.endswith(f"/runs/{RUN_ID}")
        with urllib.request.urlopen(url.rsplit("/runs/", 1)[0] + "/api/identity", timeout=5) as response:
            assert json.loads(response.read())["pid"] == child.pid
    finally:
        dashboard.stop_child(child)
    assert child.returncode is not None

    watched = subprocess.Popen(["sleep", "30"])
    port = _free_port()
    orphan = subprocess.Popen([sys.executable, "-m", "arena_evaluation.benchmark.cli", "serve", "--port", str(port), "--data-root", str(run_dir.path.parent), "--exit-with", str(watched.pid)], stdout=subprocess.PIPE, text=True)
    try:
        assert orphan.stdout is not None and orphan.stdout.readline() == f"viz: http://127.0.0.1:{port}/\n"
        watched.kill()
        watched.wait()
        assert orphan.wait(10) == 0
    finally:
        dashboard.stop_child(orphan)


def test_port_zero_prints_the_bound_port_and_stdin_eof_stops_the_dashboard(tmp_path: pathlib.Path):
    proc = subprocess.Popen([sys.executable, "-m", "arena_evaluation.benchmark.cli", "serve", "--port", "0", "--exit-on-eof", "--data-root", str(tmp_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout is not None and proc.stdin is not None
        line = proc.stdout.readline()
        port = urllib.parse.urlsplit(line.removeprefix(dashboard.VIZ_PREFIX).strip()).port
        assert line.startswith(dashboard.VIZ_PREFIX) and port
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/identity", timeout=5) as response:
            assert json.loads(response.read())["pid"] == proc.pid
        proc.stdin.close()
        assert proc.wait(10) == 0
    finally:
        dashboard.stop_child(proc)


def test_child_on_a_foreign_port_raises(run_dir: RunDir):
    with socket.socket() as foreign:
        foreign.bind(("127.0.0.1", 0))
        foreign.listen()
        with pytest.raises(dashboard.DashboardError, match="exited rc=2 before it listened"):
            dashboard.start_child("127.0.0.1", foreign.getsockname()[1], run_dir.path.parent)


@pytest.fixture
def restored_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    runner_level, runner_propagate = runner._log.level, runner._log.propagate
    try:
        yield
    finally:
        root.handlers = handlers
        root.setLevel(level)
        runner._log.setLevel(runner_level)
        runner._log.propagate = runner_propagate


@pytest.mark.usefixtures("restored_logging")
def test_runner_viz_on_a_foreign_port_exits_2_before_creating_the_run(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]):
    with socket.socket() as foreign:
        foreign.bind(("127.0.0.1", 0))
        foreign.listen()
        port = foreign.getsockname()[1]
        assert cli_main(["--data-root", str(tmp_path), "--suite", SUITE, "--contest", "[]", "--viz", "--viz-port", str(port)]) == 2
    assert f"--viz: port {port} is held by another program" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []
