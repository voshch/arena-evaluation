"""The dashboard's runtime page served by a real dashboard process on an isolated ROS domain: no runtime, input checks, and the benchmark lock."""

from __future__ import annotations

import json
import os
import pathlib
import select
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest

pytest.importorskip("arena_mcp")
pytest.importorskip("rclpy")

RUNNER = r"""
import rclpy
from arena_evaluation_msgs.msg import BenchmarkState
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

rclpy.init()
node = rclpy.create_node("benchmark_runner_stand_in")
latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
publisher = node.create_publisher(BenchmarkState, "/arena/benchmark/state", latched)
publisher.publish(BenchmarkState(run_id="r-owner", steps_total=2))
print("ready", flush=True)
input()
node.destroy_node()
rclpy.shutdown()
"""


SIM_STAND_IN = r"""
import rclpy
from arena_runtime.mcp import SIM_STATE
from arena_runtime_msgs.msg import SimState
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

rclpy.init()
node = rclpy.create_node("half_a_runtime_stand_in")
latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
publisher = node.create_publisher(SimState, SIM_STATE, latched)
publisher.publish(SimState(alive=True))
print("ready", flush=True)
input()
node.destroy_node()
rclpy.shutdown()
"""


def _ros_envs() -> Iterator[dict[str, str]]:
    """Candidate ROS domains for the test, away from the session's own; a shared host may run anything on any of them."""
    offset = int(os.environ.get("ROS_DOMAIN_ID", "0")) * 7 + os.getpid()
    for step in range(8):
        yield {**os.environ, "ROS_DOMAIN_ID": str(180 + (offset + step * 5) % 52), "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST"}


def _wait_for_line(proc: subprocess.Popen[str], expected: str, timeout: float = 20.0) -> None:
    """Read the child's stdout until a line equals `expected`, failing with every line seen when the deadline passes."""
    assert proc.stdout is not None
    fd = proc.stdout.fileno()
    seen: list[str] = []
    pending = b""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], deadline - time.monotonic())
        chunk = os.read(fd, 4096) if ready else b""
        if ready and not chunk:
            break
        pending += chunk
        *lines, pending = pending.split(b"\n")
        seen += [line.decode(errors="replace") for line in lines]
        if expected in seen:
            return
    pytest.fail(f"no {expected!r} line from the child within {timeout:.0f}s, it printed: {seen!r} then {pending!r}")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


BLOCK_VIEW_TOOLS = "import sys; sys.modules['rerun_utils.mcp'] = None; from arena_evaluation.benchmark.cli import main; raise SystemExit(main(sys.argv[1:]))"


def _serve(tmp_path: pathlib.Path, launcher: list[str]) -> Iterator[tuple[str, dict[str, str]]]:
    for env in _ros_envs():
        port = _free_port()
        proc = subprocess.Popen([sys.executable, *launcher, "serve", "--runtime", "--port", str(port), "--data-root", str(tmp_path)], stdout=subprocess.PIPE, text=True, env=env)
        try:
            _wait_for_line(proc, f"viz: http://127.0.0.1:{port}/runtime")
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runtime", timeout=60) as response:
                busy = json.loads(response.read())["running"]
            if not busy:
                yield f"http://127.0.0.1:{port}", env
                return
        finally:
            proc.terminate()
            assert proc.wait(30) is not None
    pytest.fail("every candidate ROS domain has a runtime on it")


@pytest.fixture
def dashboard(tmp_path: pathlib.Path) -> Iterator[tuple[str, dict[str, str]]]:
    """A dashboard with --runtime on the first candidate domain where no runtime answers."""
    yield from _serve(tmp_path, ["-m", "arena_evaluation.benchmark.cli"])


@pytest.fixture
def dashboard_without_view_tools(tmp_path: pathlib.Path) -> Iterator[tuple[str, dict[str, str]]]:
    """A dashboard whose process cannot import rerun_utils.mcp."""
    yield from _serve(tmp_path, ["-c", BLOCK_VIEW_TOOLS])


def _call(url: str, body: object | None = None, header: bool = True) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **({"X-Arena-Dashboard": "1"} if header else {})}, method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_runtime_page_reports_no_runtime_and_refuses_controls(dashboard: tuple[str, dict[str, str]]):
    base, env = dashboard

    status, identity = _call(f"{base}/api/identity")
    snapshot_status, snapshot = _call(f"{base}/api/runtime")
    hold_status, hold = _call(f"{base}/api/runtime/hold", {})
    forged_status, forged = _call(f"{base}/api/runtime/hold", {}, header=False)
    count_status, count = _call(f"{base}/api/runtime/spawn", {"count": 99})
    args_status, args = _call(f"{base}/api/runtime/spawn", {"launch_args": {"robot": 3}})
    unknown_status, unknown = _call(f"{base}/api/runtime/explode", {})
    misrouted_status, misrouted = _call(f"{base}/api/runtime/envs/1/hold", {})
    env_status, env_error = _call(f"{base}/api/runtime/envs/x/reset", {})

    assert status == 200 and identity["runtime"] is True
    assert snapshot_status == 200 and snapshot["running"] is False and snapshot["domain"] == env["ROS_DOMAIN_ID"] and snapshot["envs"] == []
    assert hold_status == 409 and "not available" in hold["error"]
    assert forged_status == 403 and "X-Arena-Dashboard" in forged["error"]
    assert count_status == 400 and "count must be an integer from 1 to 16" in count["error"]
    assert args_status == 400 and "launch_args" in args["error"]
    assert unknown_status == 404 and "spawn" in unknown["hint"]
    assert misrouted_status == 404 and misrouted["error"] == "hold is a runtime action"
    assert env_status == 404 and "is not a number" in env_error["error"]


def test_options_seed_and_benchmark_starts_are_checked(dashboard: tuple[str, dict[str, str]], tmp_path: pathlib.Path):
    base, _ = dashboard
    (tmp_path / "taken").mkdir()
    status, options = _call(f"{base}/api/runtime/options")
    seed = _call(f"{base}/api/runtime/envs/0/reset", {"seed": "abc"})
    no_suite = _call(f"{base}/api/runs", {"contest": "basic"})
    bad_id = _call(f"{base}/api/runs", {"suite": "basic", "contest": "basic", "run_id": "../escape"})
    taken = _call(f"{base}/api/runs", {"suite": "basic", "contest": "basic", "run_id": "taken"})
    bad_args = _call(f"{base}/api/runs", {"suite": "basic", "contest": "basic", "args": "sim:=flatsim oops"})
    forged = _call(f"{base}/api/runs", {"suite": "basic", "contest": "basic"}, header=False)

    assert status == 200 and {"robots", "worlds", "planners", "task_modes", "launch_args", "sims", "suites", "contests"} <= set(options)
    assert {"robot", "world", "robot.planner", "task.robots"} <= {arg["name"] for arg in options["launch_args"]}
    assert options["sims"][-1] == "dummy" and "basic" in options["suites"] and "basic" in options["contests"]
    assert set(options["task_modes"]) <= {"robots", "obstacles", "modules"} and options["task_modes"]
    for axis in options["task_modes"].values():
        assert isinstance(axis["multi"], bool) and axis["modes"]
        for mode in axis["modes"].values():
            for param in mode["params"]:
                assert param["name"] == f"{mode['namespace']}.{param['leaf']}"
                assert {"type", "default", "label", "description", "choices", "catalog", "range", "read_only"} <= set(param)
                assert param["label"]
    assert all("pair" not in param or param["pair"] in ("int_pair", "float_pair") for axis in options["task_modes"].values() for mode in axis["modes"].values() for param in mode["params"])
    assert "docs" not in options
    assert {"worlds", "scenarios", "robots"} <= set(options["catalogs"])
    assert seed[0] == 400 and "seed: Input should be a valid integer" in seed[1]["error"]
    assert no_suite[0] == 400 and "suite must be a suite name" in no_suite[1]["error"]
    assert bad_id[0] == 400 and "run ids are" in bad_id[1]["error"]
    assert taken[0] == 409 and "run taken already exists" in taken[1]["error"]
    assert bad_args == (400, {"error": "launch arguments are key:=value, got oops", "hint": None})
    assert forged[0] == 403


def test_a_benchmark_runner_locks_the_controls(dashboard: tuple[str, dict[str, str]]):
    base, env = dashboard
    runner = subprocess.Popen([sys.executable, "-c", RUNNER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
    try:
        _wait_for_line(runner, "ready")
        deadline = time.monotonic() + 20
        status, out = _call(f"{base}/api/runtime/release", {})
        while "owns this runtime" not in out["error"] and time.monotonic() < deadline:
            time.sleep(0.5)
            status, out = _call(f"{base}/api/runtime/release", {})
    finally:
        runner.communicate("", timeout=30)
    assert status == 409 and out["error"] == "benchmark run r-owner owns this runtime, the dashboard only watches it"


def test_catalog_answers_every_kind_offline_from_the_asset_tree(dashboard: tuple[str, dict[str, str]]):
    base, _ = dashboard
    _, options = _call(f"{base}/api/runtime/options")

    worlds_status, worlds = _call(f"{base}/api/runtime/catalog?kind=worlds")
    robots_status, robots = _call(f"{base}/api/runtime/catalog?kind=robots")
    no_world = _call(f"{base}/api/runtime/catalog?kind=scenarios")
    environments = _call(f"{base}/api/runtime/catalog?kind=environments")
    scenarios = _call(f"{base}/api/runtime/catalog?kind=scenarios&world=map_empty")
    missing_world = _call(f"{base}/api/runtime/catalog?kind=scenarios&world=no_such_world_at_all")
    unknown = _call(f"{base}/api/runtime/catalog?kind=nothing")

    assert worlds_status == 200 and worlds["kind"] == "worlds" and worlds["items"] == options["worlds"]
    assert robots_status == 200 and robots["items"] == options["robots"]
    assert no_world[0] == 400 and "world" in no_world[1]["error"]
    assert environments[0] == 200 and environments[1]["kind"] == "environments" and isinstance(environments[1]["items"], list)
    assert scenarios[0] == 200 and "preflight" in scenarios[1]["items"]
    assert missing_world[0] == 404
    assert unknown[0] == 400 and "kinds:" in unknown[1]["hint"]


def test_spawn_validates_modes_and_params_against_the_task_layer(dashboard: tuple[str, dict[str, str]]):
    base, _ = dashboard
    _, options = _call(f"{base}/api/runtime/options")
    axis, spec = next(iter(options["task_modes"].items()))

    bad_axis = _call(f"{base}/api/runtime/spawn", {"modes": {"nothing": ["x"]}})
    bad_mode = _call(f"{base}/api/runtime/spawn", {"modes": {axis: ["nothing"]}})
    undeclared = _call(f"{base}/api/runtime/spawn", {"modes": {axis: [next(iter(spec["modes"]))]}, "params": {"task.nothing.at_all": 1}})
    bad_shape = _call(f"{base}/api/runtime/spawn", {"modes": {axis: 3}})

    assert bad_axis[0] == 400 and "unknown task axis" in bad_axis[1]["error"]
    assert bad_mode[0] == 400 and "unknown" in bad_mode[1]["error"]
    assert undeclared[0] == 400 and "task.nothing.at_all" in undeclared[1]["error"]
    assert bad_shape[0] == 400 and "modes." in bad_shape[1]["error"]


def test_snapshot_reports_its_age_and_the_answer_time(dashboard: tuple[str, dict[str, str]]):
    base, _ = dashboard

    first = _call(f"{base}/api/runtime")[1]
    taken = first["now"] - first["age_s"]
    deadline = time.monotonic() + 60
    second = first
    while time.monotonic() < deadline and second["now"] - second["age_s"] <= taken:
        time.sleep(0.5)
        second = _call(f"{base}/api/runtime")[1]

    assert first["age_s"] >= 0 and second["now"] > first["now"]
    assert second["now"] - second["age_s"] > taken


def test_stopping_an_unknown_run_is_404_and_a_forged_stop_is_403(dashboard: tuple[str, dict[str, str]], tmp_path: pathlib.Path):
    base, _ = dashboard

    unknown = _call(f"{base}/api/runs/nothing/stop", {})
    forged = _call(f"{base}/api/runs/nothing/stop", {}, header=False)

    assert unknown[0] == 404 and "no run" in unknown[1]["error"]
    assert forged[0] == 403


def test_a_failing_background_refresh_replaces_the_cached_snapshot_with_the_error(dashboard: tuple[str, dict[str, str]]):
    base, env = dashboard
    good_status, good = _call(f"{base}/api/runtime")
    stand_in = subprocess.Popen([sys.executable, "-c", SIM_STAND_IN], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
    try:
        _wait_for_line(stand_in, "ready")
        deadline = time.monotonic() + 40
        status, out = _call(f"{base}/api/runtime")
        while status == 200 and time.monotonic() < deadline:
            time.sleep(0.5)
            status, out = _call(f"{base}/api/runtime")
    finally:
        stand_in.communicate("", timeout=30)
    deadline = time.monotonic() + 40
    after_status, after = _call(f"{base}/api/runtime")
    while after_status != 200 and time.monotonic() < deadline:
        time.sleep(0.5)
        after_status, after = _call(f"{base}/api/runtime")

    assert good_status == 200 and good["running"] is False
    assert status == 409 and "refresh failed" in out["error"]
    assert after_status == 200 and after["running"] is False


def test_actions_describe_each_tool_with_the_bounds_of_its_own_signature(dashboard: tuple[str, dict[str, str]]):
    base, _ = dashboard
    _, options = _call(f"{base}/api/runtime/options")
    actions = options["actions"]

    step = {param["name"]: param for param in actions["step"]["params"]}
    lockstep = {param["name"]: param for param in actions["lockstep_start"]["params"]}
    reset = {param["name"]: param for param in actions["reset"]["params"]}

    assert list(actions)[:3] == ["spawn", "despawn", "despawn_all"]
    assert actions["step"]["tool"] == "runtime_step" and actions["step"]["scope"] == "runtime" and actions["step"]["doc"]
    assert actions["reset"]["scope"] == "env" and actions["despawn_all"]["scope"] == "runtime"
    assert step["seconds"]["exclusiveMinimum"] == 0 and step["seconds"]["type"] == "number" and "timeout_s" not in step
    assert lockstep["rtf"]["minimum"] == 0 and lockstep["gated"]["type"] == "boolean" and lockstep["gated"]["default"] is True
    assert reset["seed"]["minimum"] == -1 and reset["seed"]["maximum"] == 2**31 - 1 and reset["seed"]["default"] == -1
    assert set(reset) == {"world", "seed"} and reset["world"]["type"] == "string"
    assert actions["stop_run"]["params"] and actions["stop_run"]["params"][0]["name"] == "grace_s"
    assert actions["start_run"]["params"] == [] and actions["hold"]["params"] == []
    assert all(entry["doc"] for entry in actions.values())


def test_request_bodies_are_checked_against_the_tools_own_bounds(dashboard: tuple[str, dict[str, str]]):
    base, _ = dashboard

    step = _call(f"{base}/api/runtime/step", {"seconds": 0})
    missing = _call(f"{base}/api/runtime/step", {})
    rtf = _call(f"{base}/api/runtime/lockstep_start", {"rtf": -1})
    gated = _call(f"{base}/api/runtime/lockstep_start", {"gated": "yes"})
    seed = _call(f"{base}/api/runtime/envs/0/reset", {"seed": 2**31})

    assert step[0] == 400 and step[1]["error"] == "seconds: Input should be greater than 0"
    assert missing[0] == 400 and "seconds: Field required" in missing[1]["error"]
    assert rtf[0] == 400 and rtf[1]["error"] == "rtf: Input should be greater than or equal to 0"
    assert gated[0] == 400 and gated[1]["error"].startswith("gated:")
    assert seed[0] == 400 and "seed: Input should be less than or equal to 2147483647" in seed[1]["error"]


def test_options_name_the_spawn_fields_with_their_catalogs_and_the_layers_multi_axes(dashboard: tuple[str, dict[str, str]]):
    base, _ = dashboard
    _, options = _call(f"{base}/api/runtime/options")

    fields = {field["name"]: field for field in options["spawn_fields"]}

    assert list(fields) == ["robot", "world", "robot.planner"]
    assert fields["robot"]["catalog"] == "robots" and fields["world"]["catalog"] == "worlds"
    assert fields["robot.planner"]["catalog"] is None and fields["robot.planner"]["items"] == options["planners"]
    assert fields["robot"]["description"]
    assert options["task_modes"]["modules"]["multi"] is True and options["task_modes"]["robots"]["multi"] is False


def test_spawn_params_are_checked_against_their_declared_type(dashboard: tuple[str, dict[str, str]]):
    base, _ = dashboard
    _, options = _call(f"{base}/api/runtime/options")
    wrong = {"bool": "yes", "integer": "x", "double": "x", "string": 5, "bool_array": "x", "integer_array": "x", "double_array": "x", "string_array": "x"}
    axis, mode, param = next((axis, name, param) for axis, spec in options["task_modes"].items() for name, mode in spec["modes"].items() for param in mode["params"] if param["type"] in wrong and not param["read_only"])

    answer = _call(f"{base}/api/runtime/spawn", {"modes": {axis: [mode]}, "params": {param["name"]: wrong[param["type"]]}})

    assert answer[0] == 400 and f"{param['name']} is declared as {param['type']}" in answer[1]["error"]


def test_actions_carry_the_annotations_of_their_tools_registration(dashboard: tuple[str, dict[str, str]]):
    base, _ = dashboard
    _, options = _call(f"{base}/api/runtime/options")
    actions = options["actions"]

    assert actions["despawn"]["annotations"] == {"destructive": True, "read_only": False, "idempotent": False}
    assert actions["step"]["annotations"] == {"destructive": False, "read_only": False, "idempotent": False}
    assert actions["lockstep_stop"]["annotations"]["idempotent"] is True
    assert all(set(entry["annotations"]) == {"destructive", "read_only", "idempotent"} for entry in actions.values())


def test_options_leave_out_the_view_actions_when_rerun_utils_has_no_tools(dashboard_without_view_tools: tuple[str, dict[str, str]]):
    base, _ = dashboard_without_view_tools

    status, options = _call(f"{base}/api/runtime/options")
    hold_status, _ = _call(f"{base}/api/runtime/hold", {})
    view_status, view = _call(f"{base}/api/runtime/envs/0/view", {})

    assert status == 200 and {"spawn", "despawn", "reset", "step"} <= set(options["actions"])
    assert "view" not in options["actions"] and "view_stop" not in options["actions"]
    assert hold_status == 409
    assert view_status == 404 and "no runtime action 'view'" in view["error"]


def test_view_actions_come_from_the_rerun_tools_registration(dashboard: tuple[str, dict[str, str]]):
    pytest.importorskip("rerun_utils.mcp")
    base, _ = dashboard

    _, options = _call(f"{base}/api/runtime/options")
    actions = options["actions"]

    assert actions["view"]["tool"] == "viz_rerun_start" and actions["view"]["scope"] == "env"
    assert actions["view_stop"]["tool"] == "viz_rerun_stop" and actions["view_stop"]["scope"] == "env"
    assert {param["name"] for param in actions["view"]["params"]} == set()
    assert actions["view"]["doc"] and actions["view_stop"]["doc"]
