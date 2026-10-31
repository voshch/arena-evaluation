from __future__ import annotations

import asyncio
import pathlib
import subprocess
import time
import typing

import pytest

rclpy = pytest.importorskip("rclpy")

from arena_evaluation.benchmark.config import Contest, Suite
from arena_evaluation.benchmark.progress_display import BenchmarkProgressDisplay
from arena_evaluation.benchmark.runner import (
    ARENA_NODE_FQN,
    STATE_TOPIC,
    _LATCHED,
    BenchmarkRunner,
    _EnvDied,
    _proc_starttime,
    _proc_tree,
    _SimDied,
)
from arena_evaluation.benchmark.state import Manifest, RunDir
from arena_evaluation.benchmark.step import Step, StepErrorKind, StepResult
from arena_evaluation_msgs.msg import BenchmarkState
from arena_evaluation_msgs.srv import RecordEpisode
from arena_runtime_msgs.msg import EnvRecord, EnvRegistry, HoldEntry, HoldRegistry, LockstepStatus, SimState
from rcl_interfaces.srv import GetParameters, SetParameters
from task_generator.constants import Constants
from task_generator_msgs.action import RunEpisode
from task_generator_msgs.msg import EpisodeRecord
from task_generator_msgs.srv import QueueEpisode

T = typing.TypeVar("T")


def _stage(name: str = "s1", *, timeout_peds: float | None = None, config: dict | None = None) -> Suite.Stage:
    return Suite.Stage(
        name=name,
        episodes=1,
        robot="jackal",
        map="map_empty",
        tm_robots=Constants.TaskMode.TM_Robots.RANDOM,
        tm_obstacles=Constants.TaskMode.TM_Obstacles.RANDOM,
        config={"random": {"dynamic": {"min": 2, "max": 3}}} if config is None else config,
        seed=3,
        timeout=45.0,
        timeout_peds=timeout_peds,
    )


def _step(name: str = "dwb", episodes: int = 1, reference_type: str | None = None, timeout_peds: float | None = None, config: dict | None = None) -> Step:
    return Step(contestant=Contest.Contestant(name=name, args={}), stage=_stage(timeout_peds=timeout_peds, config=config), episodes=episodes, is_reference=reference_type is not None, reference_type=reference_type)


def _run_dir(tmp_path: pathlib.Path) -> RunDir:
    manifest = Manifest(
        run_id="run",
        created_at="2026-10-13T00:00:00+00:00",
        arena_git_sha=None,
        arena_git_dirty=False,
        cli_args=[],
        env_n=1,
        headless=True,
        config_hash="0" * 40,
        simulator=None,
        scale_episodes=1.0,
        suite_name="suite",
        contest_name="contest",
        suite={},
        contest=[],
        steps=[],
    )
    return RunDir.create(tmp_path, "run", manifest)


def _drive(tmp_path: pathlib.Path, body: typing.Callable[[BenchmarkRunner], typing.Awaitable[T]], **kwargs: object) -> T:
    async def main() -> T:
        suite = Suite(name="suite", stages=[_stage()])
        contest = Contest(name="contest", description=None, contestants=[Contest.Contestant(name="dwb", args={})])
        runner = BenchmarkRunner(suite, contest, simulator=None, scale_episodes=1.0, env_n=1, run_id="run", headless=True, run_dir=_run_dir(tmp_path), **kwargs)
        try:
            return await body(runner)
        finally:
            runner.destroy_node()

    return asyncio.run(main())


def _registry(*env_ids: int) -> EnvRegistry:
    return EnvRegistry(envs=[EnvRecord(env_id=i, fqn=f"/arena/env_{i}", ready=True) for i in env_ids])


def _dead(reason: str, sec: int = 0) -> SimState:
    msg = SimState(alive=False, reason=reason)
    msg.header.stamp.sec = sec
    return msg


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


def _with_env_clients(runner: BenchmarkRunner, env_id: int) -> None:
    root = f"/arena/env_{env_id}"
    runner._episode_action_clients[env_id] = runner.create_action_client_wrapper(RunEpisode, f"{root}/lifecycle/run_episode")
    runner._episode_records[env_id] = {}
    runner._queue_clients[env_id] = runner.create_client_wrapper(QueueEpisode, f"{root}/config/queue_episode")
    runner._param_clients[env_id] = runner.create_client_wrapper(SetParameters, f"{root}/set_parameters")
    runner._param_get_clients[env_id] = runner.create_client_wrapper(GetParameters, f"{root}/get_parameters")
    runner._recorder_clients[env_id] = runner.create_client_wrapper(RecordEpisode, f"{root}/start_episode")
    runner._env_subs[env_id] = [runner.create_subscription(EpisodeRecord, f"{root}/state/episode", lambda _msg: None, 10)]
    runner._tf_namespaces[env_id] = root
    runner._triggered_episodes[env_id] = set()


def test_env_registry_reveals_new_envs_and_releases_waits_on_vanished_ones(tmp_path: pathlib.Path):
    async def body(runner: BenchmarkRunner) -> tuple[bool, bool, bool]:
        appear = asyncio.create_task(runner._await_env_visible(2))
        await _settle()
        runner._on_envs(_registry(1, 2))
        await asyncio.wait_for(appear, 5.0)
        await runner._await_env_visible(1)
        gone = asyncio.create_task(runner._wait_env_gone(1, timeout=5.0))
        await _settle()
        runner._on_envs(_registry(2))
        return await asyncio.wait_for(gone, 5.0), await runner._wait_env_gone(7, timeout=5.0), await runner._wait_env_gone(2, timeout=0.05)

    assert _drive(tmp_path, body) == (True, True, False)


def test_registry_update_from_an_executor_thread_is_scheduled_on_the_loop(tmp_path: pathlib.Path):
    async def body(runner: BenchmarkRunner) -> list[int]:
        await asyncio.get_running_loop().run_in_executor(None, runner._on_envs, _registry(4))
        await asyncio.wait_for(runner._env_visible_events[4].wait(), 5.0)
        return sorted(runner._env_records)

    assert _drive(tmp_path, body) == [4]


def test_sim_state_latches_death_and_ignores_alive_and_stale_reports(tmp_path: pathlib.Path):
    async def body(runner: BenchmarkRunner) -> list[tuple[bool, str]]:
        seen = []
        runner._on_sim_state(SimState(alive=True))
        seen.append((runner._sim_dead.is_set(), runner._sim_dead_reason))
        runner._runtime_started_at = 100.0
        runner._on_sim_state(_dead("old crash", sec=50))
        seen.append((runner._sim_dead.is_set(), runner._sim_dead_reason))
        runner._on_sim_state(_dead("", sec=150))
        seen.append((runner._sim_dead.is_set(), runner._sim_dead_reason))
        runner._on_sim_state(_dead("gz segfault", sec=151))
        seen.append((runner._sim_dead.is_set(), runner._sim_dead_reason))
        return seen

    assert _drive(tmp_path, body) == [(False, ""), (False, ""), (True, "sim reported dead"), (True, "gz segfault")]


def test_only_an_alive_report_from_the_current_runtime_marks_it_up(tmp_path: pathlib.Path):
    async def body(runner: BenchmarkRunner) -> list[bool]:
        alive = SimState(alive=True)
        seen = []
        runner._runtime_started_at = 100.0
        alive.header.stamp.sec = 50
        runner._on_sim_state(alive)
        await _settle()
        seen.append(runner._runtime_up.is_set())
        alive.header.stamp.sec = 150
        runner._on_sim_state(alive)
        await _settle()
        seen.append(runner._runtime_up.is_set())
        seen.append(runner._sim_dead.is_set())
        return seen

    assert _drive(tmp_path, body) == [False, True, False]


def test_a_stalled_spawn_marks_the_sim_dead(tmp_path: pathlib.Path):
    async def body(runner: BenchmarkRunner) -> tuple[str, bool]:
        with pytest.raises(_SimDied) as raised:
            await runner._spawn_and_setup_env(_step(), inactivity_timeout=0.0)
        return str(raised.value), runner._sim_dead.is_set()

    assert _drive(tmp_path, body) == ("spawn_env stalled for dwb/s1", True)


def test_only_foreign_holders_count_as_env_holds(tmp_path: pathlib.Path):
    async def body(runner: BenchmarkRunner) -> list[bool]:
        seen = []
        runner._on_holders(HoldRegistry(holds=[HoldEntry(caller_id=ARENA_NODE_FQN, reason="reset", count=1)]))
        seen.append(runner._env_holds)
        runner._on_holders(HoldRegistry(holds=[HoldEntry(caller_id=ARENA_NODE_FQN), HoldEntry(caller_id="/arena/env_0/task_generator_node")]))
        seen.append(runner._env_holds)
        return seen

    assert _drive(tmp_path, body) == [False, True]


def test_lockstep_status_reaches_the_monitor(tmp_path: pathlib.Path):
    async def body(runner: BenchmarkRunner) -> bool:
        runner._on_lockstep(LockstepStatus(active=True))
        runner._lockstep.open("k", time.time())
        return runner._lockstep.close("k", time.time()).active

    assert _drive(tmp_path, body) is True


def test_alive_race_returns_the_result_or_names_what_died(tmp_path: pathlib.Path):
    async def body(runner: BenchmarkRunner) -> list[object]:
        seen: list[object] = [await runner._await_alive(asyncio.sleep(0, result="done"), env_id=3, what="x")]
        runner._on_envs(_registry(3))
        waiting = asyncio.create_task(runner._await_alive(asyncio.Event().wait(), env_id=3, what="queue_episode"))
        await _settle()
        runner._on_envs(_registry())
        with pytest.raises(_EnvDied) as env_died:
            await waiting
        seen.append(str(env_died.value))
        runner._mark_sim_dead("physics exploded")
        with pytest.raises(_SimDied) as sim_died:
            await runner._await_alive(asyncio.Event().wait(), what="anything")
        seen.append(str(sim_died.value))
        seen.append(await runner._await_hb(asyncio.sleep(0, result=7), "heartbeat"))
        return seen

    assert _drive(tmp_path, body) == ["done", "env 3 disappeared from /arena/state/envs while waiting for queue_episode", "physics exploded", 7]


@pytest.mark.parametrize("method", ["_await_hb", "_await_activity"])
def test_cancelling_a_wait_cancels_the_awaited_task(tmp_path: pathlib.Path, method: str):
    async def body(runner: BenchmarkRunner) -> bool:
        inner = asyncio.ensure_future(asyncio.Event().wait())
        outer = asyncio.create_task(getattr(runner, method)(inner, "never"))
        await _settle()
        outer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await outer
        await _settle()
        return inner.cancelled()

    assert _drive(tmp_path, body) is True


def test_episode_budget_uses_the_ped_timeout_only_for_ped_references(tmp_path: pathlib.Path):
    async def body(_runner: BenchmarkRunner) -> list[float]:
        return [
            BenchmarkRunner._episode_budget(_step()),
            BenchmarkRunner._episode_budget(_step("unhindered_peds", reference_type="unhindered_peds", timeout_peds=90.0)),
            BenchmarkRunner._episode_budget(_step("unhindered_peds", reference_type="unhindered_peds")),
            BenchmarkRunner._episode_budget(_step(reference_type="unobstructed_robot", timeout_peds=90.0)),
        ]

    assert _drive(tmp_path, body) == [45.0, 90.0, 45.0, 45.0]


def test_fail_remaining_drains_the_queue_and_reports_an_abort(tmp_path: pathlib.Path):
    flushed: list[StepResult] = []

    def flush(res: StepResult) -> bool:
        flushed.append(res)
        return res.key == "b/s1"

    async def body(runner: BenchmarkRunner) -> tuple[bool, bool]:
        q: asyncio.Queue[Step] = asyncio.Queue()
        for name in ("a", "b", "c"):
            q.put_nowait(Step(contestant=Contest.Contestant(name=name, args={}), stage=_stage(), episodes=2))
        return runner._fail_remaining(q, 4, "spawn_env failed", flush), q.empty()

    assert _drive(tmp_path, body) == (True, True)
    assert [(r.key, r.status, r.env_id, r.error_kind, r.error_detail, r.episodes_total) for r in flushed] == [
        ("a/s1", "failed", 4, StepErrorKind.ENV_SETUP, "spawn_env failed", 2),
        ("b/s1", "failed", 4, StepErrorKind.ENV_SETUP, "spawn_env failed", 2),
        ("c/s1", "failed", 4, StepErrorKind.ENV_SETUP, "spawn_env failed", 2),
    ]


def test_spent_sim_death_budget_aborts_without_restarting(tmp_path: pathlib.Path):
    async def body(runner: BenchmarkRunner) -> tuple[tuple[bool, bool], int, int]:
        verdict = await runner._spend_sim_death("gz crashed", "dwb/s1")
        await runner._recover_sim("late worker", generation=5)
        return verdict, runner._retry_budget.sim_deaths, runner._sim_generation

    assert _drive(tmp_path, body, max_sim_deaths=1) == ((False, True), 1, 0)


def test_group_queue_aborts_when_the_sim_is_dead_before_the_env_spawns(tmp_path: pathlib.Path):
    flushed: list[StepResult] = []

    async def body(runner: BenchmarkRunner) -> tuple[bool, int, int]:
        runner._progress = BenchmarkProgressDisplay(title="t", total_steps=1, env_n=1, run_id="run")
        runner._mark_sim_dead("sim dead: gz exited")
        q: asyncio.Queue[Step] = asyncio.Queue()
        q.put_nowait(_step())
        aborted = await runner._run_group_queue(_step(), q, 0, lambda res: flushed.append(res) or False)
        return aborted, q.qsize(), runner._completed_groups

    assert _drive(tmp_path, body, max_sim_deaths=1) == (True, 1, 1)
    assert flushed == []


def test_env_teardown_releases_every_client_and_setup_is_idempotent(tmp_path: pathlib.Path):
    async def body(runner: BenchmarkRunner) -> list[int]:
        _with_env_clients(runner, 3)
        await runner._setup_env_clients(3, "/arena/env_3", "jackal")
        runner._teardown_env_clients(3)
        runner._teardown_env_clients(3)
        tables = (runner._episode_action_clients, runner._queue_clients, runner._param_clients, runner._param_get_clients, runner._recorder_clients, runner._env_subs, runner._tf_namespaces, runner._triggered_episodes, runner._episode_records)
        return [len(t) for t in tables]

    assert _drive(tmp_path, body) == [0] * 9


def test_pending_steps_come_from_the_run_state(tmp_path: pathlib.Path):
    async def body(runner: BenchmarkRunner) -> list[str]:
        runner._run_dir.state.write({"dwb/s1": StepResult("dwb/s1", "ok", 0, 0.0, 1.0, None, None)})
        return [s.key for s in runner._build_pending()]

    assert _drive(tmp_path, body) == []


def test_despawning_an_env_the_registry_no_longer_lists_only_tears_down_clients(tmp_path: pathlib.Path):
    async def body(runner: BenchmarkRunner) -> list[int]:
        _with_env_clients(runner, 3)
        await runner._despawn_env(3)
        await runner.teardown()
        return sorted(runner._queue_clients)

    assert _drive(tmp_path, body) == []


def test_despawn_on_a_dead_sim_only_tears_down_clients(tmp_path: pathlib.Path):
    async def body(runner: BenchmarkRunner) -> list[int]:
        _with_env_clients(runner, 3)
        runner._on_envs(_registry(3))
        runner._mark_sim_dead("gone")
        await runner._despawn_env(3)
        return sorted(runner._queue_clients)

    assert _drive(tmp_path, body) == []


def test_stage_config_on_an_env_without_clients_fails_the_step(tmp_path: pathlib.Path):
    async def body(runner: BenchmarkRunner) -> StepResult:
        return await runner._run_step(_step(episodes=3), 9, 0)

    res = _drive(tmp_path, body)

    assert (res.key, res.status, res.env_id, res.error_kind, res.episodes_total) == ("dwb/s1", "failed", 9, StepErrorKind.ENV_SETUP, 3)
    assert res.error_detail.startswith("stage config failed: ")


@pytest.mark.parametrize(
    ("reference_type", "config", "key"),
    [(None, None, "dwb/s1"), ("unobstructed_robot", None, "dwb_unobstructed_robot/s1"), ("unobstructed_robot", {"random": "preset"}, "dwb_unobstructed_robot/s1")],
)
def test_env_vanishing_during_stage_config_fails_the_step(tmp_path: pathlib.Path, reference_type: str | None, config: dict | None, key: str):
    async def body(runner: BenchmarkRunner) -> StepResult:
        _with_env_clients(runner, 3)
        runner._on_envs(_registry(3))
        task = asyncio.create_task(runner._run_step(_step(reference_type=reference_type, config=config), 3, 0))
        await _settle()
        runner._on_envs(_registry())
        return await asyncio.wait_for(task, 5.0)

    res = _drive(tmp_path, body)

    assert (res.key, res.status, res.error_kind) == (key, "failed", StepErrorKind.ENV_SETUP)
    assert "while waiting for queue_episode on env 3" in res.error_detail


def test_sim_death_during_ped_reference_stage_config_propagates(tmp_path: pathlib.Path):
    async def body(runner: BenchmarkRunner) -> None:
        _with_env_clients(runner, 3)
        runner._mark_sim_dead("gz exited")
        await runner._run_step(_step("unhindered_peds", reference_type="unhindered_peds"), 3, 0)

    with pytest.raises(_SimDied, match="gz exited"):
        _drive(tmp_path, body)


def test_step_without_episodes_is_failed(tmp_path: pathlib.Path):
    async def body(runner: BenchmarkRunner) -> StepResult:
        _with_env_clients(runner, 3)
        return await runner._run_episodes(_step(episodes=0), 3)

    res = _drive(tmp_path, body)

    assert (res.status, res.error_kind, res.episodes_run, res.episodes_total, res.lockstep) == ("failed", None, 0, 0, None)


def test_env_vanishing_before_the_episode_goal_is_accepted_fails_the_step(tmp_path: pathlib.Path):
    async def body(runner: BenchmarkRunner) -> StepResult:
        _with_env_clients(runner, 3)
        runner._progress = BenchmarkProgressDisplay(title="t", total_steps=1, env_n=1, run_id="run")
        runner._on_envs(_registry(3))
        task = asyncio.create_task(runner._run_episodes(_step(episodes=2), 3, slot_index=0))
        await _settle()
        assert runner._progress.active_slots[0]["state"] == "RUNNING"
        runner._on_envs(_registry())
        return await asyncio.wait_for(task, 5.0)

    res = _drive(tmp_path, body)

    assert (res.status, res.error_kind, res.episodes_run, res.episodes_failed, res.episodes_total) == ("failed", StepErrorKind.ENV_SETUP, 0, 0, 2)
    assert "run_episode goal on env 3" in res.error_detail
    assert res.lockstep is not None


def test_sim_death_while_sending_an_episode_goal_propagates(tmp_path: pathlib.Path):
    async def body(runner: BenchmarkRunner) -> None:
        _with_env_clients(runner, 3)
        runner._mark_sim_dead("gz exited")
        await runner._run_episodes(_step(), 3)

    with pytest.raises(_SimDied):
        _drive(tmp_path, body)


def _write_after(path: pathlib.Path, lines: list[tuple[float, str, str]]) -> typing.Awaitable[str]:
    async def writer() -> str:
        for delay, mode, text in lines:
            await asyncio.sleep(delay)
            with path.open(mode) as fh:
                fh.write(text)
        return "spawned"

    return writer()


def test_activity_wait_keeps_going_while_the_log_shows_progress(tmp_path: pathlib.Path):
    async def body(runner: BenchmarkRunner) -> str:
        log = runner._run_dir.log_path
        lines = [(0.1, "a", "[gz] loading world\n"), (0.1, "w", "x\n"), (0.1, "a", "waiting on env_0\n"), (0.1, "a", "waiting on env_0\n[gz] spawned robot\n"), (0.1, "a", "")]
        return await runner._await_activity(_write_after(log, lines), "spawn_env", inactivity_timeout=0.25, max_total_timeout=10.0, check_interval=0.05)

    assert _drive(tmp_path, body) == "spawned"


def test_activity_wait_times_out_on_a_silent_or_unreadable_log(tmp_path: pathlib.Path):
    async def body(runner: BenchmarkRunner) -> None:
        runner._run_dir.log_path.mkdir()
        await runner._await_activity(asyncio.Event().wait(), "spawn_env", inactivity_timeout=0.15, max_total_timeout=10.0, check_interval=0.05)

    with pytest.raises(TimeoutError, match="no progress for .* during spawn_env"):
        _drive(tmp_path, body)


def test_activity_wait_has_a_hard_ceiling_even_with_progress(tmp_path: pathlib.Path):
    async def body(runner: BenchmarkRunner) -> None:
        log = runner._run_dir.log_path
        lines = [(0.02, "a", f"[gz] step {i}\n") for i in range(100)]
        await runner._await_activity(_write_after(log, lines), "spawn_env", inactivity_timeout=10.0, max_total_timeout=0.2, check_interval=0.05)

    with pytest.raises(TimeoutError, match="spawn_env exceeded max timeout of 0s"):
        _drive(tmp_path, body)


def test_published_state_counts_steps_by_status(tmp_path: pathlib.Path):
    received: list[BenchmarkState] = []
    listener = rclpy.create_node("runner_state_listener")
    listener.create_subscription(BenchmarkState, STATE_TOPIC, received.append, _LATCHED)

    async def body(runner: BenchmarkRunner) -> None:
        results = {
            "a/s1": StepResult("a/s1", "ok", 0, 0.0, 1.0, None, None),
            "b/s1": StepResult("b/s1", "partial", 0, 0.0, 1.0, None, None),
            "c/s1": StepResult("c/s1", "failed", 0, 0.0, 1.0, StepErrorKind.INTERNAL, None),
            "d/s1": StepResult("d/s1", "in_progress", 0, 0.0, None, None, None),
            "e/s1": StepResult("e/s1", "skipped", 0, 0.0, 1.0, None, None),
        }
        runner._publish_state(results, 7)
        deadline = time.monotonic() + 20.0
        while not any(m.steps_total == 7 for m in received) and time.monotonic() < deadline:
            rclpy.spin_once(listener, timeout_sec=0.1)

    try:
        _drive(tmp_path, body)
    finally:
        listener.destroy_node()

    (msg,) = [m for m in received if m.steps_total == 7]
    assert (msg.run_id, msg.suite, msg.contest, msg.simulator, msg.env_n, msg.headless) == ("run", "suite", "contest", "", 1, True)
    assert (msg.steps_total, msg.steps_done, msg.steps_partial, msg.steps_failed, msg.steps_skipped, msg.steps_in_flight) == (7, 1, 1, 1, 1, 1)
    assert list(msg.active_keys) == ["d/s1"]


def test_runtime_exit_marks_the_sim_dead(tmp_path: pathlib.Path):
    async def body(runner: BenchmarkRunner) -> str:
        await runner._watch_arena_proc(subprocess.Popen(["sh", "-c", "exit 3"]))
        return runner._sim_dead_reason

    assert _drive(tmp_path, body) == "arena_runtime exited rc=3"


def test_noexit_shutdown_leaves_the_runtime_running(tmp_path: pathlib.Path):
    async def body(runner: BenchmarkRunner) -> tuple[bool, bool, object]:
        proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
        runner._arena_proc = proc
        runner._arena_watch = asyncio.create_task(runner._watch_arena_proc(proc))
        await _settle()
        await runner._shutdown_arena()
        alive = proc.poll() is None
        proc.kill()
        proc.wait()
        return alive, runner._sim_dead.is_set(), runner._arena_watch

    assert _drive(tmp_path, body, noexit=True) == (True, False, None)


def test_shutdown_interrupts_the_runtime_group_and_kills_escaped_descendants(tmp_path: pathlib.Path, runner_log: list[str]):
    proc = subprocess.Popen(["sh", "-c", "setsid sleep 30 & wait"], start_new_session=True)
    viz = subprocess.Popen(["sleep", "30"], start_new_session=True)
    deadline = time.monotonic() + 10.0
    while [pathlib.Path(f"/proc/{pid}/comm").read_text().strip() for pid in _proc_tree(proc.pid)] != ["sleep"] and time.monotonic() < deadline:
        time.sleep(0.05)
    escaped = _proc_tree(proc.pid)

    async def body(runner: BenchmarkRunner) -> tuple[object, object, bool]:
        log = runner._run_dir.log_path.open("a")
        runner._arena_proc = proc
        runner._arena_log_file = log
        runner._viz_proc = viz
        await runner._shutdown_arena()
        await runner._shutdown_arena()
        return runner._arena_log_file, runner._viz_proc, log.closed

    try:
        assert _drive(tmp_path, body) == (None, None, True)
        assert proc.poll() is not None
        assert viz.poll() is not None
        time.sleep(0.2)
        assert all(_proc_starttime(pid) != start or pathlib.Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0] == "Z" for pid, start in escaped.items())
        assert any("outlived the launch, sending SIGKILL" in m for m in runner_log)
    finally:
        for pid in escaped:
            if _proc_starttime(pid) is not None:
                subprocess.run(["kill", "-9", str(pid)], check=False)
        for p in (proc, viz):
            p.kill()
            p.wait()


@pytest.mark.parametrize("viz_state", ["none", "finished"])
def test_shutdown_without_a_live_viewer_or_log_file(tmp_path: pathlib.Path, viz_state: str):
    async def body(runner: BenchmarkRunner) -> tuple[object, object]:
        proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
        if viz_state == "finished":
            runner._viz_proc = subprocess.Popen(["true"])
            runner._viz_proc.wait()
        runner._arena_proc = proc
        await runner._shutdown_arena()
        return proc.poll(), runner._viz_proc

    returncode, viz = _drive(tmp_path, body)

    assert returncode is not None
    assert viz is None
