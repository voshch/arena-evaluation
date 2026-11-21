"""Env lifecycle and step execution mixins for the benchmark runner node."""

from __future__ import annotations

import asyncio
import collections
import contextlib
import datetime
import logging
import os
import pathlib
import signal
import subprocess
import time
import typing

import rclpy
from arena_evaluation_msgs.msg import BenchmarkState
from arena_evaluation_msgs.srv import RecordEpisode
from arena_rclpy_mixins import ActionClientWrapper, ClientWrapper
from arena_runtime.env.task_layer import decode_modes, task_layer
from arena_runtime_msgs.action import RunEpisode
from arena_runtime_msgs.msg import (
    EnvRegistry,
    EpisodeRecord,
    HoldRegistry,
    LockstepStatus,
    SimState,
)
from arena_runtime_msgs.srv import DespawnEnv, QueueEpisode, SpawnEnv
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.action.client import ClientGoalHandle
from rclpy.task import Future

from .lockstep import LockstepSummary, format_report, format_table
from .progress_display import BenchmarkProgressDisplay
from .runner_plan import (
    _ADAPTER_RESET_FAILED,
    _ARENA_ORPHAN_GRACE_S,
    _ARENA_SIGINT_GRACE_S,
    _ARENA_SIGTERM_GRACE_S,
    _CANCEL_SETTLE_S,
    _HEARTBEAT_S,
    _HUNG_EXIT_CODE,
    _LOOP_DEADLINE_S,
    _OUTCOME_LABELS,
    _SIM_STALL_S,
    _SYSTEMIC,
    ARENA_NODE_FQN,
    STATE_TOPIC,
    _all_steps_grid,
    _EnvDied,
    _is_substantive_log_line,
    _new_env_logs,
    _orphaned_env_ids,
    _proc_starttime,
    _proc_tree,
    _requeue_front,
    _SimDied,
    _SimStalled,
    _stationary_pins,
    _yaml_str,
    block_claim_key,
    build_launch_args,
    cell_verdict,
    closed_fraction,
    compromised_table,
    episode_line,
    group_pending,
    pick_block,
    resolve_planner_identity,
    robot_has_depth_camera,
    stage_request,
    step_status,
)
from .step import Step, StepErrorKind, StepResult

_T = typing.TypeVar("_T")
_log = logging.getLogger("arena_evaluation.benchmark.runner")

if typing.TYPE_CHECKING:
    from arena_rclpy_mixins import ArenaMixinNode
    from rclpy.publisher import Publisher
    from rclpy.subscription import Subscription

    from .config import Contest, Suite
    from .lockstep import LockstepMonitor
    from .runner_plan import _RetryBudget
    from .state import RunDir

    class _RunnerHost(ArenaMixinNode):
        _suite: Suite
        _contest: Contest
        _simulator: str | None
        _suite_bundle_dir: pathlib.Path | None
        _scale_episodes: float
        _env_n: int
        _run_id: str
        _headless: bool
        _run_dir: RunDir
        _retry_failed: bool
        _noexit: bool
        _lockstep_verdict: bool
        _strict: bool
        _retries: int
        _max_sim_deaths: int
        _spawn_budget: float
        _efficacy: float | None
        _world_swap: bool
        _shared: str | None
        _arena_passthrough: dict[str, str]
        _viz: bool
        _viz_proc: subprocess.Popen[bytes] | None
        _lockstep: LockstepMonitor
        _total_groups: int
        _completed_groups: int
        _sim_used: bool
        _spawn: ClientWrapper
        _despawn: ClientWrapper
        _env_records: dict[int, typing.Any]
        _env_visible_events: dict[int, asyncio.Event]
        _env_gone_events: dict[int, asyncio.Event]
        _arena_proc: subprocess.Popen[bytes] | None
        _episode_action_clients: dict[int, ActionClientWrapper]
        _queue_clients: dict[int, ClientWrapper]
        _param_clients: dict[int, ClientWrapper]
        _param_get_clients: dict[int, ClientWrapper]
        _tf_namespaces: dict[int, str]
        _episode_records: dict[int, dict[int, EpisodeRecord]]
        _env_subs: dict[int, list[Subscription]]
        _recorder_clients: dict[int, ClientWrapper]
        _triggered_episodes: dict[int, set[int]]
        _env_holds: bool
        _sim_dead: asyncio.Event
        _sim_dead_reason: str
        _sim_generation: int
        _envs_spawned: int
        _sim_recover_lock: asyncio.Lock
        _retry_budget: _RetryBudget
        _runtime_started_at: float
        _runtime_up: asyncio.Event
        _parent_episode_map: dict[tuple[str, str, int], int]
        _state_pub: Publisher

        def _build_pending(self) -> list[Step]: ...

else:

    class _RunnerHost:
        pass


class _RunnerEnvMixin(_RunnerHost):
    """Services, waits, env clients and arena-runtime process lifecycle."""

    def _set_event(self, event: asyncio.Event) -> None:
        """Set an asyncio.Event from any thread. Off the loop thread, executor callbacks included, the set is scheduled on it."""
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self.event_loop:
            event.set()
        else:
            self.event_loop.call_soon_threadsafe(event.set)

    def _on_envs(self, msg: EnvRegistry) -> None:
        new_ids = {env.env_id for env in msg.envs}
        for env_id in new_ids:
            self._set_event(self._env_visible_events.setdefault(env_id, asyncio.Event()))
        for env_id in list(self._env_gone_events):
            if env_id not in new_ids:
                self._set_event(self._env_gone_events[env_id])
        self._env_records = {env.env_id: env for env in msg.envs}

    def _on_sim_state(self, msg: SimState) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp < self._runtime_started_at:
            if not msg.alive:
                _log.debug(f"ignoring stale sim death stamped {stamp:.3f}, this runtime started at {self._runtime_started_at:.3f}")
            return
        if msg.alive:
            self._set_event(self._runtime_up)
        else:
            self._mark_sim_dead(msg.reason or "sim reported dead")

    def _mark_sim_dead(self, reason: str) -> None:
        """Latch the sim as dead. Every wait wrapped in _await_alive unblocks with _SimDied."""
        if not self._sim_dead.is_set():
            _log.error(f"sim is dead: {reason}")
        self._sim_dead_reason = reason
        self._set_event(self._sim_dead)

    def _deadman(self) -> None:
        """Loop-watchdog callback: the event loop is hung, take the runtime down and exit."""
        with contextlib.suppress(OSError):
            with self._run_dir.log_path.open("a") as handle:
                handle.write(f"benchmark: runner event loop hung for {_LOOP_DEADLINE_S:.0f}s, killing arena runtime and exiting {_HUNG_EXIT_CODE}\n")
        if self._shared is not None:
            with contextlib.suppress(Exception):
                self._run_dir.fold()
            with contextlib.suppress(OSError):
                self._run_dir.pid_path.unlink()
        proc = self._arena_proc
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        os._exit(_HUNG_EXIT_CODE)

    def _on_holders(self, msg: HoldRegistry) -> None:
        self._env_holds = any(hold.caller_id != ARENA_NODE_FQN for hold in msg.holds)

    def _on_lockstep(self, msg: LockstepStatus) -> None:
        self._lockstep.observe(msg, time.time())

    def _build_launch_args(self, step: Step) -> list[str]:
        return build_launch_args(step, self._simulator, passthrough=self._arena_passthrough)

    async def _await_env_visible(self, env_id: int) -> None:
        """Wait for env_id to appear on /arena/state/envs."""
        if env_id in self._env_records:
            return
        await self._await_hb(
            self._env_visible_events.setdefault(env_id, asyncio.Event()).wait(),
            f"env {env_id} to appear on /arena/state/envs",
        )

    def _scan_logs(
        self,
        log_path: pathlib.Path,
        ros_log_dir: pathlib.Path,
        before: set[pathlib.Path],
        positions: dict[pathlib.Path, int],
    ) -> tuple[bool, str]:
        has_progress = False
        sample_line = ""
        targets = [log_path] if log_path.exists() else []
        targets.extend(_new_env_logs(ros_log_dir, before))
        for path in targets:
            try:
                size = path.stat().st_size
                position = positions.get(path, 0)
                if size < position:
                    position = 0
                if size <= position:
                    continue
                with path.open(encoding="utf-8", errors="replace") as handle:
                    handle.seek(position)
                    text = handle.read()
                    positions[path] = handle.tell()
                for line in text.splitlines():
                    if _is_substantive_log_line(line):
                        has_progress, sample_line = True, line.strip()[:80]
                        break
            except Exception:
                pass
        return has_progress, sample_line

    async def _await_activity(
        self,
        awaitable: typing.Awaitable[_T],
        what: str,
        *,
        inactivity_timeout: float = 120.0,
        max_total_timeout: float = 600.0,
        check_interval: float = 15.0,
    ) -> _T:
        """Await task as long as non-chatter progress is detected in logs. Times out if there is no substantive progress for `inactivity_timeout` seconds, or if `max_total_timeout` is exceeded."""
        task = asyncio.ensure_future(awaitable)
        log_path = self._run_dir.log_path
        positions: dict[pathlib.Path, int] = {log_path: log_path.stat().st_size} if log_path.exists() else {}
        last_activity = time.monotonic()
        total_waited = 0.0
        ros_log_dir = self._run_dir.ros_log_dir
        before = set(ros_log_dir.glob("arena_env_*.log"))
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=check_interval)
                if task in done:
                    return task.result()
                total_waited += check_interval
                if total_waited >= max_total_timeout:
                    _log.error(f"{what} exceeded max timeout of {max_total_timeout:.0f}s")
                    task.cancel()
                    raise TimeoutError(f"{what} exceeded max timeout of {max_total_timeout:.0f}s")
                has_progress, sample_line = self._scan_logs(log_path, ros_log_dir, before, positions)
                if has_progress:
                    last_activity = time.monotonic()
                    _log.info(f"still loading {what} ({total_waited:.0f}s elapsed, active: {sample_line})")
                    continue
                silent_s = time.monotonic() - last_activity
                if silent_s >= inactivity_timeout:
                    _log.error(f"{what} stalled: no progress for {silent_s:.0f}s (total: {total_waited:.0f}s)")
                    task.cancel()
                    raise TimeoutError(f"no progress for {silent_s:.0f}s during {what}")
                _log.warning(f"waiting on {what} ({total_waited:.0f}s elapsed, no progress for {silent_s:.0f}s)")
        except asyncio.CancelledError:
            task.cancel()
            raise

    async def _await_hb(self, awaitable: typing.Awaitable[_T], what: str) -> _T:
        """Await with no timeout, warning every _HEARTBEAT_S while `what` stays pending."""
        task = asyncio.ensure_future(awaitable)
        waited = 0.0
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=_HEARTBEAT_S)
                if task in done:
                    return task.result()
                waited += _HEARTBEAT_S
                _log.warning(f"still waiting for {what} ({waited:.0f}s elapsed)")
        except asyncio.CancelledError:
            task.cancel()
            raise

    async def _await_alive(self, awaitable: typing.Awaitable[_T], *, env_id: int | None = None, what: str) -> _T:
        """Race awaitable against sim death and, when env_id is given, that env disappearing."""
        op_task = asyncio.ensure_future(awaitable)
        sim_task = asyncio.ensure_future(self._sim_dead.wait())
        watch = [sim_task]
        if env_id is not None:
            watch.append(asyncio.ensure_future(self._env_gone_events.setdefault(env_id, asyncio.Event()).wait()))
        try:
            done, _ = await asyncio.wait({op_task, *watch}, return_when=asyncio.FIRST_COMPLETED)
            if op_task in done:
                return op_task.result()
            if sim_task in done:
                raise _SimDied(self._sim_dead_reason)
            raise _EnvDied(f"env {env_id} disappeared from /arena/state/envs while waiting for {what}")
        finally:
            for pending in (op_task, *watch):
                pending.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(op_task, *watch, return_exceptions=True)

    async def _wait_env_gone(self, env_id: int, *, timeout: float | None) -> bool:
        event = asyncio.Event()
        self._env_gone_events[env_id] = event
        try:
            if env_id not in self._env_records:
                return True
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False
        finally:
            self._env_gone_events.pop(env_id, None)

    async def _await_service(self, env_id: int, client: ClientWrapper, what: str) -> None:
        await self._await_alive(
            self._await_hb(client.ensure(timeout_sec=None), what),
            env_id=env_id,
            what=what,
        )

    def _on_episode_record(self, env_id: int, msg: EpisodeRecord) -> None:
        records = self._episode_records.get(env_id)
        if records is not None:
            records[msg.episode_id] = msg
        label = _OUTCOME_LABELS.get(msg.outcome_state, f"UNKNOWN({msg.outcome_state})")
        _log.info(f"[env {env_id}] episode record: id={msg.episode_id} outcome={msg.outcome_state} ({label}) info={msg.outcome_info!r}")
        if msg.outcome_state != EpisodeRecord.RUNNING:
            return
        triggered = self._triggered_episodes.get(env_id)
        recorder = self._recorder_clients.get(env_id)
        if triggered is None or recorder is None or msg.episode_id in triggered:
            return
        triggered.add(msg.episode_id)
        _log.info(f"[env {env_id}] signalling recorder to start episode {msg.episode_id}")
        request = RecordEpisode.Request()
        request.command = RecordEpisode.Request.COMMAND_START
        request.episode_id = msg.episode_id
        future = recorder.client.call_async(request)
        future.add_done_callback(lambda completed, episode_id=msg.episode_id: self._recorder_started(env_id, episode_id, completed))

    def _recorder_started(self, env_id: int, episode_id: int, future: Future) -> None:
        error = future.exception()
        if error:
            _log.warning(f"[env {env_id}] recorder start_episode({episode_id}) failed: {error}")
        else:
            _log.info(f"[env {env_id}] recorder confirmed start of episode {episode_id}")

    async def _setup_env_clients(self, env_id: int, env_ns_root: str, robot_name: str) -> None:
        """Create per-env action client, queue_episode client, and subscriptions. Idempotent."""
        if env_id in self._episode_action_clients:
            return
        self._episode_action_clients[env_id] = self.create_action_client_wrapper(RunEpisode, f"{env_ns_root}/lifecycle/run_episode")
        self._episode_records[env_id] = {}
        queue_client = self.create_client_wrapper(QueueEpisode, f"{env_ns_root}/config/queue_episode")
        await self._await_service(env_id, queue_client, f"queue_episode service on env {env_id}")
        self._queue_clients[env_id] = queue_client
        self._param_clients[env_id] = self.create_client_wrapper(SetParameters, f"{env_ns_root}/set_parameters")
        self._param_get_clients[env_id] = self.create_client_wrapper(GetParameters, f"{env_ns_root}/get_parameters")
        what = f"tf_namespace param on env {env_id}"
        response = await self._await_alive(
            self._await_hb(
                self.await_ros(self._param_get_clients[env_id].client.call_async(GetParameters.Request(names=["tf_namespace"]))),
                what,
            ),
            env_id=env_id,
            what=what,
        )
        self._tf_namespaces[env_id] = response.values[0].string_value if response.values else ""
        self._recorder_clients[env_id] = self.create_client_wrapper(RecordEpisode, f"{env_ns_root}/start_episode")
        self._triggered_episodes[env_id] = set()

        def _on_record(msg: EpisodeRecord) -> None:
            self._on_episode_record(env_id, msg)

        self._env_subs[env_id] = [
            self.create_subscription(
                EpisodeRecord,
                f"{env_ns_root}/state/episode",
                _on_record,
                10,
            )
        ]

    def _teardown_env_clients(self, env_id: int) -> None:
        """Destroy per-env subscriptions, action client, and queue_episode client."""
        for subscription in self._env_subs.pop(env_id, []):
            self.destroy_subscription(subscription)
        for table in (
            self._episode_action_clients,
            self._queue_clients,
            self._param_clients,
            self._param_get_clients,
            self._recorder_clients,
        ):
            client = table.pop(env_id, None)
            if client is not None:
                client.client.destroy()
        for table in (
            self._tf_namespaces,
            self._triggered_episodes,
            self._episode_records,
            self._env_visible_events,
        ):
            table.pop(env_id, None)

    @staticmethod
    def _episode_budget(step: Step) -> float:
        """Sim-second budget the task generator enforces for one episode of step."""
        if step.is_reference and step.reference_type == "unhindered_peds" and step.stage.timeout_peds is not None:
            return step.stage.timeout_peds
        return step.stage.timeout

    async def _push_stage_config(self, env_id: int, step: Step) -> None:
        queue = self._queue_clients[env_id]
        request = stage_request(step)
        if step.is_reference and step.reference_type == "unhindered_peds":
            set_request = SetParameters.Request()
            set_request.parameters = _stationary_pins(task_layer(step.task_layer).mode_namespaces())
            try:
                await self._await_alive(
                    self._param_clients[env_id].call_timeout(set_request, timeout_sec=5.0),
                    env_id=env_id,
                    what=f"stationary params on env {env_id}",
                )
            except (_SimDied, _EnvDied, asyncio.CancelledError):
                raise
            except Exception as exc:
                _log.warning(f"[env {env_id}] stationary params for {step.key} failed: {exc}")
        _log.info(f"[env {env_id}] pushing stage config for {step.key} (map={step.stage.map}, modes={decode_modes(request.modes)})")
        response = await self._await_alive(
            queue.call_timeout(request, timeout_sec=10.0),
            env_id=env_id,
            what=f"queue_episode on env {env_id}",
        )
        if response is None:
            raise RuntimeError(f"queue_episode service call timed out after 10s on env {env_id} for {step.key}")
        if not response.success:
            raise RuntimeError(f"queue_episode failed for {step.key}: {response.error_msg}")
        budget = self._episode_budget(step)
        get_response = await self._await_alive(
            self._param_get_clients[env_id].call_timeout(GetParameters.Request(names=["task.episode.timeout"]), timeout_sec=10.0),
            env_id=env_id,
            what=f"episode timeout param type on env {env_id}",
        )
        if get_response is None or not get_response.values:
            raise RuntimeError(f"get_parameters(task.episode.timeout) failed on env {env_id} for {step.key}")
        budget_value = ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=round(budget)) if get_response.values[0].type == ParameterType.PARAMETER_INTEGER else ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(budget))
        timeout_request = SetParameters.Request()
        timeout_request.parameters = [Parameter(name="task.episode.timeout", value=budget_value)]
        timeout_response = await self._await_alive(
            self._param_clients[env_id].call_timeout(timeout_request, timeout_sec=10.0),
            env_id=env_id,
            what=f"episode timeout param on env {env_id}",
        )
        if timeout_response is None:
            raise RuntimeError(f"set_parameters(task.episode.timeout) timed out after 10s on env {env_id} for {step.key}")
        if not all(result.successful for result in timeout_response.results):
            reasons = "; ".join(result.reason for result in timeout_response.results if not result.successful)
            raise RuntimeError(f"set_parameters(task.episode.timeout={budget}s) rejected on env {env_id} for {step.key}: {reasons}")
        _log.info(f"[env {env_id}] stage config applied for {step.key} (episode budget {budget:.0f}s sim)")

    async def _await_episode_result(self, ac: ActionClientWrapper, goal_handle: ClientGoalHandle, env_id: int) -> RunEpisode.Impl.GetResultService.Response:
        """Await an episode result with no wall ceiling, raising _SimStalled if the sim clock freezes while no env holds the sim."""
        result_task = asyncio.ensure_future(
            self._await_alive(
                ac.await_result(goal_handle),
                env_id=env_id,
                what=f"run_episode result on env {env_id}",
            )
        )
        last_sim = self.sim_time.to_seconds()
        last_move = time.monotonic()
        try:
            while True:
                done, _ = await asyncio.wait({result_task}, timeout=1.0)
                if result_task in done:
                    response = result_task.result()
                    assert isinstance(response, RunEpisode.Impl.GetResultService.Response)
                    return response
                now_sim = self.sim_time.to_seconds()
                if now_sim > last_sim or self._env_holds:
                    last_sim, last_move = now_sim, time.monotonic()
                    continue
                if time.monotonic() - last_move >= _SIM_STALL_S:
                    raise _SimStalled(f"sim clock frozen for {_SIM_STALL_S:.0f}s on env {env_id}")
        finally:
            if not result_task.done():
                result_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.gather(result_task, return_exceptions=True)

    async def _spawn_and_setup_env(self, launch_step: Step, *, inactivity_timeout: float = 120.0) -> tuple[int, str] | None:
        """Spawn one env booting launch_step's world and set up its clients. Tracks active progress (log growth) and only times out if completely stalled."""
        registered: list[int] = []
        known = set(self._env_records)
        try:
            return await self._await_activity(
                self._spawn_and_setup_env_impl(launch_step, registered),
                f"spawn_env for {launch_step.key}",
                inactivity_timeout=inactivity_timeout,
                max_total_timeout=self._spawn_budget,
            )
        except TimeoutError:
            orphans = _orphaned_env_ids(known, self._env_records, registered)
            _log.error(f"spawn_env stalled for {launch_step.key} (no progress/output for {inactivity_timeout:.0f}s); despawning orphaned env(s) {orphans}")
            for env_id in orphans:
                await self._despawn_env(env_id)
            self._mark_sim_dead(f"spawn_env stalled for {launch_step.key}")
            raise _SimDied(self._sim_dead_reason) from None

    async def _spawn_and_setup_env_impl(self, launch_step: Step, registered: list[int]) -> tuple[int, str] | None:
        """Inner impl without timeout, called by _spawn_and_setup_env. Appends the spawned env_id to registered."""
        request = SpawnEnv.Request()
        request.ns = ""
        request.headless = self._headless
        request.launch_args = self._build_launch_args(launch_step)
        what = f"spawn_env for {launch_step.key}"
        response = await self._await_alive(self.await_ros(self._spawn.client.call_async(request)), what=what)
        if response is None or not response.success:
            message = response.error_msg if response is not None else "no response"
            if message.startswith("sim dead:"):
                self._mark_sim_dead(message)
                raise _SimDied(message)
            _log.error(f"spawn_env failed for {launch_step.key}: {message}")
            return None
        env_id = response.env_id
        registered.append(env_id)
        self._envs_spawned += 1
        await self._await_alive(
            self._await_env_visible(env_id),
            what=f"env {env_id} to appear on /arena/state/envs",
        )
        env_ns_root = self._env_records[env_id].fqn
        await self._setup_env_clients(env_id, env_ns_root, launch_step.robot)
        if self._viz and (self._viz_proc is None or self._viz_proc.poll() is not None):
            self._start_viz(env_ns_root)
        return env_id, env_ns_root

    def _start_viz(self, ns: str) -> None:
        _log.info(f"benchmark: spawning rviz for {ns}")
        try:
            self._viz_proc = subprocess.Popen(
                ["ros2", "launch", "rviz_utils", "rviz_config.launch.py", f"ns:={ns}"],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            _log.warning(f"benchmark: failed to spawn rviz: {exc}")

    async def _despawn_env(self, env_id: int) -> None:
        """Tear down clients and despawn an env, waiting for it to disappear from the registry."""
        self._teardown_env_clients(env_id)
        if self._sim_dead.is_set():
            return
        if env_id in self._env_records:
            request = DespawnEnv.Request()
            request.env_id = env_id
            try:
                await self._await_alive(
                    asyncio.wait_for(self._despawn.call_forever(request), timeout=30.0),
                    what=f"despawn_env {env_id}",
                )
                gone = await self._await_alive(
                    self._wait_env_gone(env_id, timeout=15.0),
                    what=f"env {env_id} to leave /arena/state/envs",
                )
                if not gone:
                    self._mark_sim_dead("despawn timed out")
                    return
            except _SimDied:
                return
            except TimeoutError:
                self._mark_sim_dead("despawn timed out")
                return
            except Exception as exc:
                _log.warning(f"despawn of env {env_id} failed: {exc}")
        await asyncio.sleep(2.0)

    async def _best_effort_despawn(self, env_id: int, timeout: float) -> None:
        if env_id not in self._env_records or self._sim_dead.is_set():
            return
        with contextlib.suppress(Exception):
            request = DespawnEnv.Request()
            request.env_id = env_id
            await asyncio.wait_for(self._despawn.call_forever(request), timeout=timeout)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self._wait_env_gone(env_id, timeout=timeout), timeout=timeout)

    async def _spend_sim_death(self, reason: str, step_key: str | None) -> tuple[bool, bool]:
        """Charge one sim death to the retry budget and recover the runtime. Returns (may_retry, run_must_abort)."""
        generation = self._sim_generation
        may_retry, run_abort = self._retry_budget.record(step_key, sim_death_generation=generation)
        if run_abort:
            _log.error(f"benchmark: {self._retry_budget.sim_deaths} sim deaths in this run (last: {reason}), aborting")
            return False, True
        await self._recover_sim(reason, generation)
        return may_retry, False

    async def _recover_sim(self, reason: str, generation: int) -> None:
        """Restart arena_runtime once per sim death; workers that lose the race wait for the new generation."""
        async with self._sim_recover_lock:
            if self._sim_generation != generation:
                return
            _log.error(f"sim died ({reason}), restarting arena_runtime")
            await self._restart_arena()
            self._sim_generation += 1

    async def _watch_arena_proc(self, proc: subprocess.Popen) -> None:
        """Mark the sim dead the moment the arena launch process exits."""
        returncode = await asyncio.get_running_loop().run_in_executor(None, proc.wait)
        self._mark_sim_dead(f"arena_runtime exited rc={returncode}")

    def _close_arena_log(self) -> None:
        if self._arena_log_file is not None:
            with contextlib.suppress(Exception):
                self._arena_log_file.close()
            self._arena_log_file = None

    async def _terminate_group(self, proc: subprocess.Popen, signals: typing.Sequence[tuple[int, float]]) -> None:
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, OSError):
            return
        for sig, grace in signals:
            if proc.poll() is not None:
                break
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, OSError):
                break
            except Exception as exc:
                _log.warning(f"signal {sig} failed: {exc}")
            deadline = time.time() + grace
            while time.time() < deadline and proc.poll() is None:
                await asyncio.sleep(0.1)
        if proc.poll() is None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            except Exception as exc:
                _log.warning(f"SIGKILL failed: {exc}")
            with contextlib.suppress(Exception):
                proc.kill()
                proc.wait(timeout=2.0)

    async def _shutdown_arena(self) -> None:
        watch, self._arena_watch = self._arena_watch, None
        if watch is not None:
            watch.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await watch
        if self._noexit:
            _log.info("--noexit: leaving arena_runtime.launch.py running; Ctrl+C its terminal to stop")
            return
        arena = self._arena_proc
        if arena is None or arena.poll() is not None:
            return
        print("\nbenchmark: shutting down arena runtime...", flush=True)
        try:
            os.getpgid(arena.pid)
        except (ProcessLookupError, OSError):
            self._close_arena_log()
            return
        descendants = _proc_tree(arena.pid)
        await self._terminate_group(
            arena,
            (
                (signal.SIGINT, _ARENA_SIGINT_GRACE_S),
                (signal.SIGTERM, _ARENA_SIGTERM_GRACE_S),
            ),
        )
        if self._viz_proc is not None:
            if self._viz_proc.poll() is None:
                await self._terminate_group(self._viz_proc, ((signal.SIGTERM, 2.0),))
            self._viz_proc = None
        deadline = time.time() + _ARENA_ORPHAN_GRACE_S
        while time.time() < deadline:
            if not any(_proc_starttime(pid) == start for pid, start in descendants.items()):
                break
            await asyncio.sleep(0.1)
        survivors = [pid for pid, start in descendants.items() if _proc_starttime(pid) == start]
        if survivors:
            _log.warning(f"{len(survivors)} arena process(es) outlived the launch, sending SIGKILL: {survivors}")
            for pid in survivors:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(pid, signal.SIGKILL)
        self._close_arena_log()
        print("benchmark: arena runtime shutdown complete.", flush=True)

    async def _start_arena(self) -> None:
        command = [
            "ros2",
            "launch",
            "arena_bringup",
            "arena_runtime.launch.py",
            *(f"{key}:={value}" for key, value in self._arena_passthrough.items()),
        ]
        self._arena_log_file = self._run_dir.log_path.open("a")
        ros_log_dir = self._run_dir.ros_log_dir
        ros_log_dir.mkdir(parents=True, exist_ok=True)
        proc_env = {**os.environ, "ROS_LOG_DIR": str(ros_log_dir)}
        if self._suite_bundle_dir is not None:
            worlds_dir = self._suite_bundle_dir / "worlds"
            if worlds_dir.is_dir():
                outer = proc_env.get("ARENA_WORLD_PATH")
                proc_env["ARENA_WORLD_PATH"] = f"{outer}:{worlds_dir}" if outer else str(worlds_dir)
        self._runtime_up.clear()
        self._runtime_started_at = time.time()
        self._arena_proc = subprocess.Popen(
            command,
            stdout=self._arena_log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=proc_env,
        )
        self._arena_watch = asyncio.create_task(self._watch_arena_proc(self._arena_proc), name="arena_proc_watch")
        await self._await_alive(
            self._await_hb(
                self._runtime_up.wait(),
                "the new arena_runtime_node to report the sim alive",
            ),
            what="arena_runtime_node startup",
        )
        await self._await_alive(self._spawn.ensure(timeout_sec=300.0), what="spawn_env service")
        await self._await_alive(self._despawn.ensure(timeout_sec=300.0), what="despawn_env service")

    async def _restart_arena(self) -> None:
        """Restart arena_runtime for a fresh simulator session between world switches."""
        _log.info("Restarting arena_runtime for fresh simulation session...")
        await self._shutdown_arena()
        self._env_records.clear()
        self._env_visible_events.clear()
        self._env_gone_events.clear()
        self._envs_spawned = 0
        await asyncio.sleep(1.0)
        self._sim_dead_reason = ""
        self._sim_dead.clear()
        await self._start_arena()


if typing.TYPE_CHECKING:

    class _RunnerExecHost(_RunnerHost):
        async def _await_alive(self, awaitable: typing.Awaitable[_T], *, env_id: int | None = None, what: str) -> _T: ...
        async def _await_episode_result(self, ac: ActionClientWrapper, goal_handle: ClientGoalHandle, env_id: int) -> RunEpisode.Impl.GetResultService.Response: ...
        async def _best_effort_despawn(self, env_id: int, timeout: float) -> None: ...
        async def _despawn_env(self, env_id: int) -> None: ...
        async def _push_stage_config(self, env_id: int, step: Step) -> None: ...
        async def _recover_sim(self, reason: str, generation: int) -> None: ...
        async def _restart_arena(self) -> None: ...
        async def _spawn_and_setup_env(
            self,
            launch_step: Step,
            *,
            inactivity_timeout: float = 120.0,
        ) -> tuple[int, str] | None: ...
        async def _spend_sim_death(self, reason: str, step_key: str | None) -> tuple[bool, bool]: ...
        async def _start_arena(self) -> None: ...
        def _teardown_env_clients(self, env_id: int) -> None: ...

else:

    class _RunnerExecHost:
        pass


class _RunnerExecMixin(_RunnerExecHost):
    """Episode execution, retries and the per-run scheduling loop."""

    def _update_slot(
        self,
        slot_index: int,
        env_id: int | None,
        step: Step,
        ep_idx: int,
        state: str,
        *,
        sim_start: float | None = None,
    ) -> None:
        if self._progress is None:
            return
        self._progress.update_slot(
            slot_index=slot_index,
            env_id=env_id,
            contestant=step.contestant.name,
            stage=step.stage.name,
            step_key=step.key,
            ep_idx=ep_idx,
            ep_total=step.episodes,
            state=state,
            sim_start=sim_start,
        )

    async def _stop_recording(self, env_id: int, episode_id: int, rec: EpisodeRecord) -> None:
        recorder = self._recorder_clients.get(env_id)
        if recorder is None:
            return
        state_label = _OUTCOME_LABELS.get(rec.outcome_state, str(rec.outcome_state))
        request = RecordEpisode.Request()
        request.command = RecordEpisode.Request.COMMAND_STOP
        request.episode_id = episode_id
        request.outcome_state = rec.outcome_state
        request.outcome_info = rec.outcome_info
        try:
            response = await recorder.call_timeout(request, timeout_sec=5.0)
            if response is None:
                _log.warning(f"[env {env_id}] recorder stop_episode({episode_id}) timed out")
            else:
                _log.info(f"[env {env_id}] recorder stopped episode {episode_id} ({state_label})")
        except Exception as exc:
            _log.warning(f"[env {env_id}] recorder stop_episode({episode_id}) failed: {exc}")

    def _parent_episode_id(self, step: Step, ep_idx: int, episode_id: int) -> int | None:
        if not step.is_reference:
            self._parent_episode_map[(step.contestant.name, step.stage.name, ep_idx)] = episode_id
            self._parent_episode_map[("__stage__", step.stage.name, ep_idx)] = episode_id
            return None
        key = (step.contestant.name, step.stage.name, ep_idx) if step.reference_type == "unobstructed_robot" else ("__stage__", step.stage.name, ep_idx)
        return self._parent_episode_map.get(key)

    def _append_progress(
        self,
        step: Step,
        env_id: int,
        episode_id: int,
        rec: EpisodeRecord,
        ep_idx: int,
        started_at: float,
        ended_at: float,
        lockstep: LockstepSummary | None,
    ) -> None:
        self._run_dir.progress.append(
            ts_iso=datetime.datetime.now(tz=datetime.UTC).isoformat(),
            run_id=self._run_id,
            step_key=step.key,
            contestant=step.contestant.name,
            stage=step.stage.name,
            env_id=env_id,
            episode_id=episode_id,
            episode_record=rec,
            started_at=started_at,
            ended_at=ended_at,
            parent_episode_id=self._parent_episode_id(step, ep_idx, episode_id),
            is_reference=step.is_reference,
            reference_type=step.reference_type,
            lockstep=lockstep,
        )

    async def _run_episodes(self, step: Step, env_id: int, slot_index: int = 0) -> StepResult:
        """Drive all episodes for one step. Env is already up and clients are set up."""
        started = time.time()
        episodes_run = episodes_failed = episodes_weak = episodes_compromised = 0
        worst_progress: float | None = None
        sim_s = 0.0
        stalled = False
        lockstep = None
        window: list[object] = []
        ac = self._episode_action_clients[env_id]
        record = self._env_records.get(env_id)
        env_fqn = record.fqn if record is not None else ""

        def _fold_window() -> LockstepSummary | None:
            nonlocal lockstep
            if not window:
                return None
            summary = self._lockstep.close(window.pop(), time.time())
            lockstep = summary if lockstep is None else lockstep.merge(summary)
            return summary

        def _result(
            status: typing.Literal["ok", "failed", "skipped", "partial", "in_progress"],
            error_kind: StepErrorKind | None = None,
            error_detail: str | None = None,
        ) -> StepResult:
            _fold_window()
            return StepResult(
                step.key,
                status,
                env_id,
                started,
                time.time(),
                error_kind,
                error_detail,
                episodes_run=episodes_run,
                episodes_failed=episodes_failed,
                episodes_weak=episodes_weak,
                episodes_compromised=episodes_compromised,
                episodes_worst_progress=worst_progress,
                episodes_total=step.episodes,
                sim_s=sim_s,
                lockstep=lockstep,
            )

        try:
            for ep_idx in range(step.episodes):
                self._update_slot(
                    slot_index,
                    env_id,
                    step,
                    ep_idx,
                    "RUNNING",
                    sim_start=self.sim_time.to_seconds(),
                )
                goal = RunEpisode.Goal()
                goal.world = step.stage.map
                goal.seed = step.stage.seed + ep_idx if step.stage.seed is not None else ep_idx
                window.append((step.key, ep_idx))
                self._lockstep.open(window[-1], time.time(), env=env_fqn)
                ep_started_sim = self.sim_time.to_seconds()
                ep_started_wall = time.time()
                try:
                    goal_handle = await self._await_alive(
                        asyncio.wait_for(ac.send_goal(goal), timeout=15.0),
                        env_id=env_id,
                        what=f"run_episode goal on env {env_id}",
                    )
                except TimeoutError:
                    _log.error(f"[{ep_idx + 1}/{step.episodes}] {step.key} env={env_id} send_goal timed out after 15s; abandoning env")
                    return _result(
                        "failed",
                        StepErrorKind.ENV_SETUP,
                        f"send_goal timed out after 15s on env {env_id}",
                    )
                except (_SimDied, _EnvDied, asyncio.CancelledError):
                    raise
                except Exception as exc:
                    _log.error(f"[{ep_idx + 1}/{step.episodes}] {step.key} env={env_id} goal rejected ({exc}); env action server is wedged, abandoning env")
                    return _result(
                        "failed",
                        StepErrorKind.ENV_SETUP,
                        f"run_episode goal rejected: {exc}",
                    )
                try:
                    result_obj = await self._await_episode_result(ac, goal_handle, env_id)
                except _SimStalled:
                    stalled = True
                    _log.warning(f"[{ep_idx + 1}/{step.episodes}] {step.key} env={env_id} SIM STALLED for {_SIM_STALL_S:.0f}s; cancelling and advancing")
                    try:
                        await self.await_ros(goal_handle.cancel_goal_async())
                        await asyncio.wait_for(
                            self._await_alive(
                                ac.await_result(goal_handle),
                                env_id=env_id,
                                what=f"run_episode cancel on env {env_id}",
                            ),
                            timeout=_CANCEL_SETTLE_S,
                        )
                    except (_SimDied, _EnvDied):
                        raise
                    except TimeoutError:
                        _log.error(f"[{ep_idx + 1}/{step.episodes}] {step.key} env={env_id} cancel did not settle in {_CANCEL_SETTLE_S}s; env is wedged, abandoning env")
                        return _result(
                            "failed",
                            StepErrorKind.ENV_SETUP,
                            "run_episode cancel did not settle; env wedged",
                        )
                    except Exception as exc:
                        _log.warning(f"[{ep_idx + 1}/{step.episodes}] {step.key} env={env_id} error awaiting cancel ({exc}); advancing")
                    _fold_window()
                    continue
                ep_ended_sim = self.sim_time.to_seconds()
                sim_s += ep_ended_sim - ep_started_sim
                ep_ended_wall = time.time()
                result: RunEpisode.Result = result_obj.result
                episode_id = result.episode_id
                if result.state == RunEpisode.Result.FATAL:
                    _log.error(f"[{ep_idx + 1}/{step.episodes}] {step.key} env={env_id} FATAL: {result.info} -- aborting step")
                    return _result(
                        "failed",
                        StepErrorKind.ROBOT_SETUP,
                        f"env reported FATAL: {result.info}",
                    )
                if result.state == RunEpisode.Result.FAILED and _ADAPTER_RESET_FAILED in result.info:
                    _log.error(f"[{ep_idx + 1}/{step.episodes}] {step.key} env={env_id} robot reset failed: {result.info} -- aborting step")
                    return _result(
                        "failed",
                        StepErrorKind.ROBOT_SETUP,
                        f"robot reset failed: {result.info}",
                    )
                rec = self._episode_records.get(env_id, {}).get(episode_id)
                if rec is None:
                    _log.warning(f"[{ep_idx + 1}/{step.episodes}] {step.key} env={env_id} no EpisodeRecord for episode_id={episode_id}; counted as lost")
                    _fold_window()
                    continue
                episodes_run += 1
                if rec.outcome_state == EpisodeRecord.FAILED:
                    episodes_failed += 1
                if rec.integrity:
                    episodes_compromised += 1
                fraction = closed_fraction(rec.goal_dist_start, rec.goal_dist_min)
                if rec.goal_dist_start > 0.0:
                    worst_progress = fraction if worst_progress is None else min(worst_progress, fraction)
                if self._efficacy is not None and rec.outcome_state != EpisodeRecord.SUCCESS and fraction < self._efficacy:
                    episodes_weak += 1
                _log.info(
                    episode_line(
                        f"[{ep_idx + 1}/{step.episodes}] {step.key} env={env_id}",
                        rec,
                        ep_ended_sim - ep_started_sim,
                        ep_ended_wall - ep_started_wall,
                    )
                )
                if rec.outcome_state in (
                    EpisodeRecord.SUCCESS,
                    EpisodeRecord.FAILED,
                    EpisodeRecord.SKIPPED,
                    EpisodeRecord.FATAL,
                ):
                    await self._stop_recording(env_id, episode_id, rec)
                ep_lockstep = _fold_window()
                self._append_progress(
                    step,
                    env_id,
                    episode_id,
                    rec,
                    ep_idx,
                    ep_started_sim,
                    ep_ended_sim,
                    ep_lockstep,
                )
        except (_SimDied, asyncio.CancelledError):
            raise
        except _EnvDied as exc:
            _log.warning(f"{step.key} env={env_id} env died mid-step after run={episodes_run}, failed={episodes_failed}: {exc}")
            return _result("failed", StepErrorKind.ENV_SETUP, repr(exc))
        except Exception as exc:
            _log.exception(f"{step.key} env={env_id} unexpected error mid-step after run={episodes_run}, failed={episodes_failed}")
            return _result("failed", StepErrorKind.INTERNAL, repr(exc))
        status = step_status(episodes_run, step.episodes)
        if stalled and status != "ok":
            return _result(
                status,
                StepErrorKind.EPISODE_TIMEOUT,
                f"sim stalled {_SIM_STALL_S:.0f}s",
            )
        return _result(status)

    def _failed_result(
        self,
        step: Step,
        env_id: int | None,
        error_kind: StepErrorKind,
        detail: str | None,
        *,
        started: float | None = None,
    ) -> StepResult:
        return StepResult(
            step.key,
            "failed",
            env_id,
            time.time() if started is None else started,
            time.time(),
            error_kind,
            detail,
            episodes_total=step.episodes,
        )

    async def _run_step(self, step: Step, env_id: int, slot_index: int) -> StepResult:
        """Push the stage config, then drive the step's episodes."""
        started = time.time()
        try:
            await self._push_stage_config(env_id, step)
        except (_SimDied, asyncio.CancelledError):
            raise
        except _EnvDied as exc:
            return self._failed_result(step, env_id, StepErrorKind.ENV_SETUP, repr(exc), started=started)
        except Exception as exc:
            _log.error(f"[env {env_id}] stage config failed for {step.key}: {exc}")
            return self._failed_result(
                step,
                env_id,
                StepErrorKind.ENV_SETUP,
                f"stage config failed: {exc}",
                started=started,
            )
        return await self._run_episodes(step, env_id, slot_index)

    def _fail_remaining(
        self,
        q: asyncio.Queue,
        env_id: int | None,
        detail: str,
        flush_cb: typing.Callable,
    ) -> bool:
        """Fail every step still queued. Returns True when the run must abort."""
        abort = False
        while not q.empty():
            try:
                step = q.get_nowait()
            except asyncio.QueueEmpty:
                break
            abort = flush_cb(self._failed_result(step, env_id, StepErrorKind.ENV_SETUP, detail)) or abort
        return abort

    async def _start_recorder(self, step: Step, env_id: int, env_ns_root: str, offset: int) -> asyncio.subprocess.Process:
        local_planner, inter_planner = resolve_planner_identity(step.contestant)
        params = (
            ("record_data_dir", step.record_dir),
            ("benchmark_id", self._run_id),
            ("contestant", step.contestant.name),
            ("stage", step.stage.name),
            ("map", step.stage.map),
            ("suite_name", self._suite.name),
            ("contest_name", self._contest.name),
            ("local_planner", local_planner),
            ("inter_planner", inter_planner),
            ("robot", step.robot),
        )
        arguments = [
            "run",
            "arena_evaluation",
            "record",
            "--ros-args",
            "-p",
            "use_sim_time:=true",
        ]
        for key, value in params:
            arguments += ["-p", f"{key}:={_yaml_str(value)}"]
        arguments += [
            "-p",
            f"episodes_requested:={step.episodes}",
            "-p",
            f"is_reference:={str(step.is_reference).lower()}",
        ]
        if step.reference_type:
            arguments += ["-p", f"reference_type:={_yaml_str(step.reference_type)}"]
        arguments += [
            "-p",
            f"episode_id_offset:={offset}",
            "-r",
            f"__ns:={env_ns_root}",
        ]
        tf_namespace = self._tf_namespaces.get(env_id, "")
        if tf_namespace:
            arguments += [
                "-r",
                f"/tf:={tf_namespace}/tf",
                "-r",
                f"/tf_static:={tf_namespace}/tf_static",
            ]
        process = await asyncio.create_subprocess_exec(
            "ros2",
            *arguments,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        recorder = self.create_client_wrapper(RecordEpisode, f"{env_ns_root}/start_episode")
        self._recorder_clients[env_id] = recorder
        with contextlib.suppress(Exception):
            await recorder.ensure(timeout_sec=5.0)
        return process

    async def _stop_recorder(self, process: asyncio.subprocess.Process | None) -> None:
        if process is None:
            return
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except Exception:
            with contextlib.suppress(Exception):
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)

    async def _run_group_queue(
        self,
        rep_step: Step,
        q: asyncio.Queue,
        slot_index: int,
        flush_cb: typing.Callable,
    ) -> bool:
        env_id: int | None = None
        env_ns_root = ""
        spawned_once = False
        restarted = False
        try:
            while not q.empty():
                if env_id is None:
                    self._update_slot(slot_index, None, rep_step, 0, "SPAWNING")
                    try:
                        spawned = await self._spawn_and_setup_env(rep_step)
                    except _SimDied as exc:
                        _, run_abort = await self._spend_sim_death(str(exc), None)
                        if run_abort:
                            return True
                        continue
                    if spawned is None and self._envs_spawned and not restarted:
                        restarted = True
                        await self._recover_sim(
                            f"spawn_env failed for {rep_step.key} on a used runtime",
                            self._sim_generation,
                        )
                        continue
                    if spawned is None:
                        return self._fail_remaining(
                            q,
                            None,
                            "env respawn failed after a wedged env" if spawned_once else "spawn_env failed",
                            flush_cb,
                        )
                    env_id, env_ns_root = spawned
                    spawned_once = True
                try:
                    step = q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                recorder_proc = None
                if step.record_dir is not None:
                    if self._shared is None:
                        episode_id_offset = self._global_episode_id_offset
                        self._global_episode_id_offset += step.episodes
                    else:
                        episode_id_offset = self._run_dir.reserve_episode_ids(step.episodes)
                    recorder_proc = await self._start_recorder(step, env_id, env_ns_root, episode_id_offset)
                self._update_slot(slot_index, env_id, step, 0, "SETTING_UP")
                _log.info(f"Starting step {step.key} (env={env_id}, {step.episodes} episodes)")
                sim_death_reason: str | None = None
                try:
                    step_result = await self._run_step(step, env_id, slot_index)
                except asyncio.CancelledError:
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    raise
                except _SimDied as exc:
                    sim_death_reason = str(exc) or self._sim_dead_reason
                    step_result = self._failed_result(step, env_id, StepErrorKind.SIM_DEAD, sim_death_reason)
                except _EnvDied as exc:
                    step_result = self._failed_result(step, env_id, StepErrorKind.ENV_SETUP, repr(exc))
                except Exception as exc:
                    step_result = self._failed_result(step, env_id, StepErrorKind.INTERNAL, repr(exc))
                finally:
                    await self._stop_recorder(recorder_proc)
                if step_result.status != "failed" or step_result.error_kind not in _SYSTEMIC:
                    if flush_cb(step_result):
                        return True
                    continue
                if sim_death_reason is not None:
                    self._teardown_env_clients(env_id)
                    env_id = None
                    may_retry, run_abort = await self._spend_sim_death(sim_death_reason, step.key)
                    if run_abort:
                        return True
                else:
                    may_retry, _ = self._retry_budget.record(step.key, sim_death_generation=None)
                if may_retry:
                    _log.warning(f"retrying {step.key} (attempt {self._retry_budget.attempts[step.key]}/{self._retry_budget.limit}, {step_result.error_kind}: {step_result.error_detail})")
                    _requeue_front(q, step)
                elif flush_cb(step_result):
                    return True
                if env_id is not None:
                    await self._despawn_env(env_id)
                    env_id = None
        finally:
            self._completed_groups += 1
            if env_id is not None:
                keep_alive = self._noexit and self._completed_groups == self._total_groups
                self._teardown_env_clients(env_id)
                if keep_alive:
                    _log.info(f"--noexit: keeping env {env_id} alive after last group {rep_step.key}")
                else:
                    await self._best_effort_despawn(env_id, 2.0)
        return False

    def _publish_state(self, results: typing.Mapping[str, StepResult], steps_total: int) -> None:
        if not rclpy.ok():
            return
        message = BenchmarkState()
        message.stamp = self.get_clock().now().to_msg()
        message.run_id = self._run_id
        message.suite = self._suite.name
        message.contest = self._contest.name
        message.simulator = self._simulator or ""
        message.env_n = self._env_n
        message.headless = self._headless
        message.steps_total = steps_total
        counts = collections.Counter(result.status for result in results.values())
        (
            message.steps_done,
            message.steps_partial,
            message.steps_failed,
            message.steps_skipped,
            message.steps_in_flight,
        ) = (
            counts["ok"],
            counts["partial"],
            counts["failed"],
            counts["skipped"],
            counts["in_progress"],
        )
        message.active_keys = [key for key, result in results.items() if result.status == "in_progress"]
        self._state_pub.publish(message)

    async def _worker(
        self,
        slot_index: int,
        block_queues: list[tuple],
        attached: collections.Counter,
        claim: typing.Callable[[str], bool],
        flush_cb: typing.Callable,
    ) -> bool:
        try:
            while (index := pick_block(block_queues, attached, claim)) is not None:
                rep_step, queue, _ = block_queues[index]
                self._sim_used = True
                attached[id(queue)] += 1
                try:
                    abort = await self._run_group_queue(rep_step, queue, slot_index, flush_cb)
                finally:
                    attached[id(queue)] -= 1
                if abort:
                    return True
            return False
        finally:
            if self._progress is not None:
                self._progress.clear_slot(slot_index)

    async def _run_batch(
        self,
        blocks: list[list],
        claim_keys: list[str],
        claim: typing.Callable[[str], bool],
        flush_cb: typing.Callable,
    ) -> bool:
        block_queues: list[tuple] = []
        for block, claim_key in zip(blocks, claim_keys, strict=True):
            queue: asyncio.Queue = asyncio.Queue()
            for step in block:
                queue.put_nowait(step)
            block_queues.append((block[0], queue, claim_key))
        cap = max(1, min(self._env_n, sum(len(block) for block in blocks) or 1))
        attached: collections.Counter[int] = collections.Counter()
        in_flight = {
            asyncio.create_task(
                self._worker(slot, block_queues, attached, claim, flush_cb),
                name=f"worker_{slot}",
            )
            for slot in range(cap)
        }
        try:
            while in_flight:
                done, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                if any(task.result() for task in done):
                    return True
            return False
        finally:
            for task in in_flight:
                task.cancel()
            if in_flight:
                with contextlib.suppress(Exception):
                    await asyncio.gather(*in_flight, return_exceptions=True)

    def _episode_id_offset(self) -> int:
        offset = 0
        episodes_dir = self._run_dir.path / "episodes"
        if episodes_dir.exists():
            for directory in episodes_dir.glob("episode_*"):
                try:
                    offset = max(offset, int(directory.name.split("_")[1]) + 1)
                except Exception:
                    pass
        return offset

    async def _run_steps(self) -> int:
        pending = self._build_pending()
        results: dict[str, StepResult] = dict(self._run_dir.state.steps)
        steps_total = len(results) + len(pending)
        self._publish_state(results, steps_total)
        _log.info(f"benchmark: signalled READY on {STATE_TOPIC}")
        await self._start_arena()
        self._global_episode_id_offset = self._episode_id_offset()
        self._total_groups = len(group_pending(pending, self._simulator, self._world_swap))
        step_map: dict[str, Step] = {
            step.key: step
            for step in (
                *_all_steps_grid(self._suite, self._contest, self._scale_episodes, self._run_dir.path),
                *pending,
            )
        }
        self._progress = BenchmarkProgressDisplay(
            title=f"Arena Benchmark: {self._suite.name} • {self._contest.name}",
            total_steps=steps_total,
            env_n=self._env_n,
            run_id=self._run_id,
            sim_now=lambda: self.sim_time.to_seconds(),
            live_path=self._run_dir.live_path,
            lane=self._run_dir.lane,
        )
        owned: set[str] = set()

        def _write_state() -> None:
            self._run_dir.state.write(results if self._shared is None else {key: results[key] for key in owned})

        def _flush_step_result(res: StepResult) -> bool:
            results[res.key] = res
            elapsed = (res.ended_at or time.time()) - res.started_at
            lockstep_note = f" {res.lockstep.short()}" if res.lockstep is not None and res.lockstep.active else ""
            _log.info(f"[{res.status}] {res.key} env={res.env_id} episodes={res.episodes_run}/{res.episodes_total} (failed={res.episodes_failed}) t={elapsed:.1f}s{lockstep_note}")
            owned.add(res.key)
            _write_state()
            self._publish_state(results, steps_total)
            if self._progress is not None:
                step = step_map.get(res.key)
                self._progress.log_step_completed(
                    step_key=res.key,
                    status=res.status,
                    contestant=step.contestant.name if step else res.key,
                    stage=step.stage.name if step else "",
                    episodes_run=res.episodes_run,
                    episodes_total=res.episodes_total,
                    episodes_failed=res.episodes_failed,
                    elapsed_sec=elapsed,
                    error_detail=res.error_detail,
                    sim_sec=res.sim_s,
                )
            return sum(r.episodes_run for r in results.values()) == 0 and res.status == "failed" and res.error_kind in _SYSTEMIC

        world_maps: list[str | None] = [None] if self._world_swap else list(dict.fromkeys(step.stage.map for step in pending))
        aborted_systemic = False
        self._sim_used = False

        def _claim(claim_key: str) -> bool:
            return self._shared is None or self._run_dir.claim(self._shared, claim_key)

        with self._progress:
            try:
                for world_idx, world_map in enumerate(world_maps):
                    world_steps = [step for step in pending if world_map is None or step.stage.map == world_map]
                    if not world_steps:
                        continue
                    all_blocks = group_pending(world_steps, self._simulator, self._world_swap)
                    all_claim_keys = [block_claim_key(block, world_map, self._simulator, self._world_swap) for block in all_blocks]
                    if self._shared is not None and all(self._run_dir.claimed(self._shared, key) for key in all_claim_keys):
                        _log.info(f"benchmark: every block of world {world_map} is claimed by other lanes, skipping it")
                        continue
                    # gz-sim 8 stops answering world control once a robot with a depth camera was deleted and another spawned in the same session, so such robots get a fresh runtime per block
                    fresh_per_block = self._simulator == "gazebo" and any(robot_has_depth_camera(step.robot) for step in world_steps)
                    batches = [[index] for index in range(len(all_blocks))] if fresh_per_block else [list(range(len(all_blocks)))]
                    for batch_idx, batch in enumerate(batches):
                        if (world_idx > 0 or batch_idx > 0) and self._simulator == "gazebo" and (self._shared is None or self._sim_used):
                            if batch_idx > 0:
                                _log.info("benchmark: restarting arena_runtime for the next contestant, the robot carries a depth camera")
                            await self._restart_arena()
                            self._sim_used = False
                        if await self._run_batch(
                            [all_blocks[index] for index in batch],
                            [all_claim_keys[index] for index in batch],
                            _claim,
                            _flush_step_result,
                        ):
                            aborted_systemic = True
                            _log.error("benchmark: worker hit a systemic setup failure; aborting run")
                            break
                    if aborted_systemic:
                        break
            finally:
                self._run_dir.progress.dedupe_in_place({key: result.started_at for key, result in results.items()})
                self._publish_state(results, steps_total)

        rows = [
            (
                step_map[key].contestant.name if key in step_map else key,
                step_map[key].stage.name if key in step_map else "",
                result.lockstep,
            )
            for key, result in results.items()
            if result.lockstep is not None and result.lockstep.active
        ]
        lockstep_failed = False
        if rows:
            print(f"\nbenchmark: lockstep report\n{format_report(rows)}", flush=True)
            lockstep_failed = any(summary.verdict == "fail" for _, _, summary in rows)
        compromised = compromised_table(results, step_map)
        if compromised is not None:
            print(f"\nbenchmark: compromised episodes\n{compromised}", flush=True)
        strict_failed = sorted(key for key, result in results.items() if result.error_kind is not None or result.episodes_run < result.episodes_total)
        if self._strict and strict_failed:
            print(
                f"\nbenchmark: strict verdict: {len(strict_failed)} cell(s) did not complete cleanly: {', '.join(strict_failed)}",
                flush=True,
            )
        efficacy_failed = False
        if self._efficacy is not None:
            table = [
                (
                    step_map[key].contestant.name if key in step_map else key,
                    step_map[key].stage.name if key in step_map else "",
                    cell_verdict(result),
                    f"{result.episodes_run}/{result.episodes_total}",
                    str(result.episodes_weak),
                    f"{result.episodes_worst_progress * 100:.0f}%" if result.episodes_worst_progress is not None else "-",
                )
                for key, result in sorted(results.items())
            ]
            print(
                f"\nbenchmark: preflight verdict\n{format_table(('contestant', 'stage', 'verdict', 'episodes', 'weak', 'worst_progress'), table)}",
                flush=True,
            )
            efficacy_failed = any(cell_verdict(result) in ("wedged", "weak") for result in results.values())
        if aborted_systemic:
            return 1
        return 3 if (self._lockstep_verdict and lockstep_failed) or (self._strict and strict_failed) or efficacy_failed else 0
