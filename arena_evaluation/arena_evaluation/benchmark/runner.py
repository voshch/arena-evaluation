from __future__ import annotations

import argparse
import asyncio
import collections
import contextlib
import datetime
import logging
import os
import pathlib
import re
import signal
import subprocess
import sys
import threading
import time
import types
import typing

_T = typing.TypeVar("_T")

import attrs
import rclpy
import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from arena_evaluation_msgs.msg import BenchmarkState
from arena_evaluation_msgs.srv import RecordEpisode
from arena_rclpy_mixins import ActionClientWrapper, ArenaMixinNode, ClientWrapper
from arena_rclpy_mixins.spin import start_loop_watchdog
from arena_runtime_msgs.msg import EnvRecord, EnvRegistry, LockstepStatus, SimState
from arena_runtime_msgs.srv import DespawnEnv, SpawnEnv
from arena_simulation_setup.tree import ResolverVerdict
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Bool
from task_generator.constants import Constants
from task_generator_msgs.action import RunEpisode
from task_generator_msgs.msg import EpisodeRecord
from task_generator_msgs.srv import QueueEpisode

STATE_TOPIC = "/arena/benchmark/state"
SIM_STATE_TOPIC = "/arena/state/sim"

_CANCEL_SETTLE_S = 30.0
_HEARTBEAT_S = 30.0
_ARENA_SIGINT_GRACE_S = 15.0
_ARENA_SIGTERM_GRACE_S = 5.0
_ARENA_ORPHAN_GRACE_S = 1.0
_SIM_STALL_S = 60.0
_MAX_SIM_DEATHS = 3
_LOOP_DEADLINE_S = 120.0
_HUNG_EXIT_CODE = 4

# EpisodeRecord.outcome_state labels (task_generator_msgs/msg/EpisodeRecord.msg)
_EPISODE_OUTCOME_LABELS = {0: "QUEUED", 1: "RUNNING", 2: "SUCCESS", 3: "FAILED", 4: "SKIPPED", 5: "FATAL"}


from ..storage.planner_names import split_planner_name
from .config import Contest, Suite
from .lockstep import LockstepMonitor, LockstepSummary, format_report, format_table
from .progress_display import BenchmarkProgressDisplay
from .state import (
    Manifest,
    RunDir,
    capture_git_sha,
    compute_config_hash,
    find_most_recent_resumable,
)
from .step import Step, StepErrorKind, StepResult
from .tree import ContestIdentifier, SuiteIdentifier


def _proc_starttime(pid: int) -> int | None:
    """Start tick of a pid, None once it is gone."""
    try:
        with open(f"/proc/{pid}/stat") as fh:
            stat = fh.read()
    except OSError:
        return None
    fields = stat.rpartition(")")[2].split()
    return int(fields[19])


def _proc_tree(root: int) -> dict[int, int]:
    """Descendants of root as pid -> start tick."""
    children: dict[int, list[int]] = collections.defaultdict(list)
    starts: dict[int, int] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            with open(f"/proc/{pid}/stat") as fh:
                stat = fh.read()
        except OSError:
            continue
        fields = stat.rpartition(")")[2].split()
        children[int(fields[1])].append(pid)
        starts[pid] = int(fields[19])
    tree: dict[int, int] = {}
    stack = [root]
    while stack:
        for child in children.get(stack.pop(), ()):
            if child not in tree:
                tree[child] = starts[child]
                stack.append(child)
    return tree


def _orphaned_env_ids(known: set[int], records: typing.Mapping[int, object], registered: typing.Sequence[int]) -> list[int]:
    """Env ids to despawn after a stalled spawn: registered first, then any env reserved during the wait that never went ready."""
    ids = list(dict.fromkeys(registered))
    for env_id, rec in records.items():
        if env_id in known or env_id in ids:
            continue
        if not rec.ready:
            ids.append(env_id)
    return ids


def closed_fraction(goal_dist_start: float, goal_dist_min: float) -> float:
    """Fraction of the starting goal distance closed by the closest approach, 0 when there was no goal."""
    if goal_dist_start <= 0.0:
        return 0.0
    return max(0.0, min(1.0, (goal_dist_start - goal_dist_min) / goal_dist_start))


def cell_verdict(result: StepResult) -> str:
    """Efficacy classification for one cell: wedged (never finished), weak (finished but drove badly), or ok."""
    if result.error_kind is not None or result.episodes_run < result.episodes_total:
        return "wedged"
    if result.episodes_weak > 0:
        return "weak"
    return "ok"


class _WithSteps(typing.Protocol):
    steps: dict[str, StepResult]


class _HasStateSteps(typing.Protocol):
    """Structural interface required by build_pending: any object with .state.steps."""

    @property
    def state(self) -> _WithSteps: ...


_log = logging.getLogger(__name__)

_CAP_KEYS = ("mobile", "arm", "planner")


def _launch_key(k: str) -> str:
    """Contest cap keys stay bare (`mobile`, `arm.<x>`, `planner`), launch args live under `robot.`."""
    head = k.split(".", 1)[0]
    return f"robot.{k}" if head in _CAP_KEYS else k


def build_launch_args(step: Step, simulator: str | None, passthrough: dict[str, str] | None = None) -> list[str]:
    """Return the arena launch argument list for a step, given the simulator name."""
    s = step.stage
    args = [
        *([f"sim:={simulator}"] if simulator is not None else []),
        f"robot:={s.robot}",
        f"world:={s.map}",
        f"task.robots:={s.tm_robots.value}",
        f"task.obstacles:={s.tm_obstacles.value}",
        f"run_seed:={s.seed}",
        "task.auto_reset:=false",
        "task.modules:=",
    ]
    if s.optim:
        for k, v in s.optim.items():
            args.append(f"optim.{k}:={v}")
    if step.record_dir is not None:
        args.append(f"record.dir:={step.record_dir}")
        args.append("record.auto:=false")
    own_keys = {a.split(":=", 1)[0] for a in args}
    for raw_k, v in step.contestant.args.items():
        k = _launch_key(raw_k)
        if isinstance(v, dict):
            driver = v.get("driver")
            if driver:
                if k in own_keys:
                    _log.warning(
                        "contestant %r: arg %r=%r ignored, controlled by stage",
                        step.contestant.name,
                        k,
                        driver,
                    )
                else:
                    args.append(f"{k}:={driver}")
            for ik, iv in v.items():
                if ik == "driver" or not iv:
                    continue
                fk = f"{k}.{ik}"
                if fk in own_keys:
                    _log.warning(
                        "contestant %r: arg %r=%r ignored, controlled by stage",
                        step.contestant.name,
                        fk,
                        iv,
                    )
                    continue
                args.append(f"{fk}:={iv}")
        else:
            if not v:
                continue
            if k in own_keys:
                _log.warning(
                    "contestant %r: arg %r=%r ignored, controlled by stage",
                    step.contestant.name,
                    k,
                    v,
                )
                continue
            args.append(f"{k}:={v}")

    if passthrough:
        for k, v in passthrough.items():
            if k in ("headless", "env_n", "env.n"):
                continue
            if k not in own_keys:
                args.append(f"{k}:={v}")
    return args


def _all_steps_grid(
    suite: Suite,
    contest: Contest,
    scale_episodes: float,
    record_root: pathlib.Path | None = None,
) -> list[Step]:
    """Generate all steps for a benchmark suite and contest."""
    steps: list[Step] = []
    seen: set[str] = set()

    episodes_dir = (record_root / "episodes") if record_root is not None else None

    for contestant in contest.contestants:
        for stage in suite.stages:
            main_step = Step(
                contestant=contestant,
                stage=stage,
                episodes=int(round(stage.episodes * scale_episodes)),
                record_dir=episodes_dir,
            )
            if not suite.references:
                if main_step.key in seen:
                    raise ValueError(f"duplicate step key: {main_step.key!r}")
                seen.add(main_step.key)
                steps.append(main_step)
                continue
            ref_robot_step = Step(
                contestant=contestant,
                stage=stage,
                episodes=int(round(stage.episodes * scale_episodes)),
                record_dir=episodes_dir,
                is_reference=True,
                reference_type="unobstructed_robot",
            )
            for step in (main_step, ref_robot_step):
                if step.key in seen:
                    raise ValueError(f"duplicate step key: {step.key!r}")
                seen.add(step.key)
                steps.append(step)

    if not suite.references:
        return steps

    dummy_peds_contestant = Contest.Contestant(name="unhindered_peds", args={})
    for stage in suite.stages:
        ref_peds_step = Step(
            contestant=dummy_peds_contestant,
            stage=stage,
            episodes=int(round(stage.episodes * scale_episodes)),
            record_dir=episodes_dir,
            is_reference=True,
            reference_type="unhindered_peds",
        )
        if ref_peds_step.key in seen:
            raise ValueError(f"duplicate step key: {ref_peds_step.key!r}")
        seen.add(ref_peds_step.key)
        steps.append(ref_peds_step)

    return steps


def build_pending(
    suite: Suite,
    contest: Contest,
    scale_episodes: float,
    run_dir: _HasStateSteps,
    retry_failed: bool,
    record_root: pathlib.Path,
) -> list[Step]:
    """Return the list of steps that still need to be run."""
    state_steps = run_dir.state.steps
    pending: list[Step] = []
    for step in _all_steps_grid(suite, contest, scale_episodes, record_root):
        existing = state_steps.get(step.key)
        if existing is None:
            pending.append(step)
            continue
        if existing.status == "ok":
            continue
        if existing.status == "failed" and not retry_failed:
            continue
        pending.append(step)
    return pending


def resolve_planner_identity(contestant: Contest.Contestant) -> tuple[str, str]:
    """Return (local_planner, inter_planner) for a contestant."""
    mobile = (contestant.args or {}).get("mobile")
    if isinstance(mobile, dict):
        lp = mobile.get("local_planner")
        if lp:
            ip = mobile.get("inter_planner")
            return str(lp), str(ip) if ip else "none"
    return split_planner_name(contestant.name)


def _preflight_contest(contest: Contest) -> list[str]:
    """Check nav2 contestants reference a local_planner/inter_planner that actually exists."""
    problems: list[str] = []
    try:
        share = pathlib.Path(get_package_share_directory("arena_robots")) / "config" / "nav2"
    except PackageNotFoundError:
        return ["arena_robots is not installed, cannot validate nav2 contestants"]

    for contestant in contest.contestants:
        mobile = contestant.args.get("mobile")
        if not isinstance(mobile, dict) or mobile.get("driver") != "nav2":
            continue

        lp = mobile.get("local_planner")
        if lp:
            path = share / "controllers" / lp / "controller_config.yaml"
            if not path.is_file():
                available = sorted(d.name for d in (share / "controllers").iterdir() if d.is_dir())
                problems.append(f"{contestant.name}: local_planner {lp!r} has no {path} (available: {available})")

        ip = mobile.get("inter_planner")
        if ip and ip != "none":
            path = share / "interplanners" / ip / "interplanner_config.yaml"
            if not path.is_file():
                available = sorted(d.name for d in (share / "interplanners").iterdir() if d.is_dir())
                problems.append(f"{contestant.name}: inter_planner {ip!r} has no {path} (available: {available})")

    return problems


def env_key(step: Step, simulator: str | None) -> tuple:
    """Steps with the same env_key reuse one env. Changing contestant, robot, or map forces a fresh env."""
    return (step.contestant.name, step.stage.robot, step.stage.map, simulator)


def group_pending(steps: list[Step], simulator: str | None) -> list[list[Step]]:
    groups = collections.defaultdict(list)
    peds_steps = []

    for step in steps:
        if step.is_reference and step.reference_type == "unhindered_peds":
            peds_steps.append(step)
        else:
            groups[env_key(step, simulator)].append(step)

    sorted_groups = sorted(groups.values(), key=len, reverse=True)

    for step in peds_steps:
        placed = False
        for g in sorted_groups:
            if g[0].stage.robot == step.stage.robot and g[0].stage.map == step.stage.map:
                g.append(step)
                placed = True
                break
        if not placed:
            sorted_groups.append([step])

    return sorted_groups


def _walk_dict(d: dict, prefix: str = "") -> list[Parameter]:
    """Flatten a nested dict to rcl_interfaces Parameter[] with dot-joined leaf names."""
    out: list[Parameter] = []
    for k, v in d.items():
        name = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            if "min" in v and "max" in v:
                n_name = f"{name}.n"
                pv = ParameterValue()
                val_min = v["min"]
                val_max = v["max"]
                if isinstance(val_min, int) and isinstance(val_max, int):
                    pv.type = ParameterType.PARAMETER_INTEGER_ARRAY
                    pv.integer_array_value = [val_min, val_max]
                else:
                    pv.type = ParameterType.PARAMETER_DOUBLE_ARRAY
                    pv.double_array_value = [float(val_min), float(val_max)]
                p = Parameter()
                p.name = n_name
                p.value = pv
                out.append(p)

                other = {ik: iv for ik, iv in v.items() if ik not in ("min", "max")}
                if other:
                    out.extend(_walk_dict(other, name))
                continue
            out.extend(_walk_dict(v, name))
            continue
        pv = ParameterValue()
        if isinstance(v, bool):
            pv.type = ParameterType.PARAMETER_BOOL
            pv.bool_value = v
        elif isinstance(v, int):
            pv.type = ParameterType.PARAMETER_INTEGER
            pv.integer_value = v
        elif isinstance(v, float):
            pv.type = ParameterType.PARAMETER_DOUBLE
            pv.double_value = v
        elif isinstance(v, str):
            pv.type = ParameterType.PARAMETER_STRING
            pv.string_value = v
        elif isinstance(v, list):
            if not v:
                pv.type = ParameterType.PARAMETER_STRING_ARRAY
                pv.string_array_value = []
            elif all(isinstance(x, bool) for x in v):
                pv.type = ParameterType.PARAMETER_BOOL_ARRAY
                pv.bool_array_value = v
            elif all(isinstance(x, int) for x in v):
                pv.type = ParameterType.PARAMETER_INTEGER_ARRAY
                pv.integer_array_value = v
            elif all(isinstance(x, float) for x in v):
                pv.type = ParameterType.PARAMETER_DOUBLE_ARRAY
                pv.double_array_value = v
            elif all(isinstance(x, str) for x in v):
                pv.type = ParameterType.PARAMETER_STRING_ARRAY
                pv.string_array_value = v
            else:
                raise TypeError(f"unsupported mixed list type for {name!r}")
        else:
            raise TypeError(f"unsupported param type for {name!r}: {type(v).__name__}")
        p = Parameter()
        p.name = name
        p.value = pv
        out.append(p)
    return out


def _flatten_per_mode_params(
    stage_config: dict,
    *,
    tm_obstacles: str,
    tm_robots: str,
) -> tuple[list[Parameter], list[Parameter]]:
    """Route stage.config blocks to (obstacles_params, robots_params) as leaf-keyed Parameter[]."""
    obs: list[Parameter] = []
    rob: list[Parameter] = []
    for mode, mode_dict in (stage_config or {}).items():
        if not isinstance(mode_dict, dict):
            continue
        is_scenario = mode == "scenario"
        patched: dict = {k: (pathlib.Path(val).stem if is_scenario and k == "file" and isinstance(val, str) else val) for k, val in mode_dict.items()}
        params = _walk_dict(patched)
        if mode == tm_obstacles:
            obs.extend(params)
        if mode == tm_robots:
            rob.extend(params)
    return obs, rob


_LATCHED = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)

_SYSTEMIC = (StepErrorKind.ENV_SETUP, StepErrorKind.ROBOT_SETUP, StepErrorKind.SIM_DEAD)

_SPAWN_CHATTER_PATTERNS = (
    "waiting on env_",
    "waiting on sim clock step",
    "waiting on response from",
    "still waiting for",
    "still loading spawn_env",
    "waiting on spawn_env",
    "timed out after",
    "sim cannot keep up",
)


def _is_substantive_log_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    for pat in _SPAWN_CHATTER_PATTERNS:
        if pat in s:
            return False
    return True


class _EnvDied(Exception):
    """Raised when an env disappears from /arena/state/envs while the runner was waiting on it."""


class _SimDied(Exception):
    """Raised when the sim reports itself dead or the arena runtime exits while the runner was waiting."""


class _SimStalled(Exception):
    """Raised when the sim clock stops advancing outside a reset while an episode is running."""


def _requeue_front(q: asyncio.Queue[Step], step: Step) -> None:
    """Put step back at the head of q."""
    rest: list[Step] = []
    while not q.empty():
        rest.append(q.get_nowait())
    q.put_nowait(step)
    for s in rest:
        q.put_nowait(s)


@attrs.define
class _RetryBudget:
    """Retry bookkeeping: `retries` attempts per step key, a hard cap of sim deaths per run."""

    retries: int = 1
    max_sim_deaths: int = _MAX_SIM_DEATHS
    generations: set[int] = attrs.field(factory=set)
    attempts: collections.Counter = attrs.field(factory=collections.Counter)

    @property
    def sim_deaths(self) -> int:
        return len(self.generations)

    @property
    def limit(self) -> str:
        """The retry cap as it is logged."""
        return "inf" if self.retries < 0 else str(self.retries)

    def record(self, step_key: str | None, *, sim_death_generation: int | None) -> tuple[bool, bool]:
        """Count one systemic failure. Returns (may_retry, run_must_abort)."""
        if sim_death_generation is not None:
            self.generations.add(sim_death_generation)
            if self.sim_deaths >= self.max_sim_deaths:
                return False, True
        if step_key is None:
            return False, False
        spent = self.attempts[step_key]
        self.attempts[step_key] = spent + 1
        return self.retries < 0 or spent < self.retries, False


class BenchmarkRunner(ArenaMixinNode):
    exit_code: typing.ClassVar[int] = 0

    @classmethod
    def run_main(cls, *args: object, aiomonitor: bool = False, **kwargs: object) -> None:
        """Run benchmark runner with clean lifecycle, non-blocking executor, and instant shutdown on Ctrl+C."""
        import rclpy
        from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
        from rclpy.signals import SignalHandlerOptions

        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        executor = MultiThreadedExecutor()
        node: BenchmarkRunner | None = None

        def _spin():
            try:
                executor.spin()
            except (ExternalShutdownException, Exception):
                pass

        spin_future = loop.run_in_executor(None, _spin)
        main_task: asyncio.Task | None = None

        def _sig_handler(signum: int, _frame: types.FrameType | None) -> None:
            if main_task and not main_task.done():
                loop.call_soon_threadsafe(main_task.cancel)

        prev_sigint = signal.signal(signal.SIGINT, _sig_handler)
        prev_sigterm = signal.signal(signal.SIGTERM, _sig_handler)
        watchdog_stop: threading.Event | None = None

        async def _run_app():
            nonlocal node, watchdog_stop
            node = cls(*args, **kwargs)
            node.event_loop = loop
            watchdog_stop = start_loop_watchdog(loop, node, deadline_s=_LOOP_DEADLINE_S, on_deadline=node._deadman)
            executor.add_node(node)
            try:
                await node.setup()
            except asyncio.CancelledError:
                cls.exit_code = 130
            except Exception as e:
                _log.error(f"Benchmark run error: {e!r}")
                cls.exit_code = 2

        try:
            main_task = loop.create_task(_run_app())
            loop.run_until_complete(main_task)
        except (KeyboardInterrupt, asyncio.CancelledError):
            cls.exit_code = 130
        finally:
            if watchdog_stop is not None:
                watchdog_stop.set()
            # Ensure arena runtime subprocess is terminated cleanly
            if node is not None:
                with contextlib.suppress(Exception):
                    loop.run_until_complete(node._shutdown_arena())
                if node._progress is not None:
                    node._progress.stop()
            # Clean up pending tasks in event loop
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                with contextlib.suppress(Exception):
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            executor.shutdown()
            with contextlib.suppress(Exception):
                loop.run_until_complete(spin_future)
            if node is not None:
                with contextlib.suppress(Exception):
                    executor.remove_node(node)
                    node.destroy_node()
            rclpy.try_shutdown()
            loop.close()
            signal.signal(signal.SIGINT, prev_sigint)
            signal.signal(signal.SIGTERM, prev_sigterm)

    def __init__(
        self,
        suite: Suite,
        contest: Contest,
        *,
        simulator: str | None,
        scale_episodes: float,
        env_n: int,
        run_id: str,
        headless: bool,
        run_dir: RunDir,
        retry_failed: bool = False,
        arena_passthrough: dict[str, str] | None = None,
        noexit: bool = False,
        suite_bundle_dir: pathlib.Path | None = None,
        lockstep_verdict: bool = False,
        strict: bool = False,
        retries: int = 1,
        max_sim_deaths: int = _MAX_SIM_DEATHS,
        spawn_budget: float = 600.0,
        efficacy: float | None = None,
    ) -> None:
        super().__init__("arena_benchmark_runner")
        self._suite = suite
        self._suite_bundle_dir = suite_bundle_dir
        self._contest = contest
        self._simulator = simulator
        self._scale_episodes = scale_episodes
        self._env_n = env_n
        self._run_id = run_id
        self._headless = headless
        self._run_dir = run_dir
        self._retry_failed = retry_failed
        self._arena_passthrough: dict[str, str] = dict(arena_passthrough or {})
        self._viz = self._arena_passthrough.get("viz", str(not self._headless)).lower() in ("true", "1")
        self._viz_proc: subprocess.Popen | None = None
        self._noexit = noexit
        self._lockstep_verdict = lockstep_verdict
        self._strict = strict
        self._retries = retries
        self._max_sim_deaths = max_sim_deaths
        self._spawn_budget = spawn_budget
        self._efficacy = efficacy
        self._lockstep = LockstepMonitor()
        self._total_groups = 0
        self._completed_groups = 0

        self._spawn = self.create_client_wrapper(SpawnEnv, "/arena/spawn_env")
        self._despawn = self.create_client_wrapper(DespawnEnv, "/arena/despawn_env")
        self._env_records: dict[int, EnvRecord] = {}
        self._env_visible_events: dict[int, asyncio.Event] = {}
        self._env_gone_events: dict[int, asyncio.Event] = {}

        self._arena_proc: subprocess.Popen | None = None
        self._arena_log_file = None
        self._total_groups = 0
        self._episode_action_clients: dict[int, ActionClientWrapper] = {}
        self._queue_clients: dict[int, ClientWrapper] = {}
        self._param_clients: dict[int, ClientWrapper] = {}
        self._param_get_clients: dict[int, ClientWrapper] = {}

        self._episode_records: dict[int, dict[int, EpisodeRecord]] = {}
        self._env_subs: dict[int, list] = {}
        self._recorder_clients: dict[int, ClientWrapper] = {}
        self._triggered_episodes: dict[int, set[int]] = {}
        self._env_resetting: dict[int, bool] = {}

        self._sim_dead = asyncio.Event()
        self._sim_dead_reason = ""
        self._sim_generation = 0
        self._sim_recover_lock = asyncio.Lock()
        self._retry_budget = _RetryBudget(retries=self._retries, max_sim_deaths=self._max_sim_deaths)
        self._runtime_started_at = 0.0
        self._arena_watch: asyncio.Task | None = None

        self.create_subscription(EnvRegistry, "/arena/state/envs", self._on_envs, _LATCHED)
        self.create_subscription(SimState, SIM_STATE_TOPIC, self._on_sim_state, _LATCHED)
        self.create_subscription(LockstepStatus, "/arena/state/lockstep", self._on_lockstep, _LATCHED)
        self._state_pub = self.create_publisher(BenchmarkState, STATE_TOPIC, _LATCHED)

        self._arena_proc: subprocess.Popen | None = None
        self._parent_episode_map: dict[tuple[str, str, int], int] = {}
        self._progress: BenchmarkProgressDisplay | None = None

    def _build_pending(self) -> list[Step]:
        return build_pending(
            suite=self._suite,
            contest=self._contest,
            scale_episodes=self._scale_episodes,
            run_dir=self._run_dir,
            retry_failed=self._retry_failed,
            record_root=self._run_dir.path,
        )

    def _set_event(self, event: asyncio.Event) -> None:
        """Set an asyncio.Event from any thread. Off the loop thread, executor callbacks included, the set is scheduled on it."""
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self.event_loop:
            event.set()
            return
        self.event_loop.call_soon_threadsafe(event.set)

    def _on_envs(self, msg: EnvRegistry) -> None:
        new_ids = {e.env_id for e in msg.envs}
        for env_id in new_ids:
            self._set_event(self._env_visible_events.setdefault(env_id, asyncio.Event()))
        for env_id in list(self._env_gone_events):
            if env_id not in new_ids:
                self._set_event(self._env_gone_events[env_id])
        self._env_records = {e.env_id: e for e in msg.envs}

    def _on_sim_state(self, msg: SimState) -> None:
        if msg.alive:
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp < self._runtime_started_at:
            _log.debug(f"ignoring stale sim death stamped {stamp:.3f}, this runtime started at {self._runtime_started_at:.3f}")
            return
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
            with (self._run_dir.path / "runner.log").open("a") as fh:
                fh.write(f"benchmark: runner event loop hung for {_LOOP_DEADLINE_S:.0f}s, killing arena runtime and exiting {_HUNG_EXIT_CODE}\n")
        p = self._arena_proc
        if p is not None and p.poll() is None:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        os._exit(_HUNG_EXIT_CODE)

    def _on_lockstep(self, msg: LockstepStatus) -> None:
        self._lockstep.observe(msg, time.time())

    def _build_launch_args(self, step: Step) -> list[str]:
        return build_launch_args(step, self._simulator, passthrough=self._arena_passthrough)

    async def _await_env_visible(self, env_id: int) -> None:
        """Wait for env_id to appear on /arena/state/envs."""
        if env_id in self._env_records:
            return
        await self._await_hb(self._env_visible_events.setdefault(env_id, asyncio.Event()).wait(), f"env {env_id} to appear on /arena/state/envs")

    async def _await_activity(
        self,
        awaitable: typing.Awaitable[_T],
        what: str,
        *,
        inactivity_timeout: float = 120.0,
        max_total_timeout: float = 600.0,
        check_interval: float = 15.0,
    ) -> _T:
        """Await task as long as non-chatter progress is detected in logs.
        Times out if there is no substantive progress for `inactivity_timeout` seconds,
        or if `max_total_timeout` is exceeded."""
        task = asyncio.ensure_future(awaitable)
        log_path = self._run_dir.path / "runner.log"
        last_positions: dict[pathlib.Path, int] = {}
        last_activity = time.monotonic()
        total_waited = 0.0
        ros_log_dir = pathlib.Path.home() / ".ros" / "log"

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

                has_progress = False
                sample_line = ""

                target_logs: list[pathlib.Path] = []
                if log_path.exists():
                    target_logs.append(log_path)
                if ros_log_dir.is_dir():
                    try:
                        recent_cutoff = time.time() - 3600
                        for p in ros_log_dir.glob("arena_env_*.log"):
                            if p.stat().st_mtime >= recent_cutoff:
                                target_logs.append(p)
                    except Exception:
                        pass

                for p in target_logs:
                    try:
                        curr_size = p.stat().st_size
                        last_pos = last_positions.get(p, 0)
                        if curr_size < last_pos:
                            last_pos = 0
                        if curr_size > last_pos:
                            with open(p, encoding="utf-8", errors="replace") as f:
                                f.seek(last_pos)
                                new_text = f.read()
                                last_positions[p] = f.tell()
                            for line in new_text.splitlines():
                                if _is_substantive_log_line(line):
                                    has_progress = True
                                    sample_line = line.strip()[:80]
                                    break
                    except Exception:
                        pass

                if has_progress:
                    last_activity = time.monotonic()
                    _log.info(f"still loading {what} ({total_waited:.0f}s elapsed, active: {sample_line})")
                else:
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
            for t in (op_task, *watch):
                t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(op_task, *watch, return_exceptions=True)

    async def _wait_env_gone(self, env_id: int, *, timeout: float | None) -> bool:
        ev = asyncio.Event()
        self._env_gone_events[env_id] = ev
        try:
            if env_id not in self._env_records:
                return True
            await asyncio.wait_for(ev.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False
        finally:
            self._env_gone_events.pop(env_id, None)

    async def _setup_env_clients(self, env_id: int, env_ns_root: str, robot_name: str) -> None:
        """Create per-env action client, queue_episode client, and subscriptions. Idempotent."""
        if env_id in self._episode_action_clients:
            return

        action_name = f"{env_ns_root}/lifecycle/run_episode"
        self._episode_action_clients[env_id] = self.create_action_client_wrapper(RunEpisode, action_name)
        self._episode_records[env_id] = {}

        queue_client = self.create_client_wrapper(QueueEpisode, f"{env_ns_root}/config/queue_episode")
        what = f"queue_episode service on env {env_id}"
        await self._await_alive(self._await_hb(queue_client.ensure(timeout_sec=None), what), env_id=env_id, what=what)
        self._queue_clients[env_id] = queue_client
        self._param_clients[env_id] = self.create_client_wrapper(SetParameters, f"{env_ns_root}/set_parameters")
        self._param_get_clients[env_id] = self.create_client_wrapper(GetParameters, f"{env_ns_root}/get_parameters")

        self._recorder_clients[env_id] = self.create_client_wrapper(RecordEpisode, f"{env_ns_root}/start_episode")
        self._triggered_episodes[env_id] = set()

        def _on_episode_record(msg: EpisodeRecord) -> None:
            recs = self._episode_records.get(env_id)
            if recs is not None:
                recs[msg.episode_id] = msg
            label = _EPISODE_OUTCOME_LABELS.get(msg.outcome_state, f"UNKNOWN({msg.outcome_state})")
            _log.info(f"[env {env_id}] episode record: id={msg.episode_id} outcome={msg.outcome_state} ({label}) info={msg.outcome_info!r}")
            if msg.outcome_state in (EpisodeRecord.QUEUED, EpisodeRecord.RUNNING):
                triggered = self._triggered_episodes.get(env_id)
                if triggered is not None and msg.episode_id not in triggered:
                    triggered.add(msg.episode_id)
                    rc = self._recorder_clients.get(env_id)
                    if rc is not None:
                        _log.info(f"[env {env_id}] signalling recorder to start episode {msg.episode_id}")
                        req = RecordEpisode.Request()
                        req.command = RecordEpisode.Request.COMMAND_START
                        req.episode_id = msg.episode_id
                        fut = rc.client.call_async(req)
                        fut.add_done_callback(lambda f, eid=msg.episode_id: _log.warning(f"[env {env_id}] recorder start_episode({eid}) failed: {f.exception()}") if f.exception() else _log.info(f"[env {env_id}] recorder confirmed start of episode {eid}"))

        sub_ep = self.create_subscription(
            EpisodeRecord,
            f"{env_ns_root}/state/episode",
            _on_episode_record,
            10,
        )

        self._env_resetting[env_id] = False

        def _on_resetting(msg: Bool) -> None:
            self._env_resetting[env_id] = msg.data

        sub_reset = self.create_subscription(Bool, f"{env_ns_root}/state/resetting", _on_resetting, _LATCHED)
        self._env_subs[env_id] = [sub_ep, sub_reset]

    def _teardown_env_clients(self, env_id: int) -> None:
        """Destroy per-env subscriptions, action client, and queue_episode client."""
        for sub in self._env_subs.pop(env_id, []):
            self.destroy_subscription(sub)
        ac = self._episode_action_clients.pop(env_id, None)
        if ac is not None:
            ac.client.destroy()
        qc = self._queue_clients.pop(env_id, None)
        if qc is not None:
            qc.client.destroy()
        pc = self._param_clients.pop(env_id, None)
        if pc is not None:
            pc.client.destroy()
        gc = self._param_get_clients.pop(env_id, None)
        if gc is not None:
            gc.client.destroy()
        rc = self._recorder_clients.pop(env_id, None)
        if rc is not None:
            rc.client.destroy()
        self._triggered_episodes.pop(env_id, None)
        self._episode_records.pop(env_id, None)
        self._env_visible_events.pop(env_id, None)
        self._env_resetting.pop(env_id, None)

    @staticmethod
    def _episode_budget(step: Step) -> float:
        """Sim-second budget the task generator enforces for one episode of step."""
        if step.is_reference and step.reference_type == "unhindered_peds" and step.stage.timeout_peds is not None:
            return step.stage.timeout_peds
        return step.stage.timeout

    async def _push_stage_config(self, env_id: int, step: Step) -> None:
        queue = self._queue_clients[env_id]
        req = QueueEpisode.Request()
        req.action = QueueEpisode.Request.MERGE
        req.world = step.stage.map
        req.tm_robots = step.stage.tm_robots.value
        req.tm_obstacles = step.stage.tm_obstacles.value
        req.tm_modules = []
        req.keep_modules = False
        stage_config = step.stage.config or {}
        if step.is_reference and step.reference_type == "unobstructed_robot":
            import copy

            stage_config = copy.deepcopy(stage_config)
            mode_block = stage_config.setdefault(step.stage.tm_obstacles.value, {})
            if isinstance(mode_block, dict):
                mode_block["dynamic"] = {"min": 0, "max": 0}

        obs_params, rob_params = _flatten_per_mode_params(
            stage_config,
            tm_obstacles=step.stage.tm_obstacles.value,
            tm_robots=step.stage.tm_robots.value,
        )
        if step.is_reference:
            if step.reference_type == "unobstructed_robot":
                pass
            elif step.reference_type == "unhindered_peds":
                req.tm_robots = "stationary"
                rob_params = [Parameter(name="pos_x", value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=1000.0)), Parameter(name="pos_y", value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=1000.0))]
                param_client = self._param_clients[env_id]
                set_req = SetParameters.Request()
                set_req.parameters = [Parameter(name="task.stationary.pos_x", value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=1000.0)), Parameter(name="task.stationary.pos_y", value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=1000.0))]
                try:
                    await self._await_alive(param_client.call_timeout(set_req, timeout_sec=5.0), env_id=env_id, what=f"stationary params on env {env_id}")
                except (_SimDied, _EnvDied, asyncio.CancelledError):
                    raise
                except Exception as exc:
                    _log.warning(f"[env {env_id}] stationary params for {step.key} failed: {exc}")

        req.obstacles_params = obs_params
        req.robots_params = rob_params
        _log.info(f"[env {env_id}] pushing stage config for {step.key} (map={step.stage.map}, tm_robots={req.tm_robots}, tm_obstacles={req.tm_obstacles})")
        resp = await self._await_alive(queue.call_timeout(req, timeout_sec=10.0), env_id=env_id, what=f"queue_episode on env {env_id}")
        if resp is None:
            raise RuntimeError(f"queue_episode service call timed out after 10s on env {env_id} for {step.key}")
        if not resp.success:
            raise RuntimeError(f"queue_episode failed for {step.key}: {resp.error_msg}")

        budget = self._episode_budget(step)
        get_resp = await self._await_alive(self._param_get_clients[env_id].call_timeout(GetParameters.Request(names=["timeout"]), timeout_sec=10.0), env_id=env_id, what=f"episode timeout param type on env {env_id}")
        if get_resp is None or not get_resp.values:
            raise RuntimeError(f"get_parameters(timeout) failed on env {env_id} for {step.key}")
        if get_resp.values[0].type == ParameterType.PARAMETER_INTEGER:
            budget_value = ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=round(budget))
        else:
            budget_value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(budget))
        timeout_client = self._param_clients[env_id]
        timeout_req = SetParameters.Request()
        timeout_req.parameters = [Parameter(name="timeout", value=budget_value)]
        timeout_resp = await self._await_alive(timeout_client.call_timeout(timeout_req, timeout_sec=10.0), env_id=env_id, what=f"episode timeout param on env {env_id}")
        if timeout_resp is None:
            raise RuntimeError(f"set_parameters(timeout) timed out after 10s on env {env_id} for {step.key}")
        if not all(r.successful for r in timeout_resp.results):
            reasons = "; ".join(r.reason for r in timeout_resp.results if not r.successful)
            raise RuntimeError(f"set_parameters(timeout={budget}s) rejected on env {env_id} for {step.key}: {reasons}")
        _log.info(f"[env {env_id}] stage config applied for {step.key} (episode budget {budget:.0f}s sim)")

    async def _run_episodes(
        self,
        step: Step,
        env_id: int,
        slot_index: int = 0,
    ) -> StepResult:
        """Drive all episodes for one step. Env is already up and clients are set up."""
        started = time.time()
        episodes_run = 0
        episodes_failed = 0
        episodes_weak = 0
        worst_progress: float | None = None
        sim_s = 0.0
        stalled = False
        lockstep: LockstepSummary | None = None
        window: list[object] = []
        ac = self._episode_action_clients[env_id]

        def _fold_window() -> LockstepSummary | None:
            nonlocal lockstep
            if not window:
                return None
            summary = self._lockstep.close(window.pop(), time.time())
            lockstep = summary if lockstep is None else lockstep.merge(summary)
            return summary

        def _result(status: str, error_kind: StepErrorKind | None = None, error_detail: str | None = None) -> StepResult:
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
                episodes_worst_progress=worst_progress,
                episodes_total=step.episodes,
                sim_s=sim_s,
                lockstep=lockstep,
            )

        try:
            for ep_idx in range(step.episodes):
                if self._progress is not None:
                    self._progress.update_slot(
                        slot_index=slot_index,
                        env_id=env_id,
                        contestant=step.contestant.name,
                        stage=step.stage.name,
                        step_key=step.key,
                        ep_idx=ep_idx,
                        ep_total=step.episodes,
                        state="RUNNING",
                        sim_start=self.sim_time.to_seconds(),
                    )
                goal = RunEpisode.Goal()
                goal.world = step.stage.map
                goal.seed = (step.stage.seed + ep_idx) if step.stage.seed is not None else ep_idx

                window.append((step.key, ep_idx))
                self._lockstep.open(window[-1], time.time())
                ep_started_sim = self.sim_time.to_seconds()
                ep_started_wall = time.time()

                try:
                    goal_handle = await self._await_alive(
                        asyncio.wait_for(ac.send_goal(goal), timeout=15.0),
                        env_id=env_id,
                        what=f"run_episode goal on env {env_id}",
                    )
                except TimeoutError:
                    episodes_failed += 1
                    _log.error(f"[{ep_idx + 1}/{step.episodes}] {step.key} env={env_id} send_goal timed out after 15s; abandoning env")
                    return _result("failed", StepErrorKind.ENV_SETUP, f"send_goal timed out after 15s on env {env_id}")
                except (_SimDied, _EnvDied, asyncio.CancelledError):
                    raise
                except Exception as exc:
                    episodes_failed += 1
                    _log.error(f"[{ep_idx + 1}/{step.episodes}] {step.key} env={env_id} goal rejected ({exc}); env action server is wedged, abandoning env")
                    return _result("failed", StepErrorKind.ENV_SETUP, f"run_episode goal rejected: {exc}")

                try:
                    result_obj = await self._await_episode_result(ac, goal_handle, env_id)
                except _SimStalled:
                    episodes_failed += 1
                    stalled = True
                    _log.warning(f"[{ep_idx + 1}/{step.episodes}] {step.key} env={env_id} SIM STALLED for {_SIM_STALL_S:.0f}s; cancelling and advancing")
                    try:
                        await self.await_ros(goal_handle.cancel_goal_async())
                        await asyncio.wait_for(
                            self._await_alive(ac.await_result(goal_handle), env_id=env_id, what=f"run_episode cancel on env {env_id}"),
                            timeout=_CANCEL_SETTLE_S,
                        )
                    except (_SimDied, _EnvDied):
                        raise
                    except TimeoutError:
                        _log.error(f"[{ep_idx + 1}/{step.episodes}] {step.key} env={env_id} cancel did not settle in {_CANCEL_SETTLE_S}s; env is wedged, abandoning env")
                        return _result("failed", StepErrorKind.ENV_SETUP, "run_episode cancel did not settle; env wedged")
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
                    return _result("failed", StepErrorKind.ROBOT_SETUP, f"env reported FATAL: {result.info}")

                recs = self._episode_records.get(env_id, {})
                rec = recs.get(episode_id)
                if rec is None:
                    episodes_failed += 1
                    _log.warning(f"[{ep_idx + 1}/{step.episodes}] {step.key} env={env_id} no EpisodeRecord for episode_id={episode_id}; counted as failed")
                    _fold_window()
                    continue

                episodes_run += 1
                if rec.outcome_state == EpisodeRecord.FAILED:
                    episodes_failed += 1
                frac = closed_fraction(rec.goal_dist_start, rec.goal_dist_min)
                if rec.goal_dist_start > 0.0:
                    worst_progress = frac if worst_progress is None else min(worst_progress, frac)
                if self._efficacy is not None and rec.outcome_state != EpisodeRecord.SUCCESS and frac < self._efficacy:
                    episodes_weak += 1

                state_label = {
                    EpisodeRecord.SUCCESS: "SUCCESS",
                    EpisodeRecord.FAILED: "FAILED",
                    EpisodeRecord.SKIPPED: "SKIPPED",
                }.get(rec.outcome_state, str(rec.outcome_state))
                progress_note = f" progress={frac * 100:.0f}% path={rec.path_length:.1f}m" if rec.goal_dist_start > 0.0 else ""
                _log.info(f"[{ep_idx + 1}/{step.episodes}] {step.key} env={env_id} {state_label} info={rec.outcome_info!r} sim={ep_ended_sim - ep_started_sim:.1f}s wall={ep_ended_wall - ep_started_wall:.1f}s{progress_note}")

                if rec.outcome_state in (
                    EpisodeRecord.SUCCESS,
                    EpisodeRecord.FAILED,
                    EpisodeRecord.SKIPPED,
                    EpisodeRecord.FATAL,
                ):
                    rc = self._recorder_clients.get(env_id)
                    if rc is not None:
                        req = RecordEpisode.Request()
                        req.command = RecordEpisode.Request.COMMAND_STOP
                        req.episode_id = episode_id
                        req.outcome_state = rec.outcome_state
                        req.outcome_info = rec.outcome_info
                        try:
                            resp = await rc.call_timeout(req, timeout_sec=5.0)
                            if resp is None:
                                _log.warning(f"[env {env_id}] recorder stop_episode({episode_id}) timed out")
                            else:
                                _log.info(f"[env {env_id}] recorder stopped episode {episode_id} ({state_label})")
                        except Exception as exc:
                            _log.warning(f"[env {env_id}] recorder stop_episode({episode_id}) failed: {exc}")

                parent_ep_id = None
                if not step.is_reference:
                    self._parent_episode_map[(step.contestant.name, step.stage.name, ep_idx)] = episode_id
                    self._parent_episode_map[("__stage__", step.stage.name, ep_idx)] = episode_id
                else:
                    if step.reference_type == "unobstructed_robot":
                        parent_ep_id = self._parent_episode_map.get((step.contestant.name, step.stage.name, ep_idx))
                    else:
                        parent_ep_id = self._parent_episode_map.get(("__stage__", step.stage.name, ep_idx))

                ep_lockstep = _fold_window()
                ts_iso = datetime.datetime.now(tz=datetime.UTC).isoformat()
                self._run_dir.progress.append(
                    ts_iso=ts_iso,
                    run_id=self._run_id,
                    step_key=step.key,
                    contestant=step.contestant.name,
                    stage=step.stage.name,
                    env_id=env_id,
                    episode_id=episode_id,
                    episode_record=rec,
                    started_at=ep_started_sim,
                    ended_at=ep_ended_sim,
                    parent_episode_id=parent_ep_id,
                    is_reference=step.is_reference,
                    reference_type=step.reference_type,
                    lockstep=ep_lockstep,
                )
        except (_SimDied, asyncio.CancelledError):
            raise
        except _EnvDied as exc:
            _log.warning(f"{step.key} env={env_id} env died mid-step after run={episodes_run}, failed={episodes_failed}: {exc}")
            return _result("failed", StepErrorKind.ENV_SETUP, repr(exc))
        except Exception as exc:
            _log.exception(f"{step.key} env={env_id} unexpected error mid-step after run={episodes_run}, failed={episodes_failed}")
            return _result("failed", StepErrorKind.INTERNAL, repr(exc))

        if episodes_run == 0:
            status = "failed"
        elif episodes_failed == 0:
            status = "ok"
        elif episodes_failed < episodes_run:
            status = "partial"
        else:
            status = "failed"

        if stalled and status != "ok":
            return _result(status, StepErrorKind.EPISODE_TIMEOUT, f"sim stalled {_SIM_STALL_S:.0f}s")
        return _result(status)

    async def _await_episode_result(self, ac: ActionClientWrapper, goal_handle: object, env_id: int) -> object:
        """Await an episode result with no wall ceiling, raising _SimStalled if the sim clock freezes outside a reset."""
        result_task = asyncio.ensure_future(self._await_alive(ac.await_result(goal_handle), env_id=env_id, what=f"run_episode result on env {env_id}"))
        last_sim = self.sim_time.to_seconds()
        last_move = time.monotonic()
        try:
            while True:
                done, _ = await asyncio.wait({result_task}, timeout=1.0)
                if result_task in done:
                    return result_task.result()
                now_sim = self.sim_time.to_seconds()
                if now_sim > last_sim or self._env_resetting.get(env_id, False):
                    last_sim = now_sim
                    last_move = time.monotonic()
                    continue
                if time.monotonic() - last_move >= _SIM_STALL_S:
                    raise _SimStalled(f"sim clock frozen for {_SIM_STALL_S:.0f}s on env {env_id}")
        finally:
            if not result_task.done():
                result_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.gather(result_task, return_exceptions=True)

    async def _spawn_and_setup_env(self, launch_step: Step, *, inactivity_timeout: float = 120.0) -> tuple[int, str] | None:
        """Spawn one env booting launch_step's world and set up its clients.
        Tracks active progress (log growth) and only times out if completely stalled."""
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
            if self._sim_dead.is_set():
                raise _SimDied(self._sim_dead_reason) from None
            return None

    async def _spawn_and_setup_env_impl(self, launch_step: Step, registered: list[int]) -> tuple[int, str] | None:
        """Inner impl without timeout, called by _spawn_and_setup_env. Appends the spawned env_id to registered."""
        req = SpawnEnv.Request()
        req.ns = ""
        req.headless = self._headless
        req.launch_args = self._build_launch_args(launch_step)
        what = f"spawn_env for {launch_step.key}"
        resp = await self._await_alive(self.await_ros(self._spawn.client.call_async(req)), what=what)
        if resp is None or not resp.success:
            msg = resp.error_msg if resp is not None else "no response"
            if msg.startswith("sim dead:"):
                self._mark_sim_dead(msg)
                raise _SimDied(msg)
            _log.error(f"spawn_env failed for {launch_step.key}: {msg}")
            return None
        env_id = resp.env_id
        registered.append(env_id)
        await self._await_alive(self._await_env_visible(env_id), what=f"env {env_id} to appear on /arena/state/envs")
        env_ns_root = self._env_records[env_id].fqn
        await self._setup_env_clients(env_id, env_ns_root, launch_step.stage.robot)
        if self._viz and (self._viz_proc is None or self._viz_proc.poll() is not None):
            self._start_viz(env_ns_root)
        return env_id, env_ns_root

    def _start_viz(self, ns: str) -> None:
        cmd = [
            "ros2",
            "launch",
            "rviz_utils",
            "rviz_config.launch.py",
            f"ns:={ns}",
        ]
        _log.info(f"benchmark: spawning rviz for {ns}")
        try:
            self._viz_proc = subprocess.Popen(
                cmd,
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
            dreq = DespawnEnv.Request()
            dreq.env_id = env_id
            try:
                await self._await_alive(asyncio.wait_for(self._despawn.call_forever(dreq), timeout=30.0), what=f"despawn_env {env_id}")
                gone = await self._await_alive(self._wait_env_gone(env_id, timeout=15.0), what=f"env {env_id} to leave /arena/state/envs")
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

    async def _run_step(self, step: Step, env_id: int, slot_index: int) -> StepResult:
        """Push the stage config, then drive the step's episodes."""
        started = time.time()
        try:
            await self._push_stage_config(env_id, step)
        except (_SimDied, asyncio.CancelledError):
            raise
        except _EnvDied as exc:
            return StepResult(step.key, "failed", env_id, started, time.time(), StepErrorKind.ENV_SETUP, repr(exc), episodes_total=step.episodes)
        except Exception as exc:
            _log.error(f"[env {env_id}] stage config failed for {step.key}: {exc}")
            return StepResult(step.key, "failed", env_id, started, time.time(), StepErrorKind.ENV_SETUP, f"stage config failed: {exc}", episodes_total=step.episodes)
        return await self._run_episodes(step, env_id, slot_index)

    def _fail_remaining(self, q: asyncio.Queue[Step], env_id: int | None, detail: str, flush_cb: typing.Callable[[StepResult], bool]) -> bool:
        """Fail every step still queued. Returns True when the run must abort."""
        abort = False
        while not q.empty():
            try:
                step = q.get_nowait()
            except asyncio.QueueEmpty:
                break
            res = StepResult(
                step.key,
                "failed",
                env_id,
                time.time(),
                time.time(),
                StepErrorKind.ENV_SETUP,
                detail,
                episodes_total=step.episodes,
            )
            if flush_cb(res):
                abort = True
        return abort

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

    async def _run_group_queue(self, rep_step: Step, q: asyncio.Queue[Step], slot_index: int, flush_cb: typing.Callable[[StepResult], bool]) -> bool:
        env_id: int | None = None
        env_ns_root = ""
        spawned_once = False
        try:
            while not q.empty():
                if env_id is None:
                    if self._progress is not None:
                        self._progress.update_slot(
                            slot_index=slot_index,
                            env_id=None,
                            contestant=rep_step.contestant.name,
                            stage=rep_step.stage.name,
                            step_key=rep_step.key,
                            ep_idx=0,
                            ep_total=rep_step.episodes,
                            state="SPAWNING",
                        )
                    try:
                        spawned = await self._spawn_and_setup_env(rep_step)
                    except _SimDied as exc:
                        _, run_abort = await self._spend_sim_death(str(exc), None)
                        if run_abort:
                            return True
                        continue
                    if spawned is None:
                        return self._fail_remaining(q, None, "env respawn failed after a wedged env" if spawned_once else "spawn_env failed", flush_cb)
                    env_id, env_ns_root = spawned
                    spawned_once = True

                try:
                    step = q.get_nowait()
                except asyncio.QueueEmpty:
                    break

                recorder_proc = None
                if step.record_dir is not None:
                    episode_id_offset = self._global_episode_id_offset
                    self._global_episode_id_offset += step.episodes

                    lp, ip = resolve_planner_identity(step.contestant)
                    recorder_args = [
                        "run",
                        "arena_evaluation",
                        "record",
                        "--ros-args",
                        "-p",
                        "use_sim_time:=true",
                        "-p",
                        f"record_data_dir:={step.record_dir}",
                        "-p",
                        f"benchmark_id:={self._run_id}",
                        "-p",
                        f"contestant:={step.contestant.name}",
                        "-p",
                        f"stage:={step.stage.name}",
                        "-p",
                        f"map:={step.stage.map}",
                        "-p",
                        f"suite_name:={self._suite.name}",
                        "-p",
                        f"contest_name:={self._contest.name}",
                        "-p",
                        f"local_planner:={lp}",
                        "-p",
                        f"inter_planner:={ip}",
                        "-p",
                        f"robot:={step.stage.robot}",
                        "-p",
                        f"episodes_requested:={step.episodes}",
                        "-p",
                        f"is_reference:={str(step.is_reference).lower()}",
                    ]
                    if step.reference_type:
                        recorder_args.extend(["-p", f"reference_type:={step.reference_type}"])

                    recorder_args.extend(
                        [
                            "-p",
                            f"episode_id_offset:={episode_id_offset}",
                            "-r",
                            f"__ns:={env_ns_root}",
                        ]
                    )

                    recorder_proc = await asyncio.create_subprocess_exec(
                        "ros2",
                        *recorder_args,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    rc = self.create_client_wrapper(RecordEpisode, f"{env_ns_root}/start_episode")
                    self._recorder_clients[env_id] = rc
                    with contextlib.suppress(Exception):
                        await rc.ensure(timeout_sec=5.0)
                if self._progress is not None:
                    self._progress.update_slot(
                        slot_index=slot_index,
                        env_id=env_id,
                        contestant=step.contestant.name,
                        stage=step.stage.name,
                        step_key=step.key,
                        ep_idx=0,
                        ep_total=step.episodes,
                        state="SETTING_UP",
                    )
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
                    step_result = StepResult(
                        step.key,
                        "failed",
                        env_id,
                        time.time(),
                        time.time(),
                        StepErrorKind.SIM_DEAD,
                        sim_death_reason,
                        episodes_total=step.episodes,
                    )
                except _EnvDied as exc:
                    step_result = StepResult(
                        step.key,
                        "failed",
                        env_id,
                        time.time(),
                        time.time(),
                        StepErrorKind.ENV_SETUP,
                        repr(exc),
                        episodes_total=step.episodes,
                    )
                except Exception as exc:
                    step_result = StepResult(
                        step.key,
                        "failed",
                        env_id,
                        time.time(),
                        time.time(),
                        StepErrorKind.INTERNAL,
                        repr(exc),
                        episodes_total=step.episodes,
                    )
                finally:
                    if recorder_proc is not None:
                        try:
                            os.killpg(os.getpgid(recorder_proc.pid), signal.SIGINT)
                            await asyncio.wait_for(recorder_proc.wait(), timeout=2.0)
                        except (TimeoutError, Exception):
                            with contextlib.suppress(Exception):
                                os.killpg(os.getpgid(recorder_proc.pid), signal.SIGKILL)

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
            keep_alive = self._noexit and self._completed_groups == self._total_groups and env_id is not None
            if env_id is not None:
                self._teardown_env_clients(env_id)
                if env_id in self._env_records and not keep_alive and not self._sim_dead.is_set():
                    with contextlib.suppress(Exception):
                        dreq = DespawnEnv.Request()
                        dreq.env_id = env_id
                        await asyncio.wait_for(self._despawn.call_forever(dreq), timeout=2.0)
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(self._wait_env_gone(env_id, timeout=2.0), timeout=2.0)
                if keep_alive:
                    _log.info(f"--noexit: keeping env {env_id} alive after last group {rep_step.key}")

        return False

    def _publish_state(
        self,
        results: typing.Mapping[str, StepResult],
        steps_total: int,
    ) -> None:
        if not rclpy.ok():
            return
        msg = BenchmarkState()
        msg.stamp = self.get_clock().now().to_msg()
        msg.run_id = self._run_id
        msg.suite = self._suite.name
        msg.contest = self._contest.name
        msg.simulator = self._simulator or ""
        msg.env_n = self._env_n
        msg.headless = self._headless
        msg.steps_total = steps_total
        msg.steps_done = sum(1 for r in results.values() if r.status == "ok")
        msg.steps_partial = sum(1 for r in results.values() if r.status == "partial")
        msg.steps_failed = sum(1 for r in results.values() if r.status == "failed")
        msg.steps_skipped = sum(1 for r in results.values() if r.status == "skipped")
        msg.steps_in_flight = sum(1 for r in results.values() if r.status == "in_progress")
        msg.active_keys = [k for k, r in results.items() if r.status == "in_progress"]
        self._state_pub.publish(msg)

    async def setup(self) -> None:
        try:
            BenchmarkRunner.exit_code = await self._run_steps()
        except asyncio.CancelledError:
            BenchmarkRunner.exit_code = 130
            raise
        except Exception as exc:
            _log.error(f"benchmark crashed: {exc!r}")
            BenchmarkRunner.exit_code = 2

    async def teardown(self) -> None:
        await self._shutdown_arena()

    async def _watch_arena_proc(self, proc: subprocess.Popen) -> None:
        """Mark the sim dead the moment the arena launch process exits."""
        rc = await asyncio.get_running_loop().run_in_executor(None, proc.wait)
        self._mark_sim_dead(f"arena_runtime exited rc={rc}")

    async def _shutdown_arena(self) -> None:
        watch = self._arena_watch
        self._arena_watch = None
        if watch is not None:
            watch.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await watch
        if self._noexit:
            _log.info("--noexit: leaving arena_runtime.launch.py running; Ctrl+C its terminal to stop")
            return
        p = self._arena_proc
        if p is None or p.poll() is not None:
            return

        print("\nbenchmark: shutting down arena runtime...", flush=True)
        try:
            pgid = os.getpgid(p.pid)
        except (ProcessLookupError, OSError):
            if self._arena_log_file is not None:
                with contextlib.suppress(Exception):
                    self._arena_log_file.close()
                self._arena_log_file = None
            return

        descendants = _proc_tree(p.pid)

        for sig, grace in ((signal.SIGINT, _ARENA_SIGINT_GRACE_S), (signal.SIGTERM, _ARENA_SIGTERM_GRACE_S)):
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, OSError):
                break
            except Exception as e:
                _log.warning(f"signal {sig} failed: {e}")

            deadline = time.time() + grace
            while time.time() < deadline:
                if p.poll() is not None:
                    break
                await asyncio.sleep(0.1)
            if p.poll() is not None:
                break

        if p.poll() is None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            except Exception as e:
                _log.warning(f"SIGKILL failed: {e}")
            with contextlib.suppress(Exception):
                p.kill()
                p.wait(timeout=2.0)

        if self._viz_proc is not None:
            if self._viz_proc.poll() is None:
                with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                    os.killpg(os.getpgid(self._viz_proc.pid), signal.SIGTERM)
                try:
                    self._viz_proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                        os.killpg(os.getpgid(self._viz_proc.pid), signal.SIGKILL)
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

        if self._arena_log_file is not None:
            with contextlib.suppress(Exception):
                self._arena_log_file.close()
            self._arena_log_file = None

        print("benchmark: arena runtime shutdown complete.", flush=True)

    async def _start_arena(self) -> None:
        passthrough = dict(self._arena_passthrough)
        cmd = [
            "ros2",
            "launch",
            "arena_bringup",
            "arena_runtime.launch.py",
            *(f"{k}:={v}" for k, v in passthrough.items()),
        ]

        log_path = self._run_dir.path / "runner.log"
        self._arena_log_file = log_path.open("a")

        proc_env = None
        if self._suite_bundle_dir is not None:
            worlds_dir = self._suite_bundle_dir / "worlds"
            if worlds_dir.is_dir():
                proc_env = dict(os.environ)
                outer = proc_env.get("ARENA_WORLD_PATH")
                proc_env["ARENA_WORLD_PATH"] = f"{outer}:{worlds_dir}" if outer else str(worlds_dir)

        self._runtime_started_at = time.time()
        self._arena_proc = subprocess.Popen(
            cmd,
            stdout=self._arena_log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=proc_env,
        )
        self._arena_watch = asyncio.create_task(self._watch_arena_proc(self._arena_proc), name="arena_proc_watch")

        await self._await_alive(self._spawn.ensure(timeout_sec=300.0), what="spawn_env service")
        await self._await_alive(self._despawn.ensure(timeout_sec=300.0), what="despawn_env service")

    async def _restart_arena(self) -> None:
        """Restart arena_runtime for a fresh simulator session between world switches."""
        _log.info("Restarting arena_runtime for fresh simulation session...")
        await self._shutdown_arena()
        self._env_records.clear()
        self._env_visible_events.clear()
        self._env_gone_events.clear()
        await asyncio.sleep(1.0)
        self._sim_dead_reason = ""
        self._sim_dead.clear()
        await self._start_arena()

    async def _run_steps(self) -> int:
        pending = self._build_pending()
        results: dict[str, StepResult] = dict(self._run_dir.state.steps)
        steps_total = len(results) + len(pending)
        aborted_systemic = False

        self._publish_state(results, steps_total)
        _log.info(f"benchmark: signalled READY on {STATE_TOPIC}")

        await self._start_arena()

        self._global_episode_id_offset = 0
        episodes_dir = self._run_dir.path / "episodes"
        if episodes_dir.exists():
            for d in episodes_dir.glob("episode_*"):
                try:
                    idx = int(d.name.split("_")[1])
                    self._global_episode_id_offset = max(self._global_episode_id_offset, idx + 1)
                except Exception:
                    pass

        all_blocks = group_pending(pending, self._simulator)
        self._total_groups = len(all_blocks)

        step_map: dict[str, Step] = {s.key: s for s in (*_all_steps_grid(self._suite, self._contest, self._scale_episodes, self._run_dir.path), *pending)}

        self._progress = BenchmarkProgressDisplay(
            title=f"Arena Benchmark: {self._suite.name} • {self._contest.name}",
            total_steps=steps_total,
            env_n=self._env_n,
            run_id=self._run_id,
            sim_now=lambda: self.sim_time.to_seconds(),
        )

        def _mark_step_in_progress(step: Step) -> None:
            results[step.key] = StepResult(
                step.key,
                "in_progress",
                None,
                time.time(),
                None,
                None,
                None,
                episodes_total=step.episodes,
            )
            self._run_dir.state.write(results)
            self._publish_state(results, steps_total)

        def _flush_step_result(res: StepResult) -> bool:
            results[res.key] = res
            elapsed = (res.ended_at or time.time()) - res.started_at
            lockstep_note = f" {res.lockstep.short()}" if res.lockstep is not None and res.lockstep.active else ""
            _log.info(f"[{res.status}] {res.key} env={res.env_id} episodes={res.episodes_run}/{res.episodes_total} (failed={res.episodes_failed}) t={elapsed:.1f}s{lockstep_note}")
            self._run_dir.state.write(results)
            self._publish_state(results, steps_total)

            if self._progress is not None:
                st = step_map.get(res.key)
                c_name = st.contestant.name if st else res.key
                s_name = st.stage.name if st else ""
                self._progress.log_step_completed(
                    step_key=res.key,
                    status=res.status,
                    contestant=c_name,
                    stage=s_name,
                    episodes_run=res.episodes_run,
                    episodes_total=res.episodes_total,
                    episodes_failed=res.episodes_failed,
                    elapsed_sec=elapsed,
                    error_detail=res.error_detail,
                    sim_sec=res.sim_s,
                )

            total_episodes_run = sum(r.episodes_run for r in results.values())
            if total_episodes_run > 0:
                return False
            return res.status == "failed" and res.error_kind in _SYSTEMIC

        world_maps = list(dict.fromkeys(s.stage.map for s in pending))
        in_flight: set[asyncio.Task[bool]] = set()

        with self._progress:
            try:
                for world_idx, world_map in enumerate(world_maps):
                    world_steps = [s for s in pending if s.stage.map == world_map]
                    if not world_steps:
                        continue

                    # If switching to a new world map in Gazebo, restart arena_runtime now that
                    # all previous workers are completely finished and 0 tasks are running
                    if world_idx > 0 and self._simulator == "gazebo":
                        await self._restart_arena()

                    blocks = group_pending(world_steps, self._simulator)
                    block_queues: list[tuple[Step, asyncio.Queue[Step]]] = []
                    for block in blocks:
                        q = asyncio.Queue()
                        for step in block:
                            q.put_nowait(step)
                        block_queues.append((block[0], q))

                    cap = max(1, min(self._env_n, len(world_steps) or 1))

                    async def _worker(slot_index: int, block_queues: list[tuple[Step, asyncio.Queue[Step]]]) -> bool:
                        try:
                            while True:
                                target_q = None
                                rep_step = None
                                for r_step, q in block_queues:
                                    if not q.empty():
                                        target_q = q
                                        rep_step = r_step
                                        break

                                if target_q is None:
                                    break

                                abort = await self._run_group_queue(rep_step, target_q, slot_index, _flush_step_result)
                                if abort:
                                    return True
                            return False
                        finally:
                            if self._progress is not None:
                                self._progress.clear_slot(slot_index)

                    for slot in range(cap):
                        in_flight.add(asyncio.create_task(_worker(slot, block_queues), name=f"worker_{slot}"))

                    while in_flight:
                        done, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                        for t in done:
                            abort = t.result()
                            if abort:
                                aborted_systemic = True
                                _log.error("benchmark: worker hit a systemic setup failure; aborting run")
                                for t2 in in_flight:
                                    t2.cancel()
                                with contextlib.suppress(Exception):
                                    await asyncio.gather(*in_flight, return_exceptions=True)
                                in_flight.clear()
                                break
                        if aborted_systemic:
                            break

                    if aborted_systemic:
                        break

            except asyncio.CancelledError:
                for t in in_flight:
                    t.cancel()
                await asyncio.gather(*in_flight, return_exceptions=True)
                raise
            finally:
                self._run_dir.progress.dedupe_in_place()
                self._publish_state(results, steps_total)

        rows = [(step_map[k].contestant.name if k in step_map else k, step_map[k].stage.name if k in step_map else "", r.lockstep) for k, r in results.items() if r.lockstep is not None and r.lockstep.active]
        lockstep_failed = False
        if rows:
            print(f"\nbenchmark: lockstep report\n{format_report(rows)}", flush=True)
            lockstep_failed = any(summary.verdict == "fail" for _, _, summary in rows)
        strict_failed = sorted(k for k, r in results.items() if r.error_kind is not None or r.episodes_run < r.episodes_total)
        if self._strict and strict_failed:
            print(f"\nbenchmark: strict verdict: {len(strict_failed)} cell(s) did not complete cleanly: {', '.join(strict_failed)}", flush=True)

        efficacy_failed = False
        if self._efficacy is not None:
            header = ("contestant", "stage", "verdict", "episodes", "weak", "worst_progress")
            table = []
            for k, r in sorted(results.items()):
                st = step_map.get(k)
                c_name = st.contestant.name if st else k
                s_name = st.stage.name if st else ""
                worst = f"{r.episodes_worst_progress * 100:.0f}%" if r.episodes_worst_progress is not None else "-"
                table.append((c_name, s_name, cell_verdict(r), f"{r.episodes_run}/{r.episodes_total}", str(r.episodes_weak), worst))
            print(f"\nbenchmark: preflight verdict\n{format_table(header, table)}", flush=True)
            efficacy_failed = any(cell_verdict(r) in ("wedged", "weak") for r in results.values())

        if aborted_systemic:
            return 1
        if (self._lockstep_verdict and lockstep_failed) or (self._strict and strict_failed) or efficacy_failed:
            return 3
        return 0


def _all_steps(contest: Contest, suite: Suite, scale_episodes: float) -> list[Step]:
    return _all_steps_grid(suite, contest, scale_episodes, record_root=None)


def _is_inline_contest(contest_name: str) -> bool:
    stripped = contest_name.strip()
    return stripped.startswith("[") or stripped.startswith("{")


def _is_inline_suite(suite_name: str) -> bool:
    stripped = suite_name.strip()
    return stripped.startswith("[") or stripped.startswith("{")


def _provenance(verdict: ResolverVerdict) -> dict:
    return {"resolver": repr(verdict.resolver), "path": str(verdict.path)}


def _resolve_suite_source(suite_name: str) -> tuple[pathlib.Path, pathlib.Path | None, dict | None]:
    """Resolve a suite name to (yaml path, bundle dir, provenance). Flat `<stem>.yaml`
    wins over the directory-bundle form `<stem>/suite.yaml`, whose parent is the bundle dir."""
    verdict = SuiteIdentifier(name=suite_name.removesuffix(".yaml")).resolve_source_sync()
    bundle_dir = verdict.path.parent if verdict.path.name == "suite.yaml" else None
    return verdict.path, bundle_dir, _provenance(verdict)


def _resolve_contest_source(contest_name: str) -> tuple[pathlib.Path, dict | None]:
    """Resolve a contest name to (yaml path, provenance)."""
    verdict = ContestIdentifier(name=contest_name.removesuffix(".yaml")).resolve_source_sync()
    return verdict.path, _provenance(verdict)


def _suite_bundle_dir(suite_name: str) -> pathlib.Path | None:
    """Re-derive the bundle dir for a suite name; None for inline or unresolvable names."""
    if _is_inline_suite(suite_name):
        return None
    try:
        _, bundle_dir, _ = _resolve_suite_source(suite_name)
    except FileNotFoundError:
        return None
    return bundle_dir


def _load_suite_contest(suite_name: str, contest_name: str) -> tuple[Suite, Contest, dict, list | dict, pathlib.Path | None, dict | None, dict | None]:
    suite_bundle_dir = None
    suite_provenance = None
    if _is_inline_suite(suite_name):
        suite_dict = yaml.safe_load(suite_name)
        suite = Suite.parse("inline", suite_dict)
    else:
        suite_path, suite_bundle_dir, suite_provenance = _resolve_suite_source(suite_name)
        suite_stem = suite_name.removesuffix(".yaml") if suite_provenance is not None else suite_path.stem
        suite_dict = yaml.safe_load(suite_path.read_text())
        suite = Suite.parse(suite_stem, suite_dict)

    contest_provenance = None
    if _is_inline_contest(contest_name):
        contest_dict = yaml.safe_load(contest_name)
        contest = Contest.parse("inline", contest_dict)
    else:
        contest_path, contest_provenance = _resolve_contest_source(contest_name)
        contest_stem = contest_name.removesuffix(".yaml") if contest_provenance is not None else contest_path.stem
        contest_dict = yaml.safe_load(contest_path.read_text())
        contest = Contest.parse(contest_stem, contest_dict)

    return suite, contest, suite_dict, contest_dict, suite_bundle_dir, suite_provenance, contest_provenance


def _warn_config_drift(manifest: Manifest) -> None:
    """Non-fatal: note when the on-disk suite/contest for this run's names has drifted
    from the config the run was created with. Resume always replays the stored config."""
    try:
        _, _, suite_dict, contest_dict, _, _, _ = _load_suite_contest(manifest.suite_name, manifest.contest_name)
    except FileNotFoundError:
        return
    if compute_config_hash(suite_dict, contest_dict) != manifest.config_hash:
        _log.warning(f"on-disk config for suite={manifest.suite_name!r} contest={manifest.contest_name!r} differs from this run's stored config, resuming with the stored config")


def _resolve_resume_config(
    manifest: Manifest,
) -> tuple[Suite, Contest, float, str | None]:
    """Rebuild a resumed run's config from its manifest, not from CLI args or argparse
    defaults. Resume continues the same experiment, so the manifest is authoritative."""
    return (
        Suite.parse(manifest.suite_name, manifest.suite),
        Contest.parse(manifest.contest_name, manifest.contest),
        manifest.scale_episodes,
        manifest.simulator,
    )


def _default_run_id(suite_name: str, contest_name: str) -> str:
    ts = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d-%H%M%S")
    if _is_inline_suite(suite_name):
        suite_stem = "inline"
    else:
        suite_stem = pathlib.Path(suite_name.removesuffix(".yaml")).stem
    if _is_inline_contest(contest_name):
        contest_stem = "inline"
    else:
        contest_stem = pathlib.Path(contest_name.removesuffix(".yaml")).stem
    return f"{ts}-{suite_stem}-{contest_stem}"


_KV_RE = re.compile(r"^[\w\.\-]+:=.*$")


def cli_main(argv: list[str] | None = None) -> int:
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING if sys.stderr.isatty() else logging.INFO)
    console_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%H:%M:%S"))
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers = [console_handler]
    _log.setLevel(logging.INFO)
    _log.propagate = True
    p = argparse.ArgumentParser(prog="benchmark", epilog="exit codes: 0 ok, 1 systemic abort, 2 config error or crash, 3 lockstep, strict or efficacy verdict, 4 runner hung (deadman), 130 interrupted")
    p.add_argument("--suite", default="basic")
    p.add_argument("--contest", default="basic")
    p.add_argument("--scale-episodes", type=float, default=1.0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument(
        "--resume",
        nargs="?",
        const="__auto__",
        default=None,
        help="Resume a prior run. Bare --resume picks the most recent resumable; --resume <run_id> opens that run explicitly.",
    )
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Retries per cell after a systemic failure (sim death, env or robot setup), each on a fresh env. -1 retries forever, 0 never retries. Retried attempts are not recorded.",
    )
    p.add_argument(
        "--max-sim-deaths",
        type=int,
        default=_MAX_SIM_DEATHS,
        help="Abort the run (exit 1) once this many sim deaths happen, whatever --retries allows.",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="Enable system resource profiling. Writes simulation_profile.yaml with peak/mean CPU, GPU, RAM, and disk I/O stats.",
    )
    p.add_argument(
        "--lockstep-verdict",
        action="store_true",
        help="Exit 3 when a step's lockstep report fails (a stall of 5 s or more, or no planner beat registered).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit 3 when any cell ended with an error kind or ran fewer episodes than requested, whatever the episode outcomes were.",
    )
    p.add_argument(
        "--spawn-budget",
        type=float,
        default=600.0,
        help="Maximum wall time in seconds to wait for a single env spawn, on top of the inactivity heuristic, before giving up and despawning it.",
    )
    p.add_argument(
        "--efficacy",
        type=float,
        default=None,
        metavar="FRACTION",
        help="Flag a non-SUCCESS episode as weak when it closes less than FRACTION of its starting goal distance. Prints a preflight verdict table and exits 3 on any wedged or weak cell.",
    )
    p.add_argument(
        "--noexit",
        action="store_true",
        help="On completion, leave arena_runtime.launch.py and the last env running so you can poke at it. Recording stops with the last episode as usual.",
    )
    args, extras = p.parse_known_args(argv)

    for arg in extras:
        if not _KV_RE.match(arg):
            p.error(f"unrecognized argument: {arg!r}")

    arena_passthrough: dict[str, str] = {}
    for arg in extras:
        k, v = arg.split(":=", 1)
        arena_passthrough[k] = v

    if "env_n" in arena_passthrough:
        print("benchmark: env_n:= is deprecated, use env.n:=", file=sys.stderr)
        arena_passthrough.setdefault("env.n", arena_passthrough.pop("env_n"))

    try:
        share = pathlib.Path(get_package_share_directory("arena_evaluation"))

        if args.data_root:
            data_root = pathlib.Path(args.data_root)
            print(f"benchmark: data_root from --data-root: {data_root}", file=sys.stderr)
        elif os.environ.get("ARENA_DATA_DIR"):
            data_root = pathlib.Path(os.environ["ARENA_DATA_DIR"]) / "benchmarks"
            print(f"benchmark: data_root from ARENA_DATA_DIR: {data_root}", file=sys.stderr)
        else:
            data_root = share / "data"
            print(f"benchmark: data_root from default: {data_root}", file=sys.stderr)

        if args.resume:
            resume_id = args.resume
            if resume_id == "__auto__":
                resolved = find_most_recent_resumable(data_root)
                if resolved is None:
                    print(
                        f"benchmark: no resumable runs in {data_root}",
                        file=sys.stderr,
                    )
                    return 2
                print(
                    f"benchmark: auto-resume picked run_id={resolved}",
                    file=sys.stderr,
                )
                resume_id = resolved
            run_dir = RunDir.open(data_root, resume_id)
            man = run_dir.manifest
            suite, contest, scale_episodes, simulator = _resolve_resume_config(man)
            suite_bundle_dir = _suite_bundle_dir(man.suite_name)
            if simulator is not None:
                arena_passthrough["sim"] = simulator
            _warn_config_drift(man)
            run_dir.progress.write_comment(f"resumed at {datetime.datetime.now(tz=datetime.UTC).isoformat()}")
        else:
            suite, contest, suite_dict, contest_dict, suite_bundle_dir, suite_provenance, contest_provenance = _load_suite_contest(args.suite, args.contest)
            scale_episodes = args.scale_episodes
            cfg_hash = compute_config_hash(suite_dict, contest_dict)

        arena_passthrough = {**suite.launch_args, **arena_passthrough}
        env_n = int(arena_passthrough.get("env.n", "1"))
        headless = arena_passthrough.get("headless", "false").lower() in ("true", "1")
        simulator = arena_passthrough.get("sim", None)

        steps = _all_steps(contest, suite, scale_episodes)
        if not args.resume:
            problems = _preflight_contest(contest)
            if problems:
                for problem in problems:
                    print(f"benchmark: {problem}", file=sys.stderr)
                return 2
        if not steps:
            print(
                f"benchmark: empty grid (suite={suite.name!r} contest={contest.name!r} produced no steps)",
                file=sys.stderr,
            )
            return 2
        seen: set[str] = set()
        for c in steps:
            if c.key in seen:
                print(f"benchmark: duplicate step key {c.key!r}", file=sys.stderr)
                return 2
            seen.add(c.key)

        if not args.resume:
            run_id = args.run_id or _default_run_id(args.suite, args.contest)
            sha, dirty = capture_git_sha(share.parent.parent.parent)
            steps_list = [
                {
                    "key": c.key,
                    "contestant": attrs.asdict(c.contestant),
                    "stage": {k: v.value if isinstance(v, (Constants.TaskMode.TM_Robots, Constants.TaskMode.TM_Obstacles)) else v for k, v in c.stage._asdict().items()},
                    "episodes_planned": c.episodes,
                    "is_reference": c.is_reference,
                    "reference_type": c.reference_type,
                }
                for c in steps
            ]
            manifest = Manifest(
                run_id=run_id,
                created_at=datetime.datetime.now(tz=datetime.UTC).isoformat(),
                arena_git_sha=sha,
                arena_git_dirty=dirty,
                cli_args=sys.argv[1:] if argv is None else list(argv),
                env_n=env_n,
                headless=headless,
                config_hash=cfg_hash,
                simulator=simulator,
                scale_episodes=scale_episodes,
                suite_name=suite.name,
                contest_name=contest.name,
                suite=suite_dict,
                contest=contest_dict,
                steps=steps_list,
                launch_args=dict(arena_passthrough),
                suite_provenance=suite_provenance,
                contest_provenance=contest_provenance,
            )
            run_dir = RunDir.create(data_root, run_id, manifest)
    except FileNotFoundError as exc:
        print(f"benchmark: config file not found: {exc}", file=sys.stderr)
        return 2
    except SystemExit as exc:
        return int(exc.code or 0)
    except Exception as exc:
        print(f"benchmark: {exc}", file=sys.stderr)
        return 2

    print(
        f"benchmark: prepared run_id={run_dir.manifest.run_id} steps={len(steps)} dir={run_dir.path}",
        file=sys.stderr,
    )

    run_dir.attach_log_handler(logging.getLogger())

    profiler = None
    if args.profile:
        from .profiler import SimulationProfiler

        profiler = SimulationProfiler(output_dir=run_dir.path, sample_hz=2.0)
        profiler.start()

    try:
        BenchmarkRunner.run_main(
            suite=suite,
            contest=contest,
            simulator=simulator,
            scale_episodes=scale_episodes,
            env_n=env_n,
            run_id=run_dir.manifest.run_id,
            headless=headless,
            run_dir=run_dir,
            retry_failed=args.retry_failed,
            arena_passthrough=arena_passthrough,
            noexit=args.noexit,
            suite_bundle_dir=suite_bundle_dir,
            lockstep_verdict=args.lockstep_verdict,
            strict=args.strict,
            retries=args.retries,
            max_sim_deaths=args.max_sim_deaths,
            spawn_budget=args.spawn_budget,
            efficacy=args.efficacy,
        )
    except KeyboardInterrupt:
        return 130
    finally:
        if profiler is not None:
            profiler.stop()
    return BenchmarkRunner.exit_code


if __name__ == "__main__":
    raise SystemExit(cli_main())
