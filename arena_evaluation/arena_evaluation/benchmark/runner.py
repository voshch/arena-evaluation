"""Benchmark runner node, scheduler and command-line entry point."""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import faulthandler
import logging
import os
import pathlib
import signal
import subprocess
import sys
import threading
import types
import typing

import rclpy
from ament_index_python.packages import get_package_share_directory
from arena_bringup.defaults import default_sim
from arena_evaluation_msgs.msg import BenchmarkState
from arena_rclpy_mixins import ActionClientWrapper, ArenaMixinNode, ClientWrapper
from arena_rclpy_mixins.spin import start_loop_watchdog
from arena_runtime_msgs.msg import EnvRecord, EnvRegistry, EpisodeRecord, HoldRegistry, LockstepStatus, SimState
from arena_runtime_msgs.srv import DespawnEnv, SpawnEnv

from . import dashboard, runner_cli, runner_impl, runner_plan
from .config import Contest, Suite
from .lockstep import LockstepMonitor
from .progress_display import BenchmarkProgressDisplay
from .runner_cli import data_root
from .runner_plan import (
    ARENA_NODE_FQN,
    PREPARED,
    STATE_TOPIC,
    _default_run_id,
    block_claim_key,
    build_launch_args,
    build_pending,
    cell_verdict,
    closed_fraction,
    compromised_table,
    env_key,
    episode_line,
    group_pending,
    pick_block,
    resolve_planner_identity,
    robot_has_depth_camera,
    runtime_launch_args,
    stage_params,
    stage_request,
    step_status,
)
from .state import RunDir, compute_config_hash, find_most_recent_resumable
from .step import Step

__all__ = [
    "ARENA_NODE_FQN", "BenchmarkRunner", "PREPARED", "STATE_TOPIC", "_default_run_id",
    "block_claim_key", "build_launch_args", "build_pending", "cell_verdict", "cli_main",
    "closed_fraction", "compromised_table", "data_root", "env_key", "episode_line",
    "group_pending", "pick_block", "resolve_planner_identity", "robot_has_depth_camera",
    "runtime_launch_args", "stage_params", "stage_request", "step_status",
]

_log = logging.getLogger(__name__)


class BenchmarkRunner(runner_impl._RunnerEnvMixin, runner_impl._RunnerExecMixin, ArenaMixinNode):
    exit_code: typing.ClassVar[int] = 0
    shutdown_deadline_s: typing.ClassVar[float] = runner_plan._SHUTDOWN_DEADLINE_S

    @classmethod
    def run_main(cls, *args: object, aiomonitor: bool = False, **kwargs: object) -> None:
        """Run benchmark runner with clean lifecycle, non-blocking executor, and instant shutdown on Ctrl+C."""
        from arena_rclpy_mixins.spin import create_executor
        from rclpy.executors import ExternalShutdownException
        from rclpy.signals import SignalHandlerOptions

        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        executor = create_executor()
        node: BenchmarkRunner | None = None
        factory = typing.cast("typing.Callable[..., BenchmarkRunner]", cls)

        def _spin() -> None:
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

        async def _run_app() -> None:
            nonlocal node, watchdog_stop
            node = factory(*args, **kwargs)
            node.event_loop = loop
            watchdog_stop = start_loop_watchdog(loop, node, deadline_s=runner_plan._LOOP_DEADLINE_S, on_deadline=node._deadman)
            executor.add_node(node)
            try:
                await node.setup()
            except asyncio.CancelledError:
                cls.exit_code = 130
            except Exception as exc:
                _log.error(f"Benchmark run error: {exc!r}")
                cls.exit_code = 2

        try:
            main_task = loop.create_task(_run_app())
            loop.run_until_complete(main_task)
        except (KeyboardInterrupt, asyncio.CancelledError):
            cls.exit_code = 130
        finally:
            faulthandler.dump_traceback_later(cls.shutdown_deadline_s, exit=True)
            if watchdog_stop is not None:
                watchdog_stop.set()
            if node is not None:
                with contextlib.suppress(Exception):
                    loop.run_until_complete(node._shutdown_arena())
                if node._progress is not None:
                    node._progress.stop()
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in pending:
                task.cancel()
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
            faulthandler.cancel_dump_traceback_later()

    def __init__(self, suite: Suite, contest: Contest, *, simulator: str | None, scale_episodes: float, env_n: int, run_id: str, headless: bool, run_dir: RunDir, retry_failed: bool = False, arena_passthrough: dict[str, str] | None = None, noexit: bool = False, suite_bundle_dir: pathlib.Path | None = None, lockstep_verdict: bool = False, strict: bool = False, retries: int = 1, max_sim_deaths: int = runner_plan._MAX_SIM_DEATHS, spawn_budget: float = 600.0, efficacy: float | None = None, world_swap: bool = False, shared: str | None = None) -> None:
        super().__init__("arena_benchmark_runner")
        self._suite, self._contest, self._simulator = suite, contest, simulator
        self._suite_bundle_dir, self._scale_episodes, self._env_n = suite_bundle_dir, scale_episodes, env_n
        self._run_id, self._headless, self._run_dir = run_id, headless, run_dir
        self._retry_failed, self._noexit, self._lockstep_verdict = retry_failed, noexit, lockstep_verdict
        self._strict, self._retries, self._max_sim_deaths = strict, retries, max_sim_deaths
        self._spawn_budget, self._efficacy, self._world_swap = spawn_budget, efficacy, world_swap
        self._shared = shared
        self._arena_passthrough: dict[str, str] = dict(arena_passthrough or {})
        self._viz = self._arena_passthrough.get("viz", str(not self._headless)).lower() in ("true", "1")
        self._viz_proc: subprocess.Popen | None = None
        self._lockstep = LockstepMonitor()
        self._total_groups = self._completed_groups = 0
        self._sim_used = False
        self._spawn = self.create_client_wrapper(SpawnEnv, "/arena/spawn_env")
        self._despawn = self.create_client_wrapper(DespawnEnv, "/arena/despawn_env")
        self._env_records: dict[int, EnvRecord] = {}
        self._env_visible_events: dict[int, asyncio.Event] = {}
        self._env_gone_events: dict[int, asyncio.Event] = {}
        self._arena_proc: subprocess.Popen | None = None
        self._arena_log_file = None
        self._episode_action_clients: dict[int, ActionClientWrapper] = {}
        self._queue_clients: dict[int, ClientWrapper] = {}
        self._param_clients: dict[int, ClientWrapper] = {}
        self._param_get_clients: dict[int, ClientWrapper] = {}
        self._tf_namespaces: dict[int, str] = {}
        self._episode_records: dict[int, dict[int, EpisodeRecord]] = {}
        self._env_subs: dict[int, list] = {}
        self._recorder_clients: dict[int, ClientWrapper] = {}
        self._triggered_episodes: dict[int, set[int]] = {}
        self._env_holds = False
        self._sim_dead = asyncio.Event()
        self._sim_dead_reason = ""
        self._sim_generation = self._envs_spawned = 0
        self._sim_recover_lock = asyncio.Lock()
        self._retry_budget = runner_plan._RetryBudget(retries=self._retries, max_sim_deaths=self._max_sim_deaths)
        self._runtime_started_at = 0.0
        self._runtime_up = asyncio.Event()
        self._arena_watch: asyncio.Task | None = None
        self._parent_episode_map: dict[tuple[str, str, int], int] = {}
        self._progress: BenchmarkProgressDisplay | None = None
        for message_type, topic, callback in ((EnvRegistry, "/arena/state/envs", self._on_envs), (SimState, runner_plan.SIM_STATE_TOPIC, self._on_sim_state), (LockstepStatus, "/arena/state/lockstep", self._on_lockstep), (HoldRegistry, "/arena/state/holders", self._on_holders)):
            self.create_subscription(message_type, topic, callback, runner_plan._LATCHED)
        self._state_pub = self.create_publisher(BenchmarkState, STATE_TOPIC, runner_plan._LATCHED)

    def _build_pending(self) -> list[Step]:
        return build_pending(suite=self._suite, contest=self._contest, scale_episodes=self._scale_episodes, run_dir=self._run_dir, retry_failed=self._retry_failed, record_root=self._run_dir.path)

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


def cli_main(argv: list[str] | None = None) -> int:
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING if sys.stderr.isatty() else logging.INFO)
    console_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%H:%M:%S"))
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers = [console_handler]
    _log.setLevel(logging.INFO)
    _log.propagate = True
    parser = runner_cli.build_parser(dashboard.DEFAULT_HOST, dashboard.DEFAULT_PORT)
    args, extras = parser.parse_known_args(argv)
    lane, shared_error = runner_plan._shared_lane(args.shared, args.run_id, args.resume)
    if shared_error is not None:
        print(f"benchmark: {shared_error}", file=sys.stderr)
        return 2
    arena_passthrough = runner_cli.parse_extras(parser, extras)
    run_dir: RunDir | None = None
    try:
        share = pathlib.Path(get_package_share_directory("arena_evaluation"))
        root = data_root(share, args.data_root)
        if args.viz and (problem := dashboard.port_problem(dashboard.DEFAULT_HOST, args.viz_port, root)) is not None:
            print(f"benchmark: --viz: {problem}", file=sys.stderr)
            return 2
        fresh = None
        if args.resume:
            resume_id = args.resume
            if resume_id == "__auto__":
                resolved = find_most_recent_resumable(root)
                if resolved is None:
                    print(f"benchmark: no resumable runs in {root}", file=sys.stderr)
                    return 2
                print(f"benchmark: auto-resume picked run_id={resolved}", file=sys.stderr)
                resume_id = resolved
            run_dir = RunDir.open(root, resume_id, lane=lane)
            manifest = run_dir.manifest
            suite, contest, scale_episodes, simulator = runner_plan._resolve_resume_config(manifest)
            suite_bundle_dir = runner_plan._suite_bundle_dir(manifest.suite_name)
            if simulator is not None:
                arena_passthrough["sim"] = simulator
            runner_plan._warn_config_drift(manifest)
            run_dir.progress.write_comment(f"resumed at {datetime.datetime.now(tz=datetime.UTC).isoformat()}")
        else:
            suite, contest, suite_dict, contest_dict, suite_bundle_dir, suite_provenance, contest_provenance = runner_plan._load_suite_contest(args.suite, args.contest)
            scale_episodes = args.scale_episodes
            fresh = (suite_dict, contest_dict, suite_provenance, contest_provenance, compute_config_hash(suite_dict, contest_dict))
        arena_passthrough = runtime_launch_args(suite.launch_args, arena_passthrough)
        env_n = int(arena_passthrough.get("env.n", "1"))
        headless = arena_passthrough.get("headless", "false").lower() in ("true", "1")
        simulator = arena_passthrough.setdefault("sim", default_sim())
        steps = runner_plan._all_steps(contest, suite, scale_episodes)
        if not args.resume and (problems := runner_plan._preflight_contest(contest)):
            for problem in problems:
                print(f"benchmark: {problem}", file=sys.stderr)
            return 2
        if not steps:
            print(f"benchmark: empty grid (suite={suite.name!r} contest={contest.name!r} produced no steps)", file=sys.stderr)
            return 2
        seen: set[str] = set()
        for step in steps:
            if step.key in seen:
                print(f"benchmark: duplicate step key {step.key!r}", file=sys.stderr)
                return 2
            seen.add(step.key)
        if fresh is not None:
            suite_dict, contest_dict, suite_provenance, contest_provenance, cfg_hash = fresh
            run_id = args.run_id or _default_run_id(args.suite, args.contest, os.environ.get("ARENA_LANE"))
            sha, dirty = runner_plan._arena_git_provenance(share)
            manifest = runner_cli.fresh_manifest(run_id=run_id, suite=suite, contest=contest, suite_dict=suite_dict, contest_dict=contest_dict, suite_provenance=suite_provenance, contest_provenance=contest_provenance, config_hash=cfg_hash, arena_git_sha=sha, arena_git_dirty=dirty, steps=steps, launch_args=arena_passthrough, env_n=env_n, headless=headless, simulator=simulator, scale_episodes=scale_episodes, cli_args=sys.argv[1:] if argv is None else list(argv))
            run_dir = RunDir.create(root, run_id, manifest, lane=lane)
    except FileNotFoundError as exc:
        print(f"benchmark: config file not found: {exc}", file=sys.stderr)
        return 2
    except SystemExit as exc:
        return int(exc.code or 0)
    except Exception as exc:
        print(f"benchmark: {exc}", file=sys.stderr)
        return 2
    assert run_dir is not None
    viz = None
    if args.viz:
        try:
            viz_url, viz = dashboard.start_child(dashboard.DEFAULT_HOST, args.viz_port, root, run_dir.manifest.run_id)
        except dashboard.DashboardError as exc:
            print(f"benchmark: --viz: {exc}", file=sys.stderr)
            return 2
        print(f"{dashboard.VIZ_PREFIX}{viz_url}", file=sys.stderr, flush=True)
        with run_dir.log_path.open("a") as handle:
            handle.write(f"{datetime.datetime.now(tz=datetime.UTC).isoformat()} INFO benchmark: {dashboard.VIZ_PREFIX}{viz_url}\n")
    print(f"{PREPARED}{run_dir.manifest.run_id} steps={len(steps)} dir={run_dir.path}", file=sys.stderr)
    runner_plan._snapshot_bundle_worlds(suite_bundle_dir, (step.stage.map for step in steps), run_dir.path)
    run_dir.attach_log_handler(logging.getLogger())
    if lane is not None:
        run_dir.pid_path.write_text(f"{os.getpid()}\n")
    profiler = None
    if args.profile:
        from .profiler import SimulationProfiler

        profiler = SimulationProfiler(output_dir=run_dir.path, sample_hz=2.0)
        profiler.start()
    try:
        BenchmarkRunner.run_main(suite=suite, contest=contest, simulator=simulator, scale_episodes=scale_episodes, env_n=env_n, run_id=run_dir.manifest.run_id, headless=headless, run_dir=run_dir, retry_failed=args.retry_failed, arena_passthrough=arena_passthrough, noexit=args.noexit, suite_bundle_dir=suite_bundle_dir, lockstep_verdict=args.lockstep_verdict, strict=args.strict, retries=args.retries, max_sim_deaths=args.max_sim_deaths, spawn_budget=args.spawn_budget, efficacy=args.efficacy, world_swap=args.world_swap, shared=args.shared)
    except KeyboardInterrupt:
        return 130
    finally:
        if viz is not None:
            dashboard.stop_child(viz)
        if profiler is not None:
            profiler.stop()
        if lane is not None:
            try:
                run_dir.progress.close()
                run_dir.fold()
            finally:
                with contextlib.suppress(FileNotFoundError):
                    run_dir.pid_path.unlink()
    return BenchmarkRunner.exit_code


if __name__ == "__main__":
    raise SystemExit(cli_main())
