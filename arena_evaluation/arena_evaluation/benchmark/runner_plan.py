"""Pure planning, launch-argument and configuration helpers for the benchmark runner."""

from __future__ import annotations

import asyncio
import collections
import copy
import datetime
import hashlib
import json
import logging
import os
import pathlib
import re
import shutil
import tempfile
import typing

import attrs
import yaml
from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)
from arena_runtime.env.constants import Constants
from arena_runtime.env.task_layer import (
    TM_AXES,
    TaskLayer,
    TaskModes,
    encode_modes,
    registered_modes,
    selection_args,
    task_layer,
    task_layer_name,
)
from arena_runtime_msgs.msg import EnvRecord, EpisodeRecord
from arena_runtime_msgs.srv import QueueEpisode
from arena_simulation_setup.tree import ResolverVerdict
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

from ..ingestion.metadata import IngestionMetadata
from ..storage.config_layout import CONFIG_KINDS, bundle_config_file
from ..storage.planner_names import split_planner_name
from .config import Contest, Contestant, Suite, stage_modes
from .lockstep import format_table
from .state import Manifest, capture_git_sha, compute_config_hash
from .step import Step, StepErrorKind, StepResult
from .tree import ContestIdentifier, SuiteIdentifier

STATE_TOPIC = "/arena/benchmark/state"
PREPARED = "benchmark: prepared run_id="
SIM_STATE_TOPIC = "/arena/state/sim"
ARENA_NODE_FQN = "/arena"

_CANCEL_SETTLE_S = 30.0
_HEARTBEAT_S = 30.0
_ARENA_SIGINT_GRACE_S = 15.0
_ARENA_SIGTERM_GRACE_S = 5.0
_ARENA_ORPHAN_GRACE_S = 1.0
_SIM_STALL_S = 60.0
_MAX_SIM_DEATHS = 3
_LOOP_DEADLINE_S = 120.0
_HUNG_EXIT_CODE = 4
_SHUTDOWN_DEADLINE_S = (
    _ARENA_SIGINT_GRACE_S + _ARENA_SIGTERM_GRACE_S + _ARENA_ORPHAN_GRACE_S + 60.0
)

# EpisodeRecord.outcome_state labels (arena_runtime_msgs/msg/EpisodeRecord.msg)
_OUTCOME_LABELS = {
    EpisodeRecord.QUEUED: "QUEUED",
    EpisodeRecord.RUNNING: "RUNNING",
    EpisodeRecord.SUCCESS: "SUCCESS",
    EpisodeRecord.FAILED: "FAILED",
    EpisodeRecord.SKIPPED: "SKIPPED",
    EpisodeRecord.FATAL: "FATAL",
}
_LATCHED = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)
_SYSTEMIC = (StepErrorKind.ENV_SETUP, StepErrorKind.ROBOT_SETUP, StepErrorKind.SIM_DEAD)
_RETRY_ON_RESUME = (*_SYSTEMIC, StepErrorKind.EPISODE_TIMEOUT)
_ADAPTER_RESET_FAILED = "adapter reset failed"
_CAP_KEYS = ("mobile", "arm", "planner")
_ROBOTS, _OBSTACLES, _ = TM_AXES
_KV_RE = re.compile(r"^[\w\.\-]+:=.*$")
_SHARED_NAME_RE = re.compile(r"^[\w\-][\w\.\-]*$")
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

_log = logging.getLogger("arena_evaluation.benchmark.runner")
_T = typing.TypeVar("_T")


def _arena_git_provenance(share: pathlib.Path) -> tuple[str | None, bool]:
    """Git sha and dirty flag of the Arena checkout that owns the installed share."""
    return capture_git_sha(
        pathlib.Path(IngestionMetadata.resolve_git_workspace(str(share)))
    )


def _proc_fields(pid: int) -> list[str] | None:
    try:
        stat = pathlib.Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return stat.rpartition(")")[2].split()


def _proc_starttime(pid: int) -> int | None:
    """Start tick of a pid, None once it is gone."""
    fields = _proc_fields(pid)
    return int(fields[19]) if fields is not None else None


def _proc_tree(root: int) -> dict[int, int]:
    """Descendants of root as pid -> start tick."""
    children: dict[int, list[int]] = collections.defaultdict(list)
    starts: dict[int, int] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        fields = _proc_fields(pid)
        if fields is None:
            continue
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


def _orphaned_env_ids(
    known: set[int],
    records: typing.Mapping[int, EnvRecord],
    registered: typing.Sequence[int],
) -> list[int]:
    """Env ids to despawn after a stalled spawn: registered first, then any env reserved during the wait that never went ready."""
    ids = list(dict.fromkeys(registered))
    ids += [
        env_id
        for env_id, rec in records.items()
        if env_id not in known and env_id not in ids and not rec.ready
    ]
    return ids


def closed_fraction(goal_dist_start: float, goal_dist_min: float) -> float:
    """Fraction of the starting goal distance closed by the closest approach, 0 when there was no goal."""
    if goal_dist_start <= 0.0:
        return 0.0
    return max(0.0, min(1.0, (goal_dist_start - goal_dist_min) / goal_dist_start))


def episode_line(head: str, rec: EpisodeRecord, sim_s: float, wall_s: float) -> str:
    """Per-episode log line, ending in the integrity reasons when the episode is compromised."""
    label = _OUTCOME_LABELS.get(rec.outcome_state, str(rec.outcome_state))
    progress = (
        f" progress={closed_fraction(rec.goal_dist_start, rec.goal_dist_min) * 100:.0f}% path={rec.path_length:.1f}m"
        if rec.goal_dist_start > 0.0
        else ""
    )
    integrity = f" integrity={list(rec.integrity)!r}" if rec.integrity else ""
    return f"{head} {label} info={rec.outcome_info!r} sim={sim_s:.1f}s wall={wall_s:.1f}s{progress}{integrity}"


def compromised_table(
    results: typing.Mapping[str, StepResult], step_map: typing.Mapping[str, Step]
) -> str | None:
    """Compromised and run episode counts per contestant, None when every episode was intact."""
    counts: dict[str, tuple[int, int]] = {}
    for key, result in sorted(results.items()):
        name = step.contestant.name if (step := step_map.get(key)) else key
        compromised, run = counts.get(name, (0, 0))
        counts[name] = (
            compromised + result.episodes_compromised,
            run + result.episodes_run,
        )
    table = [
        (name, str(compromised), str(run))
        for name, (compromised, run) in counts.items()
        if compromised
    ]
    return (
        format_table(("contestant", "compromised", "episodes"), table)
        if table
        else None
    )


def step_status(
    episodes_run: int, episodes_total: int
) -> typing.Literal["ok", "partial", "failed"]:
    """How completely a step recorded its episodes, whatever their outcomes."""
    if episodes_run == 0:
        return "failed"
    return "partial" if episodes_run < episodes_total else "ok"


def cell_verdict(result: StepResult) -> str:
    """Efficacy classification for one cell: wedged (never finished), weak (finished but drove badly), or ok."""
    if result.error_kind is not None or result.episodes_run < result.episodes_total:
        return "wedged"
    return "weak" if result.episodes_weak > 0 else "ok"


def _yaml_str(value: object) -> str:
    """value as a quoted YAML string, so a ros-args parameter stays a string whatever it looks like."""
    return json.dumps(str(value))


class _WithSteps(typing.Protocol):
    steps: dict[str, StepResult]


class _HasStateSteps(typing.Protocol):
    """Structural interface required by build_pending: any object with .state.steps."""

    @property
    def state(self) -> _WithSteps: ...


def _launch_key(k: str) -> str:
    """Contest cap keys stay bare (`mobile`, `arm.<x>`, `planner`), launch args live under `robot.`."""
    return f"robot.{k}" if k.split(".", 1)[0] in _CAP_KEYS else k


def runtime_launch_args(
    suite_args: dict[str, str], passthrough: dict[str, str]
) -> dict[str, str]:
    """Suite launch args under the CLI passthrough, an autostarted lockstep stepping from bringup unless `lockstep.paused` is set."""
    merged = {**suite_args, **passthrough}
    if merged.get("lockstep", "false").lower() in ("true", "1"):
        merged.setdefault("lockstep.paused", "false")
    return merged


def build_launch_args(
    step: Step, simulator: str | None, passthrough: dict[str, str] | None = None
) -> list[str]:
    """Return the arena launch argument list for a step, given the simulator name."""
    stage = step.stage
    args = [
        *([f"sim:={simulator}"] if simulator is not None else []),
        f"task_layer:={task_layer_name(step.task_layer)}",
        f"robot:={step.robot}",
        f"world:={stage.map}",
        *selection_args(
            stage_modes(stage, registered_modes(task_layer(step.task_layer)))
        ),
        *([f"run_seed:={stage.seed}"] if stage.seed is not None else []),
        "task.episode.auto_reset:=false",
    ]
    args += [f"optim.{key}:={value}" for key, value in (stage.optim or {}).items()]
    if step.record_dir is not None:
        args += [f"record.dir:={step.record_dir}", "record.auto:=false"]
    own_keys = {arg.split(":=", 1)[0] for arg in args}

    def _append(key: str, value: object) -> None:
        if key in own_keys:
            _log.warning(
                "contestant %r: arg %r=%r ignored, controlled by stage",
                step.contestant.name,
                key,
                value,
            )
        else:
            args.append(f"{key}:={value}")

    for raw_key, value in step.contestant.args.items():
        key = _launch_key(raw_key)
        if isinstance(value, dict):
            if value.get("driver"):
                _append(key, value["driver"])
            for inner_key, inner_value in value.items():
                if inner_key != "driver" and inner_value:
                    _append(f"{key}.{inner_key}", inner_value)
        elif value:
            _append(key, value)
    args += [
        f"{key}:={value}"
        for key, value in (passthrough or {}).items()
        if key not in ("headless", "env_n", "env.n") and key not in own_keys
    ]
    return args


def _all_steps_grid(
    suite: Suite,
    contest: Contest,
    scale_episodes: float,
    record_root: pathlib.Path | None = None,
) -> list[Step]:
    """Generate all steps for a benchmark suite and contest."""
    episodes_dir = record_root / "episodes" if record_root is not None else None

    def _step(
        contestant: Contestant, stage: Suite.Stage, reference_type: str | None = None
    ) -> Step:
        return Step(
            contestant=contestant,
            stage=stage,
            episodes=int(round(stage.episodes * scale_episodes)),
            record_dir=episodes_dir,
            is_reference=reference_type is not None,
            reference_type=reference_type,
            task_layer=suite.task_layer,
        )

    reference_types = (None, "unobstructed_robot") if suite.references else (None,)
    steps = [
        _step(contestant, stage, reference_type)
        for contestant in contest.contestants
        for stage in suite.stages
        for reference_type in reference_types
    ]
    if suite.references:
        steps += [
            _step(Contestant(name="unhindered_peds", args={}), stage, "unhindered_peds")
            for stage in suite.stages
        ]
    keys = [step.key for step in steps]
    if duplicate := next((key for key in keys if keys.count(key) > 1), None):
        raise ValueError(f"duplicate step key: {duplicate!r}")
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
        lost = existing.error_kind is None and existing.episodes_run == 0
        if (
            existing.status == "failed"
            and not retry_failed
            and not lost
            and existing.error_kind not in _RETRY_ON_RESUME
        ):
            continue
        pending.append(step)
    return pending


def resolve_planner_identity(contestant: Contestant) -> tuple[str, str]:
    """Return (local_planner, inter_planner) for a contestant."""
    mobile = (contestant.args or {}).get("mobile")
    if isinstance(mobile, dict) and mobile.get("local_planner"):
        return str(mobile["local_planner"]), str(mobile["inter_planner"]) if mobile.get(
            "inter_planner"
        ) else "none"
    return split_planner_name(contestant.name)


def _preflight_contest(contest: Contest) -> list[str]:
    """Check nav2 contestants reference a local_planner/inter_planner that actually exists."""
    try:
        share = (
            pathlib.Path(get_package_share_directory("arena_robots"))
            / "config"
            / "nav2"
        )
    except PackageNotFoundError:
        return ["arena_robots is not installed, cannot validate nav2 contestants"]
    checks = (
        ("local_planner", "controllers", "controller_config.yaml"),
        ("inter_planner", "interplanners", "interplanner_config.yaml"),
    )
    problems: list[str] = []
    for contestant in contest.contestants:
        mobile = contestant.args.get("mobile")
        if not isinstance(mobile, dict) or mobile.get("driver") != "nav2":
            continue
        for key, subdir, filename in checks:
            value = mobile.get(key)
            if not value or (key == "inter_planner" and value == "none"):
                continue
            path = share / subdir / value / filename
            if not path.is_file():
                available = sorted(
                    entry.name for entry in (share / subdir).iterdir() if entry.is_dir()
                )
                problems.append(
                    f"{contestant.name}: {key} {value!r} has no {path} (available: {available})"
                )
    return problems


def env_key(step: Step, simulator: str | None, world_swap: bool = False) -> tuple:
    """Steps with the same env_key reuse one env. Changing contestant, robot, or map (unless worlds swap in place) forces a fresh env."""
    return (
        (step.contestant.name, step.robot, simulator)
        if world_swap
        else (step.contestant.name, step.robot, step.stage.map, simulator)
    )


def block_claim_key(
    block: list[Step],
    world_map: str | None,
    simulator: str | None,
    world_swap: bool = False,
) -> str:
    """Claim file name of an env block, identical across the lanes of a shared run."""
    return hashlib.sha1(
        json.dumps(
            [list(env_key(block[0], simulator, world_swap)), world_map], default=str
        ).encode()
    ).hexdigest()


def pick_block(
    block_queues: typing.Sequence[tuple[Step, asyncio.Queue[Step], str]],
    attached: typing.Mapping[int, int],
    claim: typing.Callable[[str], bool],
) -> int | None:
    """Index of the first block with more queued steps than workers already on it, claimed, or None."""
    for index, (_, queue, claim_key) in enumerate(block_queues):
        if queue.qsize() > attached.get(id(queue), 0) and claim(claim_key):
            return index
    return None


def robot_has_depth_camera(robot: str) -> bool:
    """True when a `robot:=` entry resolves to a depth sensor, False for anything that does not resolve."""
    from arena_robots import assembly as arena_assembly  # noqa: PLC0415
    from arena_robots.Robot import RobotIdentifier  # noqa: PLC0415
    from arena_robots.Sensor import SensorType  # noqa: PLC0415
    from arena_robots.SetupFile import Config  # noqa: PLC0415
    from arena_runtime.env.robot_manager.robots_manager import (
        desugar_robot_entry,
        split_robot_arg,
    )  # noqa: PLC0415

    for entry in split_robot_arg(robot):
        try:
            for config in Config.parse(desugar_robot_entry(entry)):
                view = RobotIdentifier.parse(config.robot).resolve_sync()
                if view.assembly is None:
                    sensors = view.model_params.sensors
                else:
                    request, _, _ = arena_assembly.build_request(
                        view.assembly, config.parts
                    )
                    sensors = view.effective_sensors(request, frames=config.frames)
                if any(sensor.type == SensorType.DEPTH for sensor in sensors):
                    return True
        except (RuntimeError, FileNotFoundError, arena_assembly.AssemblyError):
            continue
    return False


def group_pending(
    steps: list[Step], simulator: str | None, world_swap: bool = False
) -> list[list[Step]]:
    groups: dict[tuple, list[Step]] = collections.defaultdict(list)
    peds: list[Step] = []
    for step in steps:
        if step.is_reference and step.reference_type == "unhindered_peds":
            peds.append(step)
        else:
            groups[env_key(step, simulator, world_swap)].append(step)
    sorted_groups = sorted(groups.values(), key=len, reverse=True)
    for step in peds:
        target = next(
            (
                group
                for group in sorted_groups
                if group[0].stage.robot == step.stage.robot
                and group[0].stage.map == step.stage.map
            ),
            None,
        )
        if target is None:
            target = []
            sorted_groups.append(target)
        target.append(step)
    return sorted_groups


def _param_value(name: str, value: object) -> ParameterValue:
    if isinstance(value, bool):
        return ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=value)
    for scalar_type, param_type, field in (
        (int, ParameterType.PARAMETER_INTEGER, "integer_value"),
        (float, ParameterType.PARAMETER_DOUBLE, "double_value"),
        (str, ParameterType.PARAMETER_STRING, "string_value"),
    ):
        if isinstance(value, scalar_type):
            return ParameterValue(type=param_type, **{field: value})
    if not isinstance(value, list):
        raise TypeError(f"unsupported param type for {name!r}: {type(value).__name__}")
    if not value:
        return ParameterValue(
            type=ParameterType.PARAMETER_STRING_ARRAY, string_array_value=[]
        )
    for element_type, param_type, field in (
        (bool, ParameterType.PARAMETER_BOOL_ARRAY, "bool_array_value"),
        (int, ParameterType.PARAMETER_INTEGER_ARRAY, "integer_array_value"),
        (float, ParameterType.PARAMETER_DOUBLE_ARRAY, "double_array_value"),
        (str, ParameterType.PARAMETER_STRING_ARRAY, "string_array_value"),
    ):
        if all(isinstance(element, element_type) for element in value):
            return ParameterValue(type=param_type, **{field: value})
    raise TypeError(f"unsupported mixed list type for {name!r}")


def _walk_dict(d: dict, prefix: str = "") -> list[Parameter]:
    """Flatten a nested dict to rcl_interfaces Parameter[] with dot-joined leaf names."""
    out: list[Parameter] = []
    for key, value in d.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and "min" in value and "max" in value:
            low, high = value["min"], value["max"]
            pair = (
                ParameterValue(
                    type=ParameterType.PARAMETER_INTEGER_ARRAY,
                    integer_array_value=[low, high],
                )
                if isinstance(low, int) and isinstance(high, int)
                else ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                    double_array_value=[float(low), float(high)],
                )
            )
            out.append(Parameter(name=f"{name}.n", value=pair))
            rest = {
                key: item for key, item in value.items() if key not in ("min", "max")
            }
            out += _walk_dict(rest, name) if rest else []
        elif isinstance(value, dict):
            out += _walk_dict(value, name)
        else:
            out.append(Parameter(name=name, value=_param_value(name, value)))
    return out


def stage_params(
    stage_config: dict, modes: TaskModes, namespaces: dict[str, dict[str, str]]
) -> list[Parameter]:
    """Route stage.config blocks of the selected modes to Parameter[] under the full names of the mode's namespace."""
    out: dict[str, Parameter] = {}
    for axis, names in modes.items():
        namespace_by_mode = namespaces.get(axis, {})
        for mode in names:
            block = (stage_config or {}).get(mode)
            namespace = namespace_by_mode.get(mode)
            if namespace is None or not isinstance(block, dict):
                continue
            patched = {
                key: pathlib.Path(value).stem
                if mode == "scenario" and key == "file" and isinstance(value, str)
                else value
                for key, value in block.items()
            }
            out.update((param.name, param) for param in _walk_dict(patched, namespace))
    return list(out.values())


def _human_params(stage_config: dict) -> list[Parameter]:
    """The stage.config human backend blocks as namespaced Parameter[]."""
    out: list[Parameter] = []
    for namespace in (
        *Constants.HUMAN_PARAM_NAMESPACES.values(),
        Constants.HUMAN_PARAM_RESERVED,
    ):
        block = (stage_config or {}).get(namespace)
        if isinstance(block, dict):
            out.extend(_walk_dict(block, namespace))
    return out


def _stationary_pins(namespaces: dict[str, dict[str, str]]) -> list[Parameter]:
    return [
        Parameter(
            name=f"{namespaces[_ROBOTS]['stationary']}.{leaf}",
            value=ParameterValue(
                type=ParameterType.PARAMETER_DOUBLE, double_value=1000.0
            ),
        )
        for leaf in ("pos_x", "pos_y")
    ]


def _accepts_empty_selection(layer: TaskLayer, axis: str) -> bool:
    """Whether the layer accepts the axis with no mode selected."""
    try:
        layer.validate_modes({axis: []})
    except ValueError:
        return False
    return True


def stage_request(step: Step) -> QueueEpisode.Request:
    """The QueueEpisode request that sets a step's stage: its set modes and its config blocks under the namespaces of the selected modes."""
    layer = task_layer(step.task_layer)
    modes = {
        axis: names
        for axis, names in stage_modes(step.stage, registered_modes(layer)).items()
        if names or _accepts_empty_selection(layer, axis)
    }
    namespaces = layer.mode_namespaces()
    stage_config = step.stage.config or {}
    pinned: list[Parameter] = []
    if step.is_reference and step.reference_type == "unobstructed_robot":
        stage_config = copy.deepcopy(stage_config)
        obstacle_modes = modes.get(_OBSTACLES)
        if obstacle_modes:
            for mode in obstacle_modes:
                mode_block = stage_config.setdefault(mode, {})
                if isinstance(mode_block, dict):
                    mode_block["dynamic"] = {"min": 0, "max": 0}
        else:
            pinned = [
                param
                for namespace in namespaces.get(_OBSTACLES, {}).values()
                for param in _walk_dict({"dynamic": {"min": 0, "max": 0}}, namespace)
            ]
    elif step.is_reference and step.reference_type == "unhindered_peds":
        modes[_ROBOTS] = ["stationary"]
        pinned = _stationary_pins(namespaces)
    request = QueueEpisode.Request()
    request.action = QueueEpisode.Request.MERGE
    request.world = step.stage.map
    request.modes = encode_modes(modes)
    request.params = list(
        {
            param.name: param
            for param in [*stage_params(stage_config, modes, namespaces), *pinned]
        }.values()
    )
    request.human_params = _human_params(stage_config)
    return request


def _new_env_logs(
    log_dir: pathlib.Path, before: set[pathlib.Path]
) -> list[pathlib.Path]:
    """Env logs in the runtime's own log directory that appeared after `before` was taken."""
    return sorted(
        path for path in log_dir.glob("arena_env_*.log") if path not in before
    )


def _is_substantive_log_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not any(
        pattern in stripped for pattern in _SPAWN_CHATTER_PATTERNS
    )


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
    for queued in rest:
        q.put_nowait(queued)


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

    def record(
        self, step_key: str | None, *, sim_death_generation: int | None
    ) -> tuple[bool, bool]:
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


def _is_inline(name: str) -> bool:
    return name.strip()[:1] in ("[", "{")


def _provenance(verdict: ResolverVerdict) -> dict:
    return {"resolver": repr(verdict.resolver), "path": str(verdict.path)}


def _config_kind(asset_type: str) -> str:
    """The config kind whose folder is `asset_type`: `suites` -> `suite`."""
    return next(
        kind for kind, spec in CONFIG_KINDS.items() if spec.folder == asset_type
    )


def _resolve_source(identifier: type, name: str) -> tuple[pathlib.Path, dict]:
    verdict = identifier(name=name.removesuffix(".yaml")).resolve_source_sync()
    assert verdict.path is not None
    kind = _config_kind(identifier._asset_type)
    return bundle_config_file(verdict.path, kind), _provenance(verdict)


def _config_stem(name: str) -> str:
    return (
        "inline" if _is_inline(name) else pathlib.Path(name.removesuffix(".yaml")).stem
    )


def _load_config(
    identifier: type, name: str, parse: typing.Callable[[str, dict], object]
) -> tuple[object, dict, dict | None, pathlib.Path | None]:
    if _is_inline(name):
        data = yaml.safe_load(name)
        return parse("inline", data), data, None, None
    path, provenance = _resolve_source(identifier, name)
    data = yaml.safe_load(path.read_text())
    return (
        parse(
            name.removesuffix(".yaml") if provenance is not None else path.stem, data
        ),
        data,
        provenance,
        path,
    )


def _suite_bundle_dir(suite_name: str) -> pathlib.Path | None:
    """Re-derive the bundle dir for a suite name; None for inline or unresolvable names."""
    if _is_inline(suite_name):
        return None
    try:
        path, _ = _resolve_source(SuiteIdentifier, suite_name)
    except FileNotFoundError:
        return None
    return path.parent if path.name == "suite.yaml" else None


def _load_suite_contest(
    suite_name: str, contest_name: str
) -> tuple[
    Suite, Contest, dict, list | dict, pathlib.Path | None, dict | None, dict | None
]:
    suite, suite_dict, suite_provenance, suite_path = _load_config(
        SuiteIdentifier, suite_name, Suite.parse
    )
    contest, contest_dict, contest_provenance, _ = _load_config(
        ContestIdentifier, contest_name, Contest.parse
    )
    bundle_dir = (
        suite_path.parent
        if suite_path is not None and suite_path.name == "suite.yaml"
        else None
    )
    return (
        typing.cast(Suite, suite),
        typing.cast(Contest, contest),
        suite_dict,
        contest_dict,
        bundle_dir,
        suite_provenance,
        contest_provenance,
    )


def _warn_config_drift(manifest: Manifest) -> None:
    """Non-fatal: note when the on-disk suite/contest for this run's names has drifted from the config the run was created with. Resume always replays the stored config."""
    try:
        _, _, suite_dict, contest_dict, _, _, _ = _load_suite_contest(
            manifest.suite_name, manifest.contest_name
        )
    except FileNotFoundError:
        return
    if compute_config_hash(suite_dict, contest_dict) != manifest.config_hash:
        _log.warning(
            f"on-disk config for suite={manifest.suite_name!r} contest={manifest.contest_name!r} differs from this run's stored config, resuming with the stored config"
        )


def _resolve_resume_config(
    manifest: Manifest,
) -> tuple[Suite, Contest, float, str | None]:
    """Rebuild a resumed run's config from its manifest, not from CLI args or argparse defaults. Resume continues the same experiment, so the manifest is authoritative."""
    return (
        Suite.parse(manifest.suite_name, manifest.suite),
        Contest.parse(manifest.contest_name, manifest.contest),
        manifest.scale_episodes,
        manifest.simulator,
    )


def _default_run_id(suite_name: str, contest_name: str, lane: str | None) -> str:
    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{lane if lane else ''}{'-' if lane else ''}{_config_stem(suite_name)}-{_config_stem(contest_name)}"


def _shared_lane(
    shared: str | None, run_id: str | None, resume: str | None
) -> tuple[str | None, str | None]:
    """Validate --shared arguments, returning (lane, error message)."""
    if shared is None:
        return None, None
    if not _SHARED_NAME_RE.match(shared):
        return (
            None,
            f"--shared token {shared!r} must be a plain file name (letters, digits, '_', '-', '.')",
        )
    lane = os.environ.get("ARENA_LANE", "")
    if not lane:
        return (
            None,
            "--shared needs ARENA_LANE set to this runner's lane name (e.g. ARENA_LANE=p1)",
        )
    if not _SHARED_NAME_RE.match(lane):
        return (
            None,
            f"ARENA_LANE {lane!r} must be a plain file name (letters, digits, '_', '-', '.')",
        )
    if resume == "__auto__":
        return (
            None,
            "--shared refuses bare --resume, pass --resume <run_id> so every lane opens the same run",
        )
    if resume is None and run_id is None:
        return (
            None,
            "--shared needs --run-id <id> for a fresh run or --resume <run_id>, shared by every lane",
        )
    return lane, None


def _snapshot_bundle_worlds(
    bundle_dir: pathlib.Path | None, maps: typing.Iterable[str], run_path: pathlib.Path
) -> list[str]:
    """Copy the bundle worlds the stages use into <run>/worlds, where evaluation looks maps up."""
    copied: list[str] = []
    if bundle_dir is None:
        return copied
    for name in sorted(set(maps)):
        source, target = bundle_dir / "worlds" / name, run_path / "worlds" / name
        if not source.is_dir() or target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        part = pathlib.Path(
            tempfile.mkdtemp(prefix=f".{name}.", suffix=".part", dir=target.parent)
        )
        try:
            shutil.copytree(source, part, dirs_exist_ok=True)
            part.rename(target)
            copied.append(name)
        except OSError:
            if not target.is_dir():
                raise
        finally:
            shutil.rmtree(part, ignore_errors=True)
    return copied


def _all_steps(contest: Contest, suite: Suite, scale_episodes: float) -> list[Step]:
    return _all_steps_grid(suite, contest, scale_episodes, record_root=None)
