"""The dashboard's runtime page: one runtime's envs, episodes, camera and sim controls through the runtime, episode and view MCP tools."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import dataclasses
import functools
import inspect
import os
import pathlib
import re
import shlex
import subprocess
import threading
import time
import typing

import pydantic

from .dashboard import ControlError

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from arena_runtime.env.task_layer import TaskLayer

CALL_TIMEOUT_S = 20.0
VIEWPORT_TTL_S = 10.0
OPTIONS_TTL_S = 60.0
CATALOG_TTL_S = 60.0
SNAPSHOT_S = 1.0
WATCH_S = 30.0
STOP_TIMEOUT_S = 90.0
HOLD_REASON = "dashboard"
MAX_SPAWN = 16
_CONFIG_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]*$")
_LAUNCH_TOKEN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:=")


def _int(body: dict[str, typing.Any], name: str, default: int, low: int, high: int) -> int:
    value = body.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
        raise ControlError(400, f"{name} must be an integer from {low} to {high}, got {value!r}")
    return value


@dataclasses.dataclass(frozen=True)
class _Action:
    """One dashboard action: the MCP tool it calls, its scope, the kwargs it fixes and the parameters it hides from the user."""

    tool: Callable[..., Coroutine[typing.Any, typing.Any, typing.Any]]
    scope: str
    fixed: dict[str, typing.Any] = dataclasses.field(default_factory=dict)
    hidden: tuple[str, ...] = ()
    settable: bool = True


def _viz_tools() -> tuple[Callable[..., typing.Any], Callable[..., typing.Any], Callable[..., typing.Any]] | None:
    """The rerun view tools, None when rerun_utils does not provide them."""
    try:
        from rerun_utils.mcp import viz_rerun_list, viz_rerun_start, viz_rerun_stop
    except ImportError:
        return None
    return viz_rerun_start, viz_rerun_stop, viz_rerun_list


async def _tool_result(tool: Callable[..., typing.Any], **kwargs: object) -> dict[str, typing.Any]:
    result = tool(**kwargs)
    return await result if inspect.isawaitable(result) else result


async def _live_views() -> dict[int, dict[str, typing.Any]]:
    """The alive rerun views by env id, empty when the view tools are missing."""
    viz = _viz_tools()
    if viz is None:
        return {}
    listed = await _tool_result(viz[2])
    return {view["env_id"]: {"web_port": view["web_port"], "grpc_port": view["grpc_port"], "url": view["url"]} for view in listed["views"] if view["alive"]}


@functools.cache
def _actions() -> dict[str, _Action]:
    """The MCP tool behind every dashboard action, the one place the mapping lives."""
    from arena_runtime import mcp as runtime
    from arena_runtime.env.episode_mcp import episode_reset

    from arena_evaluation.mcp.runs import evaluation_benchmark, evaluation_stop

    views: dict[str, _Action] = {}
    viz = _viz_tools()
    if viz is not None:
        start, stop, _ = viz
        views = {"view": _Action(start, "env", {"web_port": 0, "grpc_port": 0}), "view_stop": _Action(stop, "env")}
    return {
        "spawn": _Action(runtime.runtime_spawn_env, "runtime", {"wait_ready": False, "headless": True}, ("launch_args", "ns", "timeout_s")),
        "despawn": _Action(runtime.runtime_despawn_env, "env", {"wait": False}, ("timeout_s",)),
        "despawn_all": _Action(runtime.runtime_despawn_env, "runtime", {"wait": False}, ("timeout_s",)),
        "reset": _Action(episode_reset, "env", {"wait": False}, ("timeout_s",)),
        "cleanup": _Action(runtime.runtime_cleanup_env, "env"),
        **views,
        "hold": _Action(runtime.runtime_hold, "runtime", {"reason": HOLD_REASON}),
        "release": _Action(runtime.runtime_release, "runtime", {"reason": None}),
        "step": _Action(runtime.runtime_step, "runtime", hidden=("timeout_s",)),
        "lockstep_start": _Action(runtime.runtime_lockstep_start, "runtime"),
        "lockstep_pause": _Action(runtime.runtime_lockstep_pause, "runtime"),
        "lockstep_resume": _Action(runtime.runtime_lockstep_resume, "runtime"),
        "lockstep_stop": _Action(runtime.runtime_lockstep_stop, "runtime"),
        "stop_run": _Action(evaluation_stop, "run", hidden=("run_id",)),
        "start_run": _Action(evaluation_benchmark, "run", settable=False),
    }


@functools.cache
def _args_model(action: str) -> type[pydantic.BaseModel]:
    """The parameters a user sets on an action, as a pydantic model of the tool's own signature and Field metadata."""
    spec = _actions()[action]
    hints = typing.get_type_hints(spec.tool, include_extras=True)
    skip = {"env_id", *spec.fixed, *spec.hidden}
    fields: dict[str, typing.Any] = {}
    if spec.settable:
        for name, parameter in inspect.signature(spec.tool).parameters.items():
            if name not in skip:
                fields[name] = (hints[name], ... if parameter.default is inspect.Parameter.empty else parameter.default)
    return pydantic.create_model(f"{action}_args", __config__=pydantic.ConfigDict(extra="forbid"), **fields)


def _validated(action: str, body: dict[str, typing.Any]) -> dict[str, typing.Any]:
    model = _args_model(action)
    try:
        checked = model.model_validate({name: value for name, value in body.items() if name in model.model_fields}, strict=True)
    except pydantic.ValidationError as exc:
        raise ControlError(400, "; ".join(f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors())) from exc
    return {name: getattr(checked, name) for name in model.model_fields}


_TOOL_PARAM_BOUNDS = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "enum")
_PARAM_TYPES: dict[str, typing.Any] = {
    "bool": bool,
    "integer": int,
    "double": float,
    "string": str,
    "bool_array": list[bool],
    "integer_array": list[int],
    "double_array": list[float],
    "string_array": list[str],
}


def _tool_param(name: str, schema: dict[str, typing.Any]) -> dict[str, typing.Any]:
    flat = {key: value for key, value in schema.items() if key != "anyOf"}
    for branch in schema.get("anyOf", []):
        if branch.get("type") != "null":
            flat = branch | flat
    param: dict[str, typing.Any] = {"name": name, "type": flat.get("type"), "description": flat.get("description", "")}
    if "default" in flat:
        param["default"] = flat["default"]
    return param | {key: flat[key] for key in _TOOL_PARAM_BOUNDS if key in flat}


def _annotations(tool: Callable[..., typing.Any]) -> dict[str, bool]:
    from arena_mcp.core.domain import tool_spec

    hints = tool_spec(tool).annotations
    return {"destructive": bool(hints.destructive_hint), "read_only": bool(hints.read_only_hint), "idempotent": bool(hints.idempotent_hint)}


def _action_docs() -> dict[str, dict[str, typing.Any]]:
    return {
        action: {
            "tool": spec.tool.__name__,
            "doc": (spec.tool.__doc__ or "").strip().splitlines()[0],
            "scope": spec.scope,
            "annotations": _annotations(spec.tool),
            "params": [_tool_param(name, schema) for name, schema in _args_model(action).model_json_schema().get("properties", {}).items()],
        }
        for action, spec in _actions().items()
    }


def _check_param_types(params: dict[str, typing.Any], declared: dict[str, str]) -> None:
    for name, value in params.items():
        kind = declared.get(name)
        if kind is None:
            continue
        if kind not in _PARAM_TYPES:
            raise ControlError(400, f"{name} is declared as {kind}, the dashboard cannot set it")
        try:
            pydantic.TypeAdapter(_PARAM_TYPES[kind]).validate_python(value, strict=True)
        except pydantic.ValidationError as exc:
            raise ControlError(400, f"{name} is declared as {kind}: {exc.errors()[0]['msg']}") from exc


def _mapping(body: dict[str, typing.Any], name: str) -> dict[str, typing.Any]:
    value = body.get(name) or {}
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ControlError(400, f"{name} must map names to values")
    return value


def _modes(body: dict[str, typing.Any]) -> dict[str, list[str]]:
    modes = _mapping(body, "modes")
    out: dict[str, list[str]] = {}
    for axis, names in modes.items():
        listed = [names] if isinstance(names, str) else names
        if not isinstance(listed, list) or not all(isinstance(name, str) for name in listed):
            raise ControlError(400, f"modes.{axis} must be a mode name or a list of mode names")
        out[axis] = [name for name in listed if name]
    return out


def _launch_args(body: dict[str, typing.Any]) -> dict[str, str]:
    raw = body.get("launch_args", {})
    if not isinstance(raw, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in raw.items()):
        raise ControlError(400, "launch_args must map launch argument names to string values")
    return {key.strip(): value.strip() for key, value in raw.items() if key.strip() and value.strip()}


class RuntimeBridge:
    """An event loop on its own thread that runs the MCP tools on the arena_mcp ROS node for the HTTP handlers."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="dashboard-runtime", daemon=True)
        self.thread.start()
        self._viewport: tuple[float, bool] = (0.0, False)
        self._options: tuple[float, dict[str, typing.Any] | None] = (0.0, None)
        self._catalogs: dict[tuple[str, str, bool], tuple[float, list[str]]] = {}
        self._cache: tuple[float, dict[str, typing.Any] | ControlError] | None = None
        self._asked = 0.0
        self._refresher: concurrent.futures.Future[None] | None = None
        self._refresher_lock = threading.Lock()
        self._poked = asyncio.Event()
        self._views: set[int] = set()

    def run[T](self, coro: Coroutine[typing.Any, typing.Any, T], timeout: float = CALL_TIMEOUT_S) -> T:
        from arena_mcp.core import ToolError

        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return future.result(timeout)
        except TimeoutError as exc:
            future.cancel()
            raise ControlError(504, f"the runtime did not answer within {timeout:.0f}s", "the runtime may be busy resetting or dead, the summary shows its sim state") from exc
        except ToolError as exc:
            raise ControlError(409, exc.message, exc.hint) from exc

    def close(self) -> None:
        from arena_mcp.core import ROS

        for env_id in sorted(self._views):
            with contextlib.suppress(Exception):
                self.run(self._stop_view(env_id), timeout=30.0)
        with contextlib.suppress(Exception):
            self.run(ROS.shutdown(), timeout=15.0)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5.0)

    def snapshot(self) -> dict[str, typing.Any]:
        self._asked = time.monotonic()
        cache = self._cache
        if cache is None or self._asked - cache[0] > WATCH_S:
            cache = (time.monotonic(), self.run(self._snapshot()))
            self._cache = cache
        with self._refresher_lock:
            if self._refresher is None or self._refresher.done():
                self._refresher = asyncio.run_coroutine_threadsafe(self._refresh(), self.loop)
        age = time.monotonic() - cache[0]
        if isinstance(cache[1], ControlError):
            raise ControlError(cache[1].status, f"{cache[1].message} (refresh failed {age:.1f}s ago)", cache[1].hint)
        return cache[1] | {"age_s": age, "now": time.time()}

    def poke(self) -> None:
        self.loop.call_soon_threadsafe(self._poked.set)

    async def _refresh(self) -> None:
        while time.monotonic() - self._asked < WATCH_S:
            try:
                self._cache = (time.monotonic(), await self._snapshot())
            except Exception as exc:
                self._cache = (time.monotonic(), _as_control_error(exc))
            self._poked.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._poked.wait(), SNAPSHOT_S)

    def catalog(self, kind: str, world: str) -> dict[str, typing.Any]:
        return self.run(self._catalog(kind, world), timeout=70.0)

    def env_task(self, env_id: int) -> dict[str, typing.Any]:
        return self.run(self._env_task(env_id))

    def stop_run(self, run_id: str) -> dict[str, typing.Any]:
        return self.run(_actions()["stop_run"].tool(run_id), timeout=STOP_TIMEOUT_S)

    def options(self) -> dict[str, typing.Any]:
        stamp, cached = self._options
        if cached is None or time.monotonic() - stamp > OPTIONS_TTL_S:
            cached = self.run(self._load_options(), timeout=60.0)
            self._options = (time.monotonic(), cached)
        return cached

    def camera(self, env_id: int) -> bytes:
        return self.run(self._camera(env_id), timeout=40.0)

    def act(self, action: str, env_id: int | None, body: dict[str, typing.Any]) -> dict[str, typing.Any]:
        runtime_actions = {name: spec for name, spec in _actions().items() if spec.scope != "run"}
        spec = runtime_actions.get(action)
        if spec is None:
            raise ControlError(404, f"no runtime action {action!r}", f"actions: {', '.join(sorted(runtime_actions))}")
        per_env = spec.scope == "env"
        if per_env != (env_id is not None):
            raise ControlError(404, f"{action} is {'an env' if per_env else 'a runtime'} action", f"POST /api/runtime/{'envs/<env_id>/' if per_env else ''}{action}")
        handler = {"spawn": self._spawn, "despawn_all": self._despawn_all, "reset": self._reset}.get(action, self._invoke)
        try:
            return self.run(self._guarded(action, handler(action, env_id, body)), timeout=CALL_TIMEOUT_S + 10.0)
        finally:
            self.poke()

    def start_benchmark(self, data_root: pathlib.Path, body: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """Start `benchmark` detached on this ROS domain, refused while a runtime or another runner is up here."""
        from .dashboard import _RUN_ID_RE
        from .runner import _default_run_id

        suite, contest = _config_name(body, "suite"), _config_name(body, "contest")
        run_id = body.get("run_id") or _default_run_id(suite, contest, None)
        if not isinstance(run_id, str) or not _RUN_ID_RE.match(run_id):
            raise ControlError(400, f"run ids are letters, digits, '_', '.' and '-', got {run_id!r}")
        if (data_root / run_id).exists():
            raise ControlError(409, f"run {run_id} already exists in {data_root}", "pick another run id, or resume it with arena evaluation benchmark --resume")
        tokens = _launch_tokens(body.get("args", ""))
        owner, runtime_up = self.run(_domain_busy())
        if owner is not None:
            raise ControlError(409, f"benchmark run {owner} already runs on this ROS domain", "wait for it or stop it first")
        if runtime_up:
            raise ControlError(409, "a runtime is up on this ROS domain, the benchmark starts its own", "stop the runtime first (Stop on the Targets page, or arena runtime down)")
        log_dir = data_root / ".web"
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / f"{run_id}.log"
        argv = ["ros2", "run", "arena_evaluation", "benchmark", "--suite", suite, "--contest", contest, "--run-id", run_id, "--data-root", str(data_root), *tokens]
        with log.open("w") as out:
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        return {"run_id": run_id, "pid": proc.pid, "log": str(log), "command": shlex.join(argv)}

    async def _guarded(self, action: str, work: Coroutine[typing.Any, typing.Any, dict[str, typing.Any]]) -> dict[str, typing.Any]:
        owner = await _benchmark_owner()
        if owner is not None:
            work.close()
            raise ControlError(409, f"benchmark run {owner} owns this runtime, the dashboard only watches it", "stop the run first, evaluation_stop or Ctrl-C on the runner")
        return {"action": action, "result": await work}

    async def _snapshot(self) -> dict[str, typing.Any]:
        from arena_runtime.mcp import runtime_envs, runtime_state

        from arena_mcp.core import ROS

        state = await runtime_state()
        base = {"now": time.time(), "domain": os.environ.get("ROS_DOMAIN_ID", "0"), "hold_reason": HOLD_REASON}
        if not state["running"]:
            return base | {"running": False, "hint": state["hint"], "envs": [], "benchmark": None, "viewport": False}
        envs, sim_time, benchmark, viewport, views = await asyncio.gather(runtime_envs(), ROS.sim_time(), _benchmark_owner(), self._viewport_up(), _live_views())
        episodes = await asyncio.gather(*(_episode(env["env_id"]) for env in envs))
        return base | {
            "running": True,
            "state": state,
            "sim_time": sim_time,
            "envs": [env | {"episode": episode, "view": views.get(env["env_id"])} for env, episode in zip(envs, episodes, strict=True)],
            "benchmark": benchmark,
            "viewport": viewport,
        }

    async def _viewport_up(self) -> bool:
        stamp, up = self._viewport
        if time.monotonic() - stamp < VIEWPORT_TTL_S:
            return up
        try:
            from arena_cam.mcp import view_available
        except ImportError:
            up = False
        else:
            available = await view_available()
            up = any(entry["surface"] == "sim" and entry["capture"] for entry in available["viewports"])
        self._viewport = (time.monotonic(), up)
        return up

    async def _load_options(self) -> dict[str, typing.Any]:
        from arena_robots.mcp import robots_catalog

        from arena_evaluation.mcp.authoring import world_names

        robots, worlds, planners, task_modes, launch_args = await asyncio.gather(
            asyncio.to_thread(robots_catalog),
            asyncio.to_thread(world_names),
            asyncio.to_thread(_planners),
            _task_modes(),
            asyncio.to_thread(_launch_args_of_env),
        )
        suites, contests = await asyncio.gather(asyncio.to_thread(_config_names, "suite"), asyncio.to_thread(_config_names, "contest"))
        return {
            "robots": sorted({entry["model"] for entry in robots}),
            "worlds": worlds,
            "planners": planners,
            "task_modes": task_modes,
            "launch_args": launch_args,
            "sims": _sims(),
            "suites": suites,
            "contests": contests,
            "actions": _action_docs(),
            "spawn_fields": _spawn_fields(launch_args, planners),
            "catalogs": _catalog_kinds(),
        }

    async def _catalog(self, kind: str, world: str) -> dict[str, typing.Any]:
        from arena_runtime.mcp import runtime_envs, runtime_state

        kinds = _catalog_kinds()
        if kind not in kinds:
            raise ControlError(400, f"no catalog kind {kind!r}", f"kinds: {', '.join(kinds)}")
        envs = await runtime_envs() if (await runtime_state())["running"] else []
        live = bool(envs)
        key = (kind, world, live)
        stamp, items = self._catalogs.get(key, (0.0, []))
        if time.monotonic() - stamp > CATALOG_TTL_S:
            items = await _live_catalog(kind, envs[0]["env_id"], world) if live else await _offline_catalog(kind, world)
            self._catalogs[key] = (time.monotonic(), items)
        return {"kind": kind, "world": world, "items": items}

    async def _env_task(self, env_id: int) -> dict[str, typing.Any]:
        from arena_runtime.env.episode_mcp import episode_task_modes, episode_task_modes_active
        from arena_runtime.env.task_layer import TM_AXES, task_layer, task_layer_name

        active = typing.cast("dict[str, list[str]]", (await episode_task_modes_active(env_id))["modes"])
        live = await episode_task_modes(env_id, source="live")
        namespaces = task_layer(task_layer_name(typing.cast("str", live["task_layer"]))).mode_namespaces()
        params: dict[str, typing.Any] = {}
        for axis in TM_AXES:
            declared = typing.cast("dict[str, list[dict[str, typing.Any]]]", live.get(axis, {}))
            for mode in active.get(axis, []):
                for entry in declared.get(mode, []):
                    params[f"{namespaces[axis][mode]}.{entry['name']}"] = entry["value"]
        return {"modes": {axis: list(names) for axis, names in active.items()}, "params": params}

    async def _camera(self, env_id: int) -> bytes:
        try:
            from arena_cam.mcp import View, view_capture
        except ImportError as exc:
            raise ControlError(404, "arena_cam is not installed, the dashboard has no camera") from exc
        episode = await _episode(env_id)
        robots = episode.get("robots") or []
        view = View(follow=robots[0]) if robots else View(top_down=12.0)
        content = await view_capture(view=view, env_id=env_id, surface="sim")
        return content[1].data

    async def _invoke(self, action: str, env_id: int | None, body: dict[str, typing.Any]) -> dict[str, typing.Any]:
        spec = _actions()[action]
        kwargs = _validated(action, body) | ({} if env_id is None else {"env_id": env_id})
        if action == "despawn" and env_id is not None:
            await self._stop_view(env_id)
        result = await _tool_result(spec.tool, **spec.fixed, **kwargs)
        if action == "view" and result["started"]:
            self._views.add(result["env_id"])
        return result

    async def _stop_view(self, env_id: int) -> None:
        """Stop the env's rerun view if one runs, a missing view tool or a failed stop does not block the caller."""
        from arena_mcp.core import ToolError

        self._views.discard(env_id)
        viz = _viz_tools()
        if viz is not None and env_id in await _live_views():
            with contextlib.suppress(ToolError):
                await _tool_result(viz[1], env_id=env_id)

    async def _spawn(self, action: str, _env_id: int | None, body: dict[str, typing.Any]) -> dict[str, typing.Any]:
        from arena_rclpy_mixins.launch_params import value_to_launch_str
        from arena_runtime.env.task_layer import selection_args

        spec = _actions()[action]
        args = _launch_args(body)
        count = _int(body, "count", 1, 1, MAX_SPAWN)
        kwargs = _validated(action, body)
        schema = await _task_schema()
        modes = _modes(body)
        try:
            modes = schema.layer.validate_modes(modes)
        except ValueError as exc:
            raise ControlError(400, str(exc)) from exc
        declared = schema.param_types(modes)
        params = _mapping(body, "params")
        undeclared = sorted(set(params) - set(declared))
        if undeclared:
            raise ControlError(400, f"parameters not declared by the selected modes: {', '.join(undeclared)}", f"declared: {', '.join(sorted(declared)) or 'none, select modes first'}")
        _check_param_types(params, declared)
        selected = dict(token.split(":=", 1) for token in selection_args(modes))
        args |= selected | {name: value_to_launch_str(value) for name, value in params.items()}
        spawned = [await spec.tool(**spec.fixed, **kwargs, launch_args=args) for _ in range(count)]
        return {"launch_args": args, "spawned": [{"env_id": entry["env_id"], "job": entry["job"]["id"] if "job" in entry else None} for entry in spawned]}

    async def _despawn_all(self, action: str, _env_id: int | None, body: dict[str, typing.Any]) -> dict[str, typing.Any]:
        from arena_runtime.mcp import runtime_envs

        spec = _actions()[action]
        kwargs = _validated(action, body)
        despawned = []
        for env in await runtime_envs():
            await self._stop_view(env["env_id"])
            despawned.append(await spec.tool(**spec.fixed, **kwargs, env_id=env["env_id"]))
        return {"despawned": despawned}

    async def _reset(self, action: str, env_id: int | None, body: dict[str, typing.Any]) -> dict[str, typing.Any]:
        from arena_runtime.env.episode_mcp import episode_queue

        spec = _actions()[action]
        kwargs = _validated(action, body)
        modes, params = _modes(body), _mapping(body, "params")
        if modes or params:
            _check_param_types(params, (await _task_schema()).param_types(None))
            await episode_queue(env_id=env_id, modes=typing.cast("dict[str, str | list[str]]", modes) or None, params=params or None)
        return await spec.tool(**spec.fixed, **kwargs, env_id=env_id)


def _as_control_error(exc: Exception) -> ControlError:
    from arena_mcp.core import ToolError

    if isinstance(exc, ControlError):
        return exc
    if isinstance(exc, ToolError):
        return ControlError(409, exc.message, exc.hint)
    return ControlError(500, f"{type(exc).__name__}: {exc}")


def _config_name(body: dict[str, typing.Any], kind: str) -> str:
    name = body.get(kind, "")
    if not isinstance(name, str) or not _CONFIG_NAME_RE.match(name.strip()):
        raise ControlError(400, f"{kind} must be a {kind} name, got {name!r}", f"the {kind} list shows the local ones")
    return name.strip()


def _launch_tokens(args: object) -> list[str]:
    if not isinstance(args, str):
        raise ControlError(400, "args must be text of key:=value launch arguments")
    try:
        tokens = shlex.split(args)
    except ValueError as exc:
        raise ControlError(400, f"args do not parse: {exc}") from exc
    bad = [token for token in tokens if not _LAUNCH_TOKEN_RE.match(token)]
    if bad:
        raise ControlError(400, f"launch arguments are key:=value, got {', '.join(bad)}")
    return tokens


def _config_names(kind: str) -> list[str]:
    from arena_evaluation.mcp.catalog import config_names

    return config_names(kind)


async def _domain_busy() -> tuple[str | None, bool]:
    from arena_runtime.mcp import SIM_STATE

    from arena_mcp.core import ROS

    owner = await _benchmark_owner()
    return owner, (await ROS.node()).count_publishers(SIM_STATE) > 0


def _planners() -> list[str]:
    try:
        from arena_planners.resolver import list_available
    except ImportError:
        return []
    return sorted({name for names in list_available().values() for name in names})


@dataclasses.dataclass
class _TaskSchema:
    layer: TaskLayer
    namespaces: dict[str, dict[str, str]]
    registry: dict[str, dict[str, list[dict[str, typing.Any]]]]

    def param_types(self, modes: dict[str, list[str]] | None) -> dict[str, str]:
        """Full parameter name to declared type for the given modes, every mode when None."""
        return {f"{self.namespaces[axis][mode]}.{entry['name']}": entry["type"] for axis, by_mode in self.registry.items() for mode, entries in by_mode.items() if modes is None or mode in modes.get(axis, []) for entry in entries}


async def _task_schema() -> _TaskSchema:
    from arena_runtime.env.episode_mcp import episode_task_modes
    from arena_runtime.env.task_layer import TM_AXES, task_layer, task_layer_name

    layer = task_layer(task_layer_name())
    modes = await episode_task_modes(source="registry")
    registry = {axis: typing.cast("dict[str, list[dict[str, typing.Any]]]", modes[axis]) for axis in TM_AXES if axis in modes}
    return _TaskSchema(layer, layer.mode_namespaces(), registry)


def _param(namespace: str, entry: dict[str, typing.Any]) -> dict[str, typing.Any]:
    constraint = entry.get("constraint") or {}
    kind = constraint.get("kind")
    leaf = entry["name"]
    step = constraint.get("step") if kind == "range" else None
    return {
        "name": f"{namespace}.{leaf}",
        "leaf": leaf,
        "label": leaf.replace("_", " "),
        "type": entry["type"],
        "default": entry["value"],
        "description": entry.get("description", ""),
        "choices": constraint["values"] if kind == "choices" else None,
        "catalog": constraint["name"] if kind == "catalog" else None,
        "range": [constraint["lo"], constraint["hi"], step or None] if kind == "range" else None,
        "widget": ("sketch" if constraint["sketch"] else "text") if kind == "text" else None,
        "read_only": bool(entry.get("read_only", False)),
    }


async def _task_modes() -> dict[str, dict[str, typing.Any]]:
    from arena_mcp.core import ToolError

    try:
        schema = await _task_schema()
    except ToolError:
        return {}
    out: dict[str, dict[str, typing.Any]] = {}
    for axis, modes in schema.registry.items():
        out[axis] = {
            "multi": axis in schema.layer.multi_axes(),
            "modes": {mode: {"namespace": schema.namespaces[axis][mode], "params": [_param(schema.namespaces[axis][mode], entry) for entry in entries]} for mode, entries in sorted(modes.items())},
        }
    return out


def _catalog_kinds() -> list[str]:
    from arena_runtime.env.catalogs import CATALOGS

    return list(CATALOGS)


async def _live_catalog(kind: str, env_id: int, world: str) -> list[str]:
    from arena_runtime.env.episode_mcp import episode_catalog

    return await episode_catalog(typing.cast("typing.Any", kind), env_id=env_id, world=world)


async def _offline_catalog(kind: str, world: str) -> list[str]:
    from arena_runtime.env.catalogs import LISTERS

    if kind == "scenarios" and not world:
        raise ControlError(400, "scenarios need a world", "pass ?world=<name>")
    try:
        return sorted(await LISTERS[kind](world))
    except FileNotFoundError as exc:
        raise ControlError(404, f"no world {world!r}", "worlds lists the ones that exist") from exc


def _spawn_fields(launch_args: list[dict[str, str]], planners: list[str]) -> list[dict[str, typing.Any]]:
    """Launch arguments the spawn form renders itself, each with the catalog kind or the fixed list its values come from."""
    described = {arg["name"]: arg["description"] for arg in launch_args}
    sources: dict[str, tuple[str | None, list[str] | None]] = {"robot": ("robots", None), "world": ("worlds", None), "robot.planner": (None, planners)}
    return [{"name": name, "description": described.get(name, ""), "catalog": catalog, "items": items} for name, (catalog, items) in sources.items()]


def _launch_args_of_env() -> list[dict[str, str]]:
    """The arguments env.launch.py declares, with the task mode selectors."""
    from arena_rclpy_mixins.launch_argument import UnreadableLaunchFile, declared_arguments
    from arena_runtime.env.task_layer import TM_AXES, mode_param, mode_param_description

    found: dict[str, str] = {mode_param(axis): mode_param_description(axis) for axis in TM_AXES}
    try:
        declared = declared_arguments("arena_runtime", "env.launch.py")
    except UnreadableLaunchFile:
        declared = []
    for argument in declared:
        found.setdefault(argument.name, argument.description)
    return [{"name": name, "description": description} for name, description in sorted(found.items())]


def _sims() -> list[str]:
    from arena_bringup.defaults import available_sims, installed_features

    return [*available_sims(installed_features(os.environ.get("INSTALLED", ""))), "dummy"]


async def _episode(env_id: int) -> dict[str, typing.Any]:
    from arena_runtime.env.episode_mcp import episode_current

    from arena_mcp.core import ToolError

    try:
        current = await episode_current(env_id)
    except ToolError as exc:
        return {"error": exc.message}
    record = typing.cast("dict[str, typing.Any] | None", current["current"])
    if record is None:
        return {"resetting": current["resetting"], "id": None}
    return {
        "resetting": current["resetting"],
        "id": record["episode_id"],
        "outcome": record["outcome"],
        "outcome_info": record["outcome_info"],
        "world": record["world"],
        "robots": list(record["robots"]),
        "modes": record["modes"],
        "start_time": record["start_time"],
    }


async def _benchmark_owner() -> str | None:
    from arena_evaluation_msgs.msg import BenchmarkState

    from arena_mcp.core import LATCHED, ROS, ToolError

    from .runner import STATE_TOPIC

    node = await ROS.node()
    if node.count_publishers(STATE_TOPIC) == 0:
        return None
    try:
        return (await ROS.latest(BenchmarkState, STATE_TOPIC, qos=LATCHED, timeout=2.0)).run_id or "unnamed"
    except ToolError:
        return "unnamed"
