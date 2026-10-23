from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import os
import pathlib
from typing import Annotated, Any, Literal

import pydantic

from arena_mcp.core import JOBS, ROS, Job, ToolError, cli, msgs, proc
from arena_mcp.core.jobs import RUNNING, SUCCEEDED

from .base import OptionalRunId, RunId, component, config_stem, data_root, domain, is_inline, run_path

TERMINAL_STEPS = ("ok", "failed", "partial", "skipped")
BENCHMARK_JOBS = ("benchmark", "preflight")

LaunchValue = str | int | float | bool
LaunchArgs = Annotated[dict[str, LaunchValue] | None, pydantic.Field(description="launch args passed as key:=value, e.g. {'task.episode.fail_on_collision': true}, booleans written as true/false")]
ConfigArg = Annotated[str | dict[str, object] | list[object] | None, pydantic.Field(description="a config name, an inline YAML or JSON literal, or the config itself as an object")]
Confirm = Annotated[float, pydantic.Field(description="seconds to wait for the runner to prepare the run dir, a runner exiting first raises its error", ge=0, le=300)]


class Note(pydantic.BaseModel):
    label: str
    value: str


def launch_value(value: LaunchValue) -> str:
    return ("true" if value else "false") if isinstance(value, bool) else str(value)


def launch_tokens(launch_args: dict[str, LaunchValue] | None, **named: LaunchValue | None) -> list[str]:
    """key:=value tokens for the named launch args the caller set and the free-form ones, nothing for unset ones."""
    chosen = {key.replace("__", "."): value for key, value in named.items() if value is not None}
    clash = sorted(set(chosen) & set(launch_args or {}))
    if clash:
        raise ToolError(f"{', '.join(clash)} given both as a parameter and in launch_args")
    return [f"{key}:={launch_value(value)}" for key, value in {**chosen, **(launch_args or {})}.items()]


def config_value(value: str | dict[str, object] | list[object]) -> str:
    """A suite or contest argument for the runner: an inline literal as it is, objects as JSON, names checked as one path component."""
    if isinstance(value, dict | list):
        return json.dumps(value)
    return value if is_inline(value) else config_stem(value)


def utc_stamp() -> str:
    return datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d-%H%M%S")


def benchmark_argv(
    *,
    suite: str | dict[str, object] | list[object] | None,
    contest: str | dict[str, object] | list[object] | None,
    run_id: str,
    resume: bool = False,
    retry_failed: bool = False,
    retries: int | None = None,
    max_sim_deaths: int | None = None,
    lockstep_verdict: bool = False,
    strict: bool = False,
    efficacy: float | None = None,
    spawn_budget: float | None = None,
    world_swap: bool = False,
    noexit: bool = False,
    profile: bool = False,
    scale_episodes: float | None = None,
    data_root: str | None = None,
    lanes: int | None = None,
    sim: str | None = None,
    headless: bool | None = None,
    env_n: int | None = None,
    optim_obstacles: str | None = None,
    launch_args: dict[str, LaunchValue] | None = None,
) -> list[str]:
    """The `arena evaluation benchmark` arguments for these options, always naming the run id."""
    if resume:
        if suite is not None or contest is not None:
            raise ToolError("a resumed run replays its stored suite and contest, do not pass them")
        argv = ["--resume", component(run_id, "run_id")]
    else:
        if suite is None or contest is None:
            raise ToolError("a fresh benchmark needs suite and contest", hint="evaluation_suites and evaluation_contests list them")
        argv = ["--suite", config_value(suite), "--contest", config_value(contest), "--run-id", component(run_id, "run_id")]
    for flag, on in (("--retry-failed", retry_failed), ("--lockstep-verdict", lockstep_verdict), ("--strict", strict), ("--world-swap", world_swap), ("--noexit", noexit), ("--profile", profile)):
        if on:
            argv.append(flag)
    for flag, value in (("--retries", retries), ("--max-sim-deaths", max_sim_deaths), ("--efficacy", efficacy), ("--spawn-budget", spawn_budget), ("--scale-episodes", scale_episodes), ("--data-root", data_root), ("--lanes", lanes)):
        if value is not None:
            argv += [flag, str(value)]
    return argv + launch_tokens(launch_args, sim=sim, headless=headless, env__n=env_n, optim__obstacles=optim_obstacles)


def runner_pids(path: pathlib.Path, run_id: str, running: dict[str, int] | None = None) -> list[int]:
    """Live runner processes of a run: lane pid files and runners started with its --run-id."""
    from arena_evaluation.benchmark.debug import running_pids_by_run_id

    pids = set()
    for pid_file in path.glob("runner.*.pid"):
        with contextlib.suppress(OSError, ValueError):
            pid = int(pid_file.read_text().strip())
            if proc.alive(pid):
                pids.add(pid)
    found = (running_pids_by_run_id() if running is None else running).get(run_id)
    if found is not None:
        pids.add(found)
    return sorted(pids)


def run_jobs(run_id: str) -> list[Job]:
    return [job for kind in BENCHMARK_JOBS for job in JOBS.list(kind=kind, limit=500) if job.title.endswith(f" {run_id}")]


def run_status(path: pathlib.Path, *, running: dict[str, int] | None = None, light: bool = False) -> dict[str, Any]:
    """A run's progress from its manifest, the union of its lane states and, unless light, its progress rows."""
    import yaml

    from arena_evaluation.benchmark.cli import _count_by_status
    from arena_evaluation.benchmark.state import Manifest, StateFile, progress_rows

    try:
        manifest = Manifest.from_yaml((path / "manifest.yaml").read_text())
    except (OSError, TypeError, yaml.YAMLError) as exc:
        raise ToolError(f"run {path.name!r} has no readable manifest.yaml: {exc}", hint="the runner writes it once the configs load, see evaluation_console") from exc
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
        "jobs": [job.id for job in run_jobs(manifest.run_id)],
        "has_combined_metrics": (path / "combined_metrics.parquet").is_file(),
        "has_report": (path / "report.html").is_file(),
    }


def run_summary_after(path: pathlib.Path) -> dict[str, Any]:
    if not (path / "manifest.yaml").is_file():
        return {"run_dir": str(path), "prepared": False}
    try:
        status = run_status(path)
    except ToolError as exc:
        return {"run_dir": str(path), "error": exc.message}
    return {key: status[key] for key in ("run_id", "run_dir", "state", "steps_total", "steps")}


async def confirm_start(job: Job, confirm_s: float) -> dict[str, Any]:
    """Wait until the runner prepared its run or exited, raising with the log tail when it exited with an error first."""
    from arena_evaluation.benchmark.runner import PREPARED

    loop = asyncio.get_running_loop()
    deadline = loop.time() + confirm_s
    while True:
        job = JOBS.get(job.id)
        if job.state != RUNNING:
            if job.state != SUCCEEDED:
                raise ToolError(f"{job.title} exited ({job.error or job.state}) before it prepared the run:\n{JOBS.tail(job.id, 25)}", hint=f"jobs_log {job.id} has the whole output")
            return {"prepared": PREPARED in JOBS.tail(job.id, 400)}
        if PREPARED in JOBS.tail(job.id, 400):
            return {"prepared": True}
        if loop.time() >= deadline:
            return {"prepared": False, "note": f"the runner was still starting after {confirm_s:.0f}s, poll jobs_log {job.id}"}
        await asyncio.sleep(0.5)


@domain.tool(destructive=True)
async def evaluation_benchmark(
    suite: ConfigArg = None,
    contest: ConfigArg = None,
    run_id: Annotated[str | None, pydantic.Field(description="run id, <utc stamp>-<suite>-<contest> when omitted, the run to continue when resume is set")] = None,
    resume: Annotated[bool, pydantic.Field(description="continue run_id (or the most recent resumable run) from its stored config")] = False,
    retry_failed: Annotated[bool, pydantic.Field(description="on resume also rerun failed steps")] = False,
    retries: Annotated[int | None, pydantic.Field(description="retries per cell after a systemic failure, -1 forever, 0 never (runner default 1)")] = None,
    max_sim_deaths: Annotated[int | None, pydantic.Field(description="abort once this many sim deaths happened")] = None,
    lockstep_verdict: Annotated[bool, pydantic.Field(description="exit 3 when a step's lockstep report fails")] = False,
    strict: Annotated[bool, pydantic.Field(description="exit 3 when a cell ended with an error kind or ran fewer episodes than requested")] = False,
    efficacy: Annotated[float | None, pydantic.Field(description="flag non-success episodes closing less than this fraction of the goal distance as weak, exit 3 on wedged or weak cells")] = None,
    spawn_budget: Annotated[float | None, pydantic.Field(description="max wall seconds for one env spawn (runner default 600)")] = None,
    world_swap: Annotated[bool, pydantic.Field(description="swap worlds in place instead of restarting the runtime per world")] = False,
    noexit: Annotated[bool, pydantic.Field(description="leave the runtime and last env up after the run")] = False,
    profile: Annotated[bool, pydantic.Field(description="write simulation_profile.yaml with resource stats")] = False,
    scale_episodes: Annotated[float | None, pydantic.Field(description="scale every stage's episode count")] = None,
    lanes: Annotated[int | None, pydantic.Field(description="run as one shared run across this many lanes, each in a pool fork", ge=1)] = None,
    sim: Annotated[str | None, pydantic.Field(description="sim:= launch arg, the suite's launch block or the runtime default when omitted")] = None,
    headless: Annotated[bool | None, pydantic.Field(description="headless:= launch arg, only passed when set")] = None,
    env_n: Annotated[int | None, pydantic.Field(description="env.n:= parallel envs, only passed when set")] = None,
    optim_obstacles: Annotated[str | None, pydantic.Field(description="optim.obstacles:= launch arg, only passed when set")] = None,
    launch_args: LaunchArgs = None,
    data_root_dir: Annotated[str | None, pydantic.Field(description="--data-root, the run tools only read the default $ARENA_DATA_DIR/benchmarks")] = None,
    confirm_s: Confirm = 60.0,
) -> dict[str, Any]:
    """Start `arena evaluation benchmark` as a job, always with an explicit run id. Exit codes 0 ok, 1 systemic abort, 2 config error, 3 lockstep, strict or efficacy verdict, 4 runner hung, 130 interrupted."""
    from arena_evaluation.benchmark.runner import _default_run_id
    from arena_evaluation.benchmark.state import find_most_recent_resumable

    root = pathlib.Path(data_root_dir) if data_root_dir else data_root()
    if resume:
        if run_id is None:
            run_id = await asyncio.to_thread(find_most_recent_resumable, root)
            if run_id is None:
                raise ToolError(f"no resumable run in {root}")
        elif not (root / component(run_id, "run_id") / "manifest.yaml").is_file():
            raise ToolError(f"no run {run_id!r} to resume in {root}", hint="evaluation_runs lists the runs")
    else:
        if suite is None or contest is None:
            raise ToolError("a fresh benchmark needs suite and contest", hint="evaluation_suites and evaluation_contests list them")
        run_id = run_id or _default_run_id(config_value(suite), config_value(contest), os.environ.get("ARENA_LANE"))
    argv = benchmark_argv(
        suite=suite,
        contest=contest,
        run_id=run_id,
        resume=resume,
        retry_failed=retry_failed,
        retries=retries,
        max_sim_deaths=max_sim_deaths,
        lockstep_verdict=lockstep_verdict,
        strict=strict,
        efficacy=efficacy,
        spawn_budget=spawn_budget,
        world_swap=world_swap,
        noexit=noexit,
        profile=profile,
        scale_episodes=scale_episodes,
        data_root=data_root_dir,
        lanes=lanes,
        sim=sim,
        headless=headless,
        env_n=env_n,
        optim_obstacles=optim_obstacles,
        launch_args=launch_args,
    )
    path = root / run_id
    job = await cli.job("benchmark", f"benchmark {run_id}", "evaluation", "benchmark", *argv, result=lambda _job: run_summary_after(path))
    started = await confirm_start(job, confirm_s)
    return {"run_id": run_id, "run_dir": str(path), "command": "arena evaluation benchmark " + " ".join(argv), **started, "job": JOBS.get(job.id).as_dict()}


@domain.tool(destructive=True)
async def evaluation_preflight(
    planners: Annotated[list[str] | None, pydantic.Field(description="planner names (arena planners ls), bridge planners run under drl or vla, others as nav2 local planners")] = None,
    all_planners: Annotated[bool, pydantic.Field(description="every initialized planner instead of names")] = False,
    soak: Annotated[bool, pydantic.Field(description="the crowded lockstep soak instead of the short start-and-drive preflight")] = False,
    run_id: Annotated[str | None, pydantic.Field(description="run id, <utc stamp>-preflight or -soak when omitted")] = None,
    lanes: Annotated[int | None, pydantic.Field(description="spread the planners over this many lanes, each in a pool fork", ge=1)] = None,
    sim: Annotated[str | None, pydantic.Field(description="sim:= launch arg")] = None,
    launch_args: LaunchArgs = None,
    confirm_s: Confirm = 60.0,
) -> dict[str, Any]:
    """Start `arena planners test [--preflight]` as a job: per planner a wedged, weak or ok verdict (preflight) or a stall, rtf and beat table (soak), exit 3 on a failing planner."""
    if all_planners == bool(planners):
        raise ToolError("pass planner names or all_planners=true, not both or neither", hint="planners_list or arena planners ls names them")
    run_id = component(run_id or f"{utc_stamp()}-{'soak' if soak else 'preflight'}", "run_id")
    argv = [*([] if soak else ["--preflight"]), *(["--all"] if all_planners else list(planners or ())), *launch_tokens(launch_args, sim=sim), "--run-id", run_id, *(["--lanes", str(lanes)] if lanes else [])]
    path = data_root() / run_id
    job = await cli.job("preflight", f"planners test {run_id}", "planners", "test", *argv, result=lambda _job: run_summary_after(path))
    started = await confirm_start(job, confirm_s)
    return {"run_id": run_id, "run_dir": str(path), "command": "arena planners test " + " ".join(argv), **started, "job": JOBS.get(job.id).as_dict()}


@domain.tool(destructive=True)
async def evaluation_stop(run_id: RunId, grace_s: Annotated[float, pydantic.Field(description="seconds the runner gets after SIGINT to shut its runtime down before SIGTERM", gt=0, le=300)] = 60.0) -> dict[str, Any]:
    """Stop a running benchmark: SIGINT to its runner process group (job, lane runners or a CLI-started runner), escalating only after grace_s."""
    from arena_evaluation.benchmark.debug import running_processes

    path = run_path(run_id)
    groups: dict[int, str] = {}
    for job in run_jobs(run_id):
        if job.state == RUNNING and job.pid is not None and job.alive:
            groups[job.pid] = f"job {job.id}"
    for pid in await asyncio.to_thread(runner_pids, path, run_id):
        with contextlib.suppress(ProcessLookupError):
            groups.setdefault(os.getpgid(pid), f"runner {pid}")
    if not groups:
        raise ToolError(f"no runner of run {run_id!r} is running", hint="evaluation_ps lists arena processes, evaluation_kill stops leftovers")
    stopped = []
    for pgid, owner in groups.items():
        try:
            outcome = await proc.stop_group(pgid, grace=grace_s)
        except proc.ProtectedProcess as exc:
            outcome = f"refused: {exc}"
        stopped.append({"pgid": pgid, "owner": owner, "outcome": outcome})
    leftovers = [row for row in await asyncio.to_thread(running_processes) if row["kind"] in ("simulation", "arena_node")]
    return {"run_id": run_id, "stopped": stopped, "arena_processes_left": leftovers}


@domain.tool(read_only=True)
def evaluation_runs(
    limit: int = 50,
    suite: str | None = None,
    contest: str | None = None,
    state: Literal["completed", "running", "incomplete"] | None = None,
    query: Annotated[str | None, pydantic.Field(description="case-insensitive substring of the run id")] = None,
) -> dict[str, Any]:
    """Benchmark runs, newest first, with suite, contest, state and step counts, filtered by exact suite or contest, state or run id substring."""
    root = data_root()
    if not root.is_dir():
        return {"data_root": str(root), "runs": [], "more": False}
    from arena_evaluation.benchmark.debug import running_pids_by_run_id

    running = running_pids_by_run_id()
    runs = []
    unreadable = 0
    more = False
    for path in sorted((entry for entry in root.iterdir() if entry.is_dir() and (entry / "manifest.yaml").is_file()), reverse=True):
        if query and query.lower() not in path.name.lower():
            continue
        try:
            status = run_status(path, running=running, light=True)
        except ToolError:
            unreadable += 1
            continue
        if (suite and status["suite"] != suite) or (contest and status["contest"] != contest) or (state and status["state"] != state):
            continue
        if len(runs) == limit:
            more = True
            break
        runs.append({key: status[key] for key in ("run_id", "suite", "contest", "simulator", "created_at", "state", "steps_total", "steps", "lanes", "runner_pids", "has_combined_metrics", "has_report")})
    return {"data_root": str(root), "runs": runs, "more": more, "unreadable_manifests": unreadable}


@domain.tool(read_only=True)
def evaluation_status(run_id: OptionalRunId = None) -> dict[str, Any]:
    """One run's state, step counts against its manifest, active and failed steps, lanes, runner pids, jobs and artifacts."""
    return run_status(run_path(run_id))


@domain.tool(read_only=True)
async def evaluation_watch(timeout_s: Annotated[float, pydantic.Field(gt=0, le=30)] = 3.0) -> dict[str, Any]:
    """The live state a running benchmark latches on /arena/benchmark/state (what `status --watch` shows)."""
    from arena_evaluation_msgs.msg import BenchmarkState

    from arena_evaluation.benchmark.runner import STATE_TOPIC

    msg = await ROS.latest(BenchmarkState, STATE_TOPIC, timeout=timeout_s, hint="no benchmark runner publishes on this ROS domain")
    return msgs.to_dict(msg)


@domain.tool(read_only=True)
def evaluation_progress(
    run_id: OptionalRunId = None,
    last: Annotated[int | None, pydantic.Field(description="only the last rows, all rows when omitted", ge=1)] = 50,
    step_key: str | None = None,
    contestant: str | None = None,
    stage: str | None = None,
) -> dict[str, Any]:
    """Per-episode progress rows of a run, its progress.csv and every lane's file folded, optionally filtered (what `tail` streams)."""
    from arena_evaluation.benchmark.state import progress_rows

    path = run_path(run_id)
    rows = [row for row in progress_rows(path) if (step_key is None or row["step_key"] == step_key) and (contestant is None or row["contestant"] == contestant) and (stage is None or row["stage"] == stage)]
    return {"run_id": path.name, "n_rows": len(rows), "rows": rows[-last:] if last else rows}


@domain.tool(read_only=True)
def evaluation_read_episode(run_id: RunId, episode_id: int) -> dict[str, Any]:
    """An episode's metadata yaml and the files in its directory."""
    import yaml

    from arena_evaluation.storage.folder_manager import FolderManager

    path = run_path(run_id)
    directory = FolderManager(data_root=path.parent).episode_dir(path.name, episode_id)
    for name in (f"{directory.name}.yaml", "metadata.yaml"):
        meta = directory / name
        if meta.is_file():
            return {"run_id": run_id, "episode_id": episode_id, "dir": str(directory), "metadata": yaml.safe_load(meta.read_text()), "files": sorted(entry.name for entry in directory.iterdir())}
    raise ToolError(f"run {run_id!r} has no episode {episode_id}", hint="evaluation_progress lists the recorded episodes")


def load_notes(path: pathlib.Path) -> list[dict[str, str]]:
    import yaml

    if not path.is_file():
        return []
    data = yaml.safe_load(path.read_text())
    if isinstance(data, list):
        return [{"label": str(row.get("label", "")), "value": str(row.get("value", ""))} for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        return [{"label": str(key), "value": str(value)} for key, value in data.items()]
    return []


def write_notes(path: pathlib.Path, notes: list[Note], mode: str) -> list[dict[str, str]]:
    import yaml

    rows = [note.model_dump() for note in notes]
    if mode == "append":
        rows = load_notes(path) + rows
    elif mode == "merge":
        merged = {row["label"]: row for row in load_notes(path)}
        merged.update({row["label"]: row for row in rows})
        rows = list(merged.values())
    path.write_text(yaml.safe_dump(rows, sort_keys=False, allow_unicode=True))
    return rows


@domain.tool(read_only=True)
def evaluation_read_notes(run_id: RunId) -> dict[str, Any]:
    """The notes.yaml of a run as [{label, value}], what table plots render as notes."""
    return {"run_id": run_id, "notes": load_notes(run_path(run_id) / "notes.yaml")}


@domain.tool()
def evaluation_write_notes(run_id: RunId, notes: list[Note], mode: Literal["replace", "append", "merge"] = "replace") -> dict[str, Any]:
    """Write notes.yaml of a run: replace it, append rows, or merge rows by label."""
    rows = write_notes(run_path(run_id) / "notes.yaml", notes, mode)
    return {"run_id": run_id, "mode": mode, "n_notes": len(rows)}


@domain.tool()
def evaluation_append_insight(run_id: RunId, label: str, value: str) -> dict[str, Any]:
    """Append one {label, value} row to a run's notes.yaml."""
    rows = write_notes(run_path(run_id) / "notes.yaml", [Note(label=label, value=value)], "append")
    return {"run_id": run_id, "n_notes": len(rows)}


def pipeline_inputs(run_ids: list[str] | None, episode_dirs: list[str] | None) -> tuple[str, list[pathlib.Path]]:
    """The input flag and directories of a processing verb: run directories, the latest run, or episode directories."""
    if run_ids and episode_dirs:
        raise ToolError("pass run_ids or episode_dirs, not both")
    if episode_dirs:
        root = data_root()
        paths = []
        for raw in episode_dirs:
            path = pathlib.Path(raw) if pathlib.Path(raw).is_absolute() else root / raw
            if not path.is_dir() or not path.name.startswith("episode_"):
                raise ToolError(f"{raw!r} is not an episode_NNN directory", hint="relative paths start at $ARENA_DATA_DIR/benchmarks, e.g. <run_id>/episodes/episode_004")
            paths.append(path)
        return "--run-dir", paths
    return "--benchmark-dir", [run_path(run_id) for run_id in (run_ids or [None])]


def manifest_ref(reference: str, first: pathlib.Path) -> str:
    """A report manifest reference, a custom manifest written into the run directory taking precedence over a name."""
    if is_inline(reference) or pathlib.Path(reference).is_absolute():
        return reference
    local = first / f"{config_stem(reference)}.yaml"
    return str(local) if local.is_file() else reference


async def pipeline_job(
    verb: str,
    run_ids: list[str] | None,
    episode_dirs: list[str] | None,
    *,
    output_dir: str | None = None,
    generate_gifs: bool = False,
    profile: bool = False,
    workers: int | None = None,
    report_manifest: str | None = None,
    force_extract: bool = False,
) -> dict[str, Any]:
    flag, paths = pipeline_inputs(run_ids, episode_dirs)
    argv = [verb, flag, *map(str, paths)]
    if output_dir:
        argv += ["--output-dir", output_dir]
    if generate_gifs:
        argv.append("--generate-gifs")
    if profile:
        argv.append("--profile")
    if workers is not None:
        argv += ["--workers", str(workers)]
    if report_manifest:
        argv += ["--report-manifest", manifest_ref(report_manifest, paths[0])]
    if force_extract:
        argv.append("--force-extract")
    out_dir = pathlib.Path(output_dir) if output_dir else paths[0]

    def outputs(_job: Job) -> dict[str, Any]:
        found: dict[str, Any] = {}
        if verb in ("process", "run") and flag == "--benchmark-dir":
            found["combined_metrics"] = {str(path): (path / "combined_metrics.parquet").is_file() for path in paths}
        if verb in ("report", "run"):
            found["report"] = str(out_dir / "report.html") if (out_dir / "report.html").is_file() else None
        if verb in ("report", "run", "plot"):
            found["plots"] = sorted(str(plot) for plot in (out_dir / "plots").glob("*.png"))[:200] if (out_dir / "plots").is_dir() else []
        return found

    job = await cli.job(verb if verb != "run" else "pipeline", f"evaluation {verb} {' '.join(path.name for path in paths)}", "evaluation", *argv, result=outputs)
    return {"command": "arena evaluation " + " ".join(argv), "job": job.as_dict()}


RunIds = Annotated[list[str] | None, pydantic.Field(description="benchmark run ids, the most recent run when neither run_ids nor episode_dirs is given")]
EpisodeDirs = Annotated[list[str] | None, pydantic.Field(description="episode_NNN directories to handle one by one, absolute or relative to $ARENA_DATA_DIR/benchmarks")]
OutputDir = Annotated[str | None, pydantic.Field(description="where reports and plots go, the first input directory when omitted")]
Workers = Annotated[int | None, pydantic.Field(description="worker processes, CPU count when omitted")]
ReportManifest = Annotated[str | None, pydantic.Field(description="report manifest: a name (evaluation_manifests), a custom manifest in the run dir, a path, or inline {...} YAML")]
Gifs = Annotated[bool, pydantic.Field(description="also render trajectory GIFs, slow")]
Profile = Annotated[bool, pydantic.Field(description="write pipeline_profile.yaml with per-phase resource stats")]


@domain.tool()
async def evaluation_process(run_ids: RunIds = None, episode_dirs: EpisodeDirs = None, force_extract: bool = False, workers: Workers = None, profile: Profile = False) -> dict[str, Any]:
    """Start `arena evaluation process` as a job: extract MCAP topics and compute every metric into combined_metrics.parquet, failing when nothing was written."""
    return await pipeline_job("process", run_ids, episode_dirs, workers=workers, profile=profile, force_extract=force_extract)


@domain.tool()
async def evaluation_extract(run_ids: RunIds = None, episode_dirs: EpisodeDirs = None, workers: Workers = None, profile: Profile = False) -> dict[str, Any]:
    """Start `arena evaluation extract` as a job: MCAP topics into the per-episode Parquet cache, no metrics."""
    return await pipeline_job("extract", run_ids, episode_dirs, workers=workers, profile=profile)


@domain.tool()
async def evaluation_report(run_ids: RunIds = None, episode_dirs: EpisodeDirs = None, report_manifest: ReportManifest = None, output_dir: OutputDir = None, generate_gifs: Gifs = False, profile: Profile = False) -> dict[str, Any]:
    """Start `arena evaluation report` as a job: report.html and plots from existing metrics, several runs merged into one report."""
    return await pipeline_job("report", run_ids, episode_dirs, output_dir=output_dir, generate_gifs=generate_gifs, profile=profile, report_manifest=report_manifest)


@domain.tool()
async def evaluation_plot(run_ids: RunIds = None, episode_dirs: EpisodeDirs = None, report_manifest: ReportManifest = None, output_dir: OutputDir = None, generate_gifs: Gifs = False, profile: Profile = False) -> dict[str, Any]:
    """Start `arena evaluation plot` as a job: the manifest's static PNG plots only, no HTML."""
    return await pipeline_job("plot", run_ids, episode_dirs, output_dir=output_dir, generate_gifs=generate_gifs, profile=profile, report_manifest=report_manifest)


@domain.tool()
async def evaluation_pipeline(run_ids: RunIds = None, episode_dirs: EpisodeDirs = None, report_manifest: ReportManifest = None, output_dir: OutputDir = None, generate_gifs: Gifs = False, workers: Workers = None, profile: Profile = False) -> dict[str, Any]:
    """Start `arena evaluation run` as a job: re-extract, process and report in one go."""
    return await pipeline_job("run", run_ids, episode_dirs, output_dir=output_dir, generate_gifs=generate_gifs, profile=profile, workers=workers, report_manifest=report_manifest)


@domain.tool()
async def evaluation_acoustic(
    run_id: RunId,
    action: Literal["list", "animate", "snapshot"],
    episode: Annotated[str | None, pydantic.Field(description="episode id or worst, loudest-source, max-total (default worst)")] = None,
    frame: Annotated[int | None, pydantic.Field(description="snapshot frame index, the worst-case frame when omitted")] = None,
    output_format: Annotated[Literal["gif", "mp4", "frames"] | None, pydantic.Field(description="animate output format (default gif)")] = None,
    fps: int | None = None,
    max_frames: int | None = None,
    stride: int | None = None,
    downsample: int | None = None,
    dpi: int | None = None,
    vmin: Annotated[float | None, pydantic.Field(description="color floor in dBA")] = None,
    vmax: Annotated[float | None, pydantic.Field(description="color ceiling in dBA")] = None,
    door_overlay: bool = True,
    trajectories: bool = True,
    robot_trail: int | None = None,
    output: Annotated[str | None, pydantic.Field(description="output path, plots/<episode>_acoustic.* when omitted")] = None,
) -> dict[str, Any]:
    """Start `arena evaluation acoustic` as a job: list episodes with acoustic metrics, animate the acoustic field, or render one snapshot."""
    path = run_path(run_id)
    argv = ["acoustic", action, "--benchmark-dir", str(path)]
    allowed = {"list": set(), "snapshot": {"episode", "frame", "downsample", "vmin", "vmax", "trajectories", "output"}, "animate": {"episode", "fps", "max_frames", "stride", "downsample", "output_format", "dpi", "vmin", "vmax", "door_overlay", "trajectories", "robot_trail", "output"}}[action]
    values = {"episode": episode, "frame": frame, "fps": fps, "max_frames": max_frames, "stride": stride, "downsample": downsample, "output_format": output_format, "dpi": dpi, "vmin": vmin, "vmax": vmax, "robot_trail": robot_trail, "output": output}
    switches = {"door_overlay": ("--no-door-overlay", door_overlay), "trajectories": ("--no-trajectories", trajectories)}
    given = {name for name, value in values.items() if value is not None} | {name for name, (_, on) in switches.items() if not on}
    if given - allowed:
        raise ToolError(f"{action} takes no {', '.join(sorted(given - allowed))}", hint=f"{action} options: {', '.join(sorted(allowed)) or 'none'}")
    for name, value in values.items():
        if value is not None:
            argv += [f"--{'format' if name == 'output_format' else name.replace('_', '-')}", str(value)]
    argv += [flag for name, (flag, on) in switches.items() if not on]
    job = await cli.job("acoustic", f"evaluation acoustic {action} {run_id}", "evaluation", *argv, result=lambda _job: {"plots": sorted(str(plot) for plot in (path / "plots").glob("*acoustic*"))} if (path / "plots").is_dir() else {})
    return {"command": "arena evaluation " + " ".join(argv), "job": job.as_dict()}


@domain.resource("arena://evaluation/runs", mime_type="application/json", description="The 50 newest benchmark runs with state and step counts")
def runs_resource() -> dict[str, Any]:
    return evaluation_runs()


@domain.resource("arena://evaluation/runs/{run_id}/state", mime_type="application/json", description="A run's manifest summary, step counts and lane state")
def run_state_resource(run_id: str) -> dict[str, Any]:
    return run_status(run_path(run_id))


@domain.resource("arena://evaluation/runs/{run_id}/notes", mime_type="application/json", description="A run's notes.yaml as [{label, value}]")
def run_notes_resource(run_id: str) -> list[dict[str, str]]:
    return load_notes(run_path(run_id) / "notes.yaml")


@domain.resource("arena://evaluation/runs/{run_id}/report", mime_type="application/json", description="Path of a run's report.html, null before evaluation_report ran")
def run_report_resource(run_id: str) -> dict[str, str | None]:
    report = run_path(run_id) / "report.html"
    return {"run_id": run_id, "report": str(report) if report.is_file() else None}


@domain.prompt(description="Author, run, process and analyze a benchmark end to end")
def evaluation_benchmark_workflow(goal: str) -> str:
    return f"""Goal: {goal}
1. Pick or author configs: evaluation_suites, evaluation_contests, evaluation_config_schema, episode_task_modes, planners_list, assets_ls kind=world; validate with evaluation_validate_suite and evaluation_validate_contest before evaluation_create_suite or evaluation_create_contest.
2. Start it with evaluation_benchmark, then follow evaluation_status, evaluation_progress and evaluation_console; stop it with evaluation_stop.
3. When the job ended, run evaluation_process and wait for its job, then evaluation_report.
4. Analyze with evaluation_summarize, evaluation_compare_planners and evaluation_top_n, record findings with evaluation_append_insight."""
