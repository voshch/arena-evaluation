from __future__ import annotations

import asyncio
import pathlib
from typing import Annotated, Any, Literal

import pydantic

from arena_mcp.core import ToolError

from .base import OptionalRunId, domain, run_path

Kind = Literal["benchmark_runner", "simulation", "arena_node", "world_generator", "arena_cli"]


@domain.tool(read_only=True)
def evaluation_ps() -> dict[str, Any]:
    """Running arena processes (benchmark runners, arena CLI, simulators, arena nodes, world generators) with pid, kind, age and command, like `arena evaluation ps`."""
    from arena_evaluation.benchmark.debug import running_processes

    return {"processes": running_processes()}


@domain.tool(destructive=True)
async def evaluation_kill(
    pids: Annotated[list[int] | None, pydantic.Field(description="pids from evaluation_ps, each must be a classified arena process")] = None,
    kind: Annotated[Kind | None, pydantic.Field(description="every process of this kind, ignored when pids are given")] = None,
    grace_s: Annotated[float, pydantic.Field(description="seconds after SIGINT before SIGTERM, then SIGKILL", gt=0, le=300)] = 15.0,
) -> dict[str, Any]:
    """Stop arena processes by pid or kind like `arena evaluation kill`: SIGINT to each one (its group when it leads one), escalating only past grace_s, refusing other processes and this server's ancestors."""
    from arena_evaluation.benchmark.debug import kill_processes

    try:
        results = await asyncio.to_thread(kill_processes, pids=pids or None, kind=kind, grace_s=grace_s)
    except ValueError as exc:
        raise ToolError("pass pids or a kind, nothing is stopped by default", hint="evaluation_ps lists the candidates") from exc
    return {"results": results}


@domain.tool(read_only=True)
def evaluation_console(
    run_id: OptionalRunId = None,
    lines: Annotated[int, pydantic.Field(ge=1, le=5000)] = 200,
    lane: Annotated[str | None, pydantic.Field(description="a shared run's lane, its runner.<lane>.log; every lane log when omitted")] = None,
) -> dict[str, Any]:
    """The tail of a run's console log, runner.log (written for every run) or the lane logs of a shared run, and whether its runner is alive, like `arena evaluation console`."""
    from arena_evaluation.benchmark.debug import tail_console

    path = run_path(run_id)
    lanes = [lane] if lane is not None else [None, *sorted(log.name.removeprefix("runner.").removesuffix(".log") for log in path.glob("runner.*.log"))]
    tails = [tail_console(path.name, lines, lane=entry) for entry in lanes]
    logs = {pathlib.Path(tail["path"]).name: tail["lines"] for tail in tails if tail["exists"]}
    if not logs:
        raise ToolError(f"run {path.name!r} has no console log yet", hint="the runner writes runner.log once its configs load, a failed start shows in the job log (jobs_log)")
    return {"run_id": path.name, "pid": tails[0]["pid"], "alive": tails[0]["alive"], "logs": logs}
