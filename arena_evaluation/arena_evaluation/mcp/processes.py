from __future__ import annotations

import asyncio
import os
import pathlib
from typing import Annotated, Any, Literal

import pydantic

from arena_mcp.core import ToolError, proc

from .base import OptionalRunId, domain, run_path
from .runs import runner_pids

Kind = Literal["benchmark_runner", "simulation", "arena_node", "world_generator", "arena_cli"]


def arena_processes() -> list[dict[str, Any]]:
    """Arena processes as the evaluation CLI classifies them, with process group and whether this server depends on it."""
    from arena_evaluation.benchmark.debug import running_processes

    protected = proc.ancestry()
    own_group = os.getpgid(os.getpid())
    rows = []
    for row in running_processes():
        try:
            pgid = os.getpgid(row["pid"])
        except ProcessLookupError:
            continue
        rows.append({**row, "pgid": pgid, "protected": row["pid"] in protected or pgid in protected or pgid == own_group})
    return rows


@domain.tool(read_only=True)
def evaluation_ps() -> dict[str, Any]:
    """Running arena processes (benchmark runners, arena CLI, simulators, arena nodes, world generators) with pid, process group, kind, age and command."""
    return {"processes": arena_processes()}


@domain.tool(destructive=True)
async def evaluation_kill(
    pids: Annotated[list[int] | None, pydantic.Field(description="pids from evaluation_ps, each must be a classified arena process")] = None,
    kind: Annotated[Kind | None, pydantic.Field(description="every process of this kind")] = None,
    grace_s: Annotated[float, pydantic.Field(description="seconds after SIGINT before SIGTERM, then SIGKILL", gt=0, le=300)] = 15.0,
) -> dict[str, Any]:
    """Stop arena processes by pid or kind: SIGINT to each one's process group, escalating only past grace_s, refusing other processes and this server's own."""
    if not pids and kind is None:
        raise ToolError("pass pids or a kind, nothing is stopped by default", hint="evaluation_ps lists the candidates")
    rows = {row["pid"]: row for row in await asyncio.to_thread(arena_processes)}
    refused = []
    targets = []
    for pid in pids or []:
        row = rows.get(pid)
        if row is None:
            refused.append({"pid": pid, "reason": "not a running arena process"})
        elif kind is not None and row["kind"] != kind:
            refused.append({"pid": pid, "reason": f"kind {row['kind']}, not {kind}"})
        else:
            targets.append(row)
    if not pids:
        targets = [row for row in rows.values() if row["kind"] == kind]
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in targets:
        if row["protected"]:
            refused.append({"pid": row["pid"], "reason": "this server runs in it or under it"})
        else:
            groups.setdefault(row["pgid"], []).append(row)
    stopped = []
    for pgid, members in groups.items():
        try:
            outcome = await proc.stop_group(pgid, grace=grace_s)
        except proc.ProtectedProcess as exc:
            outcome = f"refused: {exc}"
        stopped.append({"pgid": pgid, "pids": [row["pid"] for row in members], "kinds": sorted({row["kind"] for row in members}), "outcome": outcome})
    return {"stopped": stopped, "refused": refused}


def tail_lines(path: pathlib.Path, lines: int) -> list[str]:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - 512 * 1024))
        text = handle.read().decode(errors="replace")
    return text.splitlines()[-lines:]


@domain.tool(read_only=True)
def evaluation_console(
    run_id: OptionalRunId = None,
    lines: Annotated[int, pydantic.Field(ge=1, le=5000)] = 200,
    lane: Annotated[str | None, pydantic.Field(description="a shared run's lane, its runner.<lane>.log; every lane log when omitted")] = None,
) -> dict[str, Any]:
    """The tail of a run's console log, runner.log (written for every run) or the lane logs of a shared run, and whether its runner is alive."""
    path = run_path(run_id)
    if lane is not None:
        logs = [path / f"runner.{lane}.log"]
    else:
        logs = [log for log in (path / "runner.log", *sorted(path.glob("runner.*.log"))) if log.is_file()]
    if not logs or not all(log.is_file() for log in logs):
        raise ToolError(f"run {path.name!r} has no console log yet", hint="the runner writes runner.log once its configs load, a failed start shows in the job log (jobs_log)")
    pids = runner_pids(path, path.name)
    return {"run_id": path.name, "runner_pids": pids, "alive": bool(pids), "logs": {log.name: tail_lines(log, lines) for log in logs}}
