"""Command-line surface and run bootstrap for the benchmark runner."""

from __future__ import annotations

import argparse
import datetime
import os
import pathlib
import sys

import attrs

from .config import Contest, Suite
from .runner_plan import _KV_RE, _MAX_SIM_DEATHS
from .state import Manifest
from .step import Step

_EPILOG = "exit codes: 0 ok, 1 systemic abort, 2 config error or crash, 3 lockstep, strict or efficacy verdict, 4 runner hung (deadman), 130 interrupted"

_OPTIONS: tuple[tuple[tuple[str, ...], dict[str, object]], ...] = (
    (("--suite",), {"default": "basic"}),
    (("--contest",), {"default": "basic"}),
    (("--scale-episodes",), {"type": float, "default": 1.0}),
    (("--run-id",), {"default": None}),
    (("--data-root",), {"default": None}),
    (
        ("--resume",),
        {
            "nargs": "?",
            "const": "__auto__",
            "default": None,
            "help": "Resume a prior run. Bare --resume picks the most recent resumable; --resume <run_id> opens that run explicitly.",
        },
    ),
    (("--retry-failed",), {"action": "store_true"}),
    (
        ("--retries",),
        {
            "type": int,
            "default": 1,
            "help": "Retries per cell after a systemic failure (sim death, env or robot setup), each on a fresh env. -1 retries forever, 0 never retries. Retried attempts are not recorded.",
        },
    ),
    (
        ("--max-sim-deaths",),
        {
            "type": int,
            "default": _MAX_SIM_DEATHS,
            "help": "Abort the run (exit 1) once this many sim deaths happen, whatever --retries allows.",
        },
    ),
    (
        ("--profile",),
        {
            "action": "store_true",
            "help": "Enable system resource profiling. Writes simulation_profile.yaml with peak/mean CPU, GPU, RAM, and disk I/O stats.",
        },
    ),
    (
        ("--lockstep-verdict",),
        {
            "action": "store_true",
            "help": "Exit 3 when a step's lockstep report fails (a stall of 5 s or more, no gated tick, or no planner beat registered or delivered).",
        },
    ),
    (
        ("--strict",),
        {
            "action": "store_true",
            "help": "Exit 3 when any cell ended with an error kind or ran fewer episodes than requested, whatever the episode outcomes were.",
        },
    ),
    (
        ("--spawn-budget",),
        {
            "type": float,
            "default": 600.0,
            "help": "Maximum wall time in seconds to wait for a single env spawn, on top of the inactivity heuristic, before giving up and despawning it.",
        },
    ),
    (
        ("--world-swap",),
        {
            "action": "store_true",
            "help": "Swap worlds in place through each env's task generator instead of restarting arena_runtime per world, so an env keeps its contestant across worlds.",
        },
    ),
    (
        ("--efficacy",),
        {
            "type": float,
            "default": None,
            "metavar": "FRACTION",
            "help": "Flag a non-SUCCESS episode as weak when it closes less than FRACTION of its starting goal distance. Prints a preflight verdict table and exits 3 on any wedged or weak cell.",
        },
    ),
    (
        ("--noexit",),
        {
            "action": "store_true",
            "help": "On completion, leave arena_runtime.launch.py and the last env running so you can poke at it. Recording stops with the last episode as usual.",
        },
    ),
    (
        ("--shared",),
        {
            "default": None,
            "metavar": "TOKEN",
            "help": "Run as one lane (named by ARENA_LANE) of a run several lanes share. Every lane gets the same TOKEN, a fresh one per launch, and the same --run-id or --resume <run_id>. Lanes claim env blocks under claims/<TOKEN>/ and fold their lane files into progress.csv and .benchmark_state.json on exit.",
        },
    ),
    (
        ("--viz",),
        {
            "action": "store_true",
            "help": "Serve the live dashboard of this run (arena evaluation serve) while it runs and print its URL. A dashboard of the same data root already on the port is reused, any other program there exits 2 before the sim starts.",
        },
    ),
)


def build_parser(default_host: str, default_port: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmark", epilog=_EPILOG)
    for flags, options in _OPTIONS:
        parser.add_argument(*flags, **options)
    parser.add_argument(
        "--viz-port",
        type=int,
        default=default_port,
        help=f"Port of the --viz dashboard on {default_host} (default {default_port}).",
    )
    return parser


def data_root(share: pathlib.Path, override: str | None) -> pathlib.Path:
    if override:
        root, source = pathlib.Path(override), "--data-root"
    elif configured := os.environ.get("ARENA_DATA_DIR"):
        root, source = pathlib.Path(configured) / "benchmarks", "ARENA_DATA_DIR"
    else:
        root, source = share / "data", "default"
    print(f"benchmark: data_root from {source}: {root}", file=sys.stderr)
    return root


def parse_extras(parser: argparse.ArgumentParser, extras: list[str]) -> dict[str, str]:
    passthrough: dict[str, str] = {}
    for arg in extras:
        if not _KV_RE.match(arg):
            parser.error(f"unrecognized argument: {arg!r}")
        key, value = arg.split(":=", 1)
        passthrough[key] = value
    if "env_n" in passthrough:
        print("benchmark: env_n:= is deprecated, use env.n:=", file=sys.stderr)
        passthrough.setdefault("env.n", passthrough.pop("env_n"))
    return passthrough


def fresh_manifest(
    *,
    run_id: str,
    suite: Suite,
    contest: Contest,
    suite_dict: dict,
    contest_dict: list | dict,
    suite_provenance: dict | None,
    contest_provenance: dict | None,
    config_hash: str,
    arena_git_sha: str | None,
    arena_git_dirty: bool,
    steps: list[Step],
    launch_args: dict[str, str],
    env_n: int,
    headless: bool,
    simulator: str,
    scale_episodes: float,
    cli_args: list[str],
) -> Manifest:
    return Manifest(
        run_id=run_id,
        created_at=datetime.datetime.now(tz=datetime.UTC).isoformat(),
        arena_git_sha=arena_git_sha,
        arena_git_dirty=arena_git_dirty,
        cli_args=cli_args,
        env_n=env_n,
        headless=headless,
        config_hash=config_hash,
        simulator=simulator,
        scale_episodes=scale_episodes,
        suite_name=suite.name,
        contest_name=contest.name,
        suite=suite_dict,
        contest=contest_dict,
        steps=[
            {
                "key": step.key,
                "contestant": attrs.asdict(step.contestant),
                "stage": step.stage._asdict(),
                "episodes_planned": step.episodes,
                "is_reference": step.is_reference,
                "reference_type": step.reference_type,
            }
            for step in steps
        ],
        launch_args=dict(launch_args),
        suite_provenance=suite_provenance,
        contest_provenance=contest_provenance,
    )
