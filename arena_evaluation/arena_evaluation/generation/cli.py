"""`evaluation generate`: base populations, prompt-driven cases and suites for the `edge_case` task modes."""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml

from .formats import first_level, is_world, levels, read_scenario
from .promptgen import prompts as P
from .promptgen.world import Grid, WorldError, _endpoint
from .propose import DEFAULT_MODEL, STORE, Unavailable, build_prompt, classify_tail, load_overrides, merge, unknown_names
from .scenario.materialize import WouldOverwrite, prune_generated, write_base
from .scenario.occupancy import world_statics
from .scenario.populate import MIN_ROUTE_CLEARANCE, PROFILES, basis_for, robot_zones, route_points
from .scenario.populate import build as build_base
from .scenario.route import MIN_EPISODE_S, ROBOT_SPEED_MPS, robot_for, route_seconds
from .scenario.zone_screen import _load_grid, _stamp_statics, ref_safe_zones, screen_zones
from .scenario.zones import zones_by_role

PROFILE_NOTES = {
    "normal": "working hours: the building's dwellers walk between rooms, pairs talk in rooms",
    "break": "break time: the same crowd in the break rooms",
    "night": "night: half the crowd, slower",
    "empty": "nobody - the no-population control",
}


def default_worlds_root() -> Path:
    """The source tree's worlds: generated scenarios are written there, and `install/` only symlinks files that existed at build time."""
    arena = Path(os.environ.get("ARENA_DIR") or Path(__file__).resolve().parents[4])
    return arena / "arena_simulation_setup" / "worlds"


def default_suites_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "benchmark" / "suites"


def _route_reachable(root: Path, zones: list[dict[str, Any]], robot: dict[str, Any]) -> bool:
    """Whether walls plus furniture leave a robot-wide path along the robot's route."""
    try:
        pts = route_points(robot, zones)
        loaded = _load_grid(root)
        if loaded is None or len(pts) < 2:
            return True
        occupied, res, ox, oy = loaded
        occupied = occupied.copy()
        _stamp_statics(occupied, world_statics(root), res, ox, oy)
        grid = Grid(occupied=occupied, res=res, ox=ox, oy=oy)
        for a, b in itertools.pairwise(pts):
            grid.plan(_endpoint(grid, a), _endpoint(grid, b))
    except WorldError:
        return False
    except Exception:  # a hint only: populate and promptgen validate the route again
        return True
    return True


def _world_parts(worlds_root: Path, world: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """A world's zones and a robot copied from one of its scenarios, else the two most separated reachable zones."""
    root = worlds_root / world
    doc = first_level(root)
    if doc is None:
        raise FileNotFoundError(f"{world}: no world.yaml")
    zones = doc.get("zones") or []

    for cand in sorted((root / "scenarios").glob("*/scenario.yaml")):
        robots = read_scenario(cand).get("robots") or []
        if robots and isinstance(robots[0], dict) and robots[0].get("start"):
            robot = dict(robots[0])
            if _route_reachable(root, zones, robot):
                return zones, robot
            # a scenario copied from another world carries that world's coordinates (acoustics_bend45_wide, 2026-08-30)
            print(f"  ! {world}: {cand.parent.name}'s robot route is not plannable on this map; ignoring it")

    named = [z for z in zones if isinstance(z, dict) and z.get("name") and z.get("corners")]
    if len(named) < 2:
        raise FileNotFoundError(f"{world}: no scenario with a robot to copy, and fewer than two named zones to fall back on. Author a scenario with a `robots:` block first.")

    def centroid(z: dict[str, Any]) -> tuple[float, float]:
        cs = z["corners"]
        return (sum(float(c["x"]) for c in cs) / len(cs), sum(float(c["y"]) for c in cs) / len(cs))

    pairs = sorted(((math.dist(centroid(a), centroid(b)), a["name"], b["name"]) for i, a in enumerate(named) for b in named[i + 1 :]), reverse=True)
    # the farthest pair can sit in disconnected rooms (arena_arena_032, _009, _065, _006, _007, 2026-08-30)
    _, start, goal = pairs[0]
    for _d, a_name, b_name in pairs:
        if _route_reachable(root, named, {"start": a_name, "goal": b_name}):
            start, goal = a_name, b_name
            break
    print(f"  ! {world}: no scenario declares a robot; using the two most separated reachable zones as endpoints - start={start!r} goal={goal!r}")
    print("  ! override with --robot-start / --robot-goal if that is not the traverse you want")
    return zones, {"start": start, "goal": goal}


def _all_zones(worlds_root: Path, world: str) -> list[dict[str, Any]]:
    """Every zone across all levels: a name only on floor 3 (`three_storied_residential`'s `3_study`) still needs a role."""
    zones: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        docs = levels(worlds_root / world)
    except yaml.YAMLError:
        docs = []
    for doc in docs:
        for zone in doc.get("zones") or []:
            if isinstance(zone, dict) and zone.get("name") and str(zone["name"]) not in seen:
                seen.add(str(zone["name"]))
                zones.append(zone)
    return zones


def _cmd_zones(args: argparse.Namespace) -> int:
    zones, _ = _world_parts(args.worlds, args.world)
    overrides = load_overrides()
    by_role = zones_by_role(zones, overrides)
    for role, names in by_role.items():
        if names:
            print(f"{role.value:9s} " + ", ".join(f"{n}*" if n in overrides else n for n in names))
    if any(n in overrides for names in by_role.values() for n in names):
        print("\n* from the reviewed mapping, not a substring rule")
    if unresolved := unknown_names(zones, overrides):
        print(f"\nunclassified: {', '.join(unresolved)}")
        print("  run `evaluation generate classify` to have a model propose roles for these")
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    # a dotted directory (`.generated`) is scratch beside the worlds, not a world
    names = sorted(args.world) if args.world else sorted(p.name for p in args.worlds.iterdir() if p.is_dir() and not p.name.startswith(".") and is_world(p))
    overrides = load_overrides()
    pending: dict[str, tuple[list[dict[str, Any]], list[str]]] = {}
    for world in dict.fromkeys(names):
        zones = _all_zones(args.worlds, world)
        if not zones:
            print(f"  ! {world}: no zones declared on any level")
            continue
        if todo := unknown_names(zones, overrides):
            pending[world] = (zones, todo)

    if not pending:
        print("every zone in every world is already classified by a rule or the reviewed mapping")
        return 0
    print(f"{sum(len(todo) for _, todo in pending.values())} unclassified zone name(s) across {len(pending)} world(s)")
    if not args.write:
        for world, (zones, todo) in pending.items():
            print(f"\n### {world}  ({len(todo)})")
            print(build_prompt(zones, todo, world=world))
        print("(dry run - this is the prompt that would be sent; pass --write to ask and store)")
        return 0

    written = 0
    for world, (zones, todo) in pending.items():
        print(f"\n### {world}")
        try:
            proposals, complaints = classify_tail(zones, world=world, names=todo, model=args.model)
        except Unavailable as exc:
            print(f"  ! {exc}")
            return 1
        for proposal in proposals:
            print(f"  {' ' if proposal.confident else '?'} {proposal.zone:20s} {proposal.role.value:8s} {proposal.why}")
        for complaint in complaints:
            print(f"  ! {complaint}")
        landed, kept = merge(proposals)
        for note in kept:
            print(f"  = {note}")
        written += landed
    print(f"\n{written} entry(ies) written to {STORE}")
    print("review them there; set `source: human` on any you change and it will not be revisited")
    return 0


def _cmd_populate(args: argparse.Namespace) -> int:
    world_root = args.worlds / args.world
    zones, robot = _world_parts(args.worlds, args.world)
    if args.robot_start:
        robot = {"start": args.robot_start, "goal": args.robot_goal or robot.get("goal")}
        print(f"  robot endpoints overridden: start={robot['start']!r} goal={robot['goal']!r}")

    overrides = load_overrides()
    if unresolved := unknown_names(zones, overrides):
        print(f"  ! {len(unresolved)} zone(s) no rule classifies: {', '.join(unresolved)}")
        print("  !   they are populated as neither work nor break; `evaluation generate classify` proposes roles")
    if (basis := basis_for(zones, overrides)) != "workplace":
        print(f"  ! zone mapping: {basis} - this world has no work zones, so the active/retreat")
        print("  !   distinction is drawn between living areas and private rooms instead")

    # homes are screened once against the longest form of the route; every base drives the same segments
    spec = robot_for(robot, zones)
    verdicts = screen_zones(zones, world_root=world_root, route=route_points(spec, zones), statics=world_statics(world_root), min_route_clearance=MIN_ROUTE_CLEARANCE, robot_zones=robot_zones(robot))
    homes = ref_safe_zones(verdicts)
    print(f"  homes: {len(homes)}/{len(verdicts)} zone(s) entirely clear of the route" + ("" if homes else " - falling back to every target zone but the robot's own"))
    for name, verdict in sorted(verdicts.items()):
        if not verdict.ref_safe and verdict.reason:
            print(f"    - {name}: {verdict.reason}")
    print(f"  route: {route_seconds(spec, zones):.0f} s at {ROBOT_SPEED_MPS} m/s ({len(spec['phases'])} leg(s); floor {MIN_EPISODE_S:.0f} s)")

    written: list[str] = []
    for profile in PROFILES:
        doc = build_base(profile, zones=zones, robot=spec, seed=args.seed, homes=homes, overrides=overrides, world=args.world)
        agents = doc.get("dynamic") or []
        if not agents and PROFILES[profile]["count"] > 0:
            print(f"  ! {profile}: no zone to populate - this base is measurable as `empty`,")
            print("  !   and every case needing a crowd will refuse it")
        talking = sum(1 for a in agents if "talker" in str(a["agent"].get("agent_type")))
        line = f"  {profile:7s} {len(agents):3d} agent(s): {talking} talking, {len(agents) - talking} walking, from {len({str(a['pose']) for a in agents})} room(s)"
        if args.write:
            try:
                out = write_base(doc, world_root=world_root, profile=profile, force=args.force, note=PROFILE_NOTES[profile])
            except WouldOverwrite as exc:
                print(f"  {profile:7s} SKIPPED - {exc}")
                continue
            written.append(out.name)
        print(line)

    if args.write and args.prune:
        keep = [*written, *(d.name for d in (world_root / "scenarios").glob("*__*"))]
        for gone in prune_generated(world_root, keep):
            print(f"  - pruned stale generated base {gone}")
    if not args.write:
        print("(dry run - pass --write to materialise)")
    return 0


def _cmd_promptgen(args: argparse.Namespace) -> int:
    from .promptgen import generate, write_result
    from .promptgen.world import WorldContext

    root = args.worlds / args.world
    prompts = P.load_sets(args.world, extra=[Path(p) for p in args.prompts or []])
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",") if i.strip()}
        prompts = [p for p in prompts if p.id in wanted]
    if not prompts:
        print(f"no prompts: none of the --ids in {P.shipped_dir()} or the --prompts files")
        return 1
    bases = [b.strip() for b in args.bases.split(",") if b.strip()] if args.bases else list(PROFILES)
    overrides = load_overrides()
    cache = Path(args.cache) if args.cache else None
    seen: dict[str, str] = {}
    counts: dict[str, int] = {}
    written = 0
    for base_id in bases:
        base_path = root / "scenarios" / base_id / "scenario.yaml"
        if not base_path.is_file():
            print(f"  ! base {base_id}: not found - run `populate --write` first")
            continue
        base_doc = read_scenario(base_path)
        try:
            world = WorldContext.load(args.worlds, args.world, base_doc, overrides)
        except WorldError as exc:
            print(f"  ! {base_id}: {exc}")
            continue
        for prompt in prompts:
            try:
                result = generate(prompt, world=world, base_id=base_id, base_doc=base_doc, base_dir=base_path.parent, seed=args.seed, cache_dir=cache, model=args.model, oracle=args.oracle)
            except Exception as exc:  # one prompt must not stop the batch
                print(f"  ! {base_id}/{prompt.id}: {exc!r}")
                if args.verbose:
                    traceback.print_exc()
                counts["crashed"] = counts.get("crashed", 0) + 1
                continue
            status = result.record.status
            if status not in ("infeasible", "denied"):
                key = result.record.dedup_key()
                if key in seen:
                    status = f"duplicate of {seen[key]}"
                else:
                    seen[key] = result.scenario_id
            counts[status.split(" ")[0]] = counts.get(status.split(" ")[0], 0) + 1
            fails = "; ".join(f"{f.step}:{f.kind}" for f in result.record.failures)
            print(f"  {result.scenario_id:40s} {status:10s} [{', '.join(result.record.steps) or '-'}]" + (f"  failures: {fails}" if fails else ""))
            if args.verbose:
                for note in result.record.warnings:
                    print(f"      - {note}")
                for failure in result.record.failures:
                    print(f"      ! {failure.step}: {failure.reason}")
            if status not in ("ok", "partial"):
                continue
            if args.write and write_result(result, root, keep_partial=not args.strict) is not None:
                written += 1
    print(f"{sum(counts.values())} generated: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) + (f"; wrote {written}" if args.write else " (dry run - pass --write)"))
    if args.write:
        print("rebuild arena_simulation_setup before launching: new scenario directories are invisible to the runtime until then")
    return 0


def _cmd_suite(args: argparse.Namespace) -> int:
    from .suite import build, scenarios

    worlds = {world: scenarios(args.worlds / world, args.match) for world in args.world}
    for world, names in worlds.items():
        print(f"  {world:28s} {len(names):4d} scenario(s)")
    if not any(worlds.values()):
        print("no scenarios matched" + ("" if args.match else ": none of these worlds has generated scenarios (pass --match for hand-authored ones)"))
        return 1
    doc = build(worlds, robot=args.robot, episodes=args.episodes, seed=args.seed, timeout=args.timeout)
    out = Path(args.out) if args.out else default_suites_dir() / f"{args.name}.yaml"
    out.write_text(yaml.safe_dump(doc, sort_keys=False))
    print(f"wrote {out}: {len(doc['stages'])} stage(s) x {args.episodes} episode(s); run with `arena evaluation benchmark --suite {out.stem} --contest <contest>`")
    return 0


def add_parsers(sub: argparse._SubParsersAction) -> None:
    generate = sub.add_parser("generate", help="Edge-case scenario generation: zones, bases, prompt-driven cases, suites.")
    verbs = generate.add_subparsers(dest="generate_command", required=True)

    def verb(name: str, fn: Callable[[argparse.Namespace], int], text: str) -> argparse.ArgumentParser:
        p = verbs.add_parser(name, help=text)
        p.add_argument("--worlds", type=Path, default=None, help="worlds root (default: the source tree's arena_simulation_setup/worlds)")
        p.set_defaults(generate_fn=fn)
        return p

    p = verb("zones", _cmd_zones, "classify a world's zones into roles")
    p.add_argument("world")

    p = verb("classify", _cmd_classify, "ask a model for the zone roles no rule covers")
    p.add_argument("world", nargs="*", help="worlds to classify; default every world")
    p.add_argument("--model", default=None, help=f"override the model (default {DEFAULT_MODEL})")
    p.add_argument("--write", action="store_true", help="call the model and store what it returns")

    p = verb("populate", _cmd_populate, "write the four base scenarios (normal, break, night, empty) for a world")
    p.add_argument("world")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--write", action="store_true")
    p.add_argument("--robot-start", default=None, help="zone name or 'x,y,yaw' for the robot start")
    p.add_argument("--robot-goal", default=None, help="zone name or 'x,y,yaw' for the robot goal")
    p.add_argument("--prune", action="store_true", help="remove generated bases this run did not write (never hand-authored ones)")
    p.add_argument("--force", action="store_true", help="replace scenarios this tool did not generate (they may be hand-authored)")

    p = verb("promptgen", _cmd_promptgen, "realise prompt x base scenarios for a world")
    p.add_argument("world")
    p.add_argument("--bases", default="", help="comma-separated base ids (default: every base)")
    p.add_argument("--ids", default="", help="comma-separated prompt ids to restrict to")
    p.add_argument("--prompts", nargs="*", help="extra prompt files")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", default=None, help="model answer cache directory")
    p.add_argument("--model", default=None)
    p.add_argument("--oracle", action="store_true", help="use each prompt's own template slots instead of a model")
    p.add_argument("--strict", action="store_true", help="do not write partial scenarios")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--write", action="store_true")

    p = verb("suite", _cmd_suite, "write a benchmark suite with one edge_case stage per (world, scenario)")
    p.add_argument("world", nargs="+")
    p.add_argument("--name", default="edge_case", help="suite name, the file stem under configs/benchmark/suites (default edge_case)")
    p.add_argument("--out", default=None, help="write here instead of configs/benchmark/suites/<name>.yaml")
    p.add_argument("--match", nargs="*", default=[], help="scenario globs (default: every scenario the generator wrote)")
    p.add_argument("--robot", default="jackal", help="the chassis; a contest compares planners on one drive family")
    p.add_argument("--episodes", type=int, default=1, help="episodes per stage, seeded seed..seed+episodes-1")
    p.add_argument("--seed", type=int, default=1000, help="first episode seed; keep screening and dataset seed blocks disjoint")
    p.add_argument("--timeout", default="240s", help="sim-time budget per episode")


def run(args: argparse.Namespace) -> int:
    if args.worlds is None:
        args.worlds = default_worlds_root()
    try:
        return int(args.generate_fn(args))
    except (FileNotFoundError, ValueError) as exc:
        print(f"evaluation generate: {exc}", file=sys.stderr)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evaluation")
    add_parsers(parser.add_subparsers(dest="command", required=True))
    return run(parser.parse_args(["generate", *(argv if argv is not None else sys.argv[1:])]))


if __name__ == "__main__":
    raise SystemExit(main())
