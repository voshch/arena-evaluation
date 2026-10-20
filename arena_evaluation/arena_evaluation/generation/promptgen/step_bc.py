"""Step 3 - Type B (tune some pedestrians) and Type C (one pedestrian against the robot)."""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import attrs

from ..scenario.populate import MODELS, for_people, speed_for
from .agents import TypeFiles, agent_type_of, mesh_for, profile_of_agent
from .lexicon import when_in
from .llm import AnswerError, Oracle, call
from .profiles import BT_ONLY, TUNABLE, ProfileError, is_behaviour_tree, profile, scaled_speed, tier_value
from .spec import A_CASES, BDecision, CDecision, Record, SchemaError, Selection, conflict_m, parse_b, parse_c
from .step_a import StepError, bake, set_type, tune_trees, walk_as_tree, walks
from .step_d import population_summary
from .world import WorldContext, WorldError

ERRATIC = "erratic"
ERRATIC_STOPS = 6
# a wanderer starting this close to the route meets the robot; fewer than ERRATIC_NEAR_MIN such are topped up beside it
ERRATIC_NEAR_M = 4.0
ERRATIC_NEAR_MIN = 3
_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))

_SYSTEM = Path(__file__).with_name("system_bc.md")


def system_prompt() -> str:
    return _SYSTEM.read_text()


def user_prompt(prompt: str, doc: Mapping[str, Any], world: WorldContext, record: Record, *, taken_fractions: Sequence[float], who: str = "") -> str:
    # Stable-prefix ordering (see step_d.user_prompt): world first, variable material last.
    parts = [world.context_for_model(), population_summary(doc, world, record.owned)]
    if who:
        parts.append(who)
    parts.append(f"prompt: {prompt}")
    decided = {k: v for k, v in record.decisions().items() if v and k in ("A", "D")}
    if decided:
        parts.append(f"already decided: {decided}")
    if taken_fractions:
        parts.append("route fractions already used by a placement (keep C at least 2 m away): " + ", ".join(f"{f:.2f}" for f in taken_fractions))
    return "\n\n".join(parts)


def decide(prompt: str, doc: Mapping[str, Any], world: WorldContext, record: Record, *, taken_fractions: Sequence[float], cache_dir: Path | None, model: str | None, oracle: Oracle | None, who: str = "") -> tuple[BDecision | None, CDecision | None]:
    system = system_prompt()
    user = user_prompt(prompt, doc, world, record, taken_fractions=taken_fractions, who=who)
    complaint = ""
    for attempt in range(2):
        answer = call(system, user + (f"\n\nYour previous answer was rejected: {complaint}. Answer again." if complaint else ""), cache_dir=cache_dir, model=model, oracle=oracle)
        record.provenance.setdefault("BC", []).append({"model": answer.model, "response_sha256": answer.sha256, "cached": answer.cached, "attempt": attempt})
        try:
            raw = answer.json()
            b = parse_b(raw.get("B"))
            if b is not None and not (isinstance(raw.get("B"), Mapping) and raw["B"].get("when")):
                # The model gave no onset: the prompt's own time phrase is the onset (the A
                # sentence keeps its own; C durations are "for ..." phrases, not onsets).
                spoken = when_in(prompt, exclude_cases=tuple(A_CASES), lexicon=None)
                if spoken is not None:
                    b = attrs.evolve(b, when=spoken)
                    record.warnings.append(f"B onset read from the prompt: {spoken.as_dict()}")
            c = parse_c(raw.get("C"))
            if c is not None:
                gap = conflict_m(world.route_length)
                for f in taken_fractions:
                    if abs(f - c.route_fraction) * world.route_length < gap:
                        if oracle is None:
                            raise SchemaError(f"conflict: C.route_fraction {c.route_fraction:g} is within {gap:g} m of a Type D placement at {f:.2f}")
                        # The oracle cannot answer a complaint: move C away from the placement along the route.
                        shift = (gap + 0.3) / max(world.route_length, 1e-9)
                        nudged = f + shift if c.route_fraction >= f else f - shift
                        if not 0.1 <= nudged <= 0.9:
                            nudged = f - shift if c.route_fraction >= f else f + shift
                        c = attrs.evolve(c, route_fraction=round(min(0.9, max(0.1, nudged)), 3))
                        record.warnings.append(f"C moved to route fraction {c.route_fraction:g} to clear the Type D placement at {f:.2f}")
            return b, c
        except (AnswerError, SchemaError) as exc:
            complaint = str(exc)
    raise StepError(f"schema: {complaint}")


def _agents(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [a for a in (doc.get("dynamic") or []) if isinstance(a, dict)]


def _zone_of_agent(agent: Mapping[str, Any], world: WorldContext) -> str:
    pose = agent.get("pose")
    if isinstance(pose, str):
        return pose
    if isinstance(pose, (list, tuple)) and len(pose) >= 2:
        return world.zone_of((float(pose[0]), float(pose[1]))) or ""
    return ""


def select(doc: Mapping[str, Any], sel: Selection, world: WorldContext, owned: set[str], files: TypeFiles, *, walking_only: bool = False) -> list[dict[str, Any]]:
    """The agents a Type B selection names, in scenario order, never one another step owns; with `walking_only`, only agents that walk."""
    pool = [a for a in _agents(doc) if str(a.get("name")) not in owned and (not walking_only or walks(a, files))]
    if sel.names:
        by_name = {str(a.get("name")): a for a in pool}
        missing = [n for n in sel.names if n not in by_name]
        if missing:
            raise StepError(f"applicability: {missing} are not unowned {'walking ' if walking_only else ''}agents of this scenario")
        return [by_name[n] for n in sel.names]
    if sel.profile:
        pool = [a for a in pool if (p := profile_of_agent(a)) is not None and p.name == sel.profile]
    if sel.zone:
        pool = [a for a in pool if _zone_of_agent(a, world) == sel.zone]
    if sel.amount == "half":
        pool = pool[: max(1, math.ceil(len(pool) / 2))]
    elif sel.amount == "all_but_one":
        pool = pool[:-1]
    if sel.n:
        pool = pool[: sel.n]
    if not pool:
        raise StepError(f"applicability: selection {sel.as_dict()} matched nobody among the unowned {'walking ' if walking_only else ''}agents")
    return pool


def _home(agent: Mapping[str, Any], world: WorldContext) -> tuple[float, float]:
    pose = agent.get("pose")
    return world.zone_sample(pose, random.Random(f"{world.name}:{agent['name']}:{ERRATIC}")) if isinstance(pose, str) else (float(pose[0]), float(pose[1]))


def _beside_route(world: WorldContext, fraction: float, side: float) -> tuple[float, float]:
    (cx, cy), (hx, hy) = world.point_at(fraction)
    return cx - hy * side, cy + hx * side


def wanderers_beside_route(doc: dict[str, Any], world: WorldContext, n: int) -> list[str]:
    """Up to `n` pedestrians standing beside the middle of the route, for an erratic route whose base has too few there."""
    rng = random.Random(f"{world.name}:wander")
    taken = {str(a.get("name")) for a in doc.get("dynamic") or [] if isinstance(a, Mapping)}
    added: list[dict[str, Any]] = []
    for i in range(4 * n):
        if len(added) == n:
            break
        fraction = 0.2 + 0.6 * ((i * 0.618) % 1.0)
        found = next((r[0] for side in (2.0, -2.0, 3.0, -3.0) if (r := _resolved(world, _beside_route(world, fraction, side))) is not None), None)
        if found is None:
            continue
        name = next(f"wander_{j}" for j in itertools.count(1) if f"wander_{j}" not in taken)
        taken.add(name)
        spot = [round(found[0], 2), round(found[1], 2), 0.0]
        added.append({"name": name, "model": MODELS[len(added) % len(MODELS)], "pose": spot, "agent": {"agent_type": "adult", "desired_velocity": speed_for("adult", rng)}, "waypoints": [spot]})
    doc["dynamic"] = [*doc.get("dynamic", []), *added]
    return [str(a["name"]) for a in added]


def _resolved(world: WorldContext, pt: tuple[float, float]) -> tuple[tuple[float, float], str] | None:
    try:
        return world.resolve(pt, within=1.5)
    except WorldError:
        return None


def erratic_trees(doc: dict[str, Any], files: TypeFiles, world: WorldContext, names: Sequence[str], *, period: float, radius: float, around: str, label: str, when: dict[str, Any]) -> list[dict[str, Any]]:
    """On a stimulus each agent switches to its own looped `erratic` sequence, each point given up after `period`, as the
    runtime `shuffle` reroutes a walker: points up to `radius` either side of the route's middle, or `radius` around where
    it starts. A reroute from outside would break a tree."""
    walk_as_tree(doc, files, world, names)
    agents = [a for a in doc.get("dynamic") or [] if isinstance(a, dict) and str(a["name"]) in names]
    for i, agent in enumerate(agents):
        home = _home(agent, world)
        steps: dict[str, Any] = {}
        for k in range(ERRATIC_STOPS):
            angle = (k + i) * _GOLDEN_ANGLE
            if around == "route":
                # clear of the start and the goal, so the robot can still leave and arrive
                target = _beside_route(world, 0.15 + 0.7 * ((k * 0.618 + i * 0.29) % 1.0), radius * math.sin(angle))
            else:
                target = (home[0] + radius * math.cos(angle), home[1] + radius * math.sin(angle))
            try:
                (x, y), _why = world.resolve(target, radius=0.3, on_route=True, within=1.0)
            except WorldError:
                continue
            steps[f"to_{len(steps) + 1}"] = {"kind": "go_to", "target_pose": {"x": round(x, 2), "y": round(y, 2), "theta": 0.0}, "patience": {"mean": period}}
        if len(steps) < 2:
            raise StepError(f"{label}: no room around {agent['name']}'s start for an erratic route")
        ref = agent_type_of(agent)
        shared = ref not in files.transient
        set_type(agent, files.switching(ref, need=ERRATIC, to=ERRATIC, steps=steps, tag=str(agent["name"]) if shared else "", retry=True))
    return [{"type": "stimulus", "label": label, "scope": list(names), "need": ERRATIC, "when": dict(when)}]


def apply_b(doc: dict[str, Any], decision: BDecision, world: WorldContext, files: TypeFiles, *, owned: set[str]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    tops_up = decision.case == "erratic_route" and decision.around == "route"
    try:
        # a tuned gait or gaze shows only on someone walking
        chosen = select(doc, decision.select, world, owned, files, walking_only=decision.case == "intrinsic")
    except StepError:
        if not tops_up:
            raise
        chosen = []
    names = [str(a["name"]) for a in chosen]
    notes: list[str] = []
    effects: list[dict[str, Any]] = []
    label = f"B:{decision.case}"

    if decision.case == "erratic_route":
        if tops_up and (near := sum(world.route_clearance(_home(a, world)) <= ERRATIC_NEAR_M for a in chosen)) < ERRATIC_NEAR_MIN:
            added = wanderers_beside_route(doc, world, ERRATIC_NEAR_MIN - near)
            names += added
            notes.append(f"erratic_route: {near} selected agent(s) start near the route, added {len(added)} beside it")
        if not names:
            raise StepError(f"applicability: selection {decision.select.as_dict()} matched nobody and no wanderer fits beside the route")
        effects.extend(erratic_trees(doc, files, world, names, period=decision.period, radius=decision.radius, around=decision.around, label=label, when=decision.when.as_effect()))
        notes.append(f"erratic_route: {len(names)} agent(s) switch to a new point around the {decision.around} every {decision.period:g} s")
        return effects, names, notes

    if decision.case == "erratic_behaviour":
        bt, plain = [], []
        for agent in chosen:
            (bt if is_behaviour_tree(files.base_doc(agent_type_of(agent))) else plain).append(agent)
        if bt:
            bake(doc, {"need_decay": "high"}, files, names=[str(a["name"]) for a in bt])
        if plain:
            effects.extend(erratic_trees(doc, files, world, [str(a["name"]) for a in plain], period=decision.period, radius=decision.radius, around=decision.around, label=label, when=decision.when.as_effect()))
            notes.append(f"erratic_behaviour: {len(plain)} plain walker(s) fell back to erratic_route")
        if bt:
            notes.append(f"erratic_behaviour: {len(bt)} behaviour-tree agent(s) with need_decay high from the start")
        return effects, names, notes

    params = dict(decision.params)
    # One change per agent type at the onset: until then the selected agents walk as the base
    # scenario wrote them. A behaviour-only parameter is dropped for plain walkers.
    by_usable: dict[tuple[tuple[str, str], ...], list[str]] = {}
    for agent in chosen:
        ref = agent_type_of(agent)
        try:
            base = files.base_doc(ref)
        except ProfileError as exc:
            raise StepError(f"intrinsic: {exc}") from None
        usable = {k: v for k, v in params.items() if k not in BT_ONLY or is_behaviour_tree(base)}
        dropped = sorted(set(params) - set(usable))
        if dropped:
            notes.append(f"intrinsic: {dropped} skipped on {agent['name']} (not a behaviour-tree type)")
        if usable:
            by_usable.setdefault(tuple(sorted(usable.items())), []).append(str(agent["name"]))
    for key, members in by_usable.items():
        usable = dict(key)
        # radius, need decay and duration have no `tune:`, so they hold from the start
        fixed = {k: v for k, v in usable.items() if k not in TUNABLE}
        if fixed:
            try:
                bake(doc, fixed, files, names=members)
            except StepError as exc:
                raise StepError(f"intrinsic: {exc}") from None
            notes.append(f"intrinsic: {fixed} on {members} from the start")
        tunable = {k: v for k, v in usable.items() if k in TUNABLE}
        if tunable:
            # every agent holds the tune itself while the need is up, a walker on its own walking tree
            walk_as_tree(doc, files, world, members)
            if tunable.get("speed") == "high":
                tunable["max_speed"] = "high"  # the engine's cap, or a high speed is a brisk walk
            effects.extend(tune_trees(doc, tunable, files, need=decision.case, label=label, when=decision.when.as_effect(), names=members))
    notes.append(f"intrinsic: {params} on {len(names)} agent(s) at {decision.when.as_dict()}")
    return effects, names, notes


def apply_c(decision: CDecision, world: WorldContext, files: TypeFiles, *, seed: int, intercept_index: int, taken_fractions: Sequence[float]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    rng = random.Random(f"{world.name}:C:{seed}")
    gap = conflict_m(world.route_length)
    for f in taken_fractions:
        if abs(f - decision.route_fraction) * world.route_length < gap:
            raise StepError(f"conflict: route fraction {decision.route_fraction:g} is within {gap:g} m of a Type D placement at {f:.2f}")
    prof = profile(decision.profile)
    ref = files.derived(prof.base, {}, prof=prof)
    label = f"C:{decision.case}"
    effect: dict[str, Any] = {"type": "intercept", "label": label, "profile": ref, "model": mesh_for(prof, rng), "route_fraction": decision.route_fraction}
    if decision.case == "blocking":
        if decision.mode == "stand":
            effect.update({"after": "stand", "duration": decision.duration, "speed": scaled_speed(prof, "mid")})
            note = f"blocking/stand: a {prof.name} stands on the route at {decision.route_fraction:g} for {decision.duration:g} s"
        else:
            effect.update({"angle": 0.0, "speed": tier_value("speed", "low", prof=prof), "after": "continue", "lead_out": 6.0, "waypoint_mode": "once"})
            note = f"blocking/lead: a {prof.name} crawls ahead of the robot from {decision.route_fraction:g}"
    else:
        effect.update({"angle": decision.angle, "speed": scaled_speed(prof, decision.speed), "after": decision.after, "duration": decision.duration, "offset": decision.offset, "waypoint_mode": "once"})
        if decision.after == "veer":
            if decision.veer_to not in world.polygons:
                raise StepError(f"approach/veer: {decision.veer_to!r} is not a zone of this world")
            if not for_people([decision.veer_to]):
                raise StepError(f"approach/veer: {decision.veer_to!r} is a room nobody walks into (wc/storage)")
            effect["veer_to"] = decision.veer_to
        note = f"approach: a {prof.name} at {decision.angle:g} deg, fraction {decision.route_fraction:g}, then {decision.after}"
    return [effect], [f"edge_{intercept_index}"], [note]
