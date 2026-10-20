"""Step 2 - Type D, pedestrian interactions. An unbroken formation is cast as walking trees a `stimulus` switches to their spots,
since HumanSim cannot retune a running walker; a breaking one keeps plain walkers rewritten by `rally` and `scatter`."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..scenario.populate import for_people
from .agents import TypeFiles, agent_type_of, clone, entry, mesh_for, unique_names
from .llm import AnswerError, Oracle, call
from .profiles import profile, scaled_speed, tune_block
from .spec import DDecision, Record, SchemaError, parse_d
from .step_a import StepError, set_type, walk_as_tree
from .world import WorldContext, WorldError

Point = tuple[float, float]
FORM = "form"

LINE_SPACING = 0.6
QUEUE_SPACING = 0.7
MIN_ROUTE_FOR_FORMATION_M = 4.0
FORM_AT_S = 4.0
FORMATION_RADIUS_M = 1.8
# too tight for the robot to pass between two members; 1.5 m let it drive through
CIRCLE_SPACING_M = 0.9
HUDDLE_RADIUS_M = 0.35


def circle_radius(n: int) -> float:
    return max(HUDDLE_RADIUS_M, CIRCLE_SPACING_M / (2.0 * math.sin(math.pi / max(2, n))))


_SYSTEM = Path(__file__).with_name("system_d.md")


def system_prompt() -> str:
    return _SYSTEM.read_text()


def population_summary(doc: Mapping[str, Any], world: WorldContext, owned: Mapping[str, Sequence[str]]) -> str:
    taken = {n for names in owned.values() for n in names}
    lines = []
    for agent in doc.get("dynamic") or []:
        if not isinstance(agent, Mapping):
            continue
        pose = agent.get("pose")
        where = pose if isinstance(pose, str) else (world.zone_of((float(pose[0]), float(pose[1]))) if isinstance(pose, (list, tuple)) else "?")
        kind = (agent.get("agent") or {}).get("agent_type", "adult") if isinstance(agent.get("agent"), Mapping) else agent.get("agent")
        flag = " (owned)" if str(agent.get("name")) in taken else ""
        lines.append(f"  - {agent.get('name')}: {kind} in {where}{flag}")
    return "population:\n" + ("\n".join(lines) if lines else "  (nobody)")


def user_prompt(prompt: str, doc: Mapping[str, Any], world: WorldContext, record: Record, *, who: str = "") -> str:
    # stable material first, so the provider's implicit prefix cache applies
    parts = [world.context_for_model(), population_summary(doc, world, record.owned)]
    if who:
        parts.append(who)
    parts.append(f"prompt: {prompt}")
    if record.a:
        parts.append(f"already decided by Step A: {record.a.as_dict()}")
    return "\n\n".join(parts)


def decide(prompt: str, doc: Mapping[str, Any], world: WorldContext, record: Record, *, cache_dir: Path | None, model: str | None, oracle: Oracle | None, who: str = "") -> DDecision | None:
    """Ask the model (or the oracle) for the Type D decision; one retry on a schema error."""
    system = system_prompt()
    user = user_prompt(prompt, doc, world, record, who=who)
    complaint = ""
    for attempt in range(2):
        answer = call(system, user + (f"\n\nYour previous answer was rejected: {complaint}. Answer again." if complaint else ""), cache_dir=cache_dir, model=model, oracle=oracle)
        record.provenance.setdefault("D", []).append({"model": answer.model, "response_sha256": answer.sha256, "cached": answer.cached, "attempt": attempt})
        try:
            raw = answer.json()
            if str(raw.get("case", "")).lower() == "none":
                return None
            return parse_d(raw)
        except (AnswerError, SchemaError) as exc:
            complaint = str(exc)
    raise StepError(f"schema: {complaint}")


def _place(decision: DDecision, world: WorldContext) -> tuple[Point, Point, bool]:
    if decision.place.get("route_fraction") is not None:
        pt, heading = world.point_at(float(decision.place["route_fraction"]))
        return pt, heading, True
    zone = str(decision.place["zone"])
    if not for_people([zone]):
        raise StepError(f"placement: {zone!r} is a room nobody is placed in (wc/storage; sealed in the inflated grid)")
    pt = world.zone_point(zone, need=circle_radius(len(decision.members)))
    _rp, heading = world.point_at(world.fraction_of(pt))
    return pt, heading, world.route_clearance(pt) < 1.5


def _circle(centre: Point, n: int, r: float | None = None) -> list[Point]:
    r = circle_radius(n) if r is None else r
    return [(centre[0] + r * math.cos(2 * math.pi * i / n), centre[1] + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


def _circle_radius(world: WorldContext, centre: Point, heading: Point, n: int) -> float:
    want = circle_radius(n)
    if world.grid is None:
        return want
    width, _l, _r = world.passage_width(centre, heading)
    return max(HUDDLE_RADIUS_M, min(want, width / 2 - 0.35))


def _take_reuse(doc: dict[str, Any], names: Sequence[str], owned: set[str]) -> list[dict[str, Any]]:
    wanted = [n for n in names if n not in owned]
    kept, taken = [], []
    for agent in doc.get("dynamic") or []:
        (taken if isinstance(agent, dict) and str(agent.get("name")) in wanted else kept).append(agent)
    doc["dynamic"] = kept
    return [clone(a) for a in taken]


def apply(doc: dict[str, Any], decision: DDecision, world: WorldContext, files: TypeFiles, *, seed: int, owned: set[str], intercept_index: int) -> tuple[list[dict[str, Any]], list[str], list[str], dict[str, Any]]:
    """Realise a Type D decision in place; `info` carries the formation centre and radius for the success criterion."""
    rng = random.Random(f"{world.name}:D:{seed}")
    notes: list[str] = []
    label = f"D:{decision.case}"
    info: dict[str, Any] = {}

    if decision.case == "stubborn":
        prof = profile(decision.profile)
        ref = files.derived(prof.base, {"repulsion": "low"}, prof=prof)
        effect = {
            "type": "intercept",
            "label": label,
            "profile": ref,
            "model": mesh_for(prof, rng),
            "angle": 180.0,
            "route_fraction": decision.route_fraction,
            "speed": scaled_speed(prof, decision.speed),
            "after": "continue",
            "lead_out": 6.0,
            "waypoint_mode": "once",
        }
        notes.append(f"stubborn: a {prof.name} walks head-on at route fraction {decision.route_fraction:g}, not yielding")
        return [effect], [f"edge_{intercept_index}"], notes, info

    if decision.place.get("route_fraction") is not None and world.route_length < MIN_ROUTE_FOR_FORMATION_M:
        raise StepError(f"applicability: the robot's first leg is {world.route_length:.1f} m, too short to place a formation on (needs {MIN_ROUTE_FOR_FORMATION_M:g} m)")
    try:
        centre, heading, on_route = _place(decision, world)
    except WorldError as exc:
        raise StepError(f"placement: {exc}") from None
    if on_route and decision.variant != "blocking":
        need = circle_radius(len(decision.members)) if decision.variant != "line" else QUEUE_SPACING * len(decision.members) / 2
        fit = world.fit_along_route(world.fraction_of(centre), need) or world.fit_along_route(world.fraction_of(centre), HUDDLE_RADIUS_M)
        if fit is not None:
            fitted, heading, fraction = fit
            if math.dist(fitted, centre) > 0.3:
                notes.append(f"placed {math.dist(fitted, centre):.1f} m along the route from the requested point, where the formation fits (fraction {fraction:.2f})")
            centre = fitted
        else:
            room = world.zone_of(centre) or world.exit_zone()
            if room is None:
                raise StepError("placement: no room to stand the formation in")
            try:
                centre = world.zone_point(room, need=need)
            except WorldError as exc:
                raise StepError(f"placement: {exc}") from None
            on_route = world.route_clearance(centre) < 1.5
            notes.append(f"no stretch of the {world.route_length:.1f} m route fits the formation; placed {'on' if on_route else 'beside'} it in {room}")

    reused = _take_reuse(doc, decision.reuse, owned) if decision.reuse else []
    profiles = [profile(m) for m in decision.members]
    variant = decision.variant

    if decision.case == "interaction_unbroken" and variant == "blocking":
        pt, heading, width = world.narrowest_passage(world.fraction_of(centre)) if on_route else (centre, heading, world.passage_width(centre, heading)[0])
        n = min(8, max(2, int(round(width / LINE_SPACING))))
        if width > n * LINE_SPACING + 0.8:
            notes.append(f"blocking: passage is {width:.1f} m wide; {n} members leave a gap")
        profiles = (profiles * 8)[:n]
        normal = (-heading[1], heading[0])
        points = [(pt[0] + normal[0] * (i - (n - 1) / 2) * LINE_SPACING, pt[1] + normal[1] * (i - (n - 1) / 2) * LINE_SPACING) for i in range(n)]
        centre = pt
    elif decision.case == "interaction_unbroken" and variant == "meet":
        points = _circle(centre, len(profiles), _circle_radius(world, centre, heading, len(profiles)))
    elif decision.case == "formation_break" and variant == "line":
        points = [(centre[0] - heading[0] * i * QUEUE_SPACING, centre[1] - heading[1] * i * QUEUE_SPACING) for i in range(len(profiles))]
    else:
        points = _circle(centre, len(profiles), _circle_radius(world, centre, heading, len(profiles)))

    # a reused agent keeps its slot through the drops below; one without a slot goes back to the scenario
    slots = [reused[i] if i < len(reused) else None for i in range(len(profiles))]
    resolved: list[Point] = []
    kept_profiles: list[Any] = []
    kept_slots: list[dict[str, Any] | None] = []
    for p, prof_i, slot in zip(points, profiles, slots, strict=True):
        radius = 0.2 if variant == "blocking" else 0.3
        try:
            q, why = world.resolve(p, radius=radius, on_route=on_route or variant == "blocking", within=1.0)
        except WorldError as exc:
            notes.append(f"dropped a member at ({p[0]:.2f}, {p[1]:.2f}): {exc}")
            continue
        if why:
            notes.append(f"member {why}")
        resolved.append(q)
        kept_profiles.append(prof_i)
        kept_slots.append(slot)
    if len(resolved) < 2:
        raise StepError(f"placement: fewer than two members could stand at ({centre[0]:.2f}, {centre[1]:.2f})")
    profiles = kept_profiles

    doc["dynamic"] = [*doc.get("dynamic", []), *(a for a in reused if all(a is not s for s in kept_slots))]
    fresh = iter(unique_names(doc, "d", sum(s is None for s in kept_slots)))
    names = [str(s["name"]) if s is not None else next(fresh) for s in kept_slots]
    here = world.zone_of(centre) or ""
    elsewhere = world.other_zones(exclude=[here])
    homes = [z for z in elsewhere if z in world.homes] or elsewhere or world.other_zones()
    if not homes:
        raise StepError("placement: no zone for the members to walk in from")
    members: list[dict[str, Any]] = []
    for i, (prof, pt) in enumerate(zip(profiles, resolved, strict=True)):
        home = homes[i % len(homes)]
        if decision.case == "interaction_unbroken" and variant == "meet":
            member = entry(names[i], prof, home, [home, pt], speed=scaled_speed(prof, "mid"), mode="once", agent_type=prof.base, rng=rng)
        else:
            away = [z for z in world.other_zones(exclude=[home]) if z != here] or [home]
            member = entry(names[i], prof, home, [home, away[(i + 1) % len(away)]], speed=scaled_speed(prof, "mid"), mode="reverse", agent_type=prof.base, rng=rng)
        if kept_slots[i] is not None:
            member["model"] = kept_slots[i].get("model", member["model"])
        members.append(member)
    doc["dynamic"] = [*doc.get("dynamic", []), *members]

    # sorted-name order is how the runtime deals targets
    spot = {name: pt for name, pt in zip(names, resolved, strict=True)}
    ordered = sorted(names)
    targets = [[round(spot[n][0], 2), round(spot[n][1], 2)] for n in ordered]
    info["formation"] = {"centre": [round(centre[0], 2), round(centre[1], 2)], "radius": FORMATION_RADIUS_M, "variant": variant}
    effects: list[dict[str, Any]] = []
    if not (decision.case == "interaction_unbroken" and variant == "meet"):
        gated = decision.when.robot_within is not None or decision.when.route_fraction is not None
        if decision.case == "formation_break" or not gated:
            form_when: dict[str, Any] = {"at": FORM_AT_S if decision.case == "formation_break" else max(FORM_AT_S, decision.when.at or 0.0)}
        else:
            form_when = decision.when.as_effect()
            if "robot_within" in form_when:
                form_when["of"] = info["formation"]["centre"]
        if decision.case == "interaction_unbroken":
            walk_as_tree(doc, files, world, ordered)
            for member, prof in zip(members, profiles, strict=True):
                x, y = spot[str(member["name"])]
                face = (-heading[0], -heading[1]) if variant == "blocking" else (centre[0] - x, centre[1] - y)
                steps = {"form": {"kind": "go_to", "target_pose": {"x": round(x, 2), "y": round(y, 2), "theta": round(math.atan2(face[1], face[0]), 3)}, "patience": {"mean": 300.0}}, "stand": {"duration": {"mean": 600.0}, "patience": {"mean": 900.0}}}
                set_type(member, files.switching(agent_type_of(member), need=FORM, to="form", steps=steps, tune=tune_block({"repulsion": "low"}, prof=prof)))
            effects.append({"type": "stimulus", "label": f"{label}:form", "scope": ordered, "need": FORM, "when": form_when})
        else:
            effects.append({"type": "rally", "label": f"{label}:form", "scope": ordered, "targets": targets, "when": form_when})
        info["formation"]["at"] = form_when
    if decision.case == "formation_break":
        when = decision.when.as_effect()
        if "robot_within" in when:
            when["of"] = info["formation"]["centre"]
            if not on_route:
                # the robot passes a formation beside the route at its clearance, never within the gate
                offset = world.route_clearance(centre)
                if offset + 1.0 > float(when["robot_within"]):
                    when["robot_within"] = round(offset + 1.0, 2)
                    notes.append(f"trigger widened to {when['robot_within']} m: the formation stands {offset:.1f} m off the route")
        elif "at" in when:
            when["at"] = max(float(when["at"]), FORM_AT_S + 3.0)
        if decision.break_kind == "scatter":
            effects.append({"type": "scatter", "label": label, "scope": ordered, "radius": decision.radius, "when": when})
        else:
            away_zones = [z for z in world.other_zones(exclude=[world.zone_of(centre) or ""])][: len(names)] or [z for z in (world.exit_zone(),) if z]
            if not away_zones:
                raise StepError("formation_break: no zone to disperse to")
            effects.append({"type": "rally", "label": label, "scope": ordered, "targets": away_zones, "when": when})
        notes.append(f"formation_break: {len(names)} walk, form a {variant} at {form_when}, then {decision.break_kind} {when}")
    else:
        notes.append(f"interaction_unbroken/{variant}: {len(names)} member(s) {'walk in and meet' if variant == 'meet' else 'walk, then form'} at ({centre[0]:.1f}, {centre[1]:.1f}), {'on' if on_route else 'off'} the route")
    return effects, names, notes, info
